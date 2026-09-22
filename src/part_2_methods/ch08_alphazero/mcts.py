"""PUCT search, evaluated by the network rather than by rollouts.

This is the one structural difference between AlphaGo's search and
AlphaZero's. AlphaGo evaluated a leaf twice -- once with the value network and
once by playing a fast random game to the end -- and mixed the two with the
parameter lambda, because in 2016 the value network alone was not trustworthy
enough and the rollout policy was fifteen hundred times faster than the
network (``benchmark.py`` reproduces that gap). AlphaZero deletes the rollout.
The value head is the only evaluator, so a leaf costs exactly one forward
pass, and lambda stops existing.

Searches for several positions are driven in lockstep so their leaves can be
evaluated in a single batch. That is the same trick the chapter describes for
AlphaGo's distributed search, doing real work here instead of illustrating:
the network is far too slow to call one position at a time, so the search
collects pending leaves across the whole fleet and sends them through
together.
"""

import math

import numpy as np
import torch


class Node:
    """A position in the tree, whose board is built only when it is needed.

    Expanding a node creates one child per legal move, but a search visits
    only a few of them. Materialising every child's board up front is the
    single most expensive thing a naive implementation does, so children
    remember the move that makes them and compute the board on first visit.
    """

    __slots__ = ("prior", "visit_count", "value_sum", "children", "_board",
                 "terminal_value", "parent", "action")

    def __init__(self, board=None, prior=0.0, parent=None, action=None):
        self._board = board
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.terminal_value = None
        self.parent = parent
        self.action = action

    def board(self, game):
        if self._board is None:
            self._board = game.apply_move(self.parent.board(game), self.action)
        return self._board

    @property
    def expanded(self):
        return bool(self.children)

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def puct_score(parent, child, c_puct):
    """Q + U, the rule that decides where the search spends its next visit."""
    # Unvisited children score on their prior alone, which is what makes the
    # policy head's opinion the thing that shapes the tree's first look.
    # max(1, ...) is the cold-start guard: on the very first simulation the
    # parent has no visits, sqrt(0) is zero, and every child would score
    # exactly zero -- making the choice of opening move arbitrary rather than
    # prior-driven.
    exploration = c_puct * child.prior * math.sqrt(
        max(1, parent.visit_count)) / (1 + child.visit_count)
    if child.visit_count == 0:
        return exploration
    # A child's Q is from the opponent's point of view, so it negates.
    return -child.q_value() + exploration


def select_child(node, c_puct):
    return max(node.children.items(),
               key=lambda item: puct_score(node, item[1], c_puct))


class BatchedMCTS:
    def __init__(self, game, net, device, c_puct=1.5,
                 dirichlet_alpha=0.6, dirichlet_weight=0.25):
        self.game = game
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

    def _evaluate(self, boards):
        """One forward pass for every pending leaf in the batch."""
        if not boards:
            return [], []
        planes = np.stack([self.game.encode(b) for b in boards])
        tensor = torch.from_numpy(planes).to(self.device)
        with torch.no_grad():
            logits, values = self.net(tensor)
            priors = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return priors, values.float().cpu().numpy()

    def _expand(self, node, priors):
        mask = self.game.legal_moves(node.board(self.game))
        masked = priors * mask
        total = masked.sum()
        # A network that puts all its mass on illegal moves still has to
        # produce a distribution, so fall back to uniform over legal moves.
        # Early in training this happens often enough to matter.
        masked = masked / total if total > 1e-8 else mask / mask.sum()
        for action in np.nonzero(mask)[0]:
            action = int(action)
            node.children[action] = Node(prior=float(masked[action]),
                                         parent=node, action=action)

    def search(self, boards, simulations, add_noise=True, rng=None,
               trace=None, roots=None):
        """Visit-count distributions for every position in ``boards``.

        Returns ``(policies, roots)``. The roots come back so the caller can
        keep the subtree under the move it plays -- see ``selfplay.py``.

        Pass a list as ``trace`` and the search appends the root distribution
        after every simulation, which is what ``animate.py`` plays back.
        """
        rng = rng or np.random.default_rng()
        if roots is None:
            roots = [Node(board) for board in boards]

        unexpanded = [r for r in roots if not r.expanded]
        if unexpanded:
            priors, _ = self._evaluate([r.board(self.game) for r in unexpanded])
            for root, prior in zip(unexpanded, priors):
                self._expand(root, prior)

        if add_noise:
            for root in roots:
                self._add_dirichlet_noise(root, rng)

        for _ in range(simulations):
            paths, leaves, pending = [], [], []
            for root in roots:
                path, leaf = self._descend(root)
                paths.append(path)
                leaves.append(leaf)
                if leaf.terminal_value is None:
                    pending.append(leaf)

            # The batch: every game in the fleet contributed at most one leaf,
            # and they all cross the device boundary once.
            priors, values = self._evaluate(
                [leaf.board(self.game) for leaf in pending])
            lookup = {}
            for index, leaf in enumerate(pending):
                self._expand(leaf, priors[index])
                lookup[id(leaf)] = float(values[index])

            for path, leaf in zip(paths, leaves):
                value = (leaf.terminal_value if leaf.terminal_value is not None
                         else lookup[id(leaf)])
                self._backup(path, value)

            if trace is not None:
                trace.append([self._visit_distribution(r) for r in roots])

        return [self._visit_distribution(root) for root in roots], roots

    def _descend(self, root):
        node = root
        path = [node]
        while node.expanded and node.terminal_value is None:
            _, node = select_child(node, self.c_puct)
            path.append(node)
        if node.terminal_value is None:
            result = self.game.outcome(node.board(self.game))
            if result is not None:
                node.terminal_value = float(result)
        return path, node

    def _backup(self, path, value):
        # Walk back to the root, flipping the sign at every level because what
        # is good for one player is bad for the other.
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, root, rng):
        """Noise at the root keeps self-play from replaying one line forever.

        Applied to the root only, and only once per search. Adding it again to
        a reused subtree would compound it every ply.
        """
        actions = list(root.children)
        noise = rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, sample in zip(actions, noise):
            child = root.children[action]
            child.prior = ((1 - self.dirichlet_weight) * child.prior
                           + self.dirichlet_weight * sample)

    def _visit_distribution(self, root):
        counts = np.zeros(self.game.action_size, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        total = counts.sum()
        return counts / total if total else counts
