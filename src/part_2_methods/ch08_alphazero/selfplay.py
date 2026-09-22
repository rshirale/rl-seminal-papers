"""Self-play, run as a fleet rather than one game at a time.

A single game of Connect Four asks the network for a few thousand
evaluations, all of them one position wide. A laptop GPU answers a batch of
256 positions in about the same wall-clock time as a batch of one -- measured
at 17.5 ms either way on Apple MPS -- so playing one game at a time wastes
almost all of the hardware. Here a few hundred games advance in lockstep and
their searches share every forward pass.

The other saving is tree reuse. Once a move is played, the subtree under the
chosen child is already built and already carries visit statistics, so it
becomes the next root rather than being thrown away and re-expanded.
"""

import numpy as np


class Example:
    """One stored position: what the search concluded, and how it turned out.

    ``player_value`` stays None until the game ends, because the label is the
    result *as the player to move in this position saw it*, and nobody knows
    that until there is a result.
    """

    __slots__ = ("board", "policy", "player_value")

    def __init__(self, board, policy):
        self.board = board
        self.policy = policy
        self.player_value = None


def play_batch(game, mcts, num_games, simulations, temperature_moves=8,
               rng=None, max_plies=None):
    """Play num_games in lockstep and return (examples, results).

    Every position is stored with the visit-count distribution the search
    produced -- the policy target, which is better than the raw network
    because the search improved on it -- and once the game ends each position
    is labelled with the result as seen by whoever was to move there.
    """
    rng = rng or np.random.default_rng()
    max_plies = max_plies or (game.board_shape[0] * game.board_shape[1] + 2)

    boards = [game.initial_board() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    finished = [None] * num_games
    live = list(range(num_games))
    ply = 0
    roots = None

    while live and ply < max_plies:
        policies, search_roots = mcts.search(
            [boards[i] for i in live], simulations, add_noise=True, rng=rng,
            roots=roots)

        still_live = []
        next_roots = []
        for slot, index in enumerate(live):
            policy = policies[slot]
            histories[index].append(Example(boards[index].copy(), policy))

            action = _choose(policy, ply, temperature_moves, rng)
            boards[index] = game.apply_move(boards[index], action)

            result = game.outcome(boards[index])
            if result is None:
                still_live.append(index)
                # Tree reuse: the chosen child becomes the next root. Detaching
                # the parent lets the rest of the old tree be collected, which
                # on a fleet of a few hundred games is the difference between
                # steady memory and none left.
                new_root = search_roots[slot].children[action]
                new_root.parent = None
                next_roots.append(new_root)
            else:
                # result is from the point of view of the player to move in the
                # new position, which is the player who did not just move.
                finished[index] = float(result)

        live = still_live
        roots = next_roots if live else None
        ply += 1

    for index in live:
        # Anything still running at the ply cap is scored as a draw.
        finished[index] = 0.0

    examples = []
    for index in range(num_games):
        final = finished[index]
        history = histories[index]
        # `final` is the result seen by the player to move in the terminal
        # position -- the one who did NOT make the last move. Walking back from
        # there the sign flips at every ply, so each stored position ends up
        # labelled with the result as its own mover saw it. Get this backwards
        # and training still runs, the loss still falls, and the agent learns
        # to lose; test_self_play_labels_the_winner_positively exists for it.
        for offset, example in enumerate(reversed(history)):
            example.player_value = -final if offset % 2 == 0 else final
        for example in history:
            for board, policy in game.symmetries(example.board, example.policy):
                examples.append((game.encode(board), policy,
                                 np.float32(example.player_value)))
    return examples, finished


def _choose(policy, ply, temperature_moves, rng):
    if ply < temperature_moves:
        # Early moves are sampled, so the fleet explores different openings
        # instead of grinding the same line a few hundred times. This is
        # AlphaZero's temperature schedule, at its two extreme settings.
        return int(rng.choice(len(policy), p=policy))
    return int(np.argmax(policy))
