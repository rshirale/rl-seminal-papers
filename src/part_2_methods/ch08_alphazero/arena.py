"""Measuring whether self-play is actually producing a stronger player.

Training loss falls whether or not the agent is learning anything useful, so
the loop needs an outside opinion. Three give it here, and they are not
equally trustworthy -- which is the part worth reading.

  * a random mover, which the agent should beat almost always within a couple
    of iterations, and which stops being informative immediately afterwards;
  * a depth-3 exact search, a real if modest bar;
  * the agent against itself, greedy and with the search noise off.

The third is the one to trust on a board whose proven value is a draw. The
value head's opinion of the empty board is the obvious metric and the weaker
one: it averages over self-play in which both sides are still imperfect, and
on a drawn board "near zero" is also what an untrained tanh head emits, so it
cannot distinguish a converged agent from an ignorant one.
"""

import numpy as np

from .solver import best_action, disc_difference


def _random_action(game, board, rng):
    legal = np.nonzero(game.legal_moves(board))[0]
    return int(rng.choice(legal))


def _agent_actions(game, mcts, boards, simulations, rng):
    policies, _ = mcts.search(boards, simulations, add_noise=False, rng=rng)
    return [int(np.argmax(p)) for p in policies]


def play_match(game, mcts, simulations, opponent, num_games, rng,
               agent_first=True):
    """Play num_games in lockstep and return (wins, draws, losses)."""
    boards = [game.initial_board() for _ in range(num_games)]
    scores = [None] * num_games
    live = list(range(num_games))
    # Ply parity decides whose turn it is; boards are always canonical.
    agent_to_move = agent_first
    ply = 0
    max_plies = game.board_shape[0] * game.board_shape[1] + 2

    while live and ply < max_plies:
        if agent_to_move:
            actions = _agent_actions(
                game, mcts, [boards[i] for i in live], simulations, rng)
        else:
            actions = [opponent(game, boards[i], rng) for i in live]

        still_live = []
        for slot, index in enumerate(live):
            boards[index] = game.apply_move(boards[index], actions[slot])
            result = game.outcome(boards[index])
            if result is None:
                still_live.append(index)
            else:
                # result belongs to the player to move now, who is the one that
                # did not just play.
                mover_score = -result
                scores[index] = mover_score if agent_to_move else -mover_score
        live = still_live
        agent_to_move = not agent_to_move
        ply += 1

    for index in live:
        scores[index] = 0.0

    wins = sum(1 for s in scores if s > 0)
    draws = sum(1 for s in scores if s == 0)
    losses = sum(1 for s in scores if s < 0)
    return wins, draws, losses


def mirror_match(game, mcts, simulations, num_games, rng, opening_plies=4):
    """The agent against itself, with the search noise off.

    On a board whose game-theoretic value is a draw, this is the sharpest
    signal the loop has. A weak agent blunders and produces decisive games; an
    agent approaching perfect play draws with itself every time, so the number
    climbs from 0 toward 1 and nothing about it can be faked by a network that
    has learned to output zero.

    The first few plies are sampled rather than taken greedily. Without that,
    every game in the batch is the same deterministic game played over and
    over, and the draw rate can only ever be exactly 0.0 or exactly 1.0 -- a
    coin flip dressed up as a measurement. The first version of this function
    had that bug and reported 1.0, 0.0, 1.0, 0.0, 0.0 across a run, which is
    what finally gave it away.
    """
    boards = [game.initial_board() for _ in range(num_games)]
    scores = [None] * num_games
    live = list(range(num_games))
    ply = 0
    max_plies = game.board_shape[0] * game.board_shape[1] + 2

    while live and ply < max_plies:
        policies, _ = mcts.search([boards[i] for i in live], simulations,
                                  add_noise=False, rng=rng)
        still_live = []
        for slot, index in enumerate(live):
            policy = policies[slot]
            if ply < opening_plies:
                action = int(rng.choice(len(policy), p=policy))
            else:
                action = int(np.argmax(policy))
            boards[index] = game.apply_move(boards[index], action)
            result = game.outcome(boards[index])
            if result is None:
                still_live.append(index)
            else:
                scores[index] = result
        live = still_live
        ply += 1

    for index in live:
        scores[index] = 0.0
    draws = sum(1 for s in scores if s == 0)
    return draws / len(scores)


def evaluate_agent(game, mcts, simulations, num_games, rng, depth=3):
    """Score the current network against random play, shallow exact search,
    and itself. Returned as a dict the training loop folds into its log."""
    half = max(1, num_games // 2)
    heuristic = disc_difference if game.name.startswith("othello") else None

    def shallow(g, board, r):
        return best_action(g, board, depth, heuristic=heuristic, rng=r)

    report = {}
    for label, opponent in (("random", _random_action), ("depth3", shallow)):
        totals = np.zeros(3, dtype=int)
        # Both colours, because a player that is only good going first is a
        # player that has learned the opening and nothing else.
        for agent_first in (True, False):
            totals += play_match(game, mcts, simulations, opponent, half,
                                 rng, agent_first=agent_first)
        wins, draws, losses = totals
        played = wins + draws + losses
        report[f"vs_{label}"] = f"{wins}/{draws}/{losses}"
        report[f"score_{label}"] = round((wins + 0.5 * draws) / played, 3)

    report["mirror_draw_rate"] = round(
        mirror_match(game, mcts, simulations, max(4, num_games // 4), rng), 3)
    return report
