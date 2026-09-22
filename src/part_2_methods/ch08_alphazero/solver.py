"""An exact solver -- the oracle the trained agent is checked against.

Almost no reinforcement learning example has a correct answer to compare
against. This one does, because van den Herik, Uiterwijk and van Rijswijck
(2002) published the game-theoretic value of every small Connect Four board
and of 6x6 Othello, and because alpha-beta with a transposition table can
reprove the smallest of them here in seconds rather than taking them on
trust.

That oracle is what turns "the loss went down" into a real measurement. An
agent measured only by its training curve looks finished; an agent measured
against a proven draw is visibly nowhere near it.

Run it:

    python -m src.part_2_methods.ch08_alphazero.solver
"""

import argparse
import time

import numpy as np

EXACT, LOWER, UPPER = 0, 1, 2


def solve(game, board, alpha=-1.0, beta=1.0, table=None):
    """Exact value of a position for the player to move.

    The transposition table stores a flag alongside each value. Without it, a
    value produced under a narrow alpha-beta window would be cached as though
    it were exact, and the solver would quietly return wrong answers on later
    lookups -- which defeats the entire point of an oracle.

    Only tractable on the small boards. The standard 6x7 board has on the
    order of 10^14 positions and is far out of reach of pure Python.
    """
    if table is None:
        table = {}
    result = game.outcome(board)
    if result is not None:
        return float(result)

    key = game.key(board)
    cached = table.get(key)
    if cached is not None:
        value, flag = cached
        if flag == EXACT:
            return value
        if flag == LOWER and value >= beta:
            return value
        if flag == UPPER and value <= alpha:
            return value

    original_alpha = alpha
    best = -1.0
    for action in np.nonzero(game.legal_moves(board))[0]:
        child = game.apply_move(board, int(action))
        # The child is canonical for the opponent, so its value negates.
        value = -solve(game, child, -beta, -alpha, table)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    if best <= original_alpha:
        flag = UPPER
    elif best >= beta:
        flag = LOWER
    else:
        flag = EXACT
    table[key] = (best, flag)
    return best


def negamax(game, board, depth, alpha=-1.0, beta=1.0, heuristic=None):
    """Depth-limited search, used as a fixed-strength baseline opponent.

    Unlike ``solve`` this is not exact, which is the point: it is a bar the
    agent can be measured against long before it is anywhere near perfect.
    """
    result = game.outcome(board)
    if result is not None:
        return float(result)
    if depth == 0:
        return heuristic(board) if heuristic else 0.0

    best = -2.0
    for action in np.nonzero(game.legal_moves(board))[0]:
        child = game.apply_move(board, int(action))
        value = -negamax(game, child, depth - 1, -beta, -alpha, heuristic)
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def best_action(game, board, depth, heuristic=None, rng=None):
    """The move a depth-limited search prefers, ties broken at random.

    Ties are broken randomly on purpose. A deterministic baseline plays one
    game over and over, and a match against it measures a single line rather
    than a player.
    """
    scored = []
    for action in np.nonzero(game.legal_moves(board))[0]:
        child = game.apply_move(board, int(action))
        value = -negamax(game, child, depth - 1, -1.0, 1.0, heuristic)
        scored.append((value, int(action)))
    best = max(score for score, _ in scored)
    tied = [action for score, action in scored if score == best]
    if rng is not None:
        return int(rng.choice(tied))
    return tied[0]


def disc_difference(board):
    """A crude Othello heuristic: who holds more of the board, scaled to the
    [-1, 1] range the search works in."""
    mine = int((board == 1).sum())
    theirs = int((board == -1).sum())
    total = mine + theirs
    return (mine - theirs) / total if total else 0.0


#: Boards small enough to settle in seconds. The rest of the published table
#: is hours of pure Python, so the chapter cites it rather than recomputing.
DEFAULT_BOARDS = [(4, 4), (4, 5), (6, 4)]


def main(argv=None):
    """Reprove the small boards and check them against the published values."""
    from .connect_four import ConnectFour

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--boards", default=",".join(f"{r}x{c}" for r, c in DEFAULT_BOARDS),
        help="comma-separated RxC list, e.g. 4x4,4x5,6x4")
    args = parser.parse_args(argv)

    boards = []
    for spec in args.boards.split(","):
        rows, cols = spec.lower().split("x")
        boards.append((int(rows), int(cols)))

    print("Connect Four, solved exactly by alpha-beta with a transposition")
    print("table, against van den Herik, Uiterwijk and van Rijswijck (2002).")
    print()
    print(f"{'board':>7}  {'solved':>7}  {'published':>9}  "
          f"{'positions':>10}  {'seconds':>8}")

    verdicts = []
    for rows, cols in boards:
        game = ConnectFour(rows, cols)
        table = {}
        started = time.time()
        value = int(solve(game, game.initial_board(), table=table))
        elapsed = time.time() - started
        published = game.solved_value()
        agrees = published is not None and value == published
        verdicts.append(agrees)
        shown = "-" if published is None else f"{published:+d}"
        print(f"{rows}x{cols:<5}  {value:+7d}  {shown:>9}  "
              f"{len(table):10,}  {elapsed:8.1f}")

    print()
    if all(verdicts):
        print("every board matches the published value.")
    else:
        print("MISMATCH: a board disagrees with the published value.")
    return verdicts


if __name__ == "__main__":
    main()
