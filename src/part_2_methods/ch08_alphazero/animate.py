"""Animations of the search, for the notebook.

Two things are worth watching rather than reading about. The first is one
move's search: the visit counts start scattered and concentrate on a few
columns as simulations accumulate, which is what "the search sharpens the
policy" means in practice -- and that sharpened distribution is exactly what
``selfplay.py`` stores as the policy target. The second is a whole game, with
the search's opinion shown beside every move.

Both return a matplotlib animation. In a notebook:

    from IPython.display import HTML
    HTML(animate_search(game, mcts).to_jshtml())
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Two hues, far enough apart to stay distinct under colour-vision deficiency.
FIRST, SECOND = "#2a78d6", "#eb6834"
EMPTY, INK, MUTED, SURFACE = "#e8e8e3", "#0b0b0b", "#8a8a85", "#fcfcfb"


def _draw_board(ax, board, game, title):
    """One position, drawn in absolute terms: first player is always blue.

    The agent's boards are canonical -- the side to move is always +1 -- so
    drawing them directly would swap the colours every ply.
    """
    rows, cols = game.board_shape
    ax.clear()
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    for r in range(rows):
        for c in range(cols):
            value = int(board[r, c])
            colour = EMPTY if value == 0 else (FIRST if value == 1 else SECOND)
            ax.add_patch(plt.Circle((c, r), 0.42, color=colour, zorder=2))


def _style_bars(ax, game, ylabel):
    cols = game.action_size
    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(cols))
    ax.set_xlabel("column", fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(axis="y", color="#e6e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d8d3")


def animate_search(game, mcts, board=None, simulations=120, interval=60):
    """Watch one move's search concentrate, simulation by simulation."""
    board = game.initial_board() if board is None else board
    trace = []
    mcts.search([board], simulations, add_noise=False, trace=trace)
    frames = [step[0] for step in trace]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4.2), dpi=80, facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1.3]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)

    def render(i):
        counts = frames[i]
        _draw_board(left, board, game, "the position - blue to move")
        right.clear()
        _style_bars(right, game, "share of visits")
        best = int(np.argmax(counts))
        colours = [FIRST if c == best else "#b9d2f0" for c in range(len(counts))]
        right.bar(range(len(counts)), counts, color=colours, width=0.68,
                  zorder=2)
        for c, value in enumerate(counts):
            if value > 0.02:
                right.text(c, value + 0.02, f"{value:.2f}", ha="center",
                           fontsize=8, color=INK)
        right.set_title(f"simulation {i + 1} of {len(frames)}", fontsize=10,
                        color=INK, loc="left", pad=10)
        return []

    fig.suptitle("One move's search", fontsize=12, color=INK, x=0.012,
                 ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.close(fig)
    return animation.FuncAnimation(fig, render, frames=len(frames),
                                   interval=interval, blit=False)


def animate_game(game, mcts, simulations=100, interval=700, opponent=None,
                 rng=None):
    """Watch a whole game, with the search's opinion beside every move."""
    rng = rng or np.random.default_rng(0)
    board = game.initial_board()
    positions, policies, chosen = [], [], []
    ply = 0
    limit = game.board_shape[0] * game.board_shape[1] + 2

    while game.outcome(board) is None and ply < limit:
        # Store the position in absolute terms so the colours never swap.
        positions.append(board.copy() if ply % 2 == 0 else -board.copy())
        if opponent is not None and ply % 2 == 1:
            action = opponent(game, board, rng)
            policies.append(None)
        else:
            distribution, _ = mcts.search([board], simulations,
                                          add_noise=False, rng=rng)
            policies.append(distribution[0])
            action = int(np.argmax(distribution[0]))
        chosen.append(action)
        board = game.apply_move(board, action)
        ply += 1

    final = board.copy() if ply % 2 == 0 else -board.copy()
    positions.append(final)
    policies.append(None)
    chosen.append(None)

    result = game.outcome(board)
    verdict = ("a draw, which is correct play on this board" if result == 0
               else f"{'second' if ply % 2 == 0 else 'first'} player wins")

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4.2), dpi=80, facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1.3]})
    for ax in (left, right):
        ax.set_facecolor(SURFACE)

    def render(i):
        last = i == len(positions) - 1
        mover = "blue" if i % 2 == 0 else "orange"
        caption = verdict if last else f"move {i + 1} - {mover} to play"
        _draw_board(left, positions[i], game, caption)
        right.clear()
        _style_bars(right, game, "share of visits")
        counts = policies[i]
        if counts is None:
            right.set_title("", fontsize=10)
            right.text(0.5, 0.5, "game over" if last else "opponent's move",
                       transform=right.transAxes, ha="center", fontsize=10,
                       color=MUTED)
        else:
            pick = chosen[i]
            colours = [FIRST if c == pick else "#b9d2f0"
                       for c in range(len(counts))]
            right.bar(range(len(counts)), counts, color=colours, width=0.68,
                      zorder=2)
            right.set_title(f"the search played column {pick}", fontsize=10,
                            color=INK, loc="left", pad=10)
        return []

    fig.suptitle("A game, and what the search considered", fontsize=12,
                 color=INK, x=0.012, ha="left", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.close(fig)
    return animation.FuncAnimation(fig, render, frames=len(positions),
                                   interval=interval, blit=False)
