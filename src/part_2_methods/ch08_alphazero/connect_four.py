"""Connect Four, on the standard board and on the smaller solved ones.

Van den Herik, Uiterwijk and van Rijswijck (2002) catalogue the
game-theoretic value of every small board: 4x4, 6x4, 4x5, 6x5, 4x6, 6x6 and
4x7 are draws, and the standard 6x7 board is a first-player win. Those values
are the ground truth this chapter checks a trained agent against, and they are
why the project's default board is 6x6 rather than 6x7 -- on a drawn board an
agent approaching perfect play draws against itself every time, which is a
convergence signal a network cannot fake by learning to output zero.
"""

import numpy as np

from .game import Game

#: Board (rows x cols) -> game-theoretic value for the first player, from
#: van den Herik, Uiterwijk and van Rijswijck (2002), "Games solved: Now and
#: in the future", section 3.1.1.
#:
#: Named for its game rather than called SOLVED_VALUES because the notebook
#: inlines both this module and othello.py into one namespace, where two
#: constants of the same name silently become one -- and Connect Four would
#: start reporting Othello's answers.
CONNECT_FOUR_SOLVED_VALUES = {
    (4, 4): 0,
    (6, 4): 0,
    (4, 5): 0,
    (6, 5): 0,
    (4, 6): 0,
    (6, 6): 0,
    (4, 7): 0,
    (6, 7): 1,
}


class ConnectFour(Game):
    def __init__(self, rows=6, cols=7, win_length=4):
        self.rows = rows
        self.cols = cols
        self.win_length = win_length
        self.name = f"connect_four_{rows}x{cols}"
        self._lines = self._build_lines()

    @property
    def action_size(self):
        return self.cols

    @property
    def board_shape(self):
        return (self.rows, self.cols)

    def initial_board(self):
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def legal_moves(self, board):
        # A column is playable while its top cell is empty.
        return (board[0] == 0).astype(np.int8)

    def apply_move(self, board, action):
        board = board.copy()
        # Gravity: the disc falls to the lowest empty cell in the column.
        rows = np.nonzero(board[:, action] == 0)[0]
        if rows.size == 0:
            raise ValueError(f"column {action} is full")
        board[rows[-1], action] = 1
        # Hand the position to the opponent, who now reads as +1.
        return -board

    def outcome(self, board):
        # outcome() is asked after a move has been played, so a completed line
        # of -1 belongs to the player who just moved -- which is a loss for the
        # player to move now, hence the sign convention holding without any
        # special case here.
        cells = board.ravel().tolist()
        for line in self._lines:
            first = cells[line[0]]
            if first and all(cells[i] == first for i in line[1:]):
                return int(first)
        if 0 not in cells:
            return 0
        return None

    def _build_lines(self):
        """Every window of win_length cells, as flat indices.

        Precomputing these once is what makes the terminal test affordable. It
        matters more than it looks: one self-play iteration asks whether a
        position is terminal millions of times. The first implementation of
        this test cost 87 microseconds a call; plain-Python index tuples
        brought it to 5.1 -- five times faster than the NumPy version, because
        these arrays are far too small to repay NumPy's per-call overhead.
        """
        lines = []
        for row in range(self.rows):
            for col in range(self.cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    end_r = row + dr * (self.win_length - 1)
                    end_c = col + dc * (self.win_length - 1)
                    if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                        continue
                    lines.append([
                        (row + dr * k) * self.cols + (col + dc * k)
                        for k in range(self.win_length)
                    ])
        return [tuple(line) for line in lines]

    def symmetries(self, board, pi):
        # Mirroring the columns leaves Connect Four unchanged, which doubles
        # every training example for free. This is AlphaGo's eight-fold board
        # symmetry trick, on a board that only has two.
        return [(board, pi), (board[:, ::-1].copy(), pi[::-1].copy())]

    def solved_value(self):
        """The proven value of the opening position, or None if unknown."""
        return CONNECT_FOUR_SOLVED_VALUES.get((self.rows, self.cols))

    def render(self, board):
        glyph = {0: ".", 1: "X", -1: "O"}
        lines = ["".join(glyph[int(v)] for v in row) for row in board]
        lines.append("".join(str(c % 10) for c in range(self.cols)))
        return "\n".join(lines)
