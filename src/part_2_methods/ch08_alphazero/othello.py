"""Othello on a 6x6 board -- the second domain, and the point of the interface.

Othello is the closer cousin to Go of the two domains here: stones sit on a
square grid, a move captures enemy stones by surrounding them, a player with
no move must pass, and the game is decided by counting territory at the end.
It is also solved at this size, and the answer is the interesting one -- van
den Herik and colleagues (2002) record 6x6 Othello as a rare genuine
*second*-player win, solved by Feinstein. An agent that comes out of self-play
convinced that moving first is an advantage here has learned something false,
and unlike almost anywhere else in reinforcement learning, we can prove it.

Nothing in the agent changes to play this. That is the whole argument: the
chapter asserts AlphaZero's generality by naming chess and shogi, and this
file lets a reader watch the same claim at a scale a laptop can finish.
"""

import numpy as np

from .game import Game

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

#: First-player value of the opening position, from van den Herik, Uiterwijk
#: and van Rijswijck (2002), section 3.4. Second player wins at 6x6. Named for
#: its game for the reason connect_four.py records.
OTHELLO_SOLVED_VALUES = {(6, 6): -1}


class Othello(Game):
    def __init__(self, size=6):
        if size % 2:
            raise ValueError("Othello needs an even board")
        self.size = size
        self.name = f"othello_{size}x{size}"

    @property
    def action_size(self):
        # One action per square, plus an explicit pass.
        return self.size * self.size + 1

    @property
    def board_shape(self):
        return (self.size, self.size)

    @property
    def pass_action(self):
        return self.size * self.size

    def initial_board(self):
        board = np.zeros((self.size, self.size), dtype=np.int8)
        mid = self.size // 2
        board[mid - 1, mid - 1] = -1
        board[mid, mid] = -1
        board[mid - 1, mid] = 1
        board[mid, mid - 1] = 1
        return board

    def _flips(self, board, row, col):
        """Discs the current player would turn by playing at (row, col)."""
        if board[row, col] != 0:
            return []
        captured = []
        for dr, dc in DIRECTIONS:
            run = []
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and board[r, c] == -1:
                run.append((r, c))
                r, c = r + dr, c + dc
            # The run only counts if our own disc closes the sandwich.
            if run and 0 <= r < self.size and 0 <= c < self.size and board[r, c] == 1:
                captured.extend(run)
        return captured

    def legal_moves(self, board):
        mask = np.zeros(self.action_size, dtype=np.int8)
        for row in range(self.size):
            for col in range(self.size):
                if self._flips(board, row, col):
                    mask[row * self.size + col] = 1
        if not mask.any():
            # With nothing to play, passing is the only legal action.
            mask[self.pass_action] = 1
        return mask

    def apply_move(self, board, action):
        board = board.copy()
        if action != self.pass_action:
            row, col = divmod(action, self.size)
            captured = self._flips(board, row, col)
            if not captured:
                raise ValueError(f"square {(row, col)} flips nothing")
            board[row, col] = 1
            for r, c in captured:
                board[r, c] = 1
        return -board

    def outcome(self, board):
        # The game is live while either side has a real move. Two passes in a
        # row end it, which is why both boards are tested rather than one.
        if self._has_move(board) or self._has_move(-board):
            return None
        mine = int((board == 1).sum())
        theirs = int((board == -1).sum())
        if mine > theirs:
            return 1
        if theirs > mine:
            return -1
        return 0

    def _has_move(self, board):
        for row in range(self.size):
            for col in range(self.size):
                if self._flips(board, row, col):
                    return True
        return False

    def symmetries(self, board, pi):
        """Othello's board has the eight symmetries of the square.

        This is the same trick AlphaGo used on the Go board, and it is worth
        eight times the training data for the cost of a few transposes. The
        pass action sits outside the grid, so it is sliced off, the square part
        is rotated, and it is put back unchanged.
        """
        out = []
        grid = pi[:-1].reshape(self.size, self.size)
        for rot in range(4):
            b = np.rot90(board, rot)
            p = np.rot90(grid, rot)
            for flip in (False, True):
                bb = np.fliplr(b) if flip else b
                pp = np.fliplr(p) if flip else p
                policy = np.concatenate([pp.ravel(), pi[-1:]])
                out.append((np.ascontiguousarray(bb),
                            np.ascontiguousarray(policy)))
        return out

    def solved_value(self):
        return OTHELLO_SOLVED_VALUES.get((self.size, self.size))

    def render(self, board):
        glyph = {0: ".", 1: "X", -1: "O"}
        return "\n".join("".join(glyph[int(v)] for v in row) for row in board)
