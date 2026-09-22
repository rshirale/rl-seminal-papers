"""The contract every domain in this chapter satisfies.

AlphaZero's headline claim is that one algorithm, with one set of
hyperparameters, masters several games. That claim only means something if the
agent talks to each game through the same narrow interface. Everything below
the interface is game rules; everything above it is the agent. Nothing in
``mcts.py``, ``selfplay.py`` or ``train.py`` imports a game module -- they
import this one, which is what lets the same agent play Connect Four and
Othello without a line changing.

The AlphaGo listings in ``alphago.py`` are the counterexample. They call
``state.take_action()``, ``state.get_liberties()`` and ``rollout.simulate()``
on an interface that is never written down anywhere in the paper, because a
Go-only system never has to say what a game *is*.

Positions are always canonical: the side to move is +1 and its opponent is -1,
so the agent never needs to know whose turn it is. ``apply_move`` negates the
board on the way out, which removes an entire category of sign bug.
"""

from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    """A two-player, zero-sum, perfect-information game."""

    name = "game"

    @property
    @abstractmethod
    def action_size(self):
        """Number of distinct actions, including a pass if the game has one."""

    @property
    @abstractmethod
    def board_shape(self):
        """Spatial shape of one plane, as (rows, cols)."""

    @property
    def input_planes(self):
        """Feature planes fed to the network: own stones, theirs, and a
        constant plane that gives the convolutions a sense of the edge."""
        return 3

    @abstractmethod
    def initial_board(self):
        """The opening position, from the first player's point of view."""

    @abstractmethod
    def legal_moves(self, board):
        """A 0/1 mask over actions, from the current player's point of view."""

    @abstractmethod
    def apply_move(self, board, action):
        """Play action and return the position with the sides swapped, so the
        result is again canonical for whoever moves next."""

    @abstractmethod
    def outcome(self, board):
        """None while the game is live, otherwise the result for the player to
        move: +1 win, -1 loss, 0 draw."""

    def solved_value(self):
        """The proven game-theoretic value of the opening position, or None.

        This is what makes the project's convergence measurable rather than
        merely plausible. Van den Herik, Uiterwijk and van Rijswijck (2002)
        catalogue the small boards; a subclass returns their entry.
        """
        return None

    def encode(self, board):
        """Stack the canonical board into planes the network can read."""
        own = (board == 1).astype(np.float32)
        theirs = (board == -1).astype(np.float32)
        ones = np.ones_like(own)
        return np.stack([own, theirs, ones])

    def key(self, board):
        """A hashable position key, for transposition tables and solvers."""
        return board.tobytes()

    def symmetries(self, board, pi):
        """Equivalent (position, policy) pairs, used to multiply training data.

        AlphaGo exploited the eight reflections and rotations of the Go board.
        Most games have fewer; the default is to claim none.
        """
        return [(board, pi)]

    def render(self, board):
        return str(board)
