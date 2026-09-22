"""AlphaGo and AlphaZero -- Chapter 8 of "RL: The Seminal Papers".

AlphaGo (Silver et al., 2016) put a neural network inside a tree search:
a policy network to say which moves are worth looking at, a value network to
say how a position is going, and Monte Carlo tree search to spend its budget
where those two agree. AlphaZero (Silver et al., 2018) then removed almost
everything specific to that design -- the human games, the hand-built
features, the rollouts, the second network -- and got a stronger player out of
what was left.

The chapter splits along that line, and so does this directory. ``alphago.py``
holds the published 19x19 architectures, which are read and measured rather
than trained: they need the 30 million position KGS corpus and a cluster. The
rest is a complete AlphaZero that trains from random weights on a laptop in
under an hour, on a board small enough to have a *proven* answer, so the agent
can be marked right or wrong rather than merely watched improving.

Only the game rules are re-exported here, for two separate reasons.

The agent needs torch, and importing it at package scope would make the rules
-- the part a reader most likely replaces with a domain of their own --
unimportable without the deep-learning stack. And ``solver``, ``train``,
``alphago``, ``benchmark`` and ``inspect_agent`` are all runnable as
``python -m``, which emits a double-import RuntimeWarning for any module the
package has already imported. Chapters 4 to 7 leave their runnable modules out
of ``__init__`` for the same pair of reasons.

Import the rest by module::

    from src.part_2_methods.ch08_alphazero.solver import solve
    from src.part_2_methods.ch08_alphazero.network import build
    from src.part_2_methods.ch08_alphazero.mcts import BatchedMCTS
    from src.part_2_methods.ch08_alphazero.train import main
"""

from .connect_four import ConnectFour
from .game import Game
from .othello import Othello

__all__ = ["ConnectFour", "Game", "Othello"]
