"""Soft Actor-Critic (SAC-v2) -- Chapter 6 of "RL: The Seminal Papers"."""

# `seeding` is deliberately not imported here. It is runnable as
# `python -m src.part_2_methods.ch06_sac.seeding`, and importing it at package
# scope makes that emit a double-import RuntimeWarning. Chapters 4 and 5 leave
# theirs out of __init__ for the same reason.

from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer
from .sac_agent import SACAgent

__all__ = ["Actor", "Critic", "ReplayBuffer", "SACAgent"]
