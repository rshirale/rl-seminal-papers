"""Deterministic seeding for the Chapter 3 training scripts.

CartPole DQN is high variance. Three unseeded 200-episode runs on the same
machine produced last-50 averages of 69.6, 76.2, and 92.1 -- a 32% spread with
nothing differing but RNG state. Without a fixed seed a reader has no way to
tell an actual bug from an unlucky draw, which is why both training scripts
accept ``--seed``.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG the chapter's training loops draw from.

    Four separate generators feed a single run: ``random`` for the epsilon-greedy
    coin flip, the environment's own RNG for episode start states and
    ``action_space.sample()``, ``np.random`` for replay-buffer sampling, and
    ``torch`` for network weight initialization. Missing any one of them leaves
    the run non-reproducible.

    Call this *before* constructing the agent, so weight initialization is
    covered. Environment seeding is separate -- see ``seed_env``.

    This pins run-to-run variance on one machine. It does not make results
    identical across platforms, PyTorch versions, or thread counts.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int) -> None:
    """Seeds an environment's episode stream and its action sampler.

    ``reset(seed=...)`` is passed once here to seed the underlying generator;
    later resets deliberately continue that stream rather than restarting it,
    which would make every episode identical.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
