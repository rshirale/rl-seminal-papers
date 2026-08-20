"""Deterministic seeding for the Chapter 4 training scripts.

DDPG on Pendulum-v1 is less volatile than CartPole DQN, but it is not stable
enough to judge from a single run. Three 200-episode runs of
``train_pendulum.py`` differing only in seed returned last-20 averages of
-140.9, -132.0, and -175.5 -- a 43-point spread with nothing else changed.
That is comfortably inside the chapter's "above -200" claim, but it is also
the margin by which a genuine regression could hide, which is why both
``train_pendulum.py`` and ``ablation.py`` take ``--seed`` and the ablation
averages over several runs rather than reporting one.

Kept deliberately identical in shape to ``ch03_dqn/seeding.py`` so the two
chapters teach one seeding habit rather than two.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG a DDPG run draws from.

    Four generators feed a single run: ``random`` for ``random.sample`` inside
    the replay buffer, ``np.random`` for the Gaussian exploration noise, the
    environment's own RNG for episode start states and ``action_space.sample()``
    during warmup, and ``torch`` for network weight initialization. Missing any
    one leaves the run non-reproducible.

    Call this *before* constructing the agent, so the actor's and critic's
    weight initialization is covered. Environment seeding is separate -- see
    ``seed_env``.

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

    ``action_space.seed`` matters more here than it did in Chapter 3: DDPG's
    warmup phase drives the environment with ``action_space.sample()``, so the
    first thousand transitions in the replay buffer come from that generator.
    Leave it unseeded and the buffer differs between runs even when everything
    else is pinned.

    ``reset(seed=...)`` is passed once to seed the underlying generator; later
    resets deliberately continue that stream rather than restarting it, which
    would make every episode identical.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
