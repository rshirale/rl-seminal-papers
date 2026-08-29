"""Deterministic seeding for the Chapter 6 training scripts.

SAC is the most seed-sensitive claim in the book, and the chapter says so
outright: the RL Zoo error bars in figure 6.12 put SAC's standard deviation on
Hopper at roughly half its mean, and "Seed variance" is one of the four
production pain points the limitations section lists. Running this module is
what turns that into a number a reader can re-run rather than take on trust.

Run it::

    python -m src.part_2_methods.ch06_sac.seeding

Kept deliberately identical in shape to ``ch03_dqn/seeding.py`` and
``ch04_ddpg/seeding.py`` so the chapters teach one seeding habit rather than
three. It diverges from Chapter 4's in one place -- the thread pin -- for the
reason given in ``set_seed``.
"""

import argparse
import random
import statistics

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG a SAC run draws from, and pins the thread count.

    Four generators feed a single run: ``random`` for ``random.sample`` inside
    the replay buffer, ``torch`` for network weight initialization *and* for
    the actor's ``rsample()`` -- SAC's exploration is the policy itself, so
    torch's generator is on the critical path in a way it never was for DDPG's
    external noise -- the environment's own RNG for episode start states, and
    ``action_space.sample()`` for the 10,000-step warmup. ``np.random`` is
    seeded too, though nothing here draws from it directly; the buffer's
    ``np.array`` conversions and any reader experiment do.

    Missing ``random`` is the easy one to miss and the expensive one: it seeds
    the minibatch draw, so leaving it out makes every gradient step in the run
    unreproducible even when the weights start identical.

    Call this *before* constructing the agent, so the actor's and critics'
    weight initialization is covered. Environment seeding is separate -- see
    ``seed_env``.

    The thread pin is part of seeding, not a performance tweak. Torch's
    intra-op parallelism changes the order floating-point work is reduced in,
    so the same seed on an 8-core machine and a 4-core one produces different
    returns. Chapter 6 prints an exact terminal transcript, which no reader
    reproduces without this. Chapter 5's ``plot_efficiency.py`` used to have to
    pin the thread count on this chapter's behalf, because this module did not
    exist; it no longer has to.

    This still does not make results identical across platforms or PyTorch
    versions.
    """
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int) -> None:
    """Seeds an environment's episode stream and its action sampler.

    ``action_space.seed`` carries more weight here than anywhere else in the
    book: SAC's warmup is 10,000 uniform random steps, a fifth of the default
    50,000-step budget, so the first fifth of the replay buffer comes entirely
    from that generator and every gradient update for the rest of the run
    samples from it.

    ``reset(seed=...)`` is passed once to seed the underlying generator; later
    resets deliberately continue that stream rather than restarting it, which
    would make every episode identical. This is Chapter 4's convention, not
    Chapter 5's per-episode reseeding: SAC updates on environment steps rather
    than on rollout boundaries, so there is no rollout boundary to make
    reproducible on its own.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)


def _demo(seeds, total_steps, printer=print):
    """Trains one configuration once per seed and reports the spread."""
    # Imported here rather than at module scope to keep this module a leaf:
    # ``train_pendulum`` imports ``set_seed`` from it, and importing the
    # trainer at module scope would make that circular. ``_score`` and
    # ``SCORE_WINDOW`` come from the ablation so this prints the same statistic
    # the chapter's tables are built from.
    if __package__:
        from .ablation import SCORE_WINDOW, _score
        from .train_pendulum import RANDOM_POLICY_BASELINE, main as train
    else:  # pragma: no cover - only used by direct script execution.
        from ablation import SCORE_WINDOW, _score
        from train_pendulum import RANDOM_POLICY_BASELINE, main as train

    seeds = tuple(seeds)

    printer(f"SAC seed variance on Pendulum-v1 - {total_steps:,} steps, "
            f"one run per seed, nothing else varied.")
    printer(f"Score = median episode return over the final {SCORE_WINDOW} "
            f"episodes. Random policy scores about "
            f"{RANDOM_POLICY_BASELINE:.0f}.\n")

    scores = []
    for seed in seeds:
        returns = train(seed=seed, total_steps=total_steps, verbose=False)
        score = _score(returns)
        scores.append(score)
        printer(f"  seed {seed:<3d} {score:>10.1f}")

    spread = max(scores) - min(scores)
    printer(f"\n  {'mean':<8} {statistics.mean(scores):>10.1f}")
    printer(f"  {'spread':<8} {spread:>10.1f}")
    printer(
        f"\nEvery run above is the same agent on the same task. The "
        f"{spread:.0f}-point"
        f"\nspread is what the seed alone is worth, so an ablation that moves"
        f"\nthe score by less than that has not been shown to do anything."
        f"\nThis is why ablation.py averages over seeds and prints a spread"
        f"\ncolumn instead of reporting its best run."
    )
    return scores


def parse_args():
    if __package__:
        from .ablation import DEFAULT_SEEDS, TOTAL_STEPS
    else:  # pragma: no cover - only used by direct script execution.
        from ablation import DEFAULT_SEEDS, TOTAL_STEPS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(DEFAULT_SEEDS),
                        help="Seeds to run, one training run each.")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help="Environment steps per run.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _demo(args.seeds, args.steps)
