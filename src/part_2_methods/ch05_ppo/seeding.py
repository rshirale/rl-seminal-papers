"""Deterministic seeding for the Chapter 5 training scripts.

PPO on Pendulum-v1 is the most seed-sensitive agent in the book so far. Three
200-episode runs of ``train_pendulum.py`` differing only in seed returned
last-50 averages of -535.5, -618.8, and -772.7 -- a 237.2-point spread with
nothing else changed. Chapter 4's DDPG moved 43 points under the same
treatment.

That spread is the number Chapter 5 leans on when it argues which
hyperparameter differences three seeds can actually resolve: a gap narrower
than it is not a result, it is a draw. Running this module is what produces
it, so the chapter's threshold has a source a reader can re-run rather than a
figure they have to take on trust.

Run this module to watch that happen::

    python -m src.part_2_methods.ch05_ppo.seeding

Kept close in shape to ``ch03_dqn/seeding.py`` and ``ch04_ddpg/seeding.py`` so
the three chapters teach one seeding habit rather than three. It diverges in
two places, both because PPO's run differs from DDPG's rather than because the
convention changed -- see ``set_seed`` for the generators PPO does not draw
from, and ``episode_seed`` for the environment convention.
"""

import argparse
import statistics

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG a PPO run draws from, and pins the thread count.

    Three generators feed a single run: ``torch`` for network weight
    initialization and for sampling actions from the policy's ``Normal``,
    ``np.random`` for the minibatch permutation in ``PPOAgent.update``, and the
    environment's own RNG for episode start states (seeded separately -- see
    ``episode_seed``). Missing any one leaves the run non-reproducible.

    Unlike Chapters 3 and 4 this does not seed the ``random`` module or the
    action space. PPO draws from neither: there is no replay buffer calling
    ``random.sample`` and no warmup phase calling ``action_space.sample()``.
    Seeding them anyway would suggest the run depends on them.

    Call this *before* constructing the agent, so the actor's and critic's
    weight initialization is covered.

    The thread pin is part of seeding here, not a performance tweak. Torch's
    intra-op parallelism changes the order floating-point work is reduced in,
    so the same seed on an 8-core machine and a 4-core one produces different
    returns -- seed 0 over 40 episodes scores -1154.9 on eight threads and
    -1181.1 on one. Chapter 5 prints an exact terminal transcript, which no
    reader reproduces without this. These networks are 64 units wide, so one
    thread costs nothing in wall time and burns about a quarter of the CPU.

    This still does not make results identical across platforms or PyTorch
    versions.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_seed(seed: int, episode: int) -> int:
    """The environment seed for one episode of a run: ``seed + episode``.

    Chapters 3 and 4 seed the environment once and let the generator's stream
    continue across resets. Chapter 5 reseeds every episode instead, because
    ``train_pendulum.py`` needs a rollout boundary to be reproducible on its
    own: PPO updates on ``update_every`` steps rather than on episode
    boundaries, so a run that is resumed or a rollout that is replayed has to
    land on the same start states either way.

    **The seeds of two runs overlap.** ``seed + episode`` means run seed 0 at
    episode 5 and run seed 1 at episode 4 both start from environment seed 5.
    Over 200 episodes, runs 0 and 1 share 199 of their 200 start states and
    runs 0 and 2 share 198, each shifted by an episode. The runs still differ,
    because weight initialization and the minibatch permutations differ, but
    their environment streams are near-copies of each other rather than
    independent draws.

    That makes the measured seed-to-seed spread a *lower* bound on the spread
    across genuinely independent seeds, which is the conservative direction for
    the claims Chapter 5 makes with it -- a difference the chapter calls
    unresolvable stays unresolvable under independent seeds. Widening the
    stride (``seed * episodes + episode``) would remove the overlap, at the
    cost of changing every number the chapter prints. Left as-is deliberately;
    the hazard is documented rather than silently fixed.
    """
    return seed + episode


def _demo(seeds, episodes, printer=print):
    """Trains one configuration once per seed and reports the spread."""
    # Imported here rather than at module scope to keep this module a leaf:
    # ``train_pendulum`` imports nothing from it, and importing the trainer at
    # module scope would make that circular. ``_score`` and ``SCORE_WINDOW``
    # come from the ablation so this prints the same statistic the chapter's
    # tables are built from, rather than a second definition of "score" that
    # happens to disagree.
    if __package__:
        from .ablation import SCORE_WINDOW, _score
        from .train_pendulum import RANDOM_POLICY_BASELINE, main as train
    else:  # pragma: no cover - only used by direct script execution.
        from ablation import SCORE_WINDOW, _score
        from train_pendulum import RANDOM_POLICY_BASELINE, main as train

    seeds = tuple(seeds)
    window = min(SCORE_WINDOW, episodes)

    printer(f"PPO seed variance on Pendulum-v1 - {episodes} episodes, "
            f"one run per seed, nothing else varied.")
    printer(f"Score = mean episode return over the final {window} episodes. "
            f"Random policy scores about {RANDOM_POLICY_BASELINE:.0f}.\n")

    scores = []
    for seed in seeds:
        result = train(seed=seed, episodes=episodes, verbose=False)
        score = _score(result, window)
        scores.append(score)
        printer(f"  seed {seed:<3d} {score:>10.1f}")

    spread = max(scores) - min(scores)
    printer(f"\n  {'mean':<8} {statistics.mean(scores):>10.1f}")
    printer(f"  {'spread':<8} {spread:>10.1f}")
    printer(
        f"\nEvery run above is the same agent on the same task. The {spread:.0f}-point"
        f"\nspread is what the seed alone is worth, so a hyperparameter change"
        f"\nthat moves the score by less than that has not been shown to do"
        f"\nanything. This is why ablation.py averages over seeds and prints a"
        f"\nspread column instead of reporting its best run."
    )
    return scores


def parse_args():
    if __package__:
        from .ablation import DEFAULT_SEEDS, EPISODES
    else:  # pragma: no cover - only used by direct script execution.
        from ablation import DEFAULT_SEEDS, EPISODES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds to run, one training run each.")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="Episodes per run.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _demo(args.seeds, args.episodes)
