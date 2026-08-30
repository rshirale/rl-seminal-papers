"""Reproduces the ablation from Mnih et al. (2015) on CartPole-v1.

The paper's ablation table is Atari data. This runs the same four variants on
the chapter's CartPole example, where a full sweep takes minutes rather than
GPU-days.

Two results are worth knowing before you read the output:

1. **The ordering is not the paper's.** On CartPole, removing the target
   network hurts far more than removing replay -- the reverse of the Atari
   result. That is not a contradiction: replay's value scales with how
   correlated consecutive observations are, and a 4-element control vector is
   far less correlated than a stream of 84x84 frames.

2. **The no-replay variant has enormous seed-to-seed variance** (measured: 33.5
   to 155.6 across three seeds, against a spread of 2.3 for the no-target-network
   variant). A single run will mislead you in either direction, which is why
   this script averages over seeds instead of taking a flag for one. The spread
   is itself the lesson: without replay, what the agent learns depends on which
   trajectory it happened to take.

Usage:
    python -m src.part_2_methods.ch03_dqn.ablation
    python -m src.part_2_methods.ch03_dqn.ablation --seeds 1 2 3 4 5 --episodes 400
"""

import argparse
import statistics

if __package__:
    from .train_cartpole import NUM_EPISODES, WARMUP_STEPS, main as train
else:  # pragma: no cover - direct script execution fallback.
    from train_cartpole import NUM_EPISODES, WARMUP_STEPS, main as train

# (label, use_replay, use_target_network) -- the paper's four rows.
VARIANTS = (
    ("Full DQN", True, True),
    ("No target network", True, False),
    ("No replay buffer", False, True),
    ("Online Q-network", False, False),
)

DEFAULT_SEEDS = (42, 7, 123)
SCORE_WINDOW = 50  # episodes averaged at the end of each run


def run_variant(use_replay, use_target_network, seeds, episodes, window):
    """Final-window score for one variant, once per seed."""
    scores = []
    for seed in seeds:
        rewards = train(
            seed=seed,
            episodes=episodes,
            use_replay=use_replay,
            use_target_network=use_target_network,
            verbose=False,
        )
        scores.append(statistics.mean(rewards[-window:]))
    return scores


def run(seeds=DEFAULT_SEEDS, episodes=300, printer=print):
    """Runs every variant over every seed. Returns {label: [score per seed]}."""
    seeds = tuple(seeds)
    results = {}
    window = min(SCORE_WINDOW, episodes)

    printer(f"Ablation on CartPole-v1 - {episodes} episodes, "
            f"seeds {', '.join(map(str, seeds))}")
    printer(f"Score = mean reward over the final {window} episodes.")

    # A CartPole episode runs ~20 steps under a random policy, so the
    # 1,000-step warmup alone eats roughly the first 50 episodes. A sweep that
    # ends shortly after that has trained for too few episodes to separate the
    # variants, and all four rows come back as the same near-random play.
    #
    # The threshold used to be WARMUP_STEPS * 2, which did not fire at 120
    # episodes even though every row was in fact identical there. Outlasting
    # warmup is not enough; there has to be training left afterwards, so the
    # bar is five times the warmup -- about 250 episodes, consistent with the
    # "use 300 or more" advice below.
    if episodes * 20 < WARMUP_STEPS * 5:
        printer(f"\n  WARNING: {episodes} episodes leaves little training "
                f"after the {WARMUP_STEPS}-step warmup.\n  If every row "
                f"matches, no variant trained long enough to differ. "
                f"Use --episodes 300 or more.")
    printer("")

    header = f"{'variant':<20}" + "".join(f"{f'seed {s}':>10}" for s in seeds)
    printer(header + f"{'mean':>10}{'spread':>9}")
    printer("-" * len(header + f"{'mean':>10}{'spread':>9}"))

    for label, use_replay, use_target in VARIANTS:
        scores = run_variant(use_replay, use_target, seeds, episodes, window)
        results[label] = scores
        cells = "".join(f"{s:>10.1f}" for s in scores)
        spread = max(scores) - min(scores)
        printer(f"{label:<20}{cells}{statistics.mean(scores):>10.1f}{spread:>9.1f}")

    printer("\nRead the spread column, not just the mean: the no-replay row is "
            "\nhighly seed-dependent, and one run of it proves nothing.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DQN ablation (replay / target network) on CartPole."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Seeds to average over. More seeds, more trustworthy - the "
             "no-replay variant especially.",
    )
    parser.add_argument(
        "--episodes", type=int, default=300,
        help=f"Episodes per run (train_cartpole trains for {NUM_EPISODES}).",
    )
    args = parser.parse_args()
    run(seeds=args.seeds, episodes=args.episodes)
