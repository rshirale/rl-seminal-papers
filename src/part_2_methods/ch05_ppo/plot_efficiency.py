"""Sample efficiency of DDPG, PPO, and SAC on Pendulum-v1, on one axis.

Table 5.3 rates the three algorithms' sample efficiency as Medium, Medium, and
High. This plots the claim instead of asserting it: one curve per algorithm,
averaged over seeds, against environment steps rather than episodes so the
comparison is in the currency that actually costs something.

The x-axis is the point. PPO is on-policy -- it collects a rollout, takes a
few epochs of gradient steps on it, and throws it away. DDPG and SAC keep
every transition in a replay buffer and revisit it many times. That is the
whole of the sample-efficiency argument, and it is visible as a horizontal gap
between the curves rather than a vertical one.

Read it with two caveats. Environment steps are not wall-clock: SAC runs a
gradient update per step and is the slowest of the three to finish despite
needing the fewest samples. And SAC's first ``WARMUP_STEPS`` are uniform
random actions, so its curve is flat by construction before it starts.

Usage:
    python -m src.part_2_methods.ch05_ppo.plot_efficiency
    python -m src.part_2_methods.ch05_ppo.plot_efficiency --figure figures

Three algorithms x three seeds is roughly half an hour on a CPU. Shrink it
with --episodes and --seeds while iterating.
"""

import argparse
import os
import statistics

import numpy as np

if __package__:
    from .ablation import SCORE_WINDOW, SMOOTH_WINDOW, _rcparams, smooth
    from .seeding import set_seed
    from .train_pendulum import RANDOM_POLICY_BASELINE
else:  # pragma: no cover - only used by direct script execution.
    from ablation import SCORE_WINDOW, SMOOTH_WINDOW, _rcparams, smooth
    from seeding import set_seed
    from train_pendulum import RANDOM_POLICY_BASELINE

EPISODES = 200
DEFAULT_SEEDS = (0, 1, 2)

# Pendulum-v1 truncates at 200 steps and never terminates, so every episode of
# every algorithm is exactly this long. That is what makes episode index and
# environment step interchangeable here, and it is why the x-axis conversion
# below is a multiplication rather than a running total.
STEPS_PER_EPISODE = 200

# Distinguished by linestyle and marker, not colour: Manning prints in black
# and white. Kept identical in shape to the STYLES table in ablation.py.
STYLES = {
    "DDPG (chapter 4)": {"color": "#117733", "ls": "-.", "marker": "^"},
    "PPO (chapter 5)":  {"color": "#0077BB", "ls": "-",  "marker": "s"},
    "SAC (chapter 6)":  {"color": "#AA3377", "ls": "--", "marker": "o"},
}


def _run_ddpg(seed, episodes):
    if __package__:
        from ..ch04_ddpg.train_pendulum import main as train
    else:  # pragma: no cover - only used by direct script execution.
        from src.part_2_methods.ch04_ddpg.train_pendulum import main as train
    return train(seed=seed, episodes=episodes, verbose=False)


def _run_ppo(seed, episodes):
    if __package__:
        from .train_pendulum import main as train
    else:  # pragma: no cover - only used by direct script execution.
        from train_pendulum import main as train
    return train(seed=seed, episodes=episodes, verbose=False).returns


def _run_sac(seed, episodes):
    if __package__:
        from ..ch06_sac.train_pendulum import main as train
    else:  # pragma: no cover - only used by direct script execution.
        from src.part_2_methods.ch06_sac.train_pendulum import main as train
    # SAC budgets in steps rather than episodes, so it is the one caller that
    # has to do the conversion itself.
    # ``.returns`` since chapter 6 gained policy diagnostics: main() hands
    # back a RunResult now, exactly as chapter 5's own trainer does.
    return train(seed=seed, total_steps=episodes * STEPS_PER_EPISODE,
                 verbose=False).returns


ALGORITHMS = (
    ("DDPG (chapter 4)", _run_ddpg),
    ("PPO (chapter 5)", _run_ppo),
    ("SAC (chapter 6)", _run_sac),
)


def run(seeds=DEFAULT_SEEDS, episodes=EPISODES, figure_dir=None,
        printer=print, algorithms=ALGORITHMS):
    """Trains every algorithm on every seed. Returns {label: [returns]}."""
    seeds = tuple(seeds)
    window = min(SCORE_WINDOW, episodes)
    results = {}

    printer(f"Sample efficiency on Pendulum-v1 - {episodes} episodes "
            f"({episodes * STEPS_PER_EPISODE:,} environment steps), "
            f"seeds {', '.join(map(str, seeds))}")
    printer(f"Score = mean episode return over the final {window} episodes. "
            f"Random policy scores about {RANDOM_POLICY_BASELINE:.0f}.\n")

    header = f"{'algorithm':<20}" + "".join(f"{f'seed {s}':>10}" for s in seeds)
    header += f"{'mean':>10}{'spread':>9}"
    printer(header)
    printer("-" * len(header))

    for label, runner in algorithms:
        curves = []
        for seed in seeds:
            # Seeding every run from here, rather than trusting each
            # trainer's own call, is what pins the thread count across all
            # three -- ch04 does not pin it itself. (Chapter 6 now does, in
            # its own seeding module; chapter 4 is the remaining one this
            # covers for.)
            set_seed(seed)
            curves.append(runner(seed, episodes)[:episodes])
        results[label] = curves

        scores = [statistics.mean(c[-window:]) for c in curves]
        cells = "".join(f"{s:>10.1f}" for s in scores)
        printer(f"{label:<20}{cells}"
                f"{statistics.mean(scores):>10.1f}"
                f"{max(scores) - min(scores):>9.1f}")

    printer("\nCompare where each curve crosses a given return, not where it "
            "\nends: the gap that matters here is horizontal, in samples.")

    if figure_dir:
        path = plot(results, figure_dir)
        printer(f"\nWrote {path}.png and {path}.svg")

    return results


def plot(results, figure_dir, basename="ch05-figure-efficiency"):
    """Writes the mean +/- std learning curves as PNG and SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _rcparams(matplotlib)
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for label, curves in results.items():
        width = min(len(c) for c in curves)
        smoothed = np.array([smooth(c[:width], SMOOTH_WINDOW) for c in curves])
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        x = np.arange(len(mean)) * STEPS_PER_EPISODE
        style = STYLES[label]
        ax.plot(x, mean, lw=2.0, markersize=4,
                markevery=max(1, len(x) // 10), label=label, **style)
        ax.fill_between(x, mean - std, mean + std,
                        color=style["color"], alpha=0.18)

    ax.axhline(RANDOM_POLICY_BASELINE, color="#999999", ls=":", lw=1.2,
               label="Random policy baseline")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Episode return ({SMOOTH_WINDOW}-ep moving avg)")
    legend = ax.legend(fontsize=8, frameon=True, facecolor="white",
                       framealpha=1.0, edgecolor="#cccccc", loc="lower right")
    legend.get_frame().set_linewidth(0.8)
    ax.set_ylim(-1750, 50)
    ax.grid(True, alpha=0.3)

    path = os.path.join(figure_dir, basename)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(DEFAULT_SEEDS),
                        help="Seeds to average over.")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="Episodes per run, for every algorithm.")
    parser.add_argument("--figure", metavar="DIR",
                        help="Write the figure to DIR as PNG and SVG.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(seeds=args.seeds, episodes=args.episodes, figure_dir=args.figure)
