"""Component ablation for Chapter 4: what actually makes DDPG work.

Trains three variants on Pendulum-v1 across several seeds and reports the
learning curves. This is the experiment behind the chapter's ablation figure.

    1. No target networks   -- the online critic computes the target it is then
       trained against, so the label moves as fast as the learner.
    2. Hard target copy     -- targets exist, but are copied wholesale on a
       DQN-style schedule instead of drifting.
    3. Full DDPG            -- soft targets at tau = 0.001, the published
       configuration.

Two notes on how this differs from an earlier version of the experiment:

  * **The second variant used to be a noise comparison** (OU against Gaussian).
    It was replaced because the chapter now teaches Gaussian noise only, and
    because soft-versus-hard target updates is the comparison the text actually
    argues for at length -- readers were being told that hard copies destabilize
    co-evolving networks and then shown a figure about something else.

  * **Report what the curves show, not what the argument wants.** Whether the
    no-target variant stays under the random-policy line is an empirical
    question, and on a task this easy it may well cross it on some seeds. The
    load-bearing evidence is the seed-to-seed spread: an unstable variant is
    one whose outcome depends on the draw. Read the spread column before the
    mean.

Usage:
    python -m src.part_2_methods.ch04_ddpg.ablation
    python -m src.part_2_methods.ch04_ddpg.ablation --seeds 0 1 2 3 4
    python -m src.part_2_methods.ch04_ddpg.ablation --figure ../../../figures

Runtime: roughly one minute per 100 episodes per run on a CPU, so the default
sweep (3 variants x 3 seeds x 200 episodes) takes on the order of half an hour.
Shrink it with --episodes and --seeds while iterating.
"""

import argparse
import os
import statistics

import numpy as np

if __package__:
    from .train_pendulum import EPISODES, main as train
else:  # pragma: no cover - direct script execution fallback.
    from train_pendulum import EPISODES, main as train

# (label, use_target_networks, target_update)
VARIANTS = (
    ("No target networks", False, "soft"),
    ("Hard target copy", True, "hard"),
    ("Full DDPG (soft targets)", True, "soft"),
)

DEFAULT_SEEDS = (0, 1, 2)
SCORE_WINDOW = 20   # episodes averaged at the end of each run
SMOOTH_WINDOW = 10  # moving average applied to the plotted curves
RANDOM_POLICY_BASELINE = -1200.0

# Distinguished by linestyle and marker, not colour: Manning prints in black
# and white, so a figure that encodes meaning in hue alone loses it on paper.
# The colours are a high/mid/low luminance set, which keeps the curves
# separable after greyscale conversion too.
STYLES = {
    "No target networks":       {"color": "#CC3311", "ls": "--",  "marker": "o"},
    "Hard target copy":         {"color": "#0077BB", "ls": "-.",  "marker": "s"},
    "Full DDPG (soft targets)": {"color": "#117733", "ls": "-",   "marker": "^"},
}


def run_variant(use_target_networks, target_update, seeds, episodes):
    """Trains one variant once per seed. Returns a list of return-curves."""
    curves = []
    for seed in seeds:
        curves.append(train(
            seed=seed,
            episodes=episodes,
            use_target_networks=use_target_networks,
            target_update=target_update,
            verbose=False,
        ))
    return curves


def smooth(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run(seeds=DEFAULT_SEEDS, episodes=EPISODES, figure_dir=None, printer=print):
    """Runs every variant over every seed. Returns {label: [curve per seed]}."""
    seeds = tuple(seeds)
    window = min(SCORE_WINDOW, episodes)
    results = {}

    printer(f"DDPG ablation on Pendulum-v1 - {episodes} episodes, "
            f"seeds {', '.join(map(str, seeds))}")
    printer(f"Score = mean episode return over the final {window} episodes. "
            f"Random policy scores about {RANDOM_POLICY_BASELINE:.0f}.\n")

    header = f"{'variant':<26}" + "".join(f"{f'seed {s}':>10}" for s in seeds)
    header += f"{'mean':>10}{'spread':>9}"
    printer(header)
    printer("-" * len(header))

    for label, use_targets, target_update in VARIANTS:
        curves = run_variant(use_targets, target_update, seeds, episodes)
        results[label] = curves
        scores = [statistics.mean(c[-window:]) for c in curves]
        cells = "".join(f"{s:>10.1f}" for s in scores)
        printer(f"{label:<26}{cells}"
                f"{statistics.mean(scores):>10.1f}"
                f"{max(scores) - min(scores):>9.1f}")

    printer("\nThe spread column is the point: a variant whose outcome swings "
            "\nwith the seed is unstable even when one of its runs looks fine.")

    if figure_dir:
        path = plot(results, episodes, figure_dir)
        printer(f"\nWrote {path}.png and {path}.svg")

    return results


def plot(results, episodes, figure_dir, basename="ch04-figure-ablation"):
    """Writes the mean +/- std learning curves as PNG and SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for label, curves in results.items():
        smoothed = np.array([smooth(c, SMOOTH_WINDOW) for c in curves])
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        x = np.arange(len(mean))
        style = STYLES[label]
        ax.plot(x, mean, lw=2.0, markersize=4,
                markevery=max(1, len(x) // 10), label=label, **style)
        ax.fill_between(x, mean - std, mean + std,
                        color=style["color"], alpha=0.18)

    ax.axhline(RANDOM_POLICY_BASELINE, color="#999999", ls=":", lw=1.2,
               label="Random policy baseline")
    ax.set_xlabel("Episode")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DDPG ablation (target networks / update rule).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Seeds to average over. More seeds, more trustworthy.",
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES,
        help="Episodes per run.",
    )
    parser.add_argument(
        "--figure", metavar="DIR", default=None,
        help="Write ch04-figure-ablation.png/.svg into DIR. Point this at the "
             "book's Chapter4/media directory to regenerate the figure.",
    )
    args = parser.parse_args()
    run(seeds=args.seeds, episodes=args.episodes, figure_dir=args.figure)
