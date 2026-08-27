"""Component ablation for Chapter 4: what actually makes DDPG work.

Trains the three variants behind the chapter's figure 4.10 on Pendulum-v1
across several seeds and reports the learning curves. Each isolates a single
design decision:

    1. No target networks -- the online critic computes the target it is then
       trained against, so the label moves as fast as the learner.
    2. Full DDPG          -- soft targets at tau = 0.001, the published
       configuration.
    3. Hard target copy   -- the targets exist, but are overwritten wholesale
       every ``hard_update_freq`` steps, DQN-style, instead of drifting.

Exploration noise is identical in all three (i.i.d. Gaussian, sigma = 0.2),
which is what lets the chapter attribute the whole gap to the target networks
rather than to the noise process.

Two notes on reading the results:

  * **The variants are not a noise comparison.** An earlier version of this
    experiment pitted OU noise against Gaussian. The chapter now teaches
    Gaussian only -- OU survives as historical context -- so the comparison
    moved to the target networks, which is what the text argues for at length.

  * **Report what the curves show, not what the argument wants.** The
    no-target variant does escape the random-policy line on this task; what it
    never does is approach the converged score. The load-bearing evidence is
    the seed-to-seed spread: an unstable variant is one whose outcome depends
    on the draw. Read the spread column before the mean.

    The same discipline applies to the hard copy. Section 4.4 warns that
    wholesale copies destabilize co-evolving actor-critic pairs, and on
    Pendulum-v1 that penalty does not materialize -- it lands within seed noise
    of the soft update. The chapter says so rather than hiding the row.

Usage:
    python -m src.part_2_methods.ch04_ddpg.ablation
    python -m src.part_2_methods.ch04_ddpg.ablation --seeds 0 1 2 3 4
    python -m src.part_2_methods.ch04_ddpg.ablation --figure ../../../figures

Runtime: the default sweep is nine 200-episode training runs, and it is slow --
measured at 92 minutes on an 8-core Intel MacBook Pro. Budget an hour and a
half, and expect that to move with your core count. If you only want to see the
shape of the result, --episodes 100 --seeds 0 1 cuts it to about a fifth and
still separates the no-target variant clearly.
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
#
# These three, in this order, are the curves figure 4.10 describes. Changing
# the set means changing the caption: it names each line by linestyle and
# marker. See STYLES below.
VARIANTS = (
    ("No target networks", False, "soft"),
    ("Full DDPG (soft targets)", True, "soft"),
    ("Hard target copy", True, "hard"),
)

DEFAULT_SEEDS = (0, 1, 2)
SCORE_WINDOW = 20   # episodes averaged at the end of each run
SMOOTH_WINDOW = 10  # moving average applied to the plotted curves
RANDOM_POLICY_BASELINE = -1200.0

# Distinguished by linestyle and marker, not colour: Manning prints in black
# and white, so a figure that encodes meaning in hue alone loses it on paper.
# Figure 4.10's caption names the variants exactly this way -- "dashed line
# with circle markers", "dash-dot line with square markers", "solid line with
# triangle markers" -- which makes this table part of the chapter's text rather
# than a presentation detail. The colours are a high/mid/low luminance set, so
# the curves stay separable after greyscale conversion too.
STYLES = {
    "No target networks":       {"color": "#CC3311", "ls": "--",  "marker": "o"},
    "Full DDPG (soft targets)": {"color": "#0077BB", "ls": "-.",  "marker": "s"},
    "Hard target copy":         {"color": "#117733", "ls": "-",   "marker": "^"},
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


def run(seeds=DEFAULT_SEEDS, episodes=EPISODES, figure_dir=None,
        printer=print, variants=VARIANTS):
    """Runs every variant over every seed. Returns {label: [curve per seed]}."""
    seeds = tuple(seeds)
    variants = tuple(variants)
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

    for label, use_targets, target_update in variants:
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
    parser.add_argument(
        "--no-hard-copy", action="store_true",
        help="Drop the DQN-style hard target copy and plot only the two "
             "target-network variants. The full three are what figure 4.10 "
             "shows; this is for a quicker sweep while iterating.",
    )
    args = parser.parse_args()
    variants = VARIANTS[:2] if args.no_hard_copy else VARIANTS
    run(seeds=args.seeds, episodes=args.episodes, figure_dir=args.figure,
        variants=variants)
