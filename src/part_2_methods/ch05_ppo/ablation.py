"""Component ablation and sensitivity sweeps for Chapter 5: what makes PPO work.

Two experiments live here, both on Pendulum-v1 across several seeds.

**Default -- the clipping ablation.** Chapter 5's central claim is that the
clipped surrogate objective is what keeps policy gradients from destroying a
working policy. The default run measures that directly:

    1. Clipping disabled -- ``eps_clip`` set large enough that the clip never
       binds, leaving every other line of the algorithm untouched.
    2. PPO as published  -- ``eps_clip = 0.2``, the value in table 5.2.

Because only the clip changes, the whole gap is attributable to it. A third
variant, an over-tight ``eps_clip = 0.05``, is available behind
``--include-tight``: it shows that the failure is two-sided, which the ``ε``
panel of ``--sweep`` covers more fully.

**``--sweep`` -- the sensitivity bowls.** The chapter's section 8 sidebar
describes "U-shaped bowls plotting performance against a single hyperparameter
value" and notes that the wider the bowl, the more forgiving the algorithm.
This mode produces those bowls for the four hyperparameters the chapter tunes,
one at a time around the published configuration.

Three notes on reading the results:

  * **Read the spread column before the mean.** PPO's seed-to-seed spread on
    Pendulum is wide -- wide enough that it routinely exceeds the difference
    between two hyperparameter settings. A gap smaller than the spread is not a
    result, and the table prints both so the two are never confused.

  * **The diagnostics are part of the finding.** ``approx_kl`` and
    ``clip_frac`` are what section 7 teaches readers to watch, so they are
    reported per variant rather than left in the training log. When clipping is
    disabled, ``approx_kl`` is the column that shows *why* the return collapsed.

  * **Report what the curves show, not what the argument wants.** Whether the
    unclipped variant lands below the random-policy line is an empirical
    question, and on a task this forgiving it may not on every seed.

Usage:
    python -m src.part_2_methods.ch05_ppo.ablation
    python -m src.part_2_methods.ch05_ppo.ablation --seeds 0 1 2 3 4
    python -m src.part_2_methods.ch05_ppo.ablation --include-tight
    python -m src.part_2_methods.ch05_ppo.ablation --sweep
    python -m src.part_2_methods.ch05_ppo.ablation --figure ../../../figures

Runtime: roughly 30 seconds per 200-episode run on a CPU, so the default
(2 variants x 3 seeds) takes about three minutes and ``--sweep`` (12 distinct
configurations x 3 seeds, shared runs cached) about twenty. Shrink either with
--episodes and --seeds while iterating.
"""

import argparse
import os
import statistics

import numpy as np

if __package__:
    from .train_pendulum import (
        EPS_CLIP, GAMMA, LAM, LR, RANDOM_POLICY_BASELINE, main as train,
    )
else:  # pragma: no cover - direct script execution fallback.
    from train_pendulum import (
        EPS_CLIP, GAMMA, LAM, LR, RANDOM_POLICY_BASELINE, main as train,
    )

# Large enough that ``torch.clamp`` never binds. Deliberately a number rather
# than a ``None`` branch in the agent: keeping the identical code path is what
# lets the ablation attribute the whole gap to the clip and not to a second
# difference introduced by the switch.
NO_CLIP = 1e9

EPISODES = 200
DEFAULT_SEEDS = (0, 1, 2)

# Wider than Chapter 4's 20-episode window. PPO's episode-to-episode variance
# on Pendulum is large enough that a 20-episode tail moves by more than the
# effect being measured, which would put noise in the score column.
SCORE_WINDOW = 50
SMOOTH_WINDOW = 10

# (label, eps_clip). This order matches the ablation's argument: the broken
# variant first, then the published one it is being contrasted against.
VARIANTS = (
    ("Clipping disabled", NO_CLIP),
    (f"PPO as published (eps = {EPS_CLIP})", EPS_CLIP),
)

# Opt-in third variant, off by default so the default run stays a clean
# two-way contrast.
TIGHT_VARIANT = ("Over-tight clip (eps = 0.05)", 0.05)

# Distinguished by linestyle and marker, not colour: Manning prints in black
# and white, so a figure that encodes meaning in hue alone loses it on paper.
# Kept identical in shape to ch04_ddpg/ablation.py so the two chapters teach
# one figure convention rather than two.
STYLES = {
    "Clipping disabled":                    {"color": "#CC3311", "ls": "--", "marker": "o"},
    f"PPO as published (eps = {EPS_CLIP})": {"color": "#0077BB", "ls": "-.", "marker": "s"},
    "Over-tight clip (eps = 0.05)":         {"color": "#117733", "ls": "-",  "marker": "^"},
}

# The sensitivity bowls. Each entry is (kwarg, values, tick labels, published
# value) and is swept one at a time around the published configuration, so
# every curve passes through the same baseline point.
SWEEPS = (
    ("eps_clip", (0.05, 0.1, 0.2, 0.4, NO_CLIP), ("0.05", "0.1", "0.2", "0.4", "none"), EPS_CLIP),
    ("lr",       (1e-4, 3e-4, 1e-3, 3e-3),       ("1e-4", "3e-4", "1e-3", "3e-3"),      LR),
    ("lam",      (0.8, 0.95, 1.0),               ("0.8", "0.95", "1.0"),                LAM),
    ("gamma",    (0.9, 0.95, 0.99),              ("0.9", "0.95", "0.99"),               GAMMA),
)

SWEEP_TITLES = {
    "eps_clip": "Clipping threshold  eps",
    "lr": "Learning rate",
    "lam": "GAE decay  lambda",
    "gamma": "Discount  gamma",
}


#: The published configuration, and the point every sweep passes through.
BASELINE = {"eps_clip": EPS_CLIP, "lr": LR, "lam": LAM, "gamma": GAMMA}


def _run_cached(cache, seed, episodes, **overrides):
    """Trains one configuration, reusing an identical earlier run.

    The cache is keyed on the *resolved* configuration rather than on the
    override, because ``eps_clip=0.2`` and ``lr=1e-3`` name the same run: each
    is the published value of its own parameter, so both resolve to the
    untouched baseline. Keying on the override instead would miss that and
    retrain the shared baseline once per sweep per seed.
    """
    config = {**BASELINE, **overrides}
    key = (seed, episodes) + tuple(sorted(config.items()))
    if key not in cache:
        cache[key] = train(seed=seed, episodes=episodes, verbose=False,
                           **config)
    return cache[key]


def _score(result, window):
    """Mean return over the final ``window`` episodes of a run."""
    return statistics.mean(result.returns[-window:])


def smooth(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run(seeds=DEFAULT_SEEDS, episodes=EPISODES, figure_dir=None,
        printer=print, variants=VARIANTS, cache=None):
    """Runs the clipping ablation. Returns {label: [RunResult per seed]}."""
    seeds = tuple(seeds)
    variants = tuple(variants)
    window = min(SCORE_WINDOW, episodes)
    cache = {} if cache is None else cache
    results = {}

    printer(f"PPO clipping ablation on Pendulum-v1 - {episodes} episodes, "
            f"seeds {', '.join(map(str, seeds))}")
    printer(f"Score = mean episode return over the final {window} episodes. "
            f"Random policy scores about {RANDOM_POLICY_BASELINE:.0f}.\n")

    header = f"{'variant':<32}" + "".join(f"{f'seed {s}':>10}" for s in seeds)
    header += f"{'mean':>10}{'spread':>9}{'approx_kl':>11}{'clip_frac':>11}"
    printer(header)
    printer("-" * len(header))

    for label, eps_clip in variants:
        runs = [_run_cached(cache, s, episodes, eps_clip=eps_clip) for s in seeds]
        results[label] = runs
        scores = [_score(r, window) for r in runs]
        kl = statistics.mean(statistics.mean(r.approx_kls) for r in runs)
        cf = statistics.mean(statistics.mean(r.clip_fracs) for r in runs)
        cells = "".join(f"{s:>10.1f}" for s in scores)
        printer(f"{label:<32}{cells}"
                f"{statistics.mean(scores):>10.1f}"
                f"{max(scores) - min(scores):>9.1f}"
                f"{kl:>11.4f}{cf:>11.3f}")

    printer("\nThe spread column is the point: a variant whose outcome swings "
            "\nwith the seed is unstable even when one of its runs looks fine."
            "\nWith clipping off, approx_kl is the column that explains the return.")

    if figure_dir:
        path = plot(results, figure_dir)
        printer(f"\nWrote {path}.png and {path}.svg")

    return results


def run_sweep(seeds=DEFAULT_SEEDS, episodes=EPISODES, figure_dir=None,
              printer=print, sweeps=SWEEPS, cache=None):
    """Runs the one-at-a-time sensitivity sweeps. Returns {param: [rows]}."""
    seeds = tuple(seeds)
    window = min(SCORE_WINDOW, episodes)
    cache = {} if cache is None else cache
    results = {}

    printer(f"PPO sensitivity sweeps on Pendulum-v1 - {episodes} episodes, "
            f"seeds {', '.join(map(str, seeds))}")
    printer("Each parameter is swept alone; the others hold at the published "
            "configuration.\n")

    header = (f"{'parameter':<12}{'value':>8}{'mean':>10}{'spread':>9}"
              f"{'approx_kl':>11}{'clip_frac':>11}")
    printer(header)
    printer("-" * len(header))

    for param, values, ticks, published in sweeps:
        rows = []
        for value, tick in zip(values, ticks):
            runs = [_run_cached(cache, s, episodes, **{param: value})
                    for s in seeds]
            scores = [_score(r, window) for r in runs]
            row = dict(
                value=value, tick=tick,
                mean=statistics.mean(scores),
                lo=min(scores), hi=max(scores),
                spread=max(scores) - min(scores),
                kl=statistics.mean(statistics.mean(r.approx_kls) for r in runs),
                clip_frac=statistics.mean(
                    statistics.mean(r.clip_fracs) for r in runs),
                published=abs(value - published) < 1e-12,
            )
            rows.append(row)
            mark = " *" if row["published"] else ""
            printer(f"{param:<12}{tick:>8}{row['mean']:>10.1f}"
                    f"{row['spread']:>9.1f}{row['kl']:>11.4f}"
                    f"{row['clip_frac']:>11.3f}{mark}")
        results[param] = rows
        printer("")

    printer("* marks the value the chapter publishes. A bowl whose floor sits "
            "\nat the starred value is the chapter's recommendation confirmed; "
            "\na gap narrower than the spread column is not a result.")

    if figure_dir:
        path = plot_sweep(results, figure_dir)
        printer(f"\nWrote {path}.png and {path}.svg")

    return results


def _rcparams(matplotlib):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot(results, figure_dir, basename="ch05-figure-clipping"):
    """Writes the mean +/- std learning curves as PNG and SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _rcparams(matplotlib)
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for label, runs in results.items():
        smoothed = np.array([smooth(r.returns, SMOOTH_WINDOW) for r in runs])
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


def plot_sweep(results, figure_dir, basename="ch05-figure-sensitivity"):
    """Writes the four sensitivity bowls as a 2x2 panel, PNG and SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _rcparams(matplotlib)
    os.makedirs(figure_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharey=True)

    for ax, (param, _, _, _) in zip(axes.ravel(), SWEEPS):
        rows = results[param]
        x = np.arange(len(rows))
        mean = [r["mean"] for r in rows]
        lo = [r["mean"] - r["lo"] for r in rows]
        hi = [r["hi"] - r["mean"] for r in rows]

        ax.errorbar(x, mean, yerr=[lo, hi], color="#0077BB", lw=1.6,
                    marker="o", markersize=5, capsize=3, elinewidth=1.2)
        # Hollow marker on the published value, filled red where the clip is off.
        for i, r in enumerate(rows):
            if r["value"] >= NO_CLIP:
                ax.plot(i, r["mean"], marker="o", markersize=6,
                        color="#CC3311", zorder=3)
            elif r["published"]:
                ax.plot(i, r["mean"], marker="o", markersize=8, mfc="white",
                        mec="#0077BB", mew=2.0, zorder=3)

        ax.axhline(RANDOM_POLICY_BASELINE, color="#999999", ls=":", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([r["tick"] for r in rows])
        ax.set_title(SWEEP_TITLES[param], fontsize=9, pad=6)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xlim(-0.5, len(rows) - 0.5)

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Return (final {SCORE_WINDOW} ep)")
    fig.tight_layout(pad=1.2)

    path = os.path.join(figure_dir, basename)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the PPO clipping ablation or the sensitivity sweeps.")
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
        help="Write the figure PNG/SVG into DIR. Point this at the book's "
             "Chapter5/media directory to regenerate it.",
    )
    parser.add_argument(
        "--include-tight", action="store_true",
        help="Add the over-tight eps = 0.05 variant as a third curve, showing "
             "that the clipping failure is two-sided.",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the sensitivity bowls instead of the clipping ablation.",
    )
    args = parser.parse_args()

    if args.sweep:
        run_sweep(seeds=args.seeds, episodes=args.episodes,
                  figure_dir=args.figure)
    else:
        variants = VARIANTS + (TIGHT_VARIANT,) if args.include_tight \
            else VARIANTS
        run(seeds=args.seeds, episodes=args.episodes,
            figure_dir=args.figure, variants=variants)
