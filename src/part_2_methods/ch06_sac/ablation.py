"""Component ablations for Chapter 6: what the entropy objective is worth.

Three experiments live here, one per exercise at the end of the chapter. All
three run on Pendulum-v1 across several seeds, and all three change exactly one
thing about an otherwise-published SAC-v2 configuration.

**Default -- the entropy ablation (exercise 1).** Chapter 6's central claim is
that folding exploration into the objective is what makes SAC work, rather than
the twin critics it shares with TD3. The default run measures that directly:

    1. Entropy bonus off -- ``alpha = 0`` with the temperature update skipped,
       leaving the twin critics, the replay buffer, the soft target updates and
       the stochastic actor all in place.
    2. SAC-v2 as published -- alpha learned by dual gradient descent.

Because only the entropy term changes, the whole gap is attributable to it.
Note what this variant is *not*: it is not DDPG. The actor is still stochastic
and still samples its actions, so the run keeps a source of exploration. What
it loses is any pressure to *keep* that spread, which is why the ablation in
the paper reports the collapse as a variance story rather than a mean one.

**``--temperature`` -- fixed temperatures (exercise 2).** Reverts to SAC-v1 by
holding alpha at 0.01, 0.2 and 1.0, and contrasts all three against the
learned temperature. alpha = 1.0 weights one nat of entropy the same as one
unit of Pendulum reward, and Pendulum's rewards run to -16 per step, so the
entropy term is nowhere near dominant -- but the entropy floor it implies is
far above what an upright-holding policy can maintain, and the run does not
converge.

**``--reward-scale`` -- reward scaling (exercise 3).** Multiplies rewards by 10
before they reach the Bellman target and reruns both the fixed and the learned
temperature. Reward scale is an implicit inverse temperature: scaling rewards
by 10 is equivalent to dividing alpha by 10. The fixed-temperature runs
therefore have to be retuned; the learned one retunes itself. This is the
experiment behind the chapter's claim that automatic tuning removes reward
scale from the hyperparameter list.

Two notes on reading the results:

  * **Read the spread column before the mean.** SAC's seed-to-seed spread is
    the chapter's own caveat on figure 6.12 -- "a single SAC run on that task
    tells you very little". A gap narrower than the spread is not a result.
    ``seeding.py`` prints the spread on its own so the threshold has a source.

  * **The score is a median, not a mean.** Pendulum-v1 resets to a uniformly
    random angle, so roughly one episode in ten starts near upright and scores
    close to zero. The chapter says to judge convergence by the median for
    exactly this reason, and these tables use the same statistic.

Usage:
    python -m src.part_2_methods.ch06_sac.ablation
    python -m src.part_2_methods.ch06_sac.ablation --seeds 0 1 2 3 4
    python -m src.part_2_methods.ch06_sac.ablation --temperature
    python -m src.part_2_methods.ch06_sac.ablation --reward-scale
    python -m src.part_2_methods.ch06_sac.ablation --figure ../../../figures

Runtime: SAC takes a gradient step per environment step, so it is the slowest
runner in the book -- roughly 70 updates a second on one CPU thread, which is
about five minutes per 30,000-step run. The default (2 variants x 3 seeds) is
around half an hour; ``--temperature`` and ``--reward-scale`` are four variants
each, so budget an hour. Shrink any of them with ``--steps`` and ``--seeds``
while iterating; the published numbers need the defaults.
"""

import argparse
import os
import statistics

import numpy as np

if __package__:
    from .train_pendulum import (
        RANDOM_POLICY_BASELINE, STEPS_PER_EPISODE, WARMUP_STEPS,
        main as train,
    )
else:  # pragma: no cover - only used by direct script execution.
    from train_pendulum import (
        RANDOM_POLICY_BASELINE, STEPS_PER_EPISODE, WARMUP_STEPS,
        main as train,
    )

#: 150 episodes, the budget the chapter's exercise 1 specifies. Shorter than
#: train_pendulum.py's 50,000 because these runs are multiplied by variants and
#: seeds, and the policy is converged well before step 30,000.
TOTAL_STEPS = 30_000
DEFAULT_SEEDS = (0, 1, 2)

#: Episodes scored at the tail of each run. 50 episodes is 10,000 steps.
SCORE_WINDOW = 50
SMOOTH_WINDOW = 10

PUBLISHED = "SAC-v2 as published (learned alpha)"
NO_ENTROPY = "Entropy bonus off (alpha = 0)"

#: (label, overrides). The broken variant first, then the published one it is
#: being contrasted against -- the order the chapter's argument runs in.
VARIANTS = (
    (NO_ENTROPY, {"auto_alpha": False, "init_alpha": 0.0}),
    (PUBLISHED, {}),
)

#: Exercise 2. The three fixed temperatures the chapter names, plus the
#: learned one they are being measured against.
TEMPERATURE_VARIANTS = (
    ("Fixed alpha = 0.01", {"auto_alpha": False, "init_alpha": 0.01}),
    ("Fixed alpha = 0.2", {"auto_alpha": False, "init_alpha": 0.2}),
    ("Fixed alpha = 1.0", {"auto_alpha": False, "init_alpha": 1.0}),
    (PUBLISHED, {}),
)

#: Exercise 3. The same two temperature regimes at reward scale 1 and 10. The
#: fixed value is 0.2, the one that works at scale 1, so the pair isolates what
#: retuning would have had to fix.
REWARD_SCALE_VARIANTS = (
    ("Fixed alpha = 0.2, rewards x1", {"auto_alpha": False, "init_alpha": 0.2}),
    ("Fixed alpha = 0.2, rewards x10",
     {"auto_alpha": False, "init_alpha": 0.2, "reward_scale": 10.0}),
    ("Learned alpha, rewards x1", {}),
    ("Learned alpha, rewards x10", {"reward_scale": 10.0}),
)

# Distinguished by linestyle and marker, not colour: Manning prints in black
# and white, so a figure that encodes meaning in hue alone loses it on paper.
# Kept identical in shape to ch04_ddpg/ablation.py and ch05_ppo/ablation.py so
# the three chapters teach one figure convention rather than three.
STYLES = {
    NO_ENTROPY:                        {"color": "#CC3311", "ls": "--", "marker": "o"},
    PUBLISHED:                         {"color": "#0077BB", "ls": "-",  "marker": "s"},
    "Fixed alpha = 0.01":              {"color": "#117733", "ls": ":",  "marker": "^"},
    "Fixed alpha = 0.2":               {"color": "#EE7733", "ls": "-.", "marker": "v"},
    "Fixed alpha = 1.0":               {"color": "#CC3311", "ls": "--", "marker": "o"},
    "Fixed alpha = 0.2, rewards x1":   {"color": "#EE7733", "ls": "-.", "marker": "v"},
    "Fixed alpha = 0.2, rewards x10":  {"color": "#CC3311", "ls": "--", "marker": "o"},
    "Learned alpha, rewards x1":       {"color": "#0077BB", "ls": "-",  "marker": "s"},
    "Learned alpha, rewards x10":      {"color": "#117733", "ls": ":",  "marker": "^"},
}


def warmup_for(total_steps):
    """The warmup budget for a run of ``total_steps``.

    Exactly the published 10,000 at the default budget and at
    ``train_pendulum.py``'s 50,000 -- ``30_000 // 3`` is 10,000 -- and
    proportionally smaller below that, so shrinking a run with ``--steps``
    while iterating still leaves gradient steps to look at. At 2,000 steps the
    published warmup would consume the entire run.
    """
    return min(WARMUP_STEPS, total_steps // 3)


def _run_cached(cache, seed, total_steps, **overrides):
    """Trains one configuration, reusing an identical earlier run.

    Keyed on the resolved overrides so that the published configuration, which
    appears in every experiment, is trained once per seed rather than once per
    experiment per seed.
    """
    overrides.setdefault("warmup_steps", warmup_for(total_steps))
    key = (seed, total_steps) + tuple(sorted(overrides.items()))
    if key not in cache:
        cache[key] = train(seed=seed, total_steps=total_steps, verbose=False,
                           **overrides)
    return cache[key]


def _returns(result):
    """The per-episode returns of a run.

    ``train_pendulum.main`` returns a ``RunResult``, but a bare list is still
    accepted so a caller holding either can be scored or plotted. Everything
    that consumes a run goes through here: when ``main`` gained its policy
    diagnostics, ``_score`` was updated and ``plot`` was not, and because no
    test passed ``figure_dir`` the mistake only surfaced an hour into a CI run,
    after every training call had already finished.
    """
    return getattr(result, "returns", result)


def _score(result, window=SCORE_WINDOW):
    """Median episode return over the final ``window`` episodes of a run.

    Median rather than mean: see the module docstring.
    """
    returns = _returns(result)
    tail = returns[-min(window, len(returns)):]
    return statistics.median(tail)


def smooth(values, window):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run(seeds=DEFAULT_SEEDS, total_steps=TOTAL_STEPS, figure_dir=None,
        printer=print, variants=VARIANTS, cache=None, title=None,
        basename="ch06-figure-entropy"):
    """Runs one ablation. Returns ``{label: [returns per seed]}``."""
    seeds = tuple(seeds)
    variants = tuple(variants)
    episodes = total_steps // STEPS_PER_EPISODE
    window = min(SCORE_WINDOW, episodes)
    cache = {} if cache is None else cache
    results = {}

    title = title or "SAC entropy ablation"
    printer(f"{title} on Pendulum-v1 - {total_steps:,} steps "
            f"({episodes} episodes), seeds {', '.join(map(str, seeds))}")
    printer(f"Score = median episode return over the final {window} episodes. "
            f"Random policy scores about {RANDOM_POLICY_BASELINE:.0f}.\n")

    header = f"{'variant':<37}" + "".join(f"{f'seed {s}':>9}" for s in seeds)
    header += f"{'median':>9}{'spread':>8}{'sigma':>8}{'entropy':>9}{'alpha':>8}"
    printer(header)
    printer("-" * len(header))

    for label, overrides in variants:
        runs = [_run_cached(cache, s, total_steps, **overrides) for s in seeds]
        results[label] = runs
        scores = [_score(r, window) for r in runs]
        cells = "".join(f"{s:>9.1f}" for s in scores)
        printer(f"{label:<37}{cells}"
                f"{statistics.median(scores):>9.1f}"
                f"{max(scores) - min(scores):>8.1f}"
                f"{statistics.mean(r.sigma for r in runs):>8.3f}"
                f"{statistics.mean(r.entropy for r in runs):>+9.3f}"
                f"{statistics.mean(r.alpha for r in runs):>8.3f}")

    printer("\nRead sigma and entropy before the return columns. Pendulum-v1 is"
            "\nforgiving enough that every variant here converges to roughly the"
            "\nsame return, so median and spread cannot tell them apart -- but"
            "\nthe policies behind those identical numbers are not alike. With"
            "\nthe entropy bonus off the actor collapses toward a delta: sigma"
            "\nfalls to near zero and entropy plunges well below the target,"
            "\nbecause differential entropy has no floor at zero. entropy here"
            "\nis E[-log pi], the quantity the temperature regulates, so read it"
            "\nagainst the SAC-v2 target of -dim(A) = -1.")

    if figure_dir:
        path = plot(results, figure_dir, basename=basename, title=title)
        printer(f"\nWrote {path}.png and {path}.svg")

    return results


def run_temperature(seeds=DEFAULT_SEEDS, total_steps=TOTAL_STEPS,
                    figure_dir=None, printer=print, cache=None):
    """Exercise 2: three fixed temperatures against the learned one."""
    return run(seeds=seeds, total_steps=total_steps, figure_dir=figure_dir,
               printer=printer, variants=TEMPERATURE_VARIANTS, cache=cache,
               title="SAC fixed-temperature comparison",
               basename="ch06-figure-temperature")


def run_reward_scale(seeds=DEFAULT_SEEDS, total_steps=TOTAL_STEPS,
                     figure_dir=None, printer=print, cache=None):
    """Exercise 3: fixed and learned temperature under a 10x reward scale."""
    return run(seeds=seeds, total_steps=total_steps, figure_dir=figure_dir,
               printer=printer, variants=REWARD_SCALE_VARIANTS, cache=cache,
               title="SAC reward-scale sensitivity",
               basename="ch06-figure-reward-scale")


def _rcparams(matplotlib):
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot(results, figure_dir, basename="ch06-figure-entropy", title=None):
    """Writes the mean +/- std learning curves as PNG and SVG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _rcparams(matplotlib)
    os.makedirs(figure_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for label, runs in results.items():
        smoothed = np.array([smooth(_returns(r), SMOOTH_WINDOW)
                             for r in runs])
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
    if title:
        ax.set_title(title, fontsize=10, pad=8)
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
        description="Run the SAC entropy ablation, the fixed-temperature "
                    "comparison, or the reward-scale experiment.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
        help="Seeds to average over. More seeds, more trustworthy.",
    )
    parser.add_argument(
        "--steps", type=int, default=TOTAL_STEPS,
        help="Environment steps per run.",
    )
    parser.add_argument(
        "--figure", metavar="DIR", default=None,
        help="Write the figure PNG/SVG into DIR.",
    )
    parser.add_argument(
        "--temperature", action="store_true",
        help="Exercise 2: fixed alpha = 0.01, 0.2 and 1.0 against learned.",
    )
    parser.add_argument(
        "--reward-scale", action="store_true",
        help="Exercise 3: fixed and learned alpha under 10x rewards.",
    )
    args = parser.parse_args()

    if args.temperature:
        run_temperature(seeds=args.seeds, total_steps=args.steps,
                        figure_dir=args.figure)
    elif args.reward_scale:
        run_reward_scale(seeds=args.seeds, total_steps=args.steps,
                         figure_dir=args.figure)
    else:
        run(seeds=args.seeds, total_steps=args.steps, figure_dir=args.figure)
