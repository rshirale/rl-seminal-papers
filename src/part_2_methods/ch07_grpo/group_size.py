"""How often a group carries no gradient -- figure 7.7, and the exercise 3 tool.

Run it::

    python -m src.part_2_methods.ch07_grpo.group_size
    python -m src.part_2_methods.ch07_grpo.group_size --figure figures
    python -m src.part_2_methods.ch07_grpo.group_size --history run.json

GRPO's baseline is the group's own mean and standard deviation, so a group
whose members all score alike produces an advantage of zero for every member
and no gradient at all. For a reward that is simply right or wrong, and a model
correct a fraction *p* of the time, that happens with probability::

    p**G + (1 - p)**G

Nothing in the default table is estimated or simulated; it is that expression.
Two things fall out of it. A small group is only safe while the model is
genuinely uncertain -- at p = 0.5 a group of four is already fine, at p = 0.9 a
group of eight still wastes a large fraction of its steps. And the problem gets
*worse as training succeeds*, because p climbs as the policy improves and the
prompts the model has mastered stop producing any signal. That is the trap
DeepSeekMath's group size of 64 buys its way out of, and the one DAPO's dynamic
sampling attacks by discarding those prompts rather than paying to generate
them.

``--history`` is the part worth running after a real training run. The chapter
claims that a *graded* reward tolerates small groups far better than a binary
one, because eight completions must land on exactly the same score rather than
merely on the same verdict. That claim is checkable: ``train_json.py
--history-out`` records every candidate's score, and this script reads them
back and computes the collapse probability at any G from the run's own measured
score distribution.

Needs nothing but the standard library. Matplotlib is imported only if you ask
for a figure.
"""

import argparse
import json
import os
from collections import Counter

#: The group sizes worth tabulating: the open-source range (4 to 16) with one
#: value either side, up to DeepSeekMath's own 64.
GROUP_SIZES = (2, 4, 8, 16, 32, 64)

#: Accuracies to plot. 0.9 is the interesting one -- it is where a successful
#: run ends up, and where small groups fall apart.
ACCURACIES = (0.5, 0.7, 0.9)

#: Default cut for the binary comparison in ``--history`` mode. 5.0 is the
#: score of an output that parses, carries exactly the schema, has the right
#: types and gets all but one field right -- i.e. the level at which a
#: right/wrong reward would plausibly call an output correct. It is a choice,
#: not a measurement, which is why it is a flag.
BINARY_THRESHOLD = 5.0


def zero_gradient_probability(p, group_size):
    """Chance that G binary-scored samples all agree, and so carry no gradient.

    Exact, not simulated: the two ways a group can be uniform are all-correct
    and all-wrong, and those are disjoint.
    """
    return p ** group_size + (1 - p) ** group_size


def collapse_probability(scores, group_size):
    """The same quantity for a graded reward, from a measured score sample.

    Treats ``scores`` as the empirical distribution a candidate is drawn from
    and returns the exact probability that ``group_size`` independent draws
    land on one value -- ``sum_s (n_s / N) ** G`` over the distinct scores.

    The independence assumption is worth naming, because it is the weak point.
    Candidates within one group are drawn for the *same* prompt from the same
    policy, so they are more alike than draws pooled across a whole run; this
    number is therefore an under-estimate of the real collapse rate. The
    observed skip rate that ``--history`` prints beside it is the measurement
    that does not make the assumption, and the two are worth reading together.
    """
    counts = Counter(scores)
    total = sum(counts.values())
    if not total:
        return float("nan")
    return sum((n / total) ** group_size for n in counts.values())


def pooled_scores(history_path):
    """Every candidate's reward from a run written by ``--history-out``."""
    with open(history_path) as handle:
        payload = json.load(handle)
    records = payload["history"]
    scores = [s for record in records for s in record.get("rewards", [])]
    if not scores:
        raise ValueError(
            f"{history_path} carries no per-candidate rewards. It was written "
            "by a version of train_json.py that recorded only group means; "
            "re-run with --history-out to produce a readable one.")
    observed = sum(r["skipped"] for r in records) / len(records)
    return scores, observed, payload.get("config", {})


def run(group_sizes=GROUP_SIZES, accuracies=ACCURACIES, history=None,
        binary_threshold=BINARY_THRESHOLD, figure_dir=None, printer=print):
    """Prints the analytic table, and the measured comparison if given one."""
    printer("Chance a group carries no gradient, for a binary reward.")
    printer("Exactly p**G + (1-p)**G -- nothing here is simulated.\n")

    header = "".join(f"{g:>10d}" for g in group_sizes)
    printer(f"  {'model accuracy':<18}{header}")
    printer(f"  {'-' * 18}{'-' * (10 * len(group_sizes))}")
    for p in accuracies:
        row = "".join(f"{zero_gradient_probability(p, g):>10.1%}"
                      for g in group_sizes)
        printer(f"  {f'p = {p:.0%} correct':<18}{row}")

    printer(
        "\nRead the p = 90% row last. That is where a run ends up if it works,"
        "\nand it is where a group of eight starts throwing away a large "
        "fraction of"
        "\nits steps -- every one of them paid for with G full generations. "
        "The cost"
        "\nof a larger group is linear; the cost of too small a group is that "
        "the run"
        "\nquietly stops learning on the prompts it has already mastered."
    )

    measured = None
    if history:
        measured = _report_history(history, group_sizes, binary_threshold,
                                   printer)

    if figure_dir:
        path = plot(group_sizes, accuracies, measured, figure_dir)
        printer(f"\nfigure written to {path}.png / .svg")
    return measured


def _report_history(history, group_sizes, binary_threshold, printer):
    """The graded-versus-binary comparison, on one run's measured scores."""
    scores, observed, config = pooled_scores(history)
    p_binary = sum(s >= binary_threshold for s in scores) / len(scores)
    graded = [collapse_probability(scores, g) for g in group_sizes]
    binary = [zero_gradient_probability(p_binary, g) for g in group_sizes]

    printer(f"\n\nMeasured on {history}")
    if config:
        printer("  run: " + ", ".join(f"{k}={v}" for k, v in config.items()))
    printer(f"  {len(scores)} candidate scores, "
            f"{len(set(scores))} distinct values")
    printer(f"  {p_binary:.0%} of them score at or above the binary "
            f"threshold of {binary_threshold:.1f}\n")

    header = "".join(f"{g:>10d}" for g in group_sizes)
    printer(f"  {'reward shape':<18}{header}")
    printer(f"  {'-' * 18}{'-' * (10 * len(group_sizes))}")
    printer(f"  {'graded (measured)':<18}"
            + "".join(f"{v:>10.1%}" for v in graded))
    printer(f"  {'binary at same p':<18}"
            + "".join(f"{v:>10.1%}" for v in binary))
    printer(
        f"\n  Observed skip rate in that run: {observed:.1%} of steps."
        f"\n  The graded row is an under-estimate -- see collapse_probability "
        f"on why -- but"
        f"\n  the gap between the two rows is the chapter's claim, measured. "
        f"To collapse, a"
        f"\n  group has to land on the same value every time: one of "
        f"{len(set(scores))} distinct scores"
        f"\n  under this reward, against one of two verdicts under a binary "
        f"one."
    )
    return {"scores": scores, "graded": graded, "binary": binary,
            "p_binary": p_binary, "observed": observed}


def plot(group_sizes, accuracies, measured, figure_dir,
         basename="ch07-figure-group-size"):
    """Writes the collapse curves as PNG and SVG."""
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

    # Distinguished by linestyle and marker as well as colour, matching
    # chapters 4 to 6: Manning prints in black and white.
    styles = [
        {"color": "#0077BB", "ls": "-", "marker": "s"},
        {"color": "#EE7733", "ls": "-.", "marker": "v"},
        {"color": "#CC3311", "ls": "--", "marker": "o"},
    ]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for p, style in zip(accuracies, styles):
        ax.plot(group_sizes,
                [zero_gradient_probability(p, g) for g in group_sizes],
                lw=2.0, markersize=4, label=f"binary reward, {p:.0%} correct",
                **style)
    if measured:
        ax.plot(group_sizes, measured["graded"], lw=2.0, markersize=4,
                color="#117733", ls=":", marker="^",
                label="graded reward, measured run")

    ax.axvspan(4, 16, color="#cccccc", alpha=0.25, lw=0)
    ax.text(8, 0.96, "open-source range", fontsize=8, ha="center",
            color="#555555")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(group_sizes))
    ax.set_xticklabels([str(g) for g in group_sizes])
    ax.set_xlabel("Group size G (every extra member is another generation)")
    ax.set_ylabel("Chance the group carries no gradient")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(fontsize=8, frameon=True, facecolor="white",
                       framealpha=1.0, edgecolor="#cccccc")
    legend.get_frame().set_linewidth(0.8)

    path = os.path.join(figure_dir, basename)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="How often a GRPO group carries no gradient, by group "
                    "size (chapter 7, exercise 3).")
    parser.add_argument("--group-sizes", type=int, nargs="+",
                        default=list(GROUP_SIZES))
    parser.add_argument("--accuracies", type=float, nargs="+",
                        default=list(ACCURACIES),
                        help="Binary-reward success rates to tabulate.")
    parser.add_argument("--history", metavar="PATH", default=None,
                        help="A run written by train_json.py --history-out; "
                             "adds the measured graded-reward comparison.")
    parser.add_argument("--binary-threshold", type=float,
                        default=BINARY_THRESHOLD,
                        help="Score at or above which a binary reward would "
                             "call an output correct.")
    parser.add_argument("--figure", metavar="DIR", default=None,
                        help="Write the figure PNG/SVG into DIR.")
    args = parser.parse_args()
    run(group_sizes=tuple(args.group_sizes),
        accuracies=tuple(args.accuracies), history=args.history,
        binary_threshold=args.binary_threshold, figure_dir=args.figure)
