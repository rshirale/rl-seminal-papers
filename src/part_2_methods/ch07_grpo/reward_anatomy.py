"""What listing 7.2 pays or charges for a range of model outputs -- figure 7.6.

Run it::

    python -m src.part_2_methods.ch07_grpo.reward_anatomy
    python -m src.part_2_methods.ch07_grpo.reward_anatomy --figure figures

Every number this prints is the reward function's own output on the candidate
beside it. Nothing is illustrative and nothing is hand-set, which is the point:
if someone edits ``rewards.py``, this table changes, and the test that pins
these eight values against the figure in the chapter fails.

The bottom three rows are the argument for graduated penalties rather than an
all-or-nothing zero. An object that is correct right up to a missing closing
brace, keys in the wrong case, and prose that never attempts JSON at all score
three visibly different numbers. Under a rule that zeroed every unparsable
output they would be indistinguishable, and the model would learn nothing from
the fact that one of them was a single character away from correct.

The second table is the other half of the argument. A reward that checked only
"does it parse" is trivially hacked by ``{}``, and the schema and content tiers
are what make that hack worth almost nothing.

Needs nothing but the standard library. Matplotlib is imported only if you ask
for a figure.
"""

import argparse
import json
import os

if __package__:
    from .dataset import make_dataset
    from .rewards import compute_json_reward, is_compliant
else:  # pragma: no cover - only used by direct script execution.
    from dataset import make_dataset
    from rewards import compute_json_reward, is_compliant

#: One ticket, used for every candidate below so the scores are comparable.
#: Drawn from the real dataset rather than hand-written, so the content tier is
#: scored against a target the trainer would actually see.
TARGET = make_dataset(1)[0]["target"]


def _candidates(target):
    """The eight outputs of figure 7.6, built from ``target``.

    Built rather than pasted so they stay correct if the dataset's first row
    changes: a hard-coded literal would silently stop matching the target and
    the content tier would quietly drop 2.0 points from every row.
    """
    exact = json.dumps(target)
    string_bool = dict(target, resolved=str(target["resolved"]).lower())
    extra_keys = dict(target, priority=1, assignee="unassigned")
    half = {"ticket_id": target["ticket_id"], "severity": target["severity"]}
    wrong_case = {"Ticket_ID": target["ticket_id"],
                  "Severity": target["severity"]}
    return [
        ("Exact target record", exact),
        ("Two invented extra keys", json.dumps(extra_keys)),
        ("Boolean sent as a string", json.dumps(string_bool)),
        ("Right answer in a code fence", "```json\n" + exact + "\n```"),
        ("Half the schema missing", json.dumps(half)),
        ("Keys in the wrong case", json.dumps(wrong_case)),
        ("Unterminated object", exact[:-1]),
        ("Prose, no JSON at all",
         "Sure! Here is the information you asked for."),
    ]


def _hacks(target):
    """Outputs that a weaker reward would pay for. None of these is compliant.

    ``{}`` is the canonical one: it parses, so a reward that checked only
    validity would pay full marks for an empty object. Here it scores 0.00 --
    better than prose, because it is at least syntactically JSON, and four
    points below the exact record, because the schema and content tiers are
    unavailable without actually reading the input.

    The last row is the reason the content tier is scored against *this
    ticket's* target rather than a constant. A model that memorized one record
    and emitted it for every prompt would be scored on whether that record is
    right for the prompt in front of it, and it is not.
    """
    other = make_dataset(2)[1]["target"]
    return [
        ("Empty object", "{}"),
        ("Schema keys, all values empty",
         json.dumps({"ticket_id": "", "severity": "", "component": "",
                     "resolved": ""})),
        ("Another ticket's record", json.dumps(other)),
    ]


def score_table(target=TARGET):
    """``[(name, candidate, reward, compliant)]`` for the figure's eight."""
    return [(name, text, compute_json_reward(text, target),
             is_compliant(text, target))
            for name, text in _candidates(target)]


def hack_table(target=TARGET):
    """The same, for the outputs a naive reward would overpay."""
    return [(name, text, compute_json_reward(text, target),
             is_compliant(text, target))
            for name, text in _hacks(target)]


def run(target=TARGET, figure_dir=None, printer=print):
    """Prints both tables, and optionally writes the figure."""
    rows = score_table(target)
    hacks = hack_table(target)

    printer("Chapter 7, figure 7.6 -- what the reward pays for each output.")
    printer(f"Target record: {json.dumps(target)}\n")
    printer(f"  {'output':<32}{'reward':>8}  {'compliant':>9}")
    printer(f"  {'-' * 32}{'-' * 8}  {'-' * 9}")
    for name, _, reward, compliant in rows:
        printer(f"  {name:<32}{reward:>8.2f}  "
                f"{'yes' if compliant else 'no':>9}")

    printer("\nWhat a parse-only reward would overpay for:\n")
    printer(f"  {'output':<32}{'reward':>8}  {'compliant':>9}")
    printer(f"  {'-' * 32}{'-' * 8}  {'-' * 9}")
    for name, _, reward, compliant in hacks:
        printer(f"  {name:<32}{reward:>8.2f}  "
                f"{'yes' if compliant else 'no':>9}")

    printer(
        "\n  Note the last row: it is strictly compliant and still wrong. "
        "Compliance is a\n  structural property, so the content tier -- "
        "scored against this ticket's own\n  target -- is the only thing "
        "standing between the policy and memorizing one\n  record."
    )

    spread = max(r for _, _, r, _ in rows) - min(r for _, _, r, _ in rows)
    printer(
        f"\nThe spread is the learning signal. These eight outputs span "
        f"{spread:.2f} points,"
        f"\nso a group containing several of them has variance to standardize "
        f"and"
        f"\ntherefore a gradient. A binary right/wrong reward would collapse "
        f"all eight"
        f"\ninto two values, and any group drawing eight of one kind would be "
        f"skipped."
        f"\nThat is the second argument for graduated penalties, beyond the "
        f"obvious one:"
        f"\nrun group_size.py to see what it is worth."
    )

    if figure_dir:
        path = plot(rows, figure_dir)
        printer(f"\nfigure written to {path}.png / .svg")
    return rows


def plot(rows, figure_dir, basename="ch07-figure-reward-anatomy"):
    """Writes the horizontal bar chart as PNG and SVG."""
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

    names = [name for name, _, _, _ in rows][::-1]
    values = [reward for _, _, reward, _ in rows][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    # Two colours, but the sign is also readable from which side of zero the
    # bar sits on and from the printed value, so the figure survives the
    # black-and-white printing the other chapters' figures are drawn for.
    colors = ["#0077BB" if v >= 0 else "#CC3311" for v in values]
    ax.barh(names, values, color=colors, height=0.62)
    ax.axvline(0, color="#444444", lw=1.0)

    for y, v in enumerate(values):
        offset = 0.12 if v >= 0 else -0.12
        ax.text(v + offset, y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)

    ax.set_xlabel("Reward")
    ax.set_xlim(-4.2, 7.2)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    path = os.path.join(figure_dir, basename)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a range of model outputs with the chapter's "
                    "rule-based reward.")
    parser.add_argument("--figure", metavar="DIR", default=None,
                        help="Write the figure PNG/SVG into DIR.")
    run(figure_dir=parser.parse_args().figure)
