"""The support-ticket dataset, generated locally rather than downloaded.

Three reasons it is synthetic. It is reproducible from a seed, so two readers
running the chapter compare the same numbers; it cannot rot when a dataset is
renamed or a mirror goes down; and it keeps the chapter runnable with no
network beyond the model weights themselves.

Each example is an unstructured note plus the record we want back. The note is
built from a template, so the *information* is always recoverable -- which
matters, because the reward's third tier scores content against this target
and an unanswerable prompt would make those points unavailable to any policy.
The task has to be one the base model can nearly do, or GRPO has nothing to
consolidate.

Like ``rewards.py``, this imports nothing outside the standard library.
"""

import random

if __package__:
    from .rewards import SCHEMA_KEYS, SEVERITIES
else:  # pragma: no cover - only used by direct script execution.
    from rewards import SCHEMA_KEYS, SEVERITIES

#: Component names the tickets refer to. Multi-word on purpose: a single token
#: would let the model score the component field from the prompt's shape alone.
COMPONENTS = ["auth service", "billing page", "search index", "checkout flow",
              "email worker", "admin console", "image uploader",
              "report builder"]

#: Four phrasings, so the model cannot learn one surface form and copy offsets
#: out of it. The fields appear in a different order in each.
TEMPLATES = [
    "Ticket {tid} came in about the {comp} failing. Marked {sev}. {state}.",
    "Issue {tid}: users report the {comp} misbehaving. Priority {sev}. {state}.",
    "Report {tid} - the {comp} is degraded, {sev} severity, {state}.",
    "Case {tid}: {comp} returned errors for several customers. Severity {sev}. "
    "{state}.",
]

#: The instruction the model sees, with the note substituted for ``{r}``.
#:
#: It names the four keys explicitly. That is not giving the answer away -- the
#: base model still has to emit them, in JSON, with the right values and the
#: right types, which is exactly what it does inconsistently and what the
#: reward is there to make consistent.
PROMPT = ("Convert the record to JSON with keys ticket_id, severity, "
          "component, resolved.\nRecord: {r}\nJSON: ")

#: The chapter's seed. Kept here rather than in the trainer so the dataset is
#: the same object whichever script builds it.
SEED = 0


def make_dataset(n=64, seed=SEED):
    """Builds ``n`` (note, target) pairs from a private ``random.Random``.

    A private generator rather than the module-level ``random`` functions:
    the trainer seeds the global RNG for torch's benefit, and a dataset that
    silently changed when a caller reseeded would be the worst kind of
    irreproducibility -- invisible until two runs disagree.

    ``resolved`` is a real ``bool``, not a string. The reward's type check
    compares against it with ``==``, and in Python ``True == "true"`` is
    ``False``, which is the whole point: a model that emits the string loses
    both the type point and the content point.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        tid = str(rng.randint(1000, 9999))
        sev = rng.choice(SEVERITIES)
        comp = rng.choice(COMPONENTS)
        done = rng.random() < 0.5
        text = rng.choice(TEMPLATES).format(
            tid=tid, comp=comp, sev=sev,
            state="Resolved" if done else "Still open")
        rows.append({
            "text": text,
            "target": {"ticket_id": tid, "severity": sev,
                       "component": comp, "resolved": done},
        })
    return rows


def format_prompt(row):
    """The prompt string for one dataset row."""
    return PROMPT.format(r=row["text"])


assert set(SCHEMA_KEYS) == {"ticket_id", "severity", "component", "resolved"}, (
    "make_dataset builds its target records key by key, so a change to "
    "SCHEMA_KEYS has to be mirrored here rather than picked up automatically."
)
