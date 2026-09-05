"""The rule-based reward that supplies Chapter 7's entire training signal.

This is listing 7.2. GRPO holds no opinion about what a good answer looks
like -- the reward function holds all of it, and nothing else in the method
does -- so this module, not ``grpo.py``, is where the task actually lives.

The task is converting an unstructured support-ticket note into a record with
four keys. It was chosen because it is the clearest case of a reward a rule can
compute: a candidate either parses and conforms or it does not, and no judgment
enters anywhere. That property, not mathematics, is what GRPO depends on.

Three tiers, in order, and the order is deliberate:

  1. **Format.** Raw JSON scores, prose and markdown fences are charged. A
     model that wraps its answer in a fence has produced something no parser
     downstream will accept.
  2. **Schema conformance.** Charged in both directions -- a missing key costs,
     and so does an invented one.
  3. **Semantic accuracy.** Scored against the *target record for that ticket*,
     never against a fixed literal. Rewarding a constant would be the easiest
     term in the function to hack, because the model could satisfy it without
     reading the input at all.

Nothing here imports torch, transformers or numpy, and that is deliberate:
the reward is the part of a GRPO pipeline that a reader is most likely to
replace with their own, and it should be runnable and testable on a bare
interpreter. ``reward_anatomy.py`` runs a range of plausible model outputs
through it and prints what each one scores.
"""

import json

#: The schema the model is being trained to emit. Both the reward and the
#: dataset generator read this list, so adding a key changes the task in one
#: place rather than two.
SCHEMA_KEYS = ["ticket_id", "severity", "component", "resolved"]

#: The closed vocabulary the ``severity`` field must be drawn from. Checked
#: case-sensitively: "High" is a formatting drift a downstream consumer would
#: reject, so the reward rejects it too.
SEVERITIES = ["low", "medium", "high", "critical"]


def extract_json(text):
    """Returns the first balanced ``{...}`` block in ``text``, or ``None``.

    Brace-counted rather than regex-matched, and deliberately not a
    ``json.loads`` on the whole string. A base model that has never been
    instruction-tuned does not stop when the object is finished -- it keeps
    generating, usually another copy of the record or a fresh invented ticket
    -- so the useful signal is the first complete object and everything after
    it is noise. Scanning to the first balanced brace ignores that tail.

    It returns the *unparsed* substring, so the caller can distinguish "no
    object was attempted" from "an object was attempted and did not parse".
    Those two failures are charged differently, and collapsing them would
    throw away the distinction the graduated penalties exist to preserve.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def compute_json_reward(output_text, target):
    """Scores one completion against the known correct record for its ticket.

    Higher is better. The range over realistic model output is about -3 to
    +6: -3 is prose that never attempts JSON, +6 is the exact target record.

    The graduated penalties are the point, and they are worth defending
    because an all-or-nothing rule is the obvious alternative. An object that
    is correct right up to a missing closing brace scores -1.00; keys in the
    wrong case score -0.40; prose scores -3.00. Under a rule that simply
    zeroed every unparsable output all three would be indistinguishable, and
    the model would learn nothing from the fact that one of them was a single
    character away from correct. That spread *is* the learning signal.

    It also buys tolerance of small groups. A group carries no gradient only
    when every member scores identically (see ``group_size.py``); a graded
    reward makes that far rarer than a binary one, which is why the chapter's
    40-step run at G = 8 never once hit the zero-variance guard.

    ``target`` is the record dict for this specific ticket -- see
    ``dataset.py``. Passing a constant would make the third tier hackable.
    """
    reward = 0.0
    text = output_text.strip()

    # Tier 1 -- format. Raw JSON only, no prose and no markdown fence.
    if text.startswith("{"):
        reward += 1.0
    else:
        reward -= 1.0

    blob = extract_json(text)
    if blob is None:
        return reward - 2.0
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        # Charged, but not zeroed: an unparsable near-miss is worth more than
        # prose, and the gap between them is what teaches the syntax.
        return reward - 2.0
    if not isinstance(data, dict):
        return reward - 1.0
    reward += 1.0                                    # parsed as an object

    # Tier 2 -- schema conformance, charged in both directions. Without the
    # `extra` term the cheapest way to a high score is to emit the schema plus
    # every plausible key the model can think of.
    missing = set(SCHEMA_KEYS) - set(data)
    extra = set(data) - set(SCHEMA_KEYS)
    reward += 1.0 if not missing else -0.5 * len(missing)
    reward -= 0.2 * len(extra)

    # Value vocabulary and value type. `"resolved": "true"` is a string, and
    # the isinstance check is what separates it from a real boolean -- the
    # single most common near-miss in this task.
    if str(data.get("severity", "")) in SEVERITIES:
        reward += 0.5
    if isinstance(data.get("resolved"), bool):
        reward += 0.5

    # Tier 3 -- content, against this ticket's own target record.
    for key in SCHEMA_KEYS:
        if key in data and data[key] == target[key]:
            reward += 0.5
    return reward


def is_compliant(output_text, target=None):
    """Strict pass/fail, used for *reporting* and never for training.

    A completion counts only if it starts with a brace, parses, carries
    exactly the four schema keys, draws its severity from the fixed
    vocabulary, and gives a real boolean rather than the string ``"true"``.

    This is deliberately not the training signal. Training on a bit would
    throw away the near-miss structure the graduated reward exists to
    preserve; reporting on the graded score would be unreadable, because a
    mean of 4.6 does not say whether anything downstream can consume the
    output. The chapter quotes both -- compliance moved 42% -> 83% while mean
    group reward moved 4.23 -> 4.61 -- and the gap between those two numbers
    is the honest picture of what 40 steps of GRPO did.

    ``target`` is accepted and ignored so this can be called wherever
    ``compute_json_reward`` is. Compliance is a structural property; content
    correctness is scored, not gated.
    """
    text = output_text.strip()
    blob = extract_json(text)
    if blob is None or not text.startswith("{"):
        return False
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return False
    return (isinstance(data, dict) and set(data) == set(SCHEMA_KEYS)
            and str(data.get("severity")) in SEVERITIES
            and isinstance(data.get("resolved"), bool))
