"""Tests for the Chapter 7 GRPO modules.

The chapter's split of dependencies is mirrored here. The reward function, the
dataset and the group-size analysis import nothing outside the standard
library, so those tests run on the lightweight setup; the objective needs torch
and is skipped without it; nothing in this file needs transformers, peft, or a
downloaded model, and nothing in it touches the network.

The reward function gets the most attention because it deserves it. In GRPO the
reward *is* the task -- the algorithm holds no opinion about what a good answer
looks like -- so a silent change to ``compute_json_reward`` changes what the
chapter trains for while every other test still passes.
"""

import importlib.util
import json
import math
import os

import pytest

from src.part_2_methods.ch07_grpo import dataset, group_size, reward_anatomy
from src.part_2_methods.ch07_grpo.rewards import (SCHEMA_KEYS, SEVERITIES,
                                                  compute_json_reward,
                                                  extract_json, is_compliant)

TARGET = dataset.make_dataset(1)[0]["target"]


# --------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------

def test_extract_json_stops_at_the_first_balanced_object():
    """A base model does not stop when the object is finished.

    It keeps generating -- usually a second copy of the record, or a fresh
    invented ticket -- so everything after the first balanced brace is noise.
    """
    text = '{"a": 1} and then {"b": 2} and some prose'
    assert extract_json(text) == '{"a": 1}'


def test_extract_json_handles_nesting():
    text = '{"outer": {"inner": 1}} trailing'
    assert extract_json(text) == '{"outer": {"inner": 1}}'


def test_extract_json_returns_none_for_prose_and_for_unterminated_objects():
    assert extract_json("no braces here") is None
    assert extract_json('{"ticket_id": "1", "severity": "high"') is None


# --------------------------------------------------------------------------
# The three tiers
# --------------------------------------------------------------------------

def test_exact_record_scores_the_maximum():
    assert compute_json_reward(json.dumps(TARGET), TARGET) == 6.0


def test_a_leading_fence_costs_exactly_the_format_points():
    """Format is charged +1 or -1, so a fence is a two-point swing and
    nothing else -- the object inside it still earns every other tier."""
    exact = json.dumps(TARGET)
    fenced = "```json\n" + exact + "\n```"
    assert compute_json_reward(exact, TARGET) - \
        compute_json_reward(fenced, TARGET) == 2.0


def test_missing_and_invented_keys_are_both_charged():
    """The `extra` term is what stops the cheapest path to a high score being
    "emit the schema plus every key you can think of"."""
    exact = json.dumps(TARGET)
    padded = json.dumps(dict(TARGET, priority=1, assignee="nobody"))
    assert compute_json_reward(padded, TARGET) == pytest.approx(
        compute_json_reward(exact, TARGET) - 0.4)

    half = json.dumps({k: TARGET[k] for k in SCHEMA_KEYS[:2]})
    assert compute_json_reward(half, TARGET) < compute_json_reward(exact,
                                                                   TARGET)


def test_a_string_boolean_loses_the_type_point_and_the_content_point():
    """`"resolved": "true"` is the single most common near-miss in this task,
    and `True == "true"` is False in Python, which is what makes it cost
    twice."""
    stringly = json.dumps(dict(TARGET, resolved=str(TARGET["resolved"]).lower()))
    assert compute_json_reward(stringly, TARGET) == pytest.approx(
        compute_json_reward(json.dumps(TARGET), TARGET) - 1.0)


def test_severity_vocabulary_is_case_sensitive():
    drifted = json.dumps(dict(TARGET, severity=TARGET["severity"].upper()))
    assert compute_json_reward(drifted, TARGET) < compute_json_reward(
        json.dumps(TARGET), TARGET)


def test_unparsable_output_is_charged_but_not_zeroed():
    """The graduated penalty is the learning signal. An object one character
    from correct must score above prose that never attempted JSON, or the
    model learns nothing from having nearly succeeded."""
    near_miss = json.dumps(TARGET)[:-1]
    prose = "Sure! Here is the information you asked for."
    assert compute_json_reward(near_miss, TARGET) > compute_json_reward(
        prose, TARGET)


def test_content_is_scored_against_this_ticket_not_a_constant():
    """Regression guard on the most hackable term in the function.

    A reward that paid for a fixed literal could be satisfied without reading
    the input at all, so a memorized record has to score below the right one.
    """
    rows = dataset.make_dataset(2)
    memorized = json.dumps(rows[1]["target"])
    assert compute_json_reward(memorized, rows[0]["target"]) < \
        compute_json_reward(json.dumps(rows[0]["target"]), rows[0]["target"])


def test_the_empty_object_hack_earns_only_the_syntax_tiers():
    """`{}` is what a parse-only reward overpays for. It parses, so it clears
    format and validity, and the schema and content tiers are unavailable."""
    assert compute_json_reward("{}", TARGET) == 0.0


def test_reward_needs_no_third_party_import():
    """`rewards.py` is the module readers replace with their own, so it has to
    run on a bare interpreter."""
    source = (dataset.__file__.rsplit("/", 1)[0] + "/rewards.py")
    with open(source) as handle:
        imports = [line for line in handle if line.startswith("import ")]
    assert imports == ["import json\n"]


# --------------------------------------------------------------------------
# is_compliant -- reporting, not training
# --------------------------------------------------------------------------

def test_compliance_requires_exactly_the_schema_and_real_types():
    assert is_compliant(json.dumps(TARGET), TARGET)
    assert not is_compliant(json.dumps(dict(TARGET, extra=1)), TARGET)
    assert not is_compliant(
        json.dumps(dict(TARGET, resolved="true")), TARGET)
    assert not is_compliant("```json\n" + json.dumps(TARGET), TARGET)
    assert not is_compliant("no json at all", TARGET)


def test_compliance_is_structural_and_does_not_check_content():
    """Which is why the reward's third tier exists: a memorized record is
    compliant and wrong, and only the content tier can tell."""
    rows = dataset.make_dataset(2)
    assert is_compliant(json.dumps(rows[1]["target"]), rows[0]["target"])


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

def test_dataset_is_reproducible_and_independent_of_the_global_rng():
    import random
    random.seed(1234)
    first = dataset.make_dataset(8)
    random.seed(9999)
    assert dataset.make_dataset(8) == first


def test_every_target_is_recoverable_from_its_prompt():
    """An unanswerable prompt would make the content tier unavailable to any
    policy, which would make that third of the reward pure noise."""
    for row in dataset.make_dataset(32):
        text = row["text"]
        assert row["target"]["ticket_id"] in text
        assert row["target"]["severity"] in text
        assert row["target"]["component"] in text
        state = "Resolved" if row["target"]["resolved"] else "Still open"
        assert state in text


def test_resolved_is_a_real_bool_and_severity_is_in_the_vocabulary():
    for row in dataset.make_dataset(32):
        assert isinstance(row["target"]["resolved"], bool)
        assert row["target"]["severity"] in SEVERITIES


def test_prompt_names_every_schema_key():
    for key in SCHEMA_KEYS:
        assert key in dataset.PROMPT


# --------------------------------------------------------------------------
# reward_anatomy -- the figure's numbers, pinned
# --------------------------------------------------------------------------

#: Figure 7.6 in the chapter. Pinned here so a change to the reward function
#: fails a test rather than silently invalidating a printed figure.
FIGURE_7_6 = {
    "Exact target record": 6.00,
    "Two invented extra keys": 5.60,
    "Boolean sent as a string": 5.00,
    "Right answer in a code fence": 4.00,
    "Half the schema missing": 2.50,
    "Keys in the wrong case": -0.40,
    "Unterminated object": -1.00,
    "Prose, no JSON at all": -3.00,
}


def test_reward_anatomy_reproduces_the_chapter_figure():
    scored = {name: value for name, _, value, _ in reward_anatomy.score_table()}
    assert scored == pytest.approx(FIGURE_7_6)


def test_reward_anatomy_runs_and_prints_both_tables():
    lines = []
    reward_anatomy.run(printer=lines.append)
    text = "\n".join(lines)
    assert "Exact target record" in text
    assert "Empty object" in text


# --------------------------------------------------------------------------
# group_size -- exercise 3
# --------------------------------------------------------------------------

def test_zero_gradient_probability_is_the_expression_and_not_a_simulation():
    assert group_size.zero_gradient_probability(0.5, 2) == 0.5
    assert group_size.zero_gradient_probability(0.9, 8) == pytest.approx(
        0.9 ** 8 + 0.1 ** 8)
    # A model that is always right never produces a gradient, at any G.
    assert group_size.zero_gradient_probability(1.0, 64) == 1.0


def test_collapse_gets_worse_as_the_policy_improves():
    """The trap the chapter names: p climbs as training succeeds, and the
    prompts the model has mastered stop producing any signal."""
    at_50 = group_size.zero_gradient_probability(0.5, 8)
    at_90 = group_size.zero_gradient_probability(0.9, 8)
    assert at_90 > at_50


def test_a_graded_reward_collapses_less_often_than_a_binary_one():
    """The chapter's argument for graduated penalties, as arithmetic.

    Eight candidates spread over the eight scores of figure 7.6 collapse far
    less often than the same eight split into two verdicts.
    """
    graded = [value for _, _, value, _ in reward_anatomy.score_table()]
    binary = [1.0 if v >= 5.0 else 0.0 for v in graded]
    for g in (2, 4, 8):
        assert group_size.collapse_probability(graded, g) < \
            group_size.collapse_probability(binary, g)


def test_collapse_probability_of_a_single_valued_sample_is_one():
    assert group_size.collapse_probability([4.0] * 10, 8) == pytest.approx(1.0)


def test_group_size_reports_history_written_by_the_trainer(tmp_path):
    """The history contract between the two scripts, without a model.

    ``group_size.py --history`` is useless if the trainer stops recording
    per-candidate scores, and that is a change nothing else would catch.
    """
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "config": {"seed": 0, "group_size": 4},
        "history": [
            {"step": 0, "rewards": [6.0, 6.0, 6.0, 6.0], "skipped": True},
            {"step": 1, "rewards": [6.0, 4.0, -1.0, 2.5], "skipped": False},
        ],
    }))
    scores, observed, config = group_size.pooled_scores(str(path))
    assert len(scores) == 8
    assert observed == 0.5
    assert config["group_size"] == 4

    lines = []
    group_size.run(history=str(path), printer=lines.append)
    assert "graded (measured)" in "\n".join(lines)


def test_pooled_scores_rejects_a_history_with_no_candidate_scores(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(
        {"history": [{"step": 0, "reward": 4.0, "skipped": False}]}))
    with pytest.raises(ValueError, match="per-candidate rewards"):
        group_size.pooled_scores(str(path))


# --------------------------------------------------------------------------
# The objective. Needs torch; the tests above deliberately do not.
# --------------------------------------------------------------------------
#
# Skipped per-test rather than by a module-level ``importorskip``, which is
# what chapters 3 to 6 use. Those chapters are torch all the way down, so
# skipping the file is the whole file's honest state. Here the reward function
# and the group-size analysis run on a bare interpreter, and a reader on the
# lightweight setup should still see them pass.

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="Chapter 7's objective needs torch (make install-llm)",
)


@requires_torch
def test_group_advantages_are_zero_mean_within_each_group():
    import torch

    from src.part_2_methods.ch07_grpo.grpo import group_advantages

    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    advantages = group_advantages(rewards, 4).view(2, 4)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(2), atol=1e-5)
    # Standardized within the group, so a tenfold larger reward scale in the
    # second group produces the same advantages. That scale invariance is what
    # makes the baseline "relative".
    assert torch.allclose(advantages[0], advantages[1], atol=1e-4)


@requires_torch
def test_a_degenerate_group_yields_zeros_rather_than_nan():
    """The 1e-4 in the denominator. Every candidate scoring alike is a real
    occurrence once a model becomes reliable on easy prompts, and without the
    guard it divides by zero."""
    import torch

    from src.part_2_methods.ch07_grpo.grpo import group_advantages

    advantages = group_advantages(torch.full((8,), 4.0), 8)
    assert torch.all(torch.isfinite(advantages))
    assert torch.allclose(advantages, torch.zeros(8))


@requires_torch
def test_group_advantages_rejects_a_ragged_batch():
    import torch

    from src.part_2_methods.ch07_grpo.grpo import group_advantages

    with pytest.raises(ValueError, match="whole number of groups"):
        group_advantages(torch.zeros(7), 4)


@requires_torch
def test_kl_penalty_is_never_negative_and_is_zero_only_at_agreement():
    """The bug this guards is not small. A naive mean log-ratio is signed, and
    when it goes negative the penalty subtracts from the loss -- rewarding
    exactly the drift it was added to prevent. The KL term is the only thing
    holding the policy near a model that still writes readable text."""
    import torch

    from src.part_2_methods.ch07_grpo.grpo import kl_penalty

    logp = torch.linspace(-5, 5, 101)
    ref = torch.zeros_like(logp)
    values = kl_penalty(logp, ref)
    assert torch.all(values >= 0)
    assert kl_penalty(torch.tensor([2.0]), torch.tensor([2.0])).item() == \
        pytest.approx(0.0, abs=1e-7)
    # Signed alternative, for contrast: it goes negative exactly where the
    # estimator must not.
    assert torch.any((ref - logp) < 0)


@requires_torch
def test_loss_is_zero_with_a_single_update_but_the_gradient_is_not():
    """The most confusing thing in the chapter's run, asserted.

    One update per exploration stage means the behavior policy is the current
    policy, the ratio is exactly 1, and the surrogate collapses to the negative
    mean of advantages that sum to zero by construction. Differentiating
    through the ratio still recovers REINFORCE-with-baseline, which is why the
    trainer reports the gradient norm rather than the loss.
    """
    import torch

    from src.part_2_methods.ch07_grpo.grpo import grpo_loss

    logp = torch.tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss = grpo_loss(logp, logp.detach(), logp.detach(), rewards, 4)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert logp.grad.abs().sum().item() > 0


@requires_torch
def test_the_gradient_pushes_toward_above_average_candidates():
    """The sign convention, which is easy to invert and hard to notice.

    The loss is minimized, so the gradient of the *loss* with respect to a
    candidate's log-probability must be negative where its advantage is
    positive -- a gradient step raises the probability of the candidates that
    beat their group.
    """
    import torch

    from src.part_2_methods.ch07_grpo.grpo import grpo_loss

    logp = torch.zeros(4, requires_grad=True)
    rewards = torch.tensor([0.0, 0.0, 0.0, 8.0])
    grpo_loss(logp, logp.detach(), logp.detach(), rewards, 4).backward()

    assert logp.grad[3] < 0            # best in group: push its logprob up
    assert torch.all(logp.grad[:3] > 0)  # the rest: push theirs down


@requires_torch
def test_clipping_binds_once_the_ratio_leaves_the_trust_region():
    """Inert in the single-update setting and load-bearing the moment a reader
    takes more than one gradient step per group, which is why it is kept."""
    import torch

    from src.part_2_methods.ch07_grpo.grpo import grpo_loss

    rewards = torch.tensor([0.0, 0.0, 0.0, 8.0])
    old = torch.zeros(4)
    drifted = torch.tensor([0.0, 0.0, 0.0, 2.0])  # ratio e^2 on the best one

    unclipped = grpo_loss(drifted, old, old, rewards, 4, clip_eps=10.0,
                          kl_beta=0.0)
    clipped = grpo_loss(drifted, old, old, rewards, 4, clip_eps=0.2,
                        kl_beta=0.0)
    assert clipped > unclipped


@requires_torch
def test_train_step_skips_a_group_with_no_variance(monkeypatch):
    """The guard DAPO's dynamic sampling generalizes, tested without a model.

    ``train_json.train_step`` talks only to a policy object, so a stub that
    returns canned completions exercises the whole step -- the skip guard, the
    optimizer step, and the history record -- with no weights and no network.
    """
    import torch

    from src.part_2_methods.ch07_grpo import train_json

    class StubPolicy:
        """Returns `texts` for every group, and a trainable log-prob."""

        def __init__(self, texts):
            self.texts = texts
            self.device = "cpu"
            self.logp = torch.zeros(len(texts), requires_grad=True)

        def sample_group(self, prompt, n, **kwargs):
            return self.texts, torch.zeros(n, 3, dtype=torch.long), \
                torch.zeros(n, 2, dtype=torch.long)

        def sequence_logprob(self, prompt_ids, gen_ids, use_adapter=True):
            return self.logp if use_adapter else self.logp.detach()

        def trainable_parameters(self):
            return [self.logp]

    row = dataset.make_dataset(1)[0]
    identical = [json.dumps(row["target"])] * 4
    policy = StubPolicy(identical)
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)

    record = train_json.train_step(policy, row, 4, optimizer)
    assert record["skipped"] is True
    assert record["loss"] is None
    assert record["rewards"] == [6.0] * 4

    varied = identical[:3] + ["not json at all"]
    policy = StubPolicy(varied)
    optimizer = torch.optim.SGD(policy.trainable_parameters(), lr=0.1)
    record = train_json.train_step(policy, row, 4, optimizer)
    assert record["skipped"] is False
    assert record["gnorm"] > 0
    assert record["rewards"][-1] == -3.0


# --------------------------------------------------------------------------
# policy.py. Also torch-only: the transformers and peft imports live inside
# `LoRAPolicy.load`, so everything below constructs the policy directly around
# the stub model in conftest and never loads a weight or touches the network.
# --------------------------------------------------------------------------
#
# This is the module where a mistake is quietest. Its own docstrings say so --
# an off-by-one in the shift is silent, and right-padding produces fluent
# nonsense and no error at all -- and until these tests it was the only module
# in the chapter with nothing checking it.

@requires_torch
def test_pick_device_prefers_cuda_and_treats_mps_as_opt_in(monkeypatch):
    """MPS is opt-in because it measured *slower* here, not because it is new.

    A future reader tidying `pick_device` into "use the best accelerator
    available" would undo a measurement -- 4.2 tokens/second against 7.4 on
    plain CPU -- rather than a stylistic choice.
    """
    import torch

    from src.part_2_methods.ch07_grpo.policy import pick_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert pick_device() == "cuda"
    assert pick_device(force_mps=True) == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True,
                        raising=False)
    # Available and still not chosen: that is the whole point of the flag.
    assert pick_device() == "cpu"

    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert pick_device(force_mps=True) == "mps"
    # Several generation ops have no MPS kernel. Without the fallback the run
    # dies partway through the first group rather than at load.
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False,
                        raising=False)
    assert pick_device(force_mps=True) == "cpu"


@requires_torch
def test_sequence_logprob_aligns_each_logit_with_the_token_it_predicts(
        ch07_oracle_lm):
    """The shift, pinned. Position ``t`` predicts token ``t + 1``.

    The stub model puts its mass on the token that actually comes next, so
    the correct alignment scores essentially zero and every other alignment
    scores about ``-confidence``. An off-by-one in either direction moves this
    number by more than seven nats -- which is the point, because in a real
    run it moves nothing a reader would notice.
    """
    import torch

    from src.part_2_methods.ch07_grpo.policy import LoRAPolicy

    policy = LoRAPolicy(ch07_oracle_lm.model, ch07_oracle_lm.tokenizer)
    prompt_ids = torch.tensor([[1, 2, 3]])
    gen_ids = torch.tensor([[4, 5, 6]])

    value = policy.sequence_logprob(prompt_ids, gen_ids).item()

    others = ch07_oracle_lm.vocab - 1
    aligned = math.log(math.exp(ch07_oracle_lm.confidence)
                       / (math.exp(ch07_oracle_lm.confidence) + others))
    misaligned = math.log(1.0 / (math.exp(ch07_oracle_lm.confidence) + others))
    assert value == pytest.approx(aligned, abs=1e-5)
    assert aligned - misaligned > 7.0


@requires_torch
def test_sequence_logprob_does_not_score_padding(ch07_oracle_lm):
    """Padding is not part of the policy's output, so it is not averaged in.

    The stub is deliberately *not* confident about pad tokens, so an unmasked
    pad position would drag the mean down by several nats. Two rows of the
    same real tokens and different pad counts have to score the same.
    """
    import torch

    from src.part_2_methods.ch07_grpo.policy import LoRAPolicy

    policy = LoRAPolicy(ch07_oracle_lm.model, ch07_oracle_lm.tokenizer)
    prompt_ids = torch.tensor([[1, 2, 3]])
    pad = ch07_oracle_lm.pad_id

    unpadded = policy.sequence_logprob(prompt_ids, torch.tensor([[4, 5]]))
    padded = policy.sequence_logprob(prompt_ids,
                                     torch.tensor([[4, 5, pad, pad]]))
    assert padded.item() == pytest.approx(unpadded.item(), abs=1e-6)

    # And the guard against a row that is nothing but padding: a zero
    # denominator would return nan and poison the whole group's loss.
    allpad = policy.sequence_logprob(prompt_ids, torch.tensor([[pad, pad]]))
    assert math.isfinite(allpad.item())


@requires_torch
def test_the_reference_branch_is_frozen_and_the_adapter_branch_is_not(
        ch07_oracle_lm):
    """The free reference policy, and the `enable_grad` that is not ambient.

    Two properties, and both are silent when broken. The reference branch has
    to run under `no_grad` -- it is never trained, and holding its graph
    doubles the memory for nothing. The adapter branch has to carry gradient
    *even when the caller is inside* `no_grad`, which is where a trainer that
    scores inside its own sampling block ends up: without the explicit
    `enable_grad` the loss would be detached and the run would train nothing,
    with no error anywhere.
    """
    import torch

    from src.part_2_methods.ch07_grpo.policy import LoRAPolicy

    policy = LoRAPolicy(ch07_oracle_lm.model, ch07_oracle_lm.tokenizer)
    prompt_ids = torch.tensor([[1, 2, 3]])
    gen_ids = torch.tensor([[4, 5, 6]])

    with torch.no_grad():
        logp = policy.sequence_logprob(prompt_ids, gen_ids, use_adapter=True)
        ref = policy.sequence_logprob(prompt_ids, gen_ids, use_adapter=False)

    assert logp.requires_grad
    assert not ref.requires_grad
    # Disabling the adapter has to actually change the distribution, or the KL
    # term is measuring the policy against itself and is always zero.
    assert ref.item() == pytest.approx(-math.log(ch07_oracle_lm.vocab),
                                       abs=1e-5)
    assert logp.item() > ref.item()


@requires_torch
def test_group_size_reads_what_the_trainer_actually_writes(tmp_path):
    """The history contract, round-tripped rather than assumed.

    The fixture above states the format by hand, which leaves the two scripts
    free to drift apart together: rename `rewards` in `write_history` and that
    test still passes while `--history` stops working. This one writes the
    file with the trainer's own function and reads it with the analysis
    script's own reader.
    """
    from src.part_2_methods.ch07_grpo import train_json

    result = train_json.RunResult(
        history=[
            {"step": 0, "reward": 2.875, "rewards": [6.0, 4.0, -1.0, 2.5],
             "loss": 0.0, "gnorm": 0.31, "skipped": False},
            {"step": 1, "reward": 6.0, "rewards": [6.0] * 4,
             "loss": None, "gnorm": None, "skipped": True},
        ],
        baseline=0.25, after=0.5, elapsed=12.0, skipped=1)

    path = train_json.write_history(result, str(tmp_path / "run.json"),
                                    seed=0, group_size=4)

    scores, observed, config = group_size.pooled_scores(path)
    assert len(scores) == 8
    assert observed == 0.5
    assert config == {"seed": 0, "group_size": 4}

    lines = []
    group_size.run(history=path, printer=lines.append)
    assert "graded (measured)" in "\n".join(lines)
