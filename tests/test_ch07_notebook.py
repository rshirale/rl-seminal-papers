"""Tests for the Chapter 7 notebook.

The notebook re-implements the reward, the dataset and the objective inline so
a reader can follow the derivation cell by cell. That duplication is a teaching
choice and it is also how a notebook drifts away from the modules it mirrors --
chapter 5's had drifted in both directions at once before anyone tested it, and
chapter 2's shipped a KeyError that every reader hit.

Three kinds of check here:

  * structure  -- the hooks, pins and seeding the other chapters' notebooks have
  * parity     -- the inline functions still agree with the modules, numerically
  * execution  -- the notebook actually runs top to bottom without raising

Most of this file needs no model and no network. Only the execution test loads
weights, and it skips itself unless they are already in the huggingface cache,
so `make test-all` never triggers a gigabyte of downloads on someone's laptop.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CH7 = ROOT / "src" / "part_2_methods" / "ch07_grpo"
NOTEBOOK = CH7 / "Chapter7_GRPO.ipynb"

MODEL_ID = "Qwen/Qwen2.5-0.5B"

has_llm_stack = all(importlib.util.find_spec(m) is not None
                    for m in ("torch", "transformers", "peft"))
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs torch")


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing `marker`.

    Located by content rather than index so inserting a cell does not silently
    break these tests -- the mistake chapter 6's suite documents.
    """
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


def executable(cell):
    """The cell with comments stripped.

    Prose and commented-out Colab lines may legitimately name things the
    executable code must not.
    """
    return "\n".join(line.split("#")[0] for line in cell.splitlines())


@pytest.fixture(scope="module")
def notebook_ns():
    """The notebook's dependency-free definitions, executed.

    Deliberately stops short of the model-loading cell: the reward, the
    dataset and the objective are the parts that must agree with the modules,
    and none of them needs weights.
    """
    # The names the setup cell would have bound. Supplied directly rather than
    # by executing that cell, which imports transformers and peft.
    namespace = {"__name__": "__notebook__", "json": json,
                 "random": __import__("random"), "SEED": 0}
    exec(compile(find_cell("SCHEMA_KEYS"), "<notebook>", "exec"), namespace)
    exec(compile(find_cell("def compute_json_reward"), "<notebook>", "exec"),
         namespace)
    return namespace


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_notebook_is_valid_json_and_every_cell_has_an_id():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    assert nb["cells"]
    assert all("id" in cell for cell in nb["cells"])


def test_colab_install_pins_match_requirements_llm():
    """The Colab cell is the only install path a Colab reader ever sees, so it
    must not drift from the repo's own bounds. Chapter 6's notebook had
    drifted this way -- it pinned a gymnasium range the repo did not use."""
    install = find_cell("pip install")
    requirements = (ROOT / "requirements-llm.txt").read_text()
    for spec in ('"transformers>=4.44,<5.0"', '"peft>=0.12,<1.0"'):
        assert spec in install, f"notebook install cell is missing {spec}"
        assert spec.strip('"') in requirements, (
            f"requirements-llm.txt does not carry {spec}")


def test_setup_cell_seeds_every_generator_and_pins_the_threads():
    """Token sampling *is* the exploration in this chapter, and it draws from
    torch's generator, so an unseeded run reports a different transcript every
    time while looking fully seeded."""
    setup = executable(find_cell("SEED = 0"))
    assert "random.seed(SEED)" in setup
    assert "np.random.seed(SEED)" in setup
    assert "torch.manual_seed(SEED)" in setup
    assert "torch.set_num_threads(1)" in setup


def test_budget_and_model_are_overridable():
    """Guards the hooks the execution test depends on."""
    assert "CH7_MODEL_ID" in find_cell("MODEL_ID")
    training = find_cell("GROUP_SIZE")
    assert "CH7_STEPS" in training
    assert "CH7_GROUP_SIZE" in training
    assert "CH7_EVAL_ROWS" in find_cell("EVAL_ROWS")
    assert "CH7_MAX_NEW_TOKENS" in find_cell("MAX_NEW_TOKENS")


def test_notebook_starts_from_a_base_model_not_the_instruct_variant():
    """The whole result depends on this. The -Instruct variant already solves
    the task, which would leave GRPO nothing to learn and turn the run into a
    demonstration that instruction tuning works."""
    cell = executable(find_cell("MODEL_ID"))
    assert "Qwen/Qwen2.5-0.5B'" in cell
    assert "-Instruct" not in cell


def test_reference_policy_comes_from_disabling_the_adapters():
    """The free reference model -- the reason LoRA and GRPO pair so well. A
    notebook that loaded a second copy of the weights instead would still run
    and would quietly double the memory."""
    cell = executable(find_cell("def sequence_logprob"))
    assert "disable_adapter()" in cell
    assert "from_pretrained" not in cell


def test_training_cell_skips_zero_variance_groups():
    """A group whose members all score alike carries no signal; every
    advantage in it is zero. This is DAPO's dynamic-sampling condition."""
    cell = executable(find_cell("GROUP_SIZE"))
    assert "rewards.std()" in cell
    assert "continue" in cell


def test_training_cell_takes_one_update_per_exploration_stage():
    """Which is what makes the loss exactly zero and the gradient large -- the
    behavior policy and the current policy are the same object."""
    cell = executable(find_cell("GROUP_SIZE"))
    assert "old_logp = logp.detach()" in cell


def test_no_magics_are_left_uncommented():
    """The execution test runs cells in-process rather than through a Jupyter
    kernel, which would resolve a kernelspec that can point at a different
    interpreter than the one running the tests."""
    for source in code_cells():
        for line in source.splitlines():
            assert not line.lstrip().startswith("%"), f"live magic: {line}"


# --------------------------------------------------------------------------
# Parity with the modules
# --------------------------------------------------------------------------

def test_notebook_reward_matches_the_module_on_every_figure_case(notebook_ns):
    """The parity check that matters most: the reward is the task."""
    from src.part_2_methods.ch07_grpo import reward_anatomy
    from src.part_2_methods.ch07_grpo.rewards import compute_json_reward

    target = reward_anatomy.TARGET
    for _, text, module_value, _ in reward_anatomy.score_table():
        assert notebook_ns["compute_json_reward"](text, target) == \
            pytest.approx(module_value)
    for _, text, module_value, _ in reward_anatomy.hack_table():
        assert notebook_ns["compute_json_reward"](text, target) == \
            pytest.approx(module_value)
    # And the module agrees with itself through the package export.
    assert compute_json_reward("{}", target) == 0.0


def test_notebook_compliance_matches_the_module(notebook_ns):
    from src.part_2_methods.ch07_grpo import reward_anatomy
    from src.part_2_methods.ch07_grpo.rewards import is_compliant

    target = reward_anatomy.TARGET
    for _, text, _, module_value in (reward_anatomy.score_table()
                                     + reward_anatomy.hack_table()):
        assert bool(notebook_ns["is_compliant"](text, target)) is module_value
        assert bool(is_compliant(text, target)) is module_value


def test_notebook_dataset_is_the_same_dataset(notebook_ns):
    """Same rows in the same order, or the notebook and the modules are
    training on different tasks while both calling it chapter 7."""
    from src.part_2_methods.ch07_grpo.dataset import make_dataset

    assert notebook_ns["make_dataset"](16) == make_dataset(16)
    assert notebook_ns["SCHEMA_KEYS"] == list(
        __import__("src.part_2_methods.ch07_grpo.rewards", fromlist=["x"]
                   ).SCHEMA_KEYS)


@requires_torch
def test_notebook_objective_matches_the_module_numerically():
    """Same tensors in, same scalar out. Listing 7.4 lives in two places and
    they have to stay one implementation."""
    import torch

    from src.part_2_methods.ch07_grpo.grpo import grpo_loss

    namespace = {"torch": torch}
    exec(compile(find_cell("def grpo_loss"), "<notebook>", "exec"), namespace)

    torch.manual_seed(0)
    logp = torch.randn(8)
    old = torch.randn(8)
    ref = torch.randn(8)
    rewards = torch.tensor([6.0, 4.0, -1.0, 2.5, 5.6, 0.0, -3.0, 5.0])

    assert namespace["grpo_loss"](logp, old, ref, rewards, 4).item() == \
        pytest.approx(grpo_loss(logp, old, ref, rewards, 4).item(), abs=1e-6)


# --------------------------------------------------------------------------
# Full execution
# --------------------------------------------------------------------------

def _weights_are_cached(model_id=MODEL_ID):
    """True only if the model is already on disk.

    The execution test is worth having and is not worth downloading a
    gigabyte for without being asked. Run the notebook or `make run-ch7-train`
    once and this test starts running.
    """
    if not has_llm_stack:
        return False
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _weights_are_cached(),
                    reason="Chapter 7's weights are not in the local cache; "
                           "run `make run-ch7-train` once to fetch them")
def test_notebook_runs_top_to_bottom(monkeypatch):
    """Executes every code cell in order, exactly as a reader would.

    Driven down to one step of a group of two with an eight-token budget: the
    point is that every cell runs and the pieces fit, not that a one-step run
    teaches the model anything.
    """
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("CH7_STEPS", "1")
    monkeypatch.setenv("CH7_GROUP_SIZE", "2")
    monkeypatch.setenv("CH7_EVAL_ROWS", "1")
    monkeypatch.setenv("CH7_MAX_NEW_TOKENS", "8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(CH7)

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    assert len(namespace["history"]) == 1
    assert 0.0 <= namespace["baseline"] <= 1.0
    assert 0.0 <= namespace["after"] <= 1.0
    # Whether the single step was taken or skipped depends on the sample, but
    # the record has to say which.
    assert "skipped" in namespace["history"][0]


@requires_torch
def test_notebook_logprob_matches_the_module(ch07_oracle_lm):
    """The one inline reimplementation that had no parity test.

    The reward, the dataset, compliance and the objective are all checked
    against their modules above. ``sequence_logprob`` was not, and it is the
    one where drift is hardest to see by eye: the shift, the grad-mode
    context and the padding mask are three details that produce plausible
    numbers when they are wrong. Both versions are driven with the same stub
    model, so this compares the code rather than the weights.

    The docstring is checked too, because it drifted: it said "Summed"
    while the line beneath it divided by the token count, contradicting the
    chapter's own note on the short-response bias.
    """
    import torch

    from src.part_2_methods.ch07_grpo.policy import LoRAPolicy

    cell = find_cell("def sequence_logprob")
    assert "Summed" not in cell, "the notebook returns a mean, not a sum"

    namespace = {"torch": torch, "model": ch07_oracle_lm.model,
                 "tokenizer": ch07_oracle_lm.tokenizer}
    exec(compile(cell, "<notebook>", "exec"), namespace)

    policy = LoRAPolicy(ch07_oracle_lm.model, ch07_oracle_lm.tokenizer)
    prompt_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
    gen_ids = torch.tensor([[4, 5, 6], [4, 5, ch07_oracle_lm.pad_id]])

    for use_adapter in (True, False):
        theirs = namespace["sequence_logprob"](prompt_ids, gen_ids,
                                               use_adapter=use_adapter)
        ours = policy.sequence_logprob(prompt_ids, gen_ids,
                                       use_adapter=use_adapter)
        assert torch.allclose(theirs.detach(), ours.detach(), atol=1e-6)
        # Not just the same numbers: the same grad mode. A notebook that
        # returned a detached tensor here would train nothing.
        assert theirs.requires_grad == ours.requires_grad == use_adapter


def test_notebook_left_pads_for_batched_generation():
    """The notebook samples a group as one batch too, so it needs this too.

    A reader who copies the setup cell into their own project and drops the
    padding line gets fluent nonsense and no error -- see the module test of
    the same name. Checked on the executable code rather than the cell text,
    so a comment mentioning left padding cannot satisfy it.
    """
    cell = executable(find_cell("padding_side"))
    assert ("padding_side = 'left'" in cell
            or 'padding_side = "left"' in cell), \
        "batched generation continues right-padded rows from their padding"
    # And the pad token itself: without one there is nothing to pad with.
    assert "pad_token = tokenizer.eos_token" in cell
