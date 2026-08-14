"""Tests for the Chapter 3 notebook.

The notebook deliberately re-implements DQN inline so readers can follow the
derivation cell by cell. That duplication is a teaching choice, but it means
the notebook can silently drift away from the module files it mirrors. These
tests cover both risks:

  * parity  -- the notebook's inline classes still behave like the modules
  * execution -- the notebook actually runs top to bottom without raising

The execution test is marked ``slow`` because it runs a real training loop.
Run everything with ``make test-all``; the default ``make test`` skips it.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

CH3 = (
    Path(__file__).resolve().parents[1]
    / "src" / "part_2_methods" / "ch03_dqn"
)
NOTEBOOK = CH3 / "Chapter3_DQN.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing `marker`. Located by content rather than
    index so inserting a cell does not silently break these tests."""
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


@pytest.fixture(scope="module")
def notebook_ns():
    """Executes the notebook's definition cells (not its training loop) and
    returns the resulting namespace."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI during tests

    namespace = {"__name__": "__notebook__"}
    for marker in ("import gymnasium", "class DQN(nn.Module)",
                   "class ReplayBuffer", "class DQNAgent"):
        exec(compile(find_cell(marker), "<notebook>", "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_notebook_is_valid_json_and_has_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 0


def test_notebook_defines_the_classes_the_chapter_describes(notebook_ns):
    for name in ("DQN", "SimpleDQN", "ReplayBuffer", "DQNAgent"):
        assert name in notebook_ns, f"notebook no longer defines {name}"


def test_training_cell_episode_count_is_overridable():
    """Guards the hook the execution test depends on."""
    assert "CH3_NUM_EPISODES" in find_cell("for episode in range(NUM_EPISODES)")


def test_colab_install_pins_match_the_requirements_files():
    """The Colab cell is the only install path a reader on Colab ever sees, so
    it must not drift from the repo's own bounds."""
    install = find_cell("%pip install")
    assert "gymnasium[classic-control]>=1.0,<2.0" in install
    # Unquoted specifiers get mangled by shell redirection.
    assert '"torch>=2.0.0"' in install


# --------------------------------------------------------------------------
# Parity with the module implementations
# --------------------------------------------------------------------------

def buffer_trace(cls):
    """Full observable state after every push, so a divergence anywhere in the
    fill-then-wrap cycle shows up. Capacity 5 with 8 pushes covers both the
    partial-fill phase (where size still increments) and wraparound -- checking
    only the saturated end state would let an off-by-one hide behind min()."""
    buf = cls(capacity=5, state_shape=(1,))
    trace = []
    for i in range(8):
        buf.push(np.array([float(i)]), i % 3, float(i),
                 np.array([-float(i)]), i % 2 == 0)
        trace.append((
            len(buf),
            buf.position,
            buf.states[:, 0].tolist(),
            buf.next_states[:, 0].tolist(),
            buf.actions.tolist(),
            buf.rewards.tolist(),
            buf.dones.tolist(),
        ))
    return trace


def test_notebook_replay_buffer_matches_module_behaviour(notebook_ns):
    from src.part_2_methods.ch03_dqn import ReplayBuffer as ModuleBuffer

    module_trace = buffer_trace(ModuleBuffer)
    notebook_trace = buffer_trace(notebook_ns["ReplayBuffer"])

    assert module_trace == notebook_trace, (
        "The notebook's inline ReplayBuffer has drifted from replay_buffer.py"
    )
    # Anchor the shared behaviour too, so both drifting together is still caught.
    assert [step[0] for step in module_trace] == [1, 2, 3, 4, 5, 5, 5, 5]
    assert [step[1] for step in module_trace] == [1, 2, 3, 4, 0, 1, 2, 3]


def test_notebook_replay_buffer_sample_matches_module(notebook_ns):
    from src.part_2_methods.ch03_dqn import ReplayBuffer as ModuleBuffer

    def sampled(cls):
        buf = cls(capacity=16, state_shape=(2,))
        for i in range(16):
            buf.push(np.full(2, i), i % 4, float(i), np.full(2, -i), False)
        np.random.seed(0)  # both call np.random.choice, so seeds align
        return [np.asarray(field).tolist() for field in buf.sample(6)]

    assert sampled(ModuleBuffer) == sampled(notebook_ns["ReplayBuffer"])


def test_notebook_replay_buffer_supports_uint8_storage(notebook_ns):
    """The memory fix in the module version must not be lost in the notebook."""
    buf = notebook_ns["ReplayBuffer"](
        capacity=4, state_shape=(4, 84, 84), state_dtype=np.uint8
    )
    assert buf.states.dtype == np.uint8


def notebook_constants(cell_source, names):
    """Reads top-level ``NAME = <number>`` assignments out of a notebook cell."""
    found = {}
    for name in names:
        match = re.search(rf"^{name}\s*=\s*([0-9_.]+)", cell_source, re.MULTILINE)
        assert match, f"{name} not found in cell"
        found[name] = float(match.group(1).replace("_", ""))
    return found


EPSILON_NAMES = ("EPSILON_START", "EPSILON_END", "EPSILON_DECAY", "WARMUP_STEPS")


def test_notebook_epsilon_constants_match_the_module():
    """Regression: the notebook trained with EPSILON_END=0.01 while
    train_cartpole.py used 0.1, so the two taught different schedules."""
    from src.part_2_methods.ch03_dqn import train_cartpole

    training_cell = find_cell("for episode in range(NUM_EPISODES)")
    notebook_values = notebook_constants(training_cell, EPSILON_NAMES)
    module_values = {name: float(getattr(train_cartpole, name)) for name in EPSILON_NAMES}

    assert notebook_values == module_values


def test_epsilon_preview_plot_matches_the_training_cell():
    """The preview chart is the reader's mental model of the schedule; if it
    disagrees with the loop below it, the chapter teaches the wrong curve."""
    preview_cell = find_cell("Epsilon decay schedule")
    training_cell = find_cell("for episode in range(NUM_EPISODES)")

    assert notebook_constants(preview_cell, EPSILON_NAMES) == \
        notebook_constants(training_cell, EPSILON_NAMES)


def test_notebook_epsilon_schedule_is_numerically_identical_to_the_module():
    """Compares the actual curve, not just the constants, so a change to the
    warmup branch is caught too."""
    from src.part_2_methods.ch03_dqn import train_cartpole as m

    def module_epsilon(step):
        if step < m.WARMUP_STEPS:
            return m.EPSILON_START
        return max(m.EPSILON_END,
                   m.EPSILON_START - (step - m.WARMUP_STEPS) / m.EPSILON_DECAY)

    preview = find_cell("Epsilon decay schedule")
    # Only the schedule block: the cell above it renders a gym frame, which
    # needs pygame and has nothing to do with the curve.
    schedule = preview.split("# --- Right: Epsilon decay schedule ---")[1].split("fig,")[0]

    namespace = {"np": np}
    exec(compile(schedule, "<preview>", "exec"), namespace)
    curve = namespace["eps_curve"]
    steps = namespace["steps_range"]

    expected = [module_epsilon(int(s)) for s in steps]
    assert np.allclose(curve, expected)


def test_notebook_uses_as_tensor_not_legacy_constructors():
    """Regression: commit b8a9ef5 moved the modules to torch.as_tensor but the
    notebook kept torch.FloatTensor, which always copies."""
    agent_cell = find_cell("class DQNAgent")
    # Prose is allowed to name the legacy constructors; executable code is not.
    code = "\n".join(line.split("#")[0] for line in agent_cell.splitlines())

    for legacy in ("torch.FloatTensor", "torch.LongTensor", "torch.BoolTensor"):
        assert legacy not in code, f"{legacy} reintroduced in the notebook"
    assert "torch.as_tensor" in code


def test_notebook_networks_match_module_output_shapes(notebook_ns):
    import torch

    from src.part_2_methods.ch03_dqn import DQN as ModuleDQN
    from src.part_2_methods.ch03_dqn import SimpleDQN as ModuleSimpleDQN

    assert notebook_ns["DQN"](4, 6)(torch.zeros(2, 4, 84, 84)).shape \
        == ModuleDQN(4, 6)(torch.zeros(2, 4, 84, 84)).shape
    assert notebook_ns["SimpleDQN"](4, 2)(torch.zeros(3, 4)).shape \
        == ModuleSimpleDQN(4, 2)(torch.zeros(3, 4)).shape


def test_notebook_agent_can_learn_on_cartpole(notebook_ns):
    import gymnasium as gym
    import torch

    env = gym.make("CartPole-v1")
    agent = notebook_ns["DQNAgent"](env)
    rng = np.random.default_rng(0)

    state, _ = env.reset(seed=0)
    for _ in range(80):
        action = agent.select_action(state, epsilon=1.0)
        nxt, reward, term, trunc, _ = env.step(action)
        agent.memory.push(state, action, reward, nxt, term)
        state = nxt if not (term or trunc) else env.reset()[0]

    before = [p.detach().clone() for p in agent.online_net.parameters()]
    for _ in range(10):
        agent.train_step(32)
    after = list(agent.online_net.parameters())

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    env.close()


# --------------------------------------------------------------------------
# Full execution
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_notebook_runs_top_to_bottom(monkeypatch):
    """Executes every code cell in order, exactly as a reader would.

    Runs in-process rather than through a Jupyter kernel on purpose: a kernel
    would resolve the ``python3`` kernelspec, which can point at a different
    interpreter than the one running the tests -- silently validating some
    other environment's packages. Every magic in this notebook is commented
    out, so in-process execution covers the same ground.
    """
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("CH3_NUM_EPISODES", "60")
    monkeypatch.chdir(CH3)  # so the module-import cell resolves

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    # The training loop must have produced a real reward history.
    assert len(namespace["rewards_history"]) == 60
    assert all(r > 0 for r in namespace["rewards_history"])
