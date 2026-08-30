"""Tests for the Chapter 2 notebook.

Chapters 3-6 each had a notebook suite; chapter 2 did not, and it is the one
notebook a reader meets first. What that cost: its outcome tally built its key
with ``f"Hazard {s}"``, which renders a tuple as ``Hazard (1, 1)`` -- with a
space -- against a dict seeded with ``"Hazard (1,1)"``. Roughly nineteen out of
twenty random walks end in a hazard, so the cell raised ``KeyError`` almost
immediately, for every reader, until someone ran it. The equivalent code in
``run_td0_gridworld.py`` had always used explicit branches and was fine, which
is the drift these tests exist to catch.

Like chapters 3-6, the notebook re-implements the chapter's environments and
algorithms inline so readers can follow them cell by cell rather than importing
from ``src/``. The tests cover both risks that duplication creates:

  * parity    -- the notebook's inline code still behaves like the modules
  * execution -- the notebook actually runs top to bottom without raising

Two differences from the modules are deliberate rather than drift, so the
parity tests below pass explicit arguments rather than relying on defaults:
``td0`` defaults to 1000 episodes in the notebook and 500 in ``algorithms.py``,
and the notebook's ``GridWorld`` omits the module's (always empty) ``walls``
list.

The execution test is marked ``slow`` because it runs a thousand training
episodes. Run everything with ``make test-all``; the default ``make test``
skips it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

CH2 = (
    Path(__file__).resolve().parents[1]
    / "src" / "part_1_foundations" / "ch02_fundamentals"
)
NOTEBOOK = CH2 / "Chapter2_Fundamentals.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing `marker`. Located by content rather than
    index so inserting a cell does not silently break these tests."""
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


@pytest.fixture(scope="module")
def notebook_ns():
    """Executes the notebook's definition cells (not its experiments) and
    returns the resulting namespace."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI during tests

    namespace = {"__name__": "__notebook__"}
    for marker in ("import numpy", "class GridWorld", "def td0"):
        exec(compile(find_cell(marker), "<notebook>", "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_notebook_is_valid_json_and_every_cell_has_an_id():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    assert nb["cells"]
    # nbformat 4.5 requires cell ids; this notebook shipped at 4.4 without them
    # long after chapters 3-6 had been normalized.
    assert all("id" in cell for cell in nb["cells"])


def test_notebook_defines_what_the_chapter_describes(notebook_ns):
    for name in ("GridWorld", "CliffWalking", "td0", "run_agent"):
        assert name in notebook_ns, f"notebook no longer defines {name}"


def test_outcome_tally_counts_every_terminal_it_can_reach(notebook_ns):
    """The regression this suite was written for.

    A random walker from (0, 0) reaches all three terminals, so any key the
    tally can write has to be one it initialized. Executing the cell is the
    check: a mismatch raises ``KeyError`` rather than miscounting.
    """
    namespace = dict(notebook_ns)
    exec(compile(find_cell("outcomes"), "<notebook>", "exec"), namespace)

    outcomes = namespace["outcomes"]
    assert sum(outcomes.values()) == 500, (
        f"episodes went uncounted: {outcomes}")
    assert set(outcomes) == {"Goal", "Hazard (1,1)", "Hazard (2,1)"}
    # Every terminal is reachable under a random policy, so a zero here means
    # the tally stopped recording one of them rather than that it never fired.
    assert all(count > 0 for count in outcomes.values()), outcomes


# --------------------------------------------------------------------------
# Parity with the module implementations
# --------------------------------------------------------------------------

def test_notebook_gridworld_transitions_match_the_module(notebook_ns):
    """Every state-action pair, not a sampled few: the grid is 12 states."""
    from environments import GridWorld as ModuleGridWorld

    nb_env, mod_env = notebook_ns["GridWorld"](), ModuleGridWorld()
    assert nb_env.terminals == mod_env.terminals

    for x in range(mod_env.width):
        for y in range(mod_env.height):
            for action in mod_env.actions:
                assert (nb_env.transition((x, y), action)
                        == mod_env.transition((x, y), action)), (
                    f"divergence at {(x, y)} / {action}")


def test_notebook_cliffwalking_transitions_match_the_module(notebook_ns):
    from environments import CliffWalking as ModuleCliff

    nb_env, mod_env = notebook_ns["CliffWalking"](), ModuleCliff()
    assert nb_env.cliff == mod_env.cliff
    assert (nb_env.start, nb_env.goal) == (mod_env.start, mod_env.goal)

    for x in range(mod_env.width):
        for y in range(mod_env.height):
            for action in mod_env.actions:
                assert (nb_env.transition((x, y), action)
                        == mod_env.transition((x, y), action)), (
                    f"divergence at {(x, y)} / {action}")


def test_notebook_td0_matches_the_module_value_table(notebook_ns):
    """Same seed and same episode count in, same value table out.

    Both implementations draw one action per step from the same RNG, so
    identical seeding makes the trajectories identical and the comparison
    exact rather than statistical.
    """
    from environments import GridWorld as ModuleGridWorld
    from algorithms import td0 as module_td0

    np.random.seed(7)
    nb_values = notebook_ns["td0"](notebook_ns["GridWorld"](), episodes=50)
    np.random.seed(7)
    mod_values = module_td0(ModuleGridWorld(), episodes=50)

    assert nb_values.keys() == mod_values.keys()
    for state, value in mod_values.items():
        assert nb_values[state] == pytest.approx(value), f"differ at {state}"


@pytest.mark.parametrize("mode", ["qlearning", "sarsa"])
def test_notebook_run_agent_matches_the_module(notebook_ns, mode):
    """Covers both branches: the off-policy max and the on-policy next action."""
    from environments import CliffWalking as ModuleCliff
    from algorithms import run_agent as module_run_agent

    np.random.seed(3)
    _, nb_history, nb_falls = notebook_ns["run_agent"](
        notebook_ns["CliffWalking"](), mode=mode, episodes=20)
    np.random.seed(3)
    _, mod_history, mod_falls = module_run_agent(
        ModuleCliff(), mode=mode, episodes=20)

    assert nb_falls == mod_falls
    assert nb_history == mod_history


# --------------------------------------------------------------------------
# Full execution
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_notebook_runs_top_to_bottom():
    """Executes every code cell in order, exactly as a reader would.

    Runs in-process rather than through a Jupyter kernel on purpose: a kernel
    would resolve the ``python3`` kernelspec, which can point at a different
    interpreter than the one running the tests -- silently validating some
    other environment's packages. This notebook uses no magics, so in-process
    execution covers the same ground.
    """
    import matplotlib
    matplotlib.use("Agg")

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    assert sum(namespace["outcomes"].values()) == 500
    # Q-Learning walks the cliff edge and SARSA detours around it, so the
    # off-policy run falls more often. This is the chapter's whole point.
    assert namespace["q_falls"] > namespace["s_falls"]
    assert len(namespace["q_hist"]) == len(namespace["s_hist"]) == 500
