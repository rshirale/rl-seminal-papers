"""Tests for the Chapter 8 notebook.

Chapters 2 through 7 each re-implement their algorithm inline in the notebook,
and every one of those duplications has drifted from its modules at least once
-- chapter 5's in both directions at the same time, chapter 2's shipping a
``KeyError`` that every reader hit on the first experiment.

Chapter 8 removes the duplication rather than testing around it. Its notebook
cells *are* the module files, with their relative imports stripped and nothing
else changed, so the parity test below is a string comparison rather than a
numerical one. That is a stronger guarantee than any of the earlier chapters
can make: there is no "the notebook is a simplified version" escape hatch, and
a change to ``mcts.py`` that is not carried into the notebook fails here
immediately.

Three kinds of check:

  * structure -- the ids, the pinned seed, the commented-out install line
  * parity    -- each module cell is its module, character for character
  * execution -- the notebook runs top to bottom, under CH8_FAST
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CH8 = ROOT / "src" / "part_2_methods" / "ch08_alphazero"
NOTEBOOK = CH8 / "Chapter8_AlphaZero.ipynb"

#: Every module the notebook inlines, in the order its cells appear. Kept as a
#: literal rather than globbed, because the order is a teaching decision -- the
#: interface before the games, the oracle before the agent -- and a glob would
#: let a reordering pass silently.
INLINED = [
    "game.py", "connect_four.py", "othello.py", "solver.py", "network.py",
    "mcts.py", "selfplay.py", "arena.py", "train.py", "animate.py",
    "alphago.py", "benchmark.py",
]


def notebook():
    return json.loads(NOTEBOOK.read_text())


def code_cells():
    return ["".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing ``marker``.

    Located by content rather than by index so inserting a cell does not
    silently break these tests -- the mistake chapter 6's suite documents.
    """
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


def inlined(name):
    """A module's source as the notebook should carry it.

    One transformation, deliberately: relative imports are dropped, because
    the notebook has no package to import from and every name they would bind
    is already in the flat namespace by the time the cell runs.
    """
    lines = (CH8 / name).read_text().splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith("from .")]
    return "\n".join(kept).rstrip() + "\n"


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_the_notebook_exists_and_is_current_nbformat():
    nb = notebook()
    assert nb["nbformat"] == 4
    # nbformat 4.5 requires cell ids; chapter 4 shipped without them once and
    # Colab rewrote the file on every open.
    assert nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert all(ids), "every cell needs an id at nbformat 4.5"
    assert len(set(ids)) == len(ids), "cell ids must be unique"


def test_no_cell_ships_stored_output():
    """Committed outputs make every diff a binary diff and leak whatever
    machine last ran the notebook into the repository."""
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_the_install_line_is_commented_out():
    """Colab already has torch and matplotlib. An uncommented pip line
    reinstalls torch over the CUDA build and costs a reader five minutes."""
    setup = find_cell("import torch")
    for line in setup.splitlines():
        if "pip install" in line:
            assert line.lstrip().startswith("#"), line


def test_the_seed_is_pinned_before_anything_random_happens():
    setup = find_cell("SEED = 0")
    assert "torch.manual_seed(SEED)" in setup
    assert "np.random.default_rng(SEED)" in setup
    cells = code_cells()
    assert cells.index(setup) < cells.index(find_cell("play_batch"))


def test_the_training_cell_holds_on_to_the_right_main():
    """``solver``, ``train``, ``alphago`` and ``benchmark`` each define a
    ``main``, and inlining them into one namespace means the last one wins.
    The notebook aliases each before the next cell clobbers it; without that,
    re-running the training cell after section 14 would run the benchmark.
    """
    training = find_cell("BUDGET_SECONDS")
    assert "train = main" in training
    assert "result = train(" in training
    benchmark = find_cell("benchmark = main")
    assert code_cells().index(training) < code_cells().index(benchmark)


def test_the_fast_hook_exists_so_the_execution_test_can_finish():
    assert "CH8_FAST" in find_cell("FAST =")


# --------------------------------------------------------------------------
# Parity: the cells are the modules
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", INLINED)
def test_each_inlined_cell_is_its_module_character_for_character(name):
    expected = inlined(name)
    # Matched on the docstring's first line, which is unique per module and
    # survives edits to the code below it.
    marker = expected.splitlines()[0]
    assert marker.startswith('"""'), f"{name} needs a module docstring"
    assert find_cell(marker) == expected, (
        f"The notebook's {name} cell has drifted from "
        f"src/part_2_methods/ch08_alphazero/{name}. Regenerate the cell from "
        f"the module rather than editing it in place."
    )


def test_the_inlined_modules_appear_in_teaching_order():
    """The interface before the games, the oracle before the agent, the
    AlphaGo listings last. A reordering is a decision, not a refactor."""
    cells = code_cells()
    positions = [cells.index(find_cell(inlined(n).splitlines()[0]))
                 for n in INLINED]
    assert positions == sorted(positions)


def test_every_module_with_a_docstring_is_either_inlined_or_deliberately_not():
    """``inspect_agent.py`` is the only module the notebook leaves out, and it
    is left out because it reads a checkpoint off disk and calls ``input()``.
    Anything else new in the directory needs a decision, not a default."""
    modules = {p.name for p in CH8.glob("*.py")} - {"__init__.py"}
    assert modules - set(INLINED) == {"inspect_agent.py"}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_notebook_runs_top_to_bottom(monkeypatch, tmp_path):
    """Executes every code cell in order, exactly as a reader would.

    Runs in-process rather than through a Jupyter kernel on purpose: a kernel
    would resolve the ``python3`` kernelspec, which can point at a different
    interpreter than the one running the tests -- silently validating some
    other environment's packages. Every magic in this notebook is commented
    out, so in-process execution covers the same ground.

    ``CH8_FAST`` shrinks the board, the fleet and the search depth. It changes
    no code path: the same self-play, the same loop, the same evaluation, on a
    4x4 board for a minute instead of 6x6 for half an hour.
    """
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("CH8_FAST", "1")
    # The loop writes runs/<game>_<timestamp>/ relative to the working
    # directory. Somewhere disposable, so a test run does not litter src/.
    monkeypatch.chdir(tmp_path)

    namespace = {"__name__": "__notebook__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}"
            )

    history = namespace["history"]
    assert len(history) >= 1
    # The evaluation branch ran, which is the half of the loop a budget that is
    # too short silently skips.
    assert "mirror_draw_rate" in history[0]
    assert namespace["game"].solved_value() == 0
    # Section 14's table is the chapter's published parameter counts.
    counts = dict(namespace["parameter_counts"]())
    assert 22_837_864 in counts.values()
