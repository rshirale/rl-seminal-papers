"""Tests for the Chapter 9 notebook.

Chapter 9 follows chapter 8 rather than chapters 2 to 7: its notebook does not
re-implement anything. Its module cells *are* the files -- chapter 6's actor
and critic, the four printed listings in ``cql.py``, one cell each, and the two
experiment scripts -- so the parity test below is a string comparison.

The scripts need one fixed transformation to run in a flat namespace, and it is
defined once, in ``inlined``, rather than left to whoever regenerates the
notebook. It removes the lines that tie a script to the repository and to the
command line, and nothing else:

  * ``from src.`` imports, and the ``L.`` and ``T.`` prefixes they bind;
  * the ``CODE_DIR =`` line, which reads ``__file__`` (the notebook sets
    ``CODE_DIR`` once in its setup cell);
  * the ``if __name__ == "__main__":`` guard. A Jupyter kernel runs every
    cell as ``__main__``, so a guarded ``main()`` fires the moment its cell is
    defined and parses the kernel's own ``-f kernel.json`` argument. That is
    not hypothetical: chapter 8's notebook kept its guards and stopped with
    ``SystemExit: 2`` at the solver cell in a real kernel, while its execution
    test passed, because it ran the cells as ``__notebook__``. The test below
    runs them as ``__main__``, with a kernel's argv, so it would catch that.

Three kinds of check:

  * structure -- the ids, no stored output, the commented install line, the
    seed, the fast hook, the ``main`` aliases
  * parity    -- each module cell is its file, character for character
  * execution -- the notebook runs top to bottom, under CH9_FAST, when the
    PointMaze dataset is already cached (CI does not install Minari)
"""

import json
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CH6 = ROOT / "src" / "part_2_methods" / "ch06_sac"
CH9 = ROOT / "src" / "part_3_applications" / "ch09_cql"
NOTEBOOK = CH9 / "Chapter9_CQL.ipynb"
DATASET_ID = "D4RL/pointmaze/umaze-v2"

#: The scripts the notebook inlines, in the order its cells appear: the
#: experiment before the check of what it produced.
SCRIPTS = ["train_cql_vs_sac.py", "eval_fqe_vs_true.py"]


def notebook():
    return json.loads(NOTEBOOK.read_text())


def code_cells():
    return ["".join(c["source"]) for c in notebook()["cells"]
            if c["cell_type"] == "code"]


def find_cell(marker):
    """The first code cell containing ``marker``, located by content so that
    inserting a cell does not silently break these tests."""
    for source in code_cells():
        if marker in source:
            return source
    raise AssertionError(f"No notebook code cell contains {marker!r}")


def inlined(path):
    """A file's source as the notebook should carry it. See the module
    docstring for what is removed and why; this is the only place it is
    defined."""
    out, guard = [], False
    for line in path.read_text().splitlines():
        if line == 'if __name__ == "__main__":':
            guard = True
            continue
        if guard:
            if line.startswith((" ", "\t")) or not line.strip():
                continue
            guard = False
        if line.startswith("from src.") or line.startswith("CODE_DIR = "):
            continue
        out.append(re.sub(r"\b[LT]\.(?=\w)", "", line))
    # Dropped import blocks leave runs of blank lines; keep at most two.
    text = re.sub(r"\n{4,}", "\n\n\n", "\n".join(out))
    return text.rstrip() + "\n"


def listings():
    """``cql.py`` split at its ``# --- Listing 9.N`` markers, one cell per
    listing; the module docstring travels with listing 9.1."""
    parts = re.split(r"(?m)^(?=# --- Listing )", (CH9 / "cql.py").read_text())
    head, body = parts[0], parts[1:]
    body[0] = head + body[0]
    return [p.strip("\n") + "\n" for p in body]


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_the_notebook_exists_and_is_current_nbformat():
    nb = notebook()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] >= 5
    ids = [c["id"] for c in nb["cells"]]
    assert all(ids), "every cell needs an id at nbformat 4.5"
    assert len(set(ids)) == len(ids), "cell ids must be unique"


def test_no_cell_ships_stored_output():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_the_install_line_is_commented_out():
    """Colab needs it and a local environment does not, so it ships
    commented, as in every other chapter."""
    for source in code_cells():
        for line in source.splitlines():
            if "pip install" in line:
                assert line.lstrip().startswith("#"), line


def test_the_seed_is_pinned_before_anything_random_happens():
    setup = find_cell("SEED = 0")
    assert "torch.manual_seed(SEED)" in setup
    assert "np.random.seed(SEED)" in setup
    cells = code_cells()
    assert cells.index(setup) < cells.index(find_cell("buffer.sample(256)"))
    # The training runs are seeded by the script, from the same constant.
    assert '"--seed", str(SEED)' in find_cell("def run(")


def test_the_fast_hook_exists_so_the_execution_test_can_finish():
    assert "CH9_FAST" in find_cell("FAST =")


def test_no_cell_keeps_a_main_guard():
    """A kernel runs cells as ``__main__``; see the module docstring."""
    for source in code_cells():
        assert '__name__ == "__main__"' not in source


def test_each_main_is_aliased_before_the_next_one_rebinds_it():
    """Both scripts define ``main``. Re-running section 7 after section 10
    must still train, not run FQE."""
    cells = code_cells()
    train_script = cells.index(inlined(CH9 / "train_cql_vs_sac.py"))
    eval_script = cells.index(inlined(CH9 / "eval_fqe_vs_true.py"))
    training = cells.index(find_cell("train = main"))
    assert train_script < training < eval_script
    assert "train()" in cells[training]
    assert cells.index(find_cell("fqe_vs_true = main")) > eval_script


def test_the_setup_cell_defines_what_the_transformation_removed():
    assert "CODE_DIR = " in find_cell("SEED = 0")


# --------------------------------------------------------------------------
# Parity: the cells are the files
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [CH6 / "actor.py", CH6 / "critic.py"],
                         ids=["actor.py", "critic.py"])
def test_chapter_6_networks_are_inlined_verbatim(path):
    expected = path.read_text().rstrip() + "\n"
    # Nothing to transform: neither file imports from the package.
    assert inlined(path) == expected
    assert expected in code_cells(), (
        f"The notebook's {path.name} cell has drifted from "
        f"src/part_2_methods/ch06_sac/{path.name}.")


@pytest.mark.parametrize("number", [1, 2, 3, 4],
                         ids=[f"listing_9.{n}" for n in (1, 2, 3, 4)])
def test_each_listing_is_its_own_cell_exactly_as_printed(number):
    expected = listings()[number - 1]
    assert expected.lstrip('"').startswith(("Listings 9.1", "# --- Listing"))
    assert f"# --- Listing 9.{number} " in expected
    assert expected in code_cells(), (
        f"The notebook's listing 9.{number} cell has drifted from cql.py. "
        f"Regenerate the cell from the file rather than editing it in place.")


def test_the_listing_cells_cover_cql_py_completely():
    """Splitting must not lose a line: the cells joined back are the file."""
    joined = "\n\n".join(c.rstrip("\n") for c in listings())
    original = (CH9 / "cql.py").read_text()
    assert re.sub(r"\s+", " ", joined).strip() == \
        re.sub(r"\s+", " ", original).strip()


@pytest.mark.parametrize("name", SCRIPTS)
def test_each_script_cell_is_its_file_after_the_fixed_transformation(name):
    expected = inlined(CH9 / name)
    assert expected in code_cells(), (
        f"The notebook's {name} cell has drifted from "
        f"src/part_3_applications/ch09_cql/{name}. Regenerate the cell from "
        f"the file rather than editing it in place.")


def test_the_transformation_only_touches_what_it_says():
    """Every line it changes is an import, CODE_DIR, the guard, or an
    ``L.``/``T.`` prefix -- checked by undoing the prefix change and
    comparing what is left."""
    for name in SCRIPTS:
        source = (CH9 / name).read_text().splitlines()
        kept = [re.sub(r"\b[LT]\.(?=\w)", "", ln) for ln in source
                if not ln.startswith(("from src.", "CODE_DIR = "))]
        kept = [ln for ln in kept
                if ln not in ('if __name__ == "__main__":', "    main()")]
        assert [ln for ln in inlined(CH9 / name).splitlines() if ln] == \
            [ln for ln in kept if ln]


def test_the_cells_appear_in_teaching_order():
    """Networks, then the listings in order, then the experiment, then its
    check. A reordering is a decision, not a refactor."""
    cells = code_cells()
    order = ([(CH6 / "actor.py").read_text().rstrip() + "\n",
              (CH6 / "critic.py").read_text().rstrip() + "\n"]
             + listings() + [inlined(CH9 / n) for n in SCRIPTS])
    positions = [cells.index(c) for c in order]
    assert positions == sorted(positions)


def test_every_module_is_inlined():
    """Anything new in the directory needs a decision, not a default."""
    modules = {p.name for p in CH9.glob("*.py")} - {"__init__.py"}
    assert modules == {"cql.py", *SCRIPTS}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def dataset_is_cached():
    try:
        import gymnasium_robotics  # noqa: F401
        import minari
    except ImportError:
        return False
    return DATASET_ID in minari.list_local_datasets()


@pytest.mark.slow
@pytest.mark.skipif(not dataset_is_cached(),
                    reason=f"needs minari, gymnasium-robotics and a cached "
                           f"{DATASET_ID} (about 530 MB)")
def test_notebook_runs_top_to_bottom(monkeypatch, tmp_path):
    """Executes every code cell in order, in-process, as a kernel would.

    In-process rather than through a Jupyter kernel for the reason chapter 8's
    suite gives: a kernelspec can point at another interpreter. What a kernel
    does differently is reproduced instead -- ``__name__`` is ``__main__`` and
    ``sys.argv`` carries ``-f kernel.json`` -- which is exactly the pair that
    breaks a guarded ``main()``.

    ``CH9_FAST`` trains for 200 steps and fits FQE for 200 on three start
    states. It changes no code path: the same loader, the same listings, the
    same logging, evaluation and checkpointing. The dataset is loaded three
    times (once by the notebook, once per training run), which is most of the
    two or three minutes this takes.
    """
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("CH9_FAST", "1")
    monkeypatch.setattr(sys, "argv",
                        ["ipykernel_launcher.py", "-f", "kernel-test.json"])
    # Runs land in ./runs/<name>/; somewhere disposable, not src/.
    monkeypatch.chdir(tmp_path)

    namespace = {"__name__": "__main__"}
    for index, source in enumerate(code_cells()):
        try:
            exec(compile(source, f"<notebook cell {index}>", "exec"),
                 namespace)
        except BaseException as exc:  # SystemExit included, on purpose
            pytest.fail(
                f"Notebook cell {index} raised {type(exc).__name__}: {exc}\n"
                f"--- cell source ---\n{source[:600]}")

    # The chapter's dataset facts, from the notebook's own buffer.
    buffer = namespace["buffer"]
    assert len(buffer.cols[0]) == 1_000_000
    assert len(buffer.starts) == 13_210

    for name in ("sac", "cql"):
        rows = namespace["read_csv"](name, "log.csv")
        assert [int(r["step"]) for r in rows] == [0, 100, 200]
        progress = namespace["read_csv"](name, "progress.csv")
        assert [int(r["step"]) for r in progress] == [100, 200]
        assert (tmp_path / "runs" / name / "paths_0000200.npz").exists()
        g_true, g_fqe = namespace["per_state"][name]
        assert len(g_true) == len(g_fqe) == 3

    # The run helper resumes rather than retrains: asking again is a no-op.
    namespace["run"]("sac", 0.0)
    assert len(namespace["read_csv"]("sac", "log.csv")) == 3
    assert sys.argv[0] == "ipykernel_launcher.py", "run() must restore argv"
