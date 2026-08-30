"""Structural checks on the chapter READMEs.

These exist because the chapter READMEs drifted, silently, in exactly the way
the companion site's contributor guide says chapter rows drift -- and unlike
the site, nothing was checking them. Chapters 1, 2 and 3 sat at 28, 37 and 43
lines against chapter 6's 138 until someone counted, and the rewrite that
followed turned up four defects that are all mechanically detectable:

  * chapter 3 described six files in a directory of eight, leaving
    ``ablation.py`` and ``seeding.py`` undocumented;
  * chapter 3 attributed ``DQNAgent`` to ``dqn_agent.py``, which defines
    ``AtariDQNAgent``; the other class lives in ``train_cartpole.py``;
  * chapter 2 omitted its own notebook, making it the one notebook in the book
    with no Colab link from its chapter;
  * chapter 3's ``ablation.py`` had no ``make`` target, so the paper's own
    ablation was reachable only by finding the file.

Every check below corresponds to one of those. What none of them can do is
tell you a *number* is wrong -- that a sigma quoted as "measured at 30,000
steps" actually came from a 6,000-step probe. That is a provenance problem, and
the convention for it (a dated "Reproduced on ... with ..." line above any
table of measurements, as chapter 4 does) is documented in ``docs/README.md``
rather than enforced here.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAKEFILE = ROOT / "Makefile"

#: Chapter number -> directory, discovered rather than hard-coded so a new
#: chapter is covered the moment it lands.
def _chapters():
    found = {}
    for part in sorted(SRC.glob("part_*")):
        for d in sorted(part.glob("ch[0-9][0-9]_*")):
            if d.is_dir():
                found[int(d.name[2:4])] = d
    return found


CHAPTERS = _chapters()
CHAPTER_IDS = [f"ch{n:02d}" for n in sorted(CHAPTERS)]
CHAPTER_DIRS = [CHAPTERS[n] for n in sorted(CHAPTERS)]


def readme(chapter_dir):
    path = chapter_dir / "README.md"
    assert path.exists(), f"{chapter_dir.name} has no README.md"
    return path.read_text()


def headings(text):
    """Level-2 headings that render as headings -- i.e. outside code fences."""
    out, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if not inside and line.startswith("## "):
            out.append(line[3:].strip())
    return out


def test_every_chapter_directory_has_a_readme():
    assert CHAPTERS, "no chapter directories discovered -- has the layout moved?"
    for number, directory in sorted(CHAPTERS.items()):
        assert (directory / "README.md").exists(), f"chapter {number} has none"


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_readme_carries_the_five_standard_sections(chapter_dir):
    """The shape documented in docs/README.md under "Chapter READMEs".

    Two headings are matched loosely on purpose. Chapter 4 opens with "What to
    run, and what to read" rather than "File Structure" -- a deliberate
    restructure -- and the running section is named for what the chapter runs
    ("Running the Loop", "Running the Benchmarks", "Running the Experiments").
    The rule is that the section exists, not that it is spelled identically.
    """
    found = headings(readme(chapter_dir))

    def has(predicate, label):
        assert any(predicate(h) for h in found), (
            f"{chapter_dir.name}/README.md has no {label} section; found {found}")

    has(lambda h: "File Structure" in h or "What to run" in h, "file-structure")
    has(lambda h: h == "Installation", "Installation")
    has(lambda h: h.startswith("Running"), "Running")
    has(lambda h: h == "Implementation Notes", "Implementation Notes")
    has(lambda h: h == "Troubleshooting", "Troubleshooting")


def file_structure_section(text):
    """The body of the file-structure section, up to the next level-2 heading.

    Scoped deliberately. Checking the whole document instead would let a module
    that is merely *mentioned* somewhere -- in a troubleshooting entry, say --
    pass for one that is listed, which is exactly the drift being guarded
    against. Level-3 headings do not terminate the section, because chapter 4
    splits its list into "Files you run", "Files you read" and "Supporting
    files".
    """
    lines, collecting, out = text.splitlines(), False, []
    for line in lines:
        if line.startswith("## "):
            title = line[3:].strip()
            collecting = "File Structure" in title or "What to run" in title
            continue
        if collecting:
            out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_every_python_module_is_documented(chapter_dir):
    """Regression: chapter 3 documented six files in a directory of eight.

    Checked against the file-structure section rather than the whole README, so
    a module that only turns up in a troubleshooting entry does not count as
    documented. The first version of this test made that mistake and let a
    deliberately reintroduced defect through.
    """
    section = file_structure_section(readme(chapter_dir))
    assert section.strip(), f"{chapter_dir.name}/README.md has no file list"

    # Each module needs an entry of its own -- a bullet or a table row that
    # *starts* with it -- not merely a mention. Substring matching over the
    # section was the second version of this test and still let a deliberately
    # removed entry through, because chapter 3's train_cartpole.py bullet
    # happens to name ablation.py in passing. Both list styles in the book are
    # accepted: "- `file.py`: ..." and chapter 4's "| `file.py` | ... |".
    entries = set(re.findall(r"^\s*[-|]\s*`([\w.]+\.py)`", section,
                             re.MULTILINE))
    missing = sorted({p.name for p in chapter_dir.glob("*.py")} - entries)
    assert not missing, (
        f"{chapter_dir.name}/README.md has no file-list entry for {missing}. "
        "Every module in the directory needs its own bullet or table row.")


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_every_notebook_is_documented_with_a_colab_link(chapter_dir):
    """Regression: chapter 2 omitted Chapter2_Fundamentals.ipynb entirely, so
    it was the one notebook in the book unreachable from its chapter."""
    text = readme(chapter_dir)
    for nb in sorted(chapter_dir.glob("*.ipynb")):
        assert nb.name in text, (
            f"{chapter_dir.name}/README.md does not mention {nb.name}")
        assert "colab.research.google.com" in text, (
            f"{chapter_dir.name}/README.md has no Colab link for {nb.name}")
        assert nb.name in text[text.index("colab.research.google.com") - 400:
                               text.index("colab.research.google.com") + 400] \
            or f"/{nb.name}" in text, (
            f"{chapter_dir.name}'s Colab link does not point at {nb.name}")


def make_targets():
    """``run-chN-...`` targets, grouped by chapter number."""
    grouped = {}
    for target in re.findall(r"^(run-ch(\d+)[a-z-]*):", MAKEFILE.read_text(),
                             re.MULTILINE):
        grouped.setdefault(int(target[1]), []).append(target[0])
    return grouped


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_every_make_target_is_documented_in_its_chapter(chapter_dir):
    """Regression: chapter 3's ablation.py had no target at all, and when one
    was added the README had to be told about it. A target a reader cannot
    discover from the chapter is a target that does not exist."""
    number = int(chapter_dir.name[2:4])
    text = readme(chapter_dir)
    undocumented = [t for t in make_targets().get(number, []) if t not in text]
    assert not undocumented, (
        f"{chapter_dir.name}/README.md does not mention {undocumented}")


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_documented_make_targets_exist_in_the_makefile(chapter_dir):
    """The other direction: a README promising a target the Makefile dropped."""
    text = readme(chapter_dir)
    declared = set(re.findall(r"^run-ch\d+[a-z-]*:", MAKEFILE.read_text(),
                              re.MULTILINE))
    declared = {t.rstrip(":") for t in declared}
    promised = set(re.findall(r"\bmake (run-ch\d+[a-z-]*)", text))
    missing = sorted(promised - declared)
    assert not missing, (
        f"{chapter_dir.name}/README.md promises {missing}, "
        "which the Makefile does not define")


#: Backticked CamelCase names that are not classes in this repo.
NOT_A_CLASS = {"README", "MDEOF", "RunResult"}


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_classes_are_attributed_to_the_module_that_defines_them(chapter_dir):
    """Regression: chapter 3's README said ``dqn_agent.py``: ``DQNAgent``.

    That module defines ``AtariDQNAgent``; ``DQNAgent`` is a different, lighter
    class in ``train_cartpole.py``, so a reader following the README to the
    Atari agent found nothing by that name.

    The convention across every chapter is that a file's primary class is named
    first after the filename, so this checks the *first* backticked CamelCase
    identifier following each ``file.py`` mention. Later names on the same line
    are free to refer elsewhere -- chapter 3's entry deliberately points at
    ``train_cartpole.py`` right after naming ``AtariDQNAgent``.
    """
    text = readme(chapter_dir)
    defined = {}
    for path in chapter_dir.glob("*.py"):
        for name in re.findall(r"^class (\w+)", path.read_text(), re.MULTILINE):
            defined.setdefault(name, set()).add(path.name)

    problems = []
    for line in text.splitlines():
        match = re.search(r"`(\w+\.py)`(.*)", line)
        if not match:
            continue
        module, rest = match.group(1), match.group(2)
        if module not in {p.name for p in chapter_dir.glob("*.py")}:
            continue
        # __init__.py re-exports its package's classes by design, so naming
        # them there is correct rather than a misattribution.
        if module == "__init__.py":
            continue
        names = [n for n in re.findall(r"`([A-Z][A-Za-z0-9]+)`", rest)
                 if n not in NOT_A_CLASS]
        if not names:
            continue
        first = names[0]
        if first in defined and module not in defined[first]:
            problems.append(
                f"{module} is credited with {first}, which is defined in "
                f"{sorted(defined[first])}")

    assert not problems, f"{chapter_dir.name}/README.md: " + "; ".join(problems)
