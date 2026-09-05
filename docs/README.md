# Companion site

The source for <https://rshirale.github.io/rl-seminal-papers/>, the companion site for *RL: The Seminal Papers*.

It is one hand-written HTML page. There is no build step, no framework, no templating, and no per-chapter pages. GitHub Pages serves this directory from `main` and runs its default Jekyll pass, but the site uses no Liquid tags and no front matter, so that pass copies the files through unchanged.

## Working on it locally

Nothing to install. Serve the directory and open it:

```bash
cd docs
python3 -m http.server 4000
# → http://localhost:4000
```

Because nothing is compiled, this renders exactly what visitors get. Installing Ruby and Jekyll is only worth it if the site later grows shared layouts or includes.

## Files

| File | What it is |
| --- | --- |
| `index.html` | The entire site — hero, chapter list, quickstart, CTA. |
| `assets/css/site.css` | Material 3 design tokens and all styling. |
| `assets/js/site.js` | Theme toggle, copy buttons, progress marking. |
| `llms.txt` | Structured summary of the book and its resources, for model consumption. Mirrors the chapter list and links in `index.html`. |
| `sitemap.xml` | Two URLs — the root and `privacy.html`. |
| `robots.txt` | Allow-all, including named AI crawlers. Points at the sitemap. |
| `privacy.html` | Analytics and privacy notice. |

## Conventions

- **Design tokens, not hex values.** Colours, radii, and surfaces are CSS custom properties on `:root` in `site.css` — `--primary` (amber `#ffb74d`), `--tertiary`, `--surface`, `--sc-low` / `--sc-high`, `--on-surface`, `--outline`, `--radius-*`. Use the token; never hard-code a colour.
- **Dark-first, with a light theme** under `[data-theme="light"]`, toggled by `#themeToggle` and `#mobileThemeToggle`. New markup that reads from the tokens themes for free.
- **Chapter rows** have a fixed shape:

  ```html
  <div class="chapter-row" data-status="live" data-chapter="N">
    <div class="ch-num done">✓</div>
    <div class="ch-body">
      <div class="ch-title"><a href="…">NAME</a></div>
      <div class="ch-sub">Author et al. (YEAR)</div>
    </div>
    <div class="ch-status">
      <span class="badge-live">Live</span>
      <button class="complete-button" type="button" aria-label="Mark Chapter N complete">Done</button>
    </div>
  </div>
  ```

  Planned chapters drop the link and the badge, and carry a clock SVG in `.ch-status` instead.
- **Copyable commands** use the `.copy-wrap` wrapper with a matching `data-copy` attribute, so the copy button picks up the right string:

  ```html
  <div class="copy-wrap">
    <code>make run-ch4-pendulum</code>
    <button class="copy-button" type="button" data-copy="make run-ch4-pendulum">Copy</button>
  </div>
  ```
- **Keep `llms.txt` in step with `index.html`.** It is a real artifact, not boilerplate — any chapter or resource link added to one belongs in the other.
- **The "available now" phrasing appears three times** in `<head>`: the meta description, `og:description`, and `twitter:description`. Change all three together.
- **Bump `sitemap.xml`** whenever `index.html` changes; `lastmod` is easy to forget.

## Chapter READMEs

Not site files, but they are the other half of what a reader lands on, and they
drift the same way `index.html` does. All six now carry the same shape:

```
# Chapter N: Title
  one paragraph on what the directory holds
  one or two on why the algorithm exists and what it changed
## File Structure          every file in the directory, notebook included
## Installation
## Running the Experiments every make target, with runtimes
## Implementation Notes    the non-obvious decisions, sourced from the code
## Troubleshooting         the failures a reader will actually hit
```

Five rules, each of them written down because it was broken. The first four
are enforced by `tests/test_readmes.py`, which fails if a chapter drifts. Since
the Chapter 7 landing, `.github/workflows/tests.yml` runs the whole suite on
every push and pull request, so these checks now fire without anyone typing
`make test`. It runs `make test-all` rather than `make test` on purpose: every
notebook execution test is marked slow, and those are the ones that catch what
a reader hits first. The fifth rule cannot be enforced at all, and is the one
to be careful about:

- **List every file.** Chapter 3's README described six files in a directory of
  eight; `ablation.py` and `seeding.py` were undocumented, and the ablation was
  reachable only by finding the file. Chapter 2 omitted its own notebook, which
  made `Chapter2_Fundamentals.ipynb` the one notebook in the book with no Colab
  link from its chapter.
- **Name the classes the code actually defines.** Chapter 3's README attributed
  `DQNAgent` to `dqn_agent.py`. That class is `AtariDQNAgent`; `DQNAgent` is a
  different, lighter class in `train_cartpole.py`. A reader following the README
  found nothing by that name.
- **Quantitative claims carry their provenance.** A number without a source
  cannot be checked, and twice now one has been wrong in a way nothing caught:
  a fabricated middle value in chapter 3's seed scores, and chapter 6's sigma
  figures labelled "measured at 30,000 steps" when they came from 6,000- and
  12,000-step probes. Chapter 4 has the pattern worth copying — a dated
  `Reproduced on YYYY-MM-DD with the defaults above:` line above the table, so
  the command, the seeds and the date travel with the numbers. Where a CI run
  produced them, cite its workflow URL. This is the one rule
  `tests/test_readmes.py` cannot enforce, which is why it is written down.
- **Measure the numbers, do not assert them.** Anything quantitative — a
  runtime, a seed spread, a fall percentage — gets run before it is written. A
  middle value for chapter 3's no-replay seed scores had crept in where only the
  range 33.5–155.6 is recorded anywhere; it was removed rather than guessed.
- **Size it to the chapter.** Chapter 1 is one 35-line script and its README is
  46 lines. Chapter 6 has seven modules and gets 138. Padding a short chapter to
  match a long one means inventing content, which is the failure this section is
  trying to prevent, not the goal.

Known variation: chapter 4 opens with "What to run, and what to read" instead
of "File Structure", splitting the directory into files you run, files you
read, and supporting files. That is a deliberate restructure and is kept — the
rule is that every file appears somewhere, not that the heading matches.

## Pending

Nothing outstanding on the site itself. Everything previously recorded here is
closed:

- **Chapter 7 status** — now live, and the whole "When adding a chapter" list
  below was worked through rather than the obvious two entries: the chapter row
  (Shao et al., 2024, with the Colab link), the three quickstart commands, the
  `make install-llm` line in step 2, the Colab link row in step 4, a GRPO tab in
  Papers → Code with its `paperExamples` entry in `site.js`, the `progressLine`
  first-paint text (`0 of 6` → `0 of 7`), the "Core Algorithms" and "Reasoning
  Pipelines" learn cards — the second of which still said GRPO was a *later*
  chapter — both learning-path cards and their `pathAdvice` strings in
  `site.js`, the JSON-LD topic list, the three `<head>` descriptions, the CTA
  line, `llms.txt` (summary, chapter line, and eleven resource links), and
  `sitemap.xml`. The `deep` path advice was separately stale, routing readers
  only as far as Chapter 5; it now ends at Chapter 6.

  Chapter 7 is the first chapter whose stack is optional, so the site says so
  where it matters. `install-full` still reads "Chapters 3–6" rather than 3–7,
  which is deliberate: `requirements-llm.txt` carries torch itself, so a
  Chapter 7 reader needs `install-llm` and nothing else, and widening the
  `install-full` note would have promised a stack that does not actually run
  the chapter. `install-llm` is in turn called out as needed by the trainer
  alone — the reward function and the group-size analysis run on a bare
  interpreter.

- **Chapter 4 run commands** — added to the quickstart, along with the ablation's
  real runtime. Ninety-two minutes is not what a reader budgets for a line
  sitting between two three-minute ones.
- **Chapter 4 detail in `llms.txt`** — its module list is now as full as chapter
  3's. Chapter 5 was the one lagging by the time anyone checked, and has been
  brought level in the same pass.
- **Chapter 6 status** — now live. The chapter shipped with a README, the three
  exercise runners, a seeding module and a test suite, so the site was brought
  up with it: the chapter row, the quickstart commands, the `llms.txt` chapter
  line and its nine resource links, the three `<head>` descriptions (1–5 → 1–6),
  the "Key Takeaways" line that still called SAC planned, and `sitemap.xml`. It
  had previously been settled the other way, with the root `README.md` marked in
  progress; that marker is gone.
- **Chapter 6 leftovers** — a second pass caught what the first one missed,
  all of it places that name the live chapters without saying "Chapter 6":
  the `llms.txt` summary line (still "1–5"), the Papers → Code tab strip (which
  had no SAC entry), the Colab link row in quickstart step 4, the "Deep RL
  practitioner" learning-path card, the `make install-full` note, the
  `progressLine` first-paint text ("0 of 5"), the JSON-LD topic list, and
  `make run-ch6-seeding`, which had a Chapter 5 counterpart on the page but no
  Chapter 6 one.
- **Chapter README drift** — chapters 1, 2 and 3 were 28, 37 and 43 lines
  against chapter 6's 138, and were file manifests rather than chapter
  companions. All three now carry Implementation Notes and Troubleshooting, and
  the shape they have to land in is written down under **Chapter READMEs**
  above. Three factual errors surfaced in the rewrite: chapter 3 listed six of
  its eight files and misnamed the Atari agent class, and chapter 2 omitted its
  own notebook. Chapter 3's `ablation.py` also had no `make` target, unlike
  chapters 4–6, so the paper's own ablation was reachable only by finding the
  file; smoke-running the new target then showed its too-short-run guard failing
  to fire at 120 episodes, which is fixed and covered by a test.

- **Chapter 2's notebook was untested.** Chapters 3–6 each had an execution
  suite; chapter 2, the notebook a reader opens first, had none. It shipped a
  `KeyError` in its opening experiment — the outcome tally built its key with
  `f"Hazard {s}"`, which renders `Hazard (1, 1)` against a dict seeded with
  `"Hazard (1,1)"` — and roughly nineteen of twenty episodes end in a hazard,
  so the cell failed for every reader who ran it. `run_td0_gridworld.py` had
  the same tally written with explicit branches and was fine, which is what
  notebook drift looks like. `tests/test_ch02_notebook.py` now covers parity
  with the modules and full execution, and the notebook was normalized to
  nbformat 4.5 with cell ids like the other four.
- **`make run-ch3-ablation` was missing from both quickstarts.** The target
  landed with the chapter 3 README rewrite and `tests/test_readmes.py` checks
  the chapter documents it — but nothing checks the root `README.md` or
  `index.html`, and it was the only chapter ablation absent from both. Added to
  each, and `llms.txt` gained chapter 2's two runner scripts, which it listed
  for every other chapter.

- **Chapter 3 runtime** — quoted as two minutes on the site and in `make help`,
  and three in `src/part_2_methods/ch03_dqn/README.md`. Three is correct; the
  other two now say so. A runtime that appears in the chapter README, `make
  help`, and the quickstart note is three places to change, not one.

When adding a chapter, the files that need touching together are:

- `Makefile` — the new run targets, and the `install-full` line in `help`, which
  names the chapter range.
- `index.html` — the chapter row, the quickstart commands, the Colab link row in
  step 4, the Papers → Code tab (plus its entry in `paperExamples` in
  `site.js`), the `progressLine` fallback text, the `make install-full` note,
  the "Deep RL practitioner" path card, the JSON-LD topic list, and the three
  `<head>` descriptions.
- `llms.txt` — the summary line at the top, the chapter list, *and* the resource
  links. Anything linked from `index.html` belongs here too, including links
  that are not files: Chapter 6's ablation workflow on the Actions tab is one.
- `sitemap.xml` — `lastmod` on the root URL.
- The chapter's own `README.md` under `src/…/chNN_*/` — see **Chapter READMEs**
  above for the shape it has to land in. A chapter whose code ships before its
  README does is how chapters 1–3 ended up at a third the length of 4–6.

A chapter that lands in the row list but not the quickstart is the failure mode
this section keeps catching; the second-pass list above is what that failure
mode looks like once the obvious two are covered.

## Before you start

If you keep a second checkout of this repository, pull first. The site files here move independently of the chapter code, and an older clone can be missing `assets/` and `privacy.html` entirely.
