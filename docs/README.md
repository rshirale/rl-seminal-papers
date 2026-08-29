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

## Pending

Nothing outstanding on the site itself. Everything previously recorded here is
closed:

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

A chapter that lands in the row list but not the quickstart is the failure mode
this section keeps catching; the second-pass list above is what that failure
mode looks like once the obvious two are covered.

## Before you start

If you keep a second checkout of this repository, pull first. The site files here move independently of the chapter code, and an older clone can be missing `assets/` and `privacy.html` entirely.
