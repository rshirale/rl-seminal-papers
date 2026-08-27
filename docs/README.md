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

Nothing outstanding on the site itself. The three items previously recorded here
are closed:

- **Chapter 4 run commands** — added to the quickstart, along with the ablation's
  real runtime. Ninety-two minutes is not what a reader budgets for a line
  sitting between two three-minute ones.
- **Chapter 4 detail in `llms.txt`** — its module list is now as full as chapter
  3's. Chapter 5 was the one lagging by the time anyone checked, and has been
  brought level in the same pass.
- **Chapter 6 status** — settled as planned. `index.html` and `llms.txt` already
  said so; the root `README.md` was the file out of step, listing a Colab badge
  that read as shipped. It now marks the chapter in progress while keeping the
  notebook reachable, because the code does run.

When adding a chapter, the files that need touching together are `index.html`
(chapter row *and* quickstart command), `llms.txt` (the chapter list *and* the
resource links), the three `<head>` descriptions, and `sitemap.xml`. A chapter
that lands in the row list but not the quickstart is the failure mode this
section keeps catching.

## Before you start

If you keep a second checkout of this repository, pull first. The site files here move independently of the chapter code, and an older clone can be missing `assets/` and `privacy.html` entirely.
