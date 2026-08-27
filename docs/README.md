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

### Add the Chapter 4 run commands to the quickstart

The "Run an experiment" step currently offers only `make run-ch2-cliff` and `make run-ch3-cartpole`. Chapter 4 ships two Make targets and neither is listed. Add both, in the `.copy-wrap` pattern above:

```
make run-ch4-pendulum — DDPG on Pendulum-v1 (~3 min)
make run-ch4-ablation — Which components DDPG actually needs (~90 min)
```

Put the runtime on the ablation row. It is nine 200-episode training runs, measured at 92 minutes on an 8-core Intel CPU. Without that note, a reader who copies the command will reasonably conclude it has hung.

### Bring Chapter 4 up to Chapter 3's level of detail in `llms.txt`

Chapter 3 lists six individual source files. Chapter 4 lists only the notebook, the Colab link, and the directory tree, though it now has a fuller module set and a README that separates the files you run from the files you read. Add:

```
- Chapter 4 README: …/blob/main/src/part_2_methods/ch04_ddpg/README.md
- Chapter 4 Actor network: …/actor.py
- Chapter 4 Critic network: …/critic.py
- Chapter 4 Replay buffer: …/replay_buffer.py
- Chapter 4 Gaussian exploration noise: …/gaussian_noise.py
- Chapter 4 DDPG agent: …/ddpg_agent.py
- Chapter 4 Pendulum training script: …/train_pendulum.py
- Chapter 4 Component ablation: …/ablation.py
- Chapter 4 Seeding helper (reproducible runs): …/seeding.py
```

Chapter 3 links its README nowhere either — worth adding both in the same pass.

### Confirm the Chapter 6 status

`index.html` lists Chapter 6 (SAC) as planned, and `llms.txt` marks it `(planned)`, while `src/part_2_methods/ch06_sac/` exists with an implementation and a notebook, and the root `README.md` lists Chapter 6 with a Colab badge. The site and the repository disagree. Confirm the intended status before changing either.

## Before you start

If you keep a second checkout of this repository, pull first. The site files here move independently of the chapter code, and an older clone can be missing `assets/` and `privacy.html` entirely.
