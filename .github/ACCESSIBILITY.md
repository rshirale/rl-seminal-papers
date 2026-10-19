# Accessibility

This repository and its [companion website](https://rshirale.github.io/rl-seminal-papers/)
should be usable by everyone reading the book, whatever device, input method or
assistive technology they use.

## What the companion site supports

- **Keyboard navigation.** A "Skip to main content" link comes first in the
  tab order, every interactive element shows a visible focus outline, and the
  reading-path cards respond to Enter and Space. In the paper-to-code tabs,
  the arrow keys move between papers and Home and End jump to the first and
  last.
- **Reduced motion.** If your system asks for reduced motion, the site turns
  off its decorative animations and starts the Q-value animation paused. A
  Play/Pause button controls it either way.
- **Light and dark themes**, switched with the button in the header. Your
  choice is remembered on that device.
- **Screen readers.** The page declares its language, controls have
  accessible names, and the Q-Learning playground's readout is announced as it
  updates. The playground grid is also provided as a table that screen readers
  can move through cell by cell, giving each cell's best action and Q-value
  and marking the agent, the goal and the learned path.

## Known limitations

- **Figures** in the notebooks and chapter scripts are images with no text
  alternative beyond the explanation around them.
- The site hasn't been formally audited against WCAG.

## Supported environments

The site is built for current desktop and mobile browsers. Its automated
check (`make site-check`) runs in headless Chrome, so other browsers are
checked by hand rather than on every change. The notebooks run in Jupyter and
Google Colab, and the code supports Python 3.10–3.13.

## Reporting a barrier

If something on the site or in the code is hard or impossible for you to use,
please [open an issue](https://github.com/rshirale/rl-seminal-papers/issues/new/choose)
or email rahul.shirale@live.com. Tell us what you were trying to do, what
happened, and which browser or assistive technology you were using.
Accessibility problems are treated as bugs.
