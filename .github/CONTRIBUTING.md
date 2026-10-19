# Contributing

Thanks for helping improve the companion code for **Reinforcement Learning: The
Foundational Ideas**. Bug reports, fixes and clearer explanations are all
welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## What belongs here

- **Code that doesn't run or gives wrong results.** A chapter script, `make`
  target or notebook that fails, or an implementation that doesn't match the
  math in its paper.
- **Setup problems** on a platform or Python version the README says is
  supported (Python 3.10–3.13).
- **Mistakes in this repository's docs**: the README, chapter READMEs and the
  companion site under `docs/`.

Errors in the book's own text are best reported through the book's errata
channel at Manning. If the error is in a code listing that also lives here,
open an issue here as well.

Security problems should not go in a public issue. See
[SECURITY.md](SECURITY.md).

## Reporting a bug

Open an issue with the **Bug report** template. The most useful thing you can
include is the output of:

```bash
make doctor
```

It shows the active interpreter, platform, CPU architecture and the version of
every dependency. Most "it doesn't run for me" reports turn out to be the wrong
virtual environment, and the first two lines make that obvious.

## Making a change

1. Fork the repository and create a branch from `main`.
2. Set up an environment as described in the [README](../README.md#-quick-start),
   then install the test tools:

   ```bash
   make install-full    # or the install target for the chapter you're changing
   make install-test
   ```

3. Make your change. Keep it to one fix or feature per pull request.
4. Run the tests:

   ```bash
   make test        # fast suite
   make test-all    # also runs every chapter notebook top to bottom
   ```

   CI runs `make test-all` on Python 3.10, 3.12 and 3.13, so run it locally
   when you touch a notebook or anything a notebook imports.

5. Open a pull request and fill in the template.

### Things the tests check that are easy to miss

- **Notebooks must match their modules.** Each chapter's notebook is tested
  against the Python files next to it, and from chapter 8 on, the notebook's
  code cells are the module source, character for character. If you change a
  module, update the matching notebook cell. The parity tests will tell you
  which one.
- **Chapter READMEs must be complete.** `tests/test_readmes.py` checks that
  every module, notebook and `make` target in a chapter is listed in that
  chapter's README. A new file or target needs a README entry too. The
  expected layout is described in [docs/README.md](../docs/README.md).

### Style

- Match the code around you: naming, comment density, docstring style.
- Follow the paper's notation where the chapter does, so the code still reads
  next to the equations.
- Don't add new dependencies without discussing it in an issue first. Each
  chapter's install footprint is kept small on purpose.

## License

Contributions are released under the repository's [MIT License](../LICENSE).
