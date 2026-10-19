# Security Policy

This repository is educational code that accompanies a book. It is not a
service and has no releases, so fixes go to the `main` branch only.

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately
in either of these ways:

- **GitHub:** use [Report a vulnerability](https://github.com/rshirale/rl-seminal-papers/security/advisories/new)
  under the repository's **Security** tab.
- **Email:** rahul.shirale@live.com

Please include what you found, the steps to reproduce it, and the files or
`make` targets involved. You should get a reply within a week.

## In scope

- Code in this repository that could harm the person running it, for example
  unsafe loading of downloaded files or checkpoints, or a workflow in
  `.github/workflows/` that exposes secrets.
- The companion site under `docs/`, published at
  https://rshirale.github.io/rl-seminal-papers/.

Vulnerabilities in third-party packages (PyTorch, Gymnasium, transformers,
Minari and others) should be reported to those projects. If a pinned version
here is affected, open an ordinary issue asking for the pin to be raised.
