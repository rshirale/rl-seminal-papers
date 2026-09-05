import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CH2 = ROOT / "src" / "part_1_foundations" / "ch02_fundamentals"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CH2) not in sys.path:
    sys.path.insert(0, str(CH2))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end runs (notebook execution, training loops). "
        "Skipped by `make test`, included by `make test-all`.",
    )


@pytest.fixture
def ch07_oracle_lm():
    """A stub causal LM for Chapter 7's ``sequence_logprob``, and its tokenizer.

    That function shifts the logits by one before gathering, and an off-by-one
    there is silent: the run trains, the numbers look plausible, and every
    log-probability is attributed to the wrong token. This model turns the
    alignment into something loud. Its position ``t`` puts almost all of its
    mass on the token that actually appears at ``t + 1``, so the correct
    alignment scores about -0.002 per token and any other alignment scores
    about ``-CONFIDENCE``.

    Two details are there to make the rest of the function observable. The
    model is not confident about padding -- where the next token is the pad id
    it puts its mass elsewhere, which is what makes an unmasked pad position
    visible in the mean. And it carries one trainable parameter, so a test can
    tell the adapter branch (which must carry gradient, even when the caller
    is inside ``no_grad``) from the reference branch (which must not).

    Lives here rather than in ``test_grpo.py`` because the notebook parity
    test needs the same stub: the notebook re-implements this function inline,
    and the two have to stay one implementation.
    """
    import contextlib
    from types import SimpleNamespace

    import torch

    vocab, confidence, pad_id = 7, 8.0, 0
    decoy = vocab - 1

    class OracleLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.adapter = True

        def forward(self, ids):
            index = torch.zeros_like(ids).unsqueeze(-1)
            nxt = ids[:, 1:]
            index[:, :-1, 0] = torch.where(nxt == pad_id,
                                           torch.full_like(nxt, decoy), nxt)
            logits = torch.zeros(*ids.shape, vocab).scatter(2, index,
                                                            confidence)
            if not self.adapter:
                # The reference policy is a different distribution -- flat --
                # so a test can tell the two branches apart by their value.
                logits = torch.zeros_like(logits)
            return SimpleNamespace(logits=logits * self.scale)

        @contextlib.contextmanager
        def disable_adapter(self):
            """Stands in for peft's context manager of the same name."""
            self.adapter = False
            try:
                yield
            finally:
                self.adapter = True

    return SimpleNamespace(
        model=OracleLM(),
        tokenizer=SimpleNamespace(pad_token_id=pad_id),
        vocab=vocab, confidence=confidence, pad_id=pad_id,
    )
