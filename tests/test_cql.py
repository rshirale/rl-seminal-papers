"""Tests for the Chapter 9 CQL modules.

The listings in ``cql.py`` are printed in the chapter, so these tests pin down
what the text claims about them rather than how they are written. The critic
loss must reduce to the plain TD loss at ``cql_alpha=0`` -- exercise 9.1's SAC
baseline depends on it -- and must be linear in ``cql_alpha``, which is what
lets ``probe_metrics`` read the penalty off as the difference of two calls.
FQE must recover a value it can know exactly. And a resumed training run must
not log the same steps twice.

None of this needs the PointMaze dataset, so Minari is stubbed when it is not
installed; ``load_minari`` is the only code that touches it.
"""

import csv
import sys
import types

import pytest

torch = pytest.importorskip("torch")

try:
    import minari  # noqa: F401
except ImportError:
    sys.modules["minari"] = types.ModuleType("minari")

from src.part_2_methods.ch06_sac.actor import Actor
from src.part_2_methods.ch06_sac.critic import Critic
from src.part_3_applications.ch09_cql import cql as L
from src.part_3_applications.ch09_cql.train_cql_vs_sac import truncate_log

STATE_DIM, ACTION_DIM = 6, 2


def make_batch(b=32, done=0.0):
    s = torch.randn(b, STATE_DIM)
    a = torch.rand(b, ACTION_DIM) * 2 - 1
    r = torch.rand(b, 1)
    s2 = torch.randn(b, STATE_DIM)
    d = torch.full((b, 1), done)
    return [s, a, r, s2, d]


def loss_at(alpha, nets, batch):
    torch.manual_seed(1)
    return L.cql_critic_loss(*nets, batch, cql_alpha=alpha).item()


@pytest.fixture
def nets():
    torch.manual_seed(0)
    return Critic(STATE_DIM, ACTION_DIM), Critic(STATE_DIM, ACTION_DIM), \
        Actor(STATE_DIM, ACTION_DIM)


def test_zero_alpha_leaves_the_plain_td_loss(nets):
    critic, critic_target, actor = nets
    batch = make_batch()
    s, a, r, s2, d = batch
    torch.manual_seed(1)
    with torch.no_grad():
        a2, _ = actor(s2)
        y = r + 0.99 * (1 - d) * torch.min(*critic_target(s2, a2))
        q1, q2 = critic(s, a)
        td = (torch.nn.functional.mse_loss(q1, y)
              + torch.nn.functional.mse_loss(q2, y)).item()
    assert loss_at(0.0, nets, batch) == pytest.approx(td, rel=1e-6)


def test_loss_is_linear_in_alpha(nets):
    batch = make_batch()
    l0, l1, l5 = (loss_at(x, nets, batch) for x in (0.0, 1.0, 5.0))
    assert l5 - l0 == pytest.approx(5 * (l1 - l0), rel=1e-4)


def test_loss_gradient_reaches_the_critic_only(nets):
    critic, critic_target, actor = nets
    L.cql_critic_loss(critic, critic_target, actor, make_batch()).backward()
    assert all(p.grad is not None for p in critic.parameters())
    assert all(p.grad is None for p in critic_target.parameters())
    assert all(p.grad is None for p in actor.parameters())


def test_offline_buffer_samples_aligned_columns():
    import numpy as np
    xs = [np.arange(5)[:, None], np.arange(5, 8)[:, None]]
    buf = L.OfflineBuffer(xs, [x * 10 for x in xs])
    a, b = buf.sample(64)
    assert a.shape == b.shape == (64, 1)
    assert a.dtype == torch.float32
    assert torch.equal(b, a * 10)


def test_fqe_recovers_a_known_value():
    # Every transition ends the episode with reward 1, so the true value of
    # every start state is exactly 1 whatever the policy does.
    torch.manual_seed(0)
    s, a, r, s2, d = make_batch(b=64, done=1.0)
    buf = L.OfflineBuffer([s.numpy()], [a.numpy()], [torch.ones(64, 1).numpy()],
                          [s2.numpy()], [d.numpy()])
    buf.starts = s[:8]
    estimate = L.fitted_q_evaluation(Actor(STATE_DIM, ACTION_DIM), buf,
                                     Critic(STATE_DIM, ACTION_DIM), steps=500)
    assert estimate.item() == pytest.approx(1.0, abs=0.05)


def test_truncate_log_drops_rows_a_resume_would_repeat(tmp_path):
    path = tmp_path / "progress.csv"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(
            [["step", "q_loss"], [5000, 1.0], [10000, 2.0], [15000, 3.0]])
    truncate_log(path, 12500)
    with open(path, newline="") as f:
        assert list(csv.reader(f)) == [
            ["step", "q_loss"], ["5000", "1.0"], ["10000", "2.0"]]
