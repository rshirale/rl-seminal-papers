import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.part_2_methods.ch05_ppo.ppo_agent import PPOAgent


def make_agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return PPOAgent(
        state_dim=3,
        action_dim=1,
        max_action=2.0,
        batch_size=16,
        k_epochs=1,
    )


def test_actions_are_bounded_and_diagnostics_are_finite():
    agent = make_agent()

    for _ in range(100):
        action, logprob, value = agent.select_action(np.zeros(3, dtype=np.float32))

        assert action.shape == (1,)
        assert np.all(action >= -2.0)
        assert np.all(action <= 2.0)
        assert np.isfinite(logprob)
        assert np.isfinite(value)


def test_update_runs_on_a_small_rollout():
    agent = make_agent()
    state = np.zeros(3, dtype=np.float32)
    rollouts = []

    for _ in range(16):
        action, logprob, value = agent.select_action(state)
        rollouts.append((state, action, -1.0, state, 0.0, logprob, value))

    approx_kl, clip_fraction = agent.update(rollouts)

    assert np.isfinite(approx_kl)
    assert np.isfinite(clip_fraction)
    assert 0.0 <= clip_fraction <= 1.0


def test_empty_update_is_a_noop():
    assert make_agent().update([]) == (0.0, 0.0)
