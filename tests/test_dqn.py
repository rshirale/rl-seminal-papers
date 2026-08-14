"""Chapter 3 (DQN) module tests.

Covers the replay buffer, both Q-networks, and the Atari agent. Several tests
are explicit regressions for bugs that were live in this chapter; those are
marked with a "Regression:" note so they are not "simplified" away later.
"""

import numpy as np
import pytest
import torch

from src.part_2_methods.ch03_dqn import DQN, DQNAgent, ReplayBuffer, SimpleDQN

CPU = torch.device("cpu")


def make_agent(**overrides):
    """An Atari agent small enough to construct in a test."""
    kwargs = dict(
        input_channels=4,
        num_actions=6,
        device=CPU,
        buffer_capacity=200,
        batch_size=8,
        warmup_steps=16,
    )
    kwargs.update(overrides)
    return DQNAgent(**kwargs)


def frame(rng, channels=4):
    """A raw uint8 frame stack, exactly as the Atari wrappers emit it."""
    return rng.integers(0, 256, (channels, 84, 84), dtype=np.uint8)


# --------------------------------------------------------------------------
# ReplayBuffer
# --------------------------------------------------------------------------

def test_push_then_sample_returns_stored_values():
    buf = ReplayBuffer(capacity=10, state_shape=(3,))
    buf.push(np.array([1.0, 2.0, 3.0]), 2, 0.5, np.array([4.0, 5.0, 6.0]), True)

    states, actions, rewards, next_states, dones = buf.sample(1)

    assert np.allclose(states[0], [1.0, 2.0, 3.0])
    assert np.allclose(next_states[0], [4.0, 5.0, 6.0])
    assert actions[0] == 2
    assert rewards[0] == pytest.approx(0.5)
    assert bool(dones[0]) is True


def test_len_tracks_each_push_before_capacity_is_reached():
    """Checked push by push: asserting only the saturated value would hide an
    off-by-one in the size update, since min() clamps it away."""
    buf = ReplayBuffer(capacity=5, state_shape=(1,))
    for expected in range(1, 6):
        buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), False)
        assert len(buf) == expected


def test_len_saturates_at_capacity_and_writes_wrap_around():
    buf = ReplayBuffer(capacity=4, state_shape=(1,))
    for i in range(6):
        buf.push(np.array([float(i)]), 0, 0.0, np.array([0.0]), False)

    assert len(buf) == 4          # saturated, not 6
    assert buf.position == 2      # wrapped: 6 % 4
    # The two oldest entries (0, 1) were overwritten by (4, 5).
    assert sorted(buf.states[:, 0].tolist()) == [2.0, 3.0, 4.0, 5.0]


def test_state_dtype_is_honoured():
    """Regression: the uint8 buffer is what keeps Atari memory tractable."""
    buf = ReplayBuffer(capacity=4, state_shape=(4, 84, 84), state_dtype=np.uint8)
    assert buf.states.dtype == np.uint8
    assert buf.next_states.dtype == np.uint8


def test_uint8_storage_is_four_times_smaller_than_float32():
    shape = (4, 84, 84)
    small = ReplayBuffer(capacity=100, state_shape=shape, state_dtype=np.uint8)
    large = ReplayBuffer(capacity=100, state_shape=shape, state_dtype=np.float32)
    assert large.states.nbytes == 4 * small.states.nbytes


def test_sample_returns_fields_in_documented_order():
    buf = ReplayBuffer(capacity=32, state_shape=(2,))
    for _ in range(16):
        buf.push(np.zeros(2), 1, 1.0, np.ones(2), False)

    states, actions, rewards, next_states, dones = buf.sample(8)

    assert states.shape == (8, 2)
    assert next_states.shape == (8, 2)
    assert actions.shape == rewards.shape == dones.shape == (8,)
    assert np.all(states == 0.0) and np.all(next_states == 1.0)


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

def test_dqn_accepts_atari_stack_and_emits_one_value_per_action():
    q = DQN(input_channels=4, num_actions=6)(torch.zeros(2, 4, 84, 84))
    assert q.shape == (2, 6)


def test_simple_dqn_emits_one_value_per_action():
    q = SimpleDQN(input_dim=4, num_actions=2)(torch.zeros(5, 4))
    assert q.shape == (5, 2)


# --------------------------------------------------------------------------
# DQNAgent
# --------------------------------------------------------------------------

def test_agent_stores_frames_as_uint8():
    """Regression: storing float32 here needed ~22.6 GB at capacity=100k."""
    assert make_agent().memory.states.dtype == np.uint8


def test_agent_normalizes_uint8_frames_into_unit_range():
    agent = make_agent()
    raw = np.full((4, 84, 84), 255, dtype=np.uint8)

    out = agent._states_to_tensor(raw)

    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.ones_like(out))


def test_normalization_matches_explicit_divide():
    agent = make_agent()
    raw = frame(np.random.default_rng(0))
    expected = torch.as_tensor(raw.astype(np.float32) / 255.0)
    assert torch.allclose(agent._states_to_tensor(raw), expected)


def test_normalization_does_not_mutate_the_source_array():
    """The in-place div_ must land on a copy, never on buffer storage."""
    agent = make_agent()
    raw = np.full((4, 84, 84), 200, dtype=np.uint8)

    agent._states_to_tensor(raw)

    assert raw.dtype == np.uint8
    assert np.all(raw == 200)


def test_select_action_returns_a_valid_action():
    agent = make_agent()
    state = frame(np.random.default_rng(1))
    assert agent.select_action(state, epsilon=0.0) in range(6)
    assert agent.select_action(state, epsilon=1.0) in range(6)


def test_greedy_action_is_deterministic():
    agent = make_agent()
    state = frame(np.random.default_rng(2))
    choices = {agent.select_action(state, epsilon=0.0) for _ in range(10)}
    assert len(choices) == 1


def test_select_action_leaves_network_in_train_mode():
    """select_action flips to eval() internally; it must restore train()."""
    agent = make_agent()
    agent.select_action(frame(np.random.default_rng(3)), epsilon=0.0)
    assert agent.online_net.training is True


def test_learning_updates_online_weights():
    agent = make_agent()
    rng = np.random.default_rng(4)
    before = [p.detach().clone() for p in agent.online_net.parameters()]

    state = frame(rng)
    for _ in range(40):
        nxt = frame(rng)
        agent.step(state, int(rng.integers(0, 6)), 1.0, nxt, False)
        state = nxt

    after = list(agent.online_net.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_warmup_shorter_than_batch_size_does_not_crash():
    """Regression: sample() draws without replacement, so learning must also
    wait for the buffer to hold batch_size transitions."""
    agent = make_agent(warmup_steps=2, batch_size=16)
    rng = np.random.default_rng(5)

    state = frame(rng)
    for _ in range(30):
        nxt = frame(rng)
        agent.step(state, 0, 0.0, nxt, False)  # must not raise
        state = nxt


def test_target_network_starts_synced_and_resyncs_on_demand():
    agent = make_agent()

    def params_equal():
        return all(
            torch.equal(a, b)
            for a, b in zip(agent.online_net.parameters(),
                            agent.target_net.parameters())
        )

    assert params_equal()  # identical weights at construction

    with torch.no_grad():
        next(agent.online_net.parameters()).add_(1.0)
    assert not params_equal()

    agent._update_target_network()
    assert params_equal()


def test_target_network_stays_frozen():
    assert make_agent().target_net.training is False


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def draw_from_every_rng():
    """One sample from each generator a training run depends on."""
    import random

    return (
        random.random(),
        float(np.random.random()),
        float(torch.rand(1)),
    )


def test_set_seed_makes_every_generator_reproducible():
    from src.part_2_methods.ch03_dqn.seeding import set_seed

    set_seed(42)
    first = draw_from_every_rng()
    set_seed(42)
    assert draw_from_every_rng() == first


def test_different_seeds_produce_different_draws():
    """Guards against a set_seed that freezes everything to a constant."""
    from src.part_2_methods.ch03_dqn.seeding import set_seed

    set_seed(42)
    a = draw_from_every_rng()
    set_seed(7)
    assert draw_from_every_rng() != a


def test_seeding_covers_network_initialization():
    """Weight init must be inside the seeded region, or agents built from the
    same seed still differ."""
    from src.part_2_methods.ch03_dqn.seeding import set_seed

    set_seed(42)
    first = [p.detach().clone() for p in make_agent().online_net.parameters()]
    set_seed(42)
    second = list(make_agent().online_net.parameters())

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_seed_env_makes_episode_starts_reproducible():
    import gymnasium as gym

    from src.part_2_methods.ch03_dqn.seeding import seed_env

    def first_states(seed):
        env = gym.make("CartPole-v1")
        seed_env(env, seed)
        states = [env.reset()[0].tolist() for _ in range(3)]
        actions = [int(env.action_space.sample()) for _ in range(5)]
        env.close()
        return states, actions

    assert first_states(42) == first_states(42)
    assert first_states(42) != first_states(7)


def test_cartpole_main_accepts_a_seed_and_defaults_to_unseeded():
    """The flag must be opt-in, so omitting it preserves existing behaviour."""
    import inspect

    from src.part_2_methods.ch03_dqn import train_cartpole

    signature = inspect.signature(train_cartpole.main)
    assert "seed" in signature.parameters
    assert signature.parameters["seed"].default is None
