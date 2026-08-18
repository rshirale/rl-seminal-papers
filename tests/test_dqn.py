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


# --------------------------------------------------------------------------
# Atari wrappers (train_atari)
#
# The wrappers never run during the CartPole example the chapter walks through,
# so nothing else in this suite exercises them. Every test below is an explicit
# regression for a bug that shipped.
# --------------------------------------------------------------------------

import gymnasium as gym

SHAPE = (4, 4, 3)


def atari_module():
    """train_atari imports ale_py and cv2 at module scope."""
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")
    from src.part_2_methods.ch03_dqn import train_atari

    return train_atari


class _SkipStub(gym.Env):
    """Emits a scripted sequence of flat frames and can end the episode early.

    Frame values are deliberately non-monotonic so "max over the last two
    frames" and "max over every frame in the block" give different answers.
    """

    def __init__(self, values, terminate_after=None):
        self.observation_space = gym.spaces.Box(0, 255, SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.values = values
        self.terminate_after = terminate_after
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(SHAPE, np.uint8), {}

    def step(self, action):
        value = self.values[self.t]
        self.t += 1
        terminated = self.terminate_after is not None and self.t >= self.terminate_after
        return np.full(SHAPE, value, np.uint8), 1.0, terminated, False, {}


def test_max_and_skip_pools_only_the_final_two_frames():
    """The flicker fix is a max over the last *two* frames, not over the block."""
    m = atari_module()
    env = m.MaxAndSkipEnv(_SkipStub([9, 1, 2, 3]), skip=4)
    env.reset()

    obs, reward, terminated, _, _ = env.step(0)

    assert obs.max() == 3, "expected max(2, 3) from the last two frames"
    assert reward == 4.0
    assert not terminated


def test_max_and_skip_returns_the_terminal_frame_when_the_episode_ends_early():
    """Regression: the buffer was only written at i == skip-2 and skip-1, so an
    episode ending on the first skipped frame broke out before either write.
    The returned frame was then max-pooled entirely from the *previous* action's
    frames, and the frame the agent actually died on never reached the replay
    buffer. Terminal transitions are exactly the ones the Bellman target treats
    specially, so corrupting them is not a harmless edge case."""
    m = atari_module()
    env = m.MaxAndSkipEnv(_SkipStub([5, 6, 7, 8], terminate_after=1), skip=4)
    env.reset()  # seeds the buffer with zeros

    obs, reward, terminated, _, _ = env.step(0)

    assert terminated
    assert obs.max() == 5, "terminal frame was dropped in favour of stale buffer contents"
    assert reward == 1.0, "reward should cover only the steps actually taken"


class _FireStub(gym.Env):
    """Minimal FIRE-on-reset environment that can die on a chosen action once."""

    def __init__(self, die_on_action):
        self.observation_space = gym.spaces.Box(0, 255, SHAPE, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self.die_on_action = die_on_action
        self.reset_kwargs = []

    def get_action_meanings(self):
        return ["NOOP", "FIRE", "UP", "DOWN"]

    def reset(self, **kwargs):
        self.reset_kwargs.append(kwargs)
        return np.full(SHAPE, 7, np.uint8), {"from_reset": len(self.reset_kwargs)}

    def step(self, action):
        terminated = action == self.die_on_action
        if terminated:
            self.die_on_action = None  # die at most once, so recovery can succeed
        return np.full(SHAPE, 99, np.uint8), 0.0, terminated, False, {"from_step": True}


def test_fire_reset_returns_the_post_reset_frame_not_the_terminal_one():
    """Regression: the recovery reset's return value was discarded, so reset()
    handed back the terminal frame from the episode that had just ended — the
    agent began the next episode looking at a game-over screen."""
    m = atari_module()
    stub = _FireStub(die_on_action=2)
    env = m.FireResetEnv(stub)

    obs, info = env.reset()

    assert obs.max() == 7, "returned the terminal frame instead of the fresh reset"
    assert info.get("from_reset"), "returned a hardcoded {} instead of the env's info"


def test_fire_reset_does_not_replay_the_same_seed_on_recovery():
    """Regression: recovery resets reused the caller's seed, which replays the
    identical failing episode instead of recovering from it."""
    m = atari_module()
    stub = _FireStub(die_on_action=2)
    env = m.FireResetEnv(stub)

    env.reset(seed=1234)

    assert stub.reset_kwargs[0] == {"seed": 1234}, "the initial reset must honour the seed"
    assert len(stub.reset_kwargs) > 1, "expected a recovery reset after the episode died"
    assert all("seed" not in kw for kw in stub.reset_kwargs[1:])


def test_atari_loop_stores_terminated_not_done():
    """Regression: the loop stored `done` (terminated or truncated), so a
    time-limit truncation had its bootstrap target zeroed by _learn(). Asserted
    on the source because the flag only exists inside main()'s loop.
    train_cartpole.py:110 has always made this distinction."""
    import inspect

    m = atari_module()
    source = inspect.getsource(m.main)
    code = "\n".join(line.split("#")[0] for line in source.splitlines())

    assert "agent.step(state, action, reward, next_state, terminated)" in code
    assert "next_state, done)" not in code
