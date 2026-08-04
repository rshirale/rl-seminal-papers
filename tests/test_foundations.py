import numpy as np

from algorithms import run_agent, td0
from environments import CliffWalking, GridWorld


def test_gridworld_terminal_transitions():
    env = GridWorld()

    state, reward, done = env.transition((2, 2), "Right")

    assert state == (3, 2)
    assert reward == 10
    assert done is True


def test_td0_returns_finite_values():
    values = td0(GridWorld(), episodes=10)

    assert len(values) == 12
    assert all(np.isfinite(value) for value in values.values())


def test_cliff_transition_resets_to_start():
    env = CliffWalking()

    state, reward, done = env.transition((0, 0), "Right")

    assert state == env.start
    assert reward == -100
    assert done is False


def test_q_learning_and_sarsa_return_expected_shapes():
    env = CliffWalking()

    for mode in ("qlearning", "sarsa"):
        q_values, rewards, falls = run_agent(env, mode=mode, episodes=3)

        assert len(q_values) == env.width * env.height * len(env.actions)
        assert len(rewards) == 3
        assert falls >= 0
