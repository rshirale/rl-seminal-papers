"""
Train DDPG on Pendulum-v1.

Usage:
    python src/part_2_methods/ch04_ddpg/train_pendulum.py

Matches Algorithm 1 from Lillicrap et al. (2015), "Continuous Control
with Deep Reinforcement Learning."  Pendulum-v1 is a low-dimensional
continuous-control benchmark well suited for verifying DDPG: the state
is (cos θ, sin θ, θ̇) and the single action is torque ∈ [-2, 2].
Reward is roughly –(θ² + 0.1 θ̇² + 0.001 τ²), so the goal is to keep
the pole upright at near-zero velocity and minimal torque.  A well-trained
agent typically achieves episode returns above –200.
"""

import gymnasium as gym
import numpy as np

from .ddpg_agent import DDPGAgent

EPISODES      = 300
MAX_STEPS     = 200   # Pendulum-v1 default horizon
WARMUP_EPS    = 10    # collect experience before training
PRINT_EVERY   = 10

ENV_ID        = "Pendulum-v1"


def main():
    env = gym.make(ENV_ID)
    state_dim  = env.observation_space.shape[0]   # 3
    action_dim = env.action_space.shape[0]         # 1
    max_action = float(env.action_space.high[0])   # 2.0

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )

    returns = []

    for ep in range(1, EPISODES + 1):
        state, _ = env.reset()
        agent.reset_noise()
        ep_return = 0.0

        for _ in range(MAX_STEPS):
            explore = ep > WARMUP_EPS
            action  = agent.select_action(state, explore=explore)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, float(done))

            if ep > WARMUP_EPS:
                agent.train()

            state     = next_state
            ep_return += reward

            if done:
                break

        returns.append(ep_return)

        if ep % PRINT_EVERY == 0:
            avg = np.mean(returns[-PRINT_EVERY:])
            print(f"Episode {ep:4d} | Return: {ep_return:8.1f} | "
                  f"Avg-{PRINT_EVERY}: {avg:8.1f}")

    env.close()
    print("\nTraining complete.")
    print(f"Final {PRINT_EVERY}-episode average: "
          f"{np.mean(returns[-PRINT_EVERY:]):.1f}")


if __name__ == "__main__":
    main()
