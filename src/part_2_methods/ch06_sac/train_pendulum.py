"""
Train SAC (SAC-v2) on Pendulum-v1.

Usage:
    python src/part_2_methods/ch06_sac/train_pendulum.py

Matches Algorithm 1 from Haarnoja et al. (2018), "Soft Actor-Critic
Algorithms and Applications."  Pendulum-v1 is a low-dimensional
continuous-control benchmark well suited for verifying SAC: the state is
(cos θ, sin θ, θ̇) and the single action is torque ∈ [-2, 2].  The agent
takes random actions during a short warmup to seed the replay buffer, then
runs one gradient update per environment step.

A well-trained agent settles around episode returns of roughly –150, while
the temperature α decays from 1.0 toward ~0.05 as the policy sharpens and
needs less encouragement to explore.  Trains in a few minutes on CPU.
"""

import gymnasium as gym
import numpy as np
import torch

from sac_agent import SACAgent

SEED          = 42
ENV_ID        = "Pendulum-v1"
TOTAL_STEPS   = 50_000
WARMUP_STEPS  = 10_000   # random actions before the first gradient update
PRINT_EVERY   = 10       # episodes


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make(ENV_ID)
    env.action_space.seed(SEED)
    state_dim  = env.observation_space.shape[0]   # 3
    action_dim = env.action_space.shape[0]         # 1
    max_action = float(env.action_space.high[0])   # 2.0

    agent = SACAgent(state_dim, action_dim, max_action)

    state, _ = env.reset(seed=SEED)
    ep_return = 0.0
    returns = []
    episode = 0

    for step in range(1, TOTAL_STEPS + 1):
        if len(agent.replay) < WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store the termination flag only: a time-limit truncation is not a
        # true terminal state, so the target must still bootstrap from
        # next_state. Pendulum-v1 only ever truncates, never terminates.
        agent.store(state, action, reward, next_state, float(terminated))

        state      = next_state
        ep_return += reward

        if len(agent.replay) >= WARMUP_STEPS:
            agent.train()

        if done:
            episode += 1
            returns.append(ep_return)
            if episode % PRINT_EVERY == 0:
                avg = np.mean(returns[-PRINT_EVERY:])
                print(f"Episode {episode:4d} | Step {step:6d} | "
                      f"Return: {ep_return:8.1f} | "
                      f"Avg-{PRINT_EVERY}: {avg:8.1f} | "
                      f"alpha: {agent.alpha:.3f}")
            state, _ = env.reset()
            ep_return = 0.0

    env.close()
    print("\nTraining complete.")
    if returns:
        print(f"Final {PRINT_EVERY}-episode average: "
              f"{np.mean(returns[-PRINT_EVERY:]):.1f}")


if __name__ == "__main__":
    main()
