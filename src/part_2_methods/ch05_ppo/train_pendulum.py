import argparse

import gymnasium as gym
import numpy as np
import torch

if __package__:
    from .ppo_agent import PPOAgent
else:  # pragma: no cover - only used by direct script execution.
    from ppo_agent import PPOAgent


def parse_args():
    """Parse options for a reproducible, configurable training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--update-every", type=int, default=2048)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def main():
    """Train PPO on Pendulum-v1."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = PPOAgent(
        state_dim, action_dim, max_action,
        lr=1e-3, gamma=args.gamma, lam=0.95,
        eps_clip=0.2, k_epochs=10, batch_size=64
    )

    rollouts = []
    timestep = 0
    ep_rewards = []
    approx_kl = 0.0
    clip_frac = 0.0

    try:
        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode)
            ep_reward = 0.0

            for _ in range(env.spec.max_episode_steps):
                timestep += 1
                action, logprob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # True termination stops bootstrapping; time-limit truncation does not.
                rollouts.append(
                    (state, action, reward, next_state,
                     float(terminated), logprob, value)
                )
                state = next_state
                ep_reward += reward

                if timestep % args.update_every == 0:
                    approx_kl, clip_frac = agent.update(rollouts)
                    rollouts = []

                if terminated or truncated:
                    break

            ep_rewards.append(ep_reward)
            if episode % args.log_every == 0:
                avg_reward = np.mean(ep_rewards[-args.log_every:])
                print(
                    f"Episode {episode:3d} | Reward: {avg_reward:8.1f} | "
                    f"approx_kl: {approx_kl:.3f} | clip_frac: {clip_frac:.2f}"
                )

        if rollouts:
            approx_kl, clip_frac = agent.update(rollouts)
    finally:
        env.close()


if __name__ == "__main__":
    main()
