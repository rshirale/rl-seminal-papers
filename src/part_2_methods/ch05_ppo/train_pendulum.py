import gymnasium as gym
import numpy as np
import torch
from ppo_agent import PPOAgent

SEED = 42
MAX_EPISODES = 400
MAX_TIMESTEPS = 200
UPDATE_EVERY = 2048
LOG_EVERY = 50

torch.manual_seed(SEED)
np.random.seed(SEED)

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = PPOAgent(
    state_dim, action_dim, max_action,
    lr=1e-3, gamma=0.9, lam=0.95,
    eps_clip=0.2, k_epochs=10, batch_size=64
)

rollouts = []
timestep = 0
ep_rewards = []
approx_kl = 0.0
clip_frac = 0.0

for episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset(seed=SEED + episode)
    ep_reward = 0.0

    for t in range(MAX_TIMESTEPS):
        timestep += 1

        action, logprob, value = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        rollouts.append(
            (state, action, reward, next_state,
             float(done or truncated), logprob, value)
        )
        state = next_state
        ep_reward += reward

        if timestep % UPDATE_EVERY == 0:
            approx_kl, clip_frac = agent.update(rollouts)
            rollouts = []

        if done or truncated:
            break

    ep_rewards.append(ep_reward)

    if episode % LOG_EVERY == 0:
        avg_reward = np.mean(ep_rewards[-LOG_EVERY:])
        print(
            f"Episode {episode:3d} | "
            f"Reward: {avg_reward:8.1f} | "
            f"approx_kl: {approx_kl:.3f} | "
            f"clip_frac: {clip_frac:.2f}"
        )

env.close()
