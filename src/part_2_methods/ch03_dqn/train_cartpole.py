import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_network import SimpleDQN
from replay_buffer import ReplayBuffer

BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10_000
TARGET_UPDATE_FREQ = 500
WARMUP_STEPS = 1_000
NUM_EPISODES = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgentCartPole:
    """Lightweight DQN agent using SimpleDQN for CartPole-v1."""
    def __init__(self, env, learning_rate=1e-4, gamma=GAMMA):
        self.env = env
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity=100000)

        input_dim = env.observation_space.shape[0]
        self.online_net = SimpleDQN(input_dim, self.num_actions).to(device)
        self.target_net = SimpleDQN(input_dim, self.num_actions).to(device)
        self._update_target_network()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.huber_loss = nn.SmoothL1Loss()
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.online_net(state_t).argmax().item()

    def _update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q[dones] = 0.0
            targets = rewards + (self.gamma * next_q)

        loss = self.huber_loss(current_q, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1


def main():
    env = gym.make("CartPole-v1")
    agent = DQNAgentCartPole(env)
    total_steps = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            epsilon = max(
                EPSILON_END,
                EPSILON_START - total_steps / EPSILON_DECAY
            )
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward
            total_steps += 1
            done = terminated or truncated

            if total_steps >= WARMUP_STEPS:
                agent.train_step(BATCH_SIZE)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent._update_target_network()

            if done:
                break

        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Eps: {epsilon:.3f}"
        )


if __name__ == "__main__":
    main()
