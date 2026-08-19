import argparse
import random

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

if __package__:
    from .dqn_network import SimpleDQN
    from .replay_buffer import ReplayBuffer
    from .seeding import seed_env, set_seed
else:  # pragma: no cover - direct script execution fallback.
    from dqn_network import SimpleDQN
    from replay_buffer import ReplayBuffer
    from seeding import seed_env, set_seed

BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10_000
TARGET_UPDATE_FREQ = 500
WARMUP_STEPS = 1_000
NUM_EPISODES = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Lightweight DQN agent using SimpleDQN for CartPole-v1.

    ``use_replay`` and ``use_target_network`` exist to reproduce the ablation
    in Mnih et al. (2015): switch either off and the corresponding failure mode
    from the chapter reappears. Both default to on, so ordinary training is
    unchanged.
    """
    def __init__(self, env, learning_rate=1e-4, gamma=GAMMA,
                 use_replay=True, use_target_network=True):
        self.env = env
        self.num_actions = env.action_space.n
        self.gamma = gamma
        self.use_replay = use_replay
        self.use_target_network = use_target_network
        self.memory = ReplayBuffer(
            capacity=100000,
            state_shape=env.observation_space.shape,
        )

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
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return self.online_net(state_t).argmax().item()

    def _update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return
        # Without replay the agent learns from the transitions it just
        # generated - consecutive, highly correlated, and biased toward
        # wherever the agent currently is. That is the failure replay fixes.
        draw = self.memory.sample if self.use_replay else self.memory.sample_recent
        states, actions, rewards, next_states, dones = draw(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32).to(device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(device)
        actions = torch.as_tensor(actions, dtype=torch.long).to(device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.as_tensor(dones, dtype=torch.bool).to(device)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            # Without a target network the TD target is computed from the same
            # weights being updated: every step moves the goal it is chasing.
            bootstrap = self.target_net if self.use_target_network else self.online_net
            next_q = bootstrap(next_states).max(1)[0]
            next_q[dones] = 0.0
            targets = rewards + (self.gamma * next_q)

        loss = self.huber_loss(current_q, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps_done += 1


def main(seed: int | None = None, episodes: int = NUM_EPISODES,
         use_replay: bool = True, use_target_network: bool = True,
         verbose: bool = True):
    """Trains on CartPole-v1 and returns the per-episode rewards.

    The two ablation switches are passed straight through to the agent, so the
    four rows of the chapter's ablation table are four calls to this function.
    """
    env = gym.make("CartPole-v1")

    # Seed before building the agent so weight initialization is covered too.
    if seed is not None:
        set_seed(seed)
        seed_env(env, seed)

    agent = DQNAgent(
        env, use_replay=use_replay, use_target_network=use_target_network
    )
    total_steps = 0
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            if total_steps < WARMUP_STEPS:
                epsilon = EPSILON_START
            else:
                epsilon = max(
                    EPSILON_END,
                    EPSILON_START - (total_steps - WARMUP_STEPS) / EPSILON_DECAY
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

                if (agent.use_target_network
                        and total_steps % TARGET_UPDATE_FREQ == 0):
                    agent._update_target_network()

            if done:
                break

        rewards.append(episode_reward)
        if verbose:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} | "
                f"Eps: {epsilon:.3f}"
            )

    return rewards


if __name__ == "__main__":
    # Parsed here rather than inside main() so importing this module (as the
    # tests do) never touches sys.argv.
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix every RNG for a reproducible run. Omitted by default, which "
             "keeps the original non-deterministic behaviour.",
    )
    parser.add_argument(
        "--episodes", type=int, default=NUM_EPISODES,
        help="Episodes to train for.",
    )
    parser.add_argument(
        "--no-replay", action="store_true",
        help="Ablation: learn from the most recent transitions instead of a "
             "random minibatch. Reproduces the 'no replay buffer' row of the "
             "chapter's ablation table.",
    )
    parser.add_argument(
        "--no-target-network", action="store_true",
        help="Ablation: bootstrap from the online network, so the TD target "
             "moves with every update. Reproduces the 'no target network' row.",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        episodes=args.episodes,
        use_replay=not args.no_replay,
        use_target_network=not args.no_target_network,
    )
