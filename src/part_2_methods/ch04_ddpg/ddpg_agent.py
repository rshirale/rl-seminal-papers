import copy

import numpy as np
import torch
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic
from .ou_noise import OUNoise
from .replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent from Lillicrap et al. (2015),
    "Continuous Control with Deep Reinforcement Learning."

    Implements Algorithm 1:
      - Deterministic actor  μ(s | θᵘ)  trained via policy gradient
      - Action-value critic  Q(s,a | θQ)  trained via Bellman backup
      - Soft target updates (τ = 0.001) for both actor and critic targets
      - Ornstein-Uhlenbeck exploration noise
      - Experience replay buffer (capacity 1,000,000)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 0.001,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 1_000_000,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        self.replay = ReplayBuffer(buffer_size)
        self.noise = OUNoise(action_dim)

    def select_action(self, state: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        """Return a clipped action, optionally with OU exploration noise."""
        s_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor(s_t).cpu().numpy()
        if explore:
            action += self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def store(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)

    def train(self):
        """
        Sample one mini-batch and run one critic + actor + soft-update step.
        Returns (critic_loss, actor_loss) or (None, None) if buffer not ready.
        """
        if len(self.replay) < self.batch_size:
            return None, None

        s, a, r, ns, d = [t.to(self.device)
                          for t in self.replay.sample(self.batch_size)]

        with torch.no_grad():
            na = self.actor_target(ns)
            nq = self.critic_target(ns, na)
            y  = r + self.gamma * (1 - d) * nq

        c_loss = F.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        a_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return c_loss.item(), a_loss.item()

    def reset_noise(self):
        self.noise.reset()

    def _soft_update(self, target: torch.nn.Module,
                     source: torch.nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
