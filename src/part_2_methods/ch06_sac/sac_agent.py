import copy

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from .actor import Actor
    from .critic import Critic
    from .replay_buffer import ReplayBuffer
else:  # pragma: no cover - only used by direct script execution.
    from actor import Actor
    from critic import Critic
    from replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic (SAC-v2) agent from Haarnoja et al. (2018),
    "Soft Actor-Critic Algorithms and Applications."

    Implements Algorithm 1:
      - Stochastic squashed-Gaussian actor  π_φ(a | s)
      - Twin soft critics  Q_θ1, Q_θ2  with min-clipped Bellman targets
      - Target critics updated by soft EMA (τ = 0.005); no target actor,
        since the stochastic policy already provides principled exploration
      - Automatic temperature tuning: α is the Lagrange multiplier of an
        entropy constraint, learned in log-space toward target_entropy
      - One gradient update per environment step (off-policy replay)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        target_entropy: float = None,
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
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Automatic temperature: optimize log α so that α = exp(log α) > 0.
        # The default target entropy H̄ = −dim(A) is the SAC-v2 heuristic.
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay = ReplayBuffer(buffer_size)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Sample an action; use the mean action when deterministic=True."""
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(s_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def store(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)

    def train(self):
        """
        Run one critic + actor + temperature update and a soft target sync.
        Returns (critic_loss, actor_loss, alpha), or (None, None, None) if
        the buffer does not yet hold a full mini-batch.
        """
        if len(self.replay) < self.batch_size:
            return None, None, None

        s, a, r, ns, d = [t.to(self.device)
                          for t in self.replay.sample(self.batch_size)]
        alpha = self.log_alpha.exp().detach()

        # --- Critic update: soft Bellman target with min-clipped twin Q ---
        with torch.no_grad():
            na, log_pi_next = self.actor(ns)
            q1_t, q2_t = self.critic_target(ns, na)
            soft_v = torch.min(q1_t, q2_t) - alpha * log_pi_next
            y = r + self.gamma * (1 - d) * soft_v

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update: maximize min-Q minus the entropy penalty ---
        a_new, log_pi = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_new)
        actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Temperature update: drive entropy toward the target ---
        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.critic_target, self.critic)

        return (
            critic_loss.item(),
            actor_loss.item(),
            self.log_alpha.exp().item(),
        )

    def _soft_update(self, target: torch.nn.Module,
                     source: torch.nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
