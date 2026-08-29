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
    """Soft Actor-Critic (SAC-v2), Algorithm 1 of Haarnoja et al. (2018b).

    Composes the four pieces the chapter builds separately:

      * a stochastic squashed-Gaussian actor ``pi_phi(a | s)``,
      * twin soft critics ``Q_theta1``, ``Q_theta2``, whose minimum forms the
        Bellman target,
      * frozen target copies of both critics, soft-updated at rate ``tau``
        (there is no target *actor*: the stochastic policy already smooths the
        target, which is what TD3 has to simulate with injected noise),
      * a uniform replay buffer.

    ``train()`` performs the three gradient steps of the algorithm's inner loop
    -- critic, actor, temperature -- followed by the soft target update.

    The temperature is the Lagrange multiplier of an entropy constraint and is
    learned in log-space so ``alpha = exp(log_alpha)`` stays strictly positive.
    ``auto_alpha=False`` and ``init_alpha`` turn that off, which is what the
    chapter's exercises 1 and 2 ablate: ``init_alpha=0.0`` with
    ``auto_alpha=False`` is SAC with the entropy bonus removed entirely, and a
    fixed non-zero ``init_alpha`` is SAC-v1 with a hand-set temperature.

    ``reward_scale`` multiplies the reward inside the Bellman target. It is the
    handle for exercise 3: reward scale acts as an implicit inverse
    temperature, so scaling rewards by 10 is equivalent to dividing alpha by
    10, and the point of the exercise is that the auto-tuner absorbs that while
    a fixed temperature does not.

    Args:
        state_dim: observation dimensionality.
        action_dim: action dimensionality.
        max_action: symmetric action bound; the actor's tanh is scaled by it.
        gamma: discount factor.
        tau: soft target update rate; SAC-v2's 0.005, applied every step.
        actor_lr, critic_lr, alpha_lr: all 3e-4 in the paper, and usually kept
            identical to each other. Unlike DDPG, the critic does not lead.
        batch_size: transitions per gradient update.
        buffer_size: replay capacity.
        target_entropy: the entropy floor H-bar. Defaults to the SAC-v2
            heuristic -dim(A), which is -1 on Pendulum-v1.
        auto_alpha: learn the temperature by dual gradient descent (SAC-v2) or
            hold it fixed at ``init_alpha`` (SAC-v1, and the exercises).
        init_alpha: starting temperature; the value alpha holds at when
            ``auto_alpha=False``. 1.0 is what the chapter's figure 6.6 starts
            from.
        reward_scale: multiplies the reward in the Bellman target.
        device: overrides automatic CUDA/CPU selection.
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
        target_entropy: float | None = None,
        auto_alpha: bool = True,
        init_alpha: float = 1.0,
        reward_scale: float = 1.0,
        device: torch.device | str | None = None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.auto_alpha = auto_alpha
        self.reward_scale = reward_scale
        self.device = torch.device(device) if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim,
                           max_action=max_action).to(self.device)
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

        # Optimizing log_alpha rather than alpha is what keeps the temperature
        # strictly positive: exp() of any real is positive, so the optimizer
        # can never step it negative and no clamp is needed.
        if auto_alpha and init_alpha <= 0.0:
            raise ValueError(
                "init_alpha must be positive when auto_alpha=True: alpha is "
                "optimized as log_alpha, and log(0) has no finite gradient. "
                "Pass auto_alpha=False to run with the entropy bonus off."
            )
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = target_entropy
        self.log_alpha = torch.tensor(
            [np.log(init_alpha)] if init_alpha > 0 else [-np.inf],
            dtype=torch.float32, device=self.device,
            requires_grad=auto_alpha,
        )
        # A third optimizer, over a single scalar. Its learning rate is
        # independent of the actor's and the critic's.
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr) \
            if auto_alpha else None

        self.replay = ReplayBuffer(buffer_size)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        """Samples an action; returns the mean action when ``deterministic``.

        SAC needs no exploration flag and no noise process: the policy is a
        distribution, so sampling from it *is* the exploration. ``deterministic``
        exists for evaluation only.
        """
        s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(s_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def store(self, s, a, r, ns, done) -> None:
        self.replay.push(s, a, r, ns, done)

    def train(self):
        """One critic + actor + temperature update, then a soft target sync.

        Returns ``(critic_loss, actor_loss, alpha)``, or ``(None, None, None)``
        if the buffer does not yet hold a full minibatch.
        """
        if len(self.replay) < self.batch_size:
            return None, None, None

        s, a, r, ns, d = [t.to(self.device)
                          for t in self.replay.sample(self.batch_size)]
        alpha = self.log_alpha.exp().detach()

        # --- Critic update -------------------------------------------------
        # Both expectations in the soft Bellman target become sampling here:
        # the outer one over the environment is the minibatch drawn from the
        # buffer, the inner one over the policy is the actor's sample at ns.
        with torch.no_grad():
            na, log_pi_next = self.actor(ns)
            q1_t, q2_t = self.critic_target(ns, na)
            # Pessimistic: min over the two frozen target critics. For the
            # target to be inflated, both would have to overestimate the same
            # state-action pair at once.
            soft_v = torch.min(q1_t, q2_t) - alpha * log_pi_next
            y = self.reward_scale * r + self.gamma * (1 - d) * soft_v

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update --------------------------------------------------
        # Two competing terms: -min(Q1, Q2) pulls toward actions the critics
        # rate highly, alpha * log_pi punishes certainty. As the policy
        # sharpens its density grows without bound and log_pi -> +inf, so the
        # term being minimized blows up. That is what prevents collapse.
        a_new, log_pi = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_new)
        actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Temperature update --------------------------------------------
        # Dual gradient descent on J(alpha). Entropy below the floor makes the
        # gradient positive and alpha rises, strengthening the penalty; above
        # it, alpha falls. The mechanism is bidirectional.
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item(), self.alpha

    def _soft_update(self, target: torch.nn.Module,
                     source: torch.nn.Module) -> None:
        """Polyak averaging, every step: theta_bar <- tau*theta + (1-tau)*theta_bar."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def save(self, path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
