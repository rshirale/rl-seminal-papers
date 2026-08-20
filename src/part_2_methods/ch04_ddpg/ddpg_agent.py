import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from .actor import Actor
    from .critic import Critic
    from .gaussian_noise import GaussianNoise
    from .replay_buffer import ReplayBuffer
else:  # pragma: no cover - direct script execution fallback.
    from actor import Actor
    from critic import Critic
    from gaussian_noise import GaussianNoise
    from replay_buffer import ReplayBuffer

SOFT = "soft"
HARD = "hard"


class DDPGAgent:
    """Deep Deterministic Policy Gradient, Algorithm 1 of Lillicrap et al. (2015).

    Composes the five pieces the chapter builds separately:

      * a deterministic actor mu(s | theta^mu) trained through the critic,
      * an action-value critic Q(s, a | theta^Q) trained by Bellman backup,
      * frozen target copies of both, soft-updated at rate tau,
      * Gaussian exploration noise,
      * a uniform experience replay buffer.

    The ``use_target_networks`` and ``target_update`` switches exist to
    reproduce the chapter's ablation: turn targets off and the online critic
    becomes its own label-generator; set ``target_update="hard"`` and the
    targets jump on a DQN-style schedule instead of drifting. Both default to
    the published configuration, so ordinary training is unaffected. This
    mirrors the ablation switches on Chapter 3's ``DQNAgent``.

    Args:
        state_dim: observation dimensionality.
        action_dim: action dimensionality.
        max_action: symmetric action bound; the actor's tanh is scaled by it.
        gamma: discount factor.
        tau: soft update rate. The paper tested 0.01 and found instability on
            several environments; 0.001 was stable across all of them.
        actor_lr, critic_lr: the paper's 1e-4 / 1e-3. The critic learns ten
            times faster on purpose -- the actor's gradient is only as good as
            the critic producing it, so the value estimate has to lead.
        batch_size: transitions per update.
        buffer_size: replay capacity.
        sigma: exploration noise scale.
        sigma_final, sigma_decay_steps: optional linear annealing of ``sigma``.
        critic_weight_decay: L2 penalty on the critic. The paper uses 1e-2;
            this defaults to 0.0 to match the chapter's listings, which use
            plain Adam.
        use_target_networks: ablation switch. False makes the online networks
            serve as their own targets.
        target_update: ``"soft"`` (Polyak, every step) or ``"hard"`` (wholesale
            copy every ``hard_update_freq`` steps, as in DQN).
        hard_update_freq: copy interval when ``target_update="hard"``.
        device: overrides automatic CUDA/CPU selection.
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
        sigma: float = 0.2,
        sigma_final: float | None = None,
        sigma_decay_steps: int = 100_000,
        critic_weight_decay: float = 0.0,
        use_target_networks: bool = True,
        target_update: str = SOFT,
        hard_update_freq: int = 1_000,
        device: torch.device | str | None = None,
    ):
        if target_update not in (SOFT, HARD):
            raise ValueError(
                f"target_update must be {SOFT!r} or {HARD!r}, got {target_update!r}"
            )

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.use_target_networks = use_target_networks
        self.target_update = target_update
        self.hard_update_freq = hard_update_freq
        self.train_steps = 0
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        if use_target_networks:
            # Built fresh and loaded rather than deep-copied, to match listing
            # 4.4 line for line.
            self.actor_target = Actor(
                state_dim, action_dim, max_action).to(self.device)
            self.critic_target = Critic(state_dim, action_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            for p in self.actor_target.parameters():
                p.requires_grad = False
            for p in self.critic_target.parameters():
                p.requires_grad = False
        else:
            # Ablation: the online networks bootstrap from themselves. Every
            # gradient step moves the target it is regressing toward.
            self.actor_target = self.actor
            self.critic_target = self.critic

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr,
            weight_decay=critic_weight_decay,
        )

        self.replay = ReplayBuffer(buffer_size)
        self.noise = GaussianNoise(
            action_dim, sigma=sigma,
            sigma_final=sigma_final, decay_steps=sigma_decay_steps,
        )

    def select_action(self, state: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        """Returns a clipped action, optionally with exploration noise.

        Pass ``explore=False`` to evaluate: the whole point of a deterministic
        policy is that its greedy behaviour is just a forward pass.
        """
        s_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(s_t).cpu().numpy()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def store(self, s, a, r, ns, done) -> None:
        """Records one transition.

        ``done`` must be the *termination* flag, not ``terminated or
        truncated``. See the note in ``train_pendulum.main``: zeroing the
        bootstrap on a time-limit cutoff teaches the agent that the world ends
        after 200 steps.
        """
        self.replay.push(s, a, r, ns, done)

    def train(self):
        """Runs one critic step, one actor step, and the target update.

        Returns ``(critic_loss, actor_loss)``, or ``(None, None)`` while the
        buffer holds less than one batch.
        """
        if len(self.replay) < self.batch_size:
            return None, None

        s, a, r, ns, d = [t.to(self.device)
                          for t in self.replay.sample(self.batch_size)]

        # --- Critic update: regress Q(s, a) onto the Bellman target ---
        with torch.no_grad():
            na = self.actor_target(ns)
            nq = self.critic_target(ns, na)
            y = r + self.gamma * (1.0 - d) * nq

        c_loss = F.mse_loss(self.critic(s, a), y)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        # --- Actor update: ascend Q(s, mu(s)) ---
        # The actor's loss backpropagates *through* the critic to reach the
        # actor's weights, so autograd would otherwise populate a full set of
        # critic gradients that nothing consumes -- only actor_opt.step() runs.
        # Freezing the critic for the duration skips building that half of the
        # graph. Behaviour is identical; the work is not.
        self._set_critic_requires_grad(False)
        a_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()
        self._set_critic_requires_grad(True)

        self.train_steps += 1
        self._update_targets()

        return c_loss.item(), a_loss.item()

    def reset_noise(self) -> None:
        self.noise.reset()

    def _set_critic_requires_grad(self, flag: bool) -> None:
        for p in self.critic.parameters():
            p.requires_grad = flag

    def _update_targets(self) -> None:
        if not self.use_target_networks:
            return
        if self.target_update == SOFT:
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
        elif self.train_steps % self.hard_update_freq == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

    def _soft_update(self, target: torch.nn.Module,
                     source: torch.nn.Module) -> None:
        """Polyak averaging: theta' <- tau * theta + (1 - tau) * theta'.

        Written inside ``torch.no_grad()`` and using ``copy_`` on the tensors
        themselves. Reaching through ``.data`` also works but bypasses
        autograd's bookkeeping rather than satisfying it, which is why it is
        discouraged in current PyTorch.
        """
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.copy_(self.tau * sp + (1.0 - self.tau) * tp)

    def save(self, path) -> None:
        """Writes actor, critic, and both optimizers to one checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "train_steps": self.train_steps,
        }, path)

    def load(self, path) -> None:
        """Restores a checkpoint and re-syncs the target networks."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.train_steps = ckpt.get("train_steps", 0)
        if self.use_target_networks:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
