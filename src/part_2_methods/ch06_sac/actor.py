import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """
    Stochastic squashed-Gaussian policy from Haarnoja et al. (2018),
    "Soft Actor-Critic ..." (Appendix C).

    A shared two-layer backbone feeds separate mean and log-std heads. The
    mathematical requirement behind a *stochastic* actor is the maximum
    entropy objective: a deterministic policy is a Dirac delta with
    negative-infinite entropy, so SAC cannot use one.

    Actions are drawn with the reparameterization trick, squashed with tanh
    to respect the action bounds, and scaled by max_action. The returned
    log-probability includes the tanh change-of-variables correction *and*
    the action-scale term, so the entropy estimate that drives training is
    unbiased (as in Stable-Baselines3 and CleanRL).
    """

    def __init__(self, state_dim: int, action_dim: int,
                 max_action: float = 1.0, hidden: int = 256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        """
        Return (action, log_prob).

        With deterministic=True the mean action (tanh(μ)·max_action) is
        returned for evaluation and log_prob is None — the paper notes the
        mean action typically yields slightly higher returns at eval time.
        Otherwise an action is sampled via the reparameterization trick and
        its corrected log-probability is returned.
        """
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean) * self.max_action, None

        dist = Normal(mean, std)
        u = dist.rsample()                 # reparameterized: u = μ + σ·ε
        y = torch.tanh(u)                  # squash to (-1, 1)
        action = y * self.max_action

        # log π(a|s) = log μ(u|s) − Σ log(max_action·(1 − tanh²(u)))
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(self.max_action * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob
