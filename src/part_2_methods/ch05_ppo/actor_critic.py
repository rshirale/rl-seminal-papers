import torch
import torch.nn as nn
from torch.distributions import Normal


class SquashedNormal:
    """Normal policy transformed to the environment's bounded action range."""

    def __init__(self, mean, std, max_action):
        self.base = Normal(mean, std)
        self.max_action = max_action

    def sample(self):
        """Sample an action in ``[-max_action, max_action]``."""
        return torch.tanh(self.base.sample()) * self.max_action

    def log_prob(self, action):
        """Return the log probability of a scaled, squashed action."""
        scaled = torch.clamp(action / self.max_action, -1 + 1e-6, 1 - 1e-6)
        pre_tanh = torch.atanh(scaled)
        correction = torch.log(1 - scaled.square() + 1e-6)
        action_scale_correction = torch.log(
            torch.as_tensor(self.max_action, device=action.device)
        )
        return self.base.log_prob(pre_tanh) - correction - action_scale_correction

    def entropy(self):
        """Return the base entropy as a stable entropy approximation."""
        return self.base.entropy()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(
            torch.zeros(1, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mu)
        return SquashedNormal(mu, std, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)
