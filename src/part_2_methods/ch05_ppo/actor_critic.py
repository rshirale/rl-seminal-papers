import torch
import torch.nn as nn
from torch.distributions import Normal


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
        # The state-independent log_std parameter is a modern best practice
        # not explicitly detailed in the original paper.
        self.log_std = nn.Parameter(
            torch.zeros(1, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        x = self.net(state)
        # Squash the mean into the environment's action bounds. Samples
        # themselves are unbounded; the environment clips them.
        mu = torch.tanh(self.mu(x)) * self.max_action
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # A separate backbone from the actor's: the two networks share no
        # parameters.
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)
