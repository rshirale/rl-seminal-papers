import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Deterministic policy network μ(s | θᵘ) from Lillicrap et al. (2015).

    Two hidden layers of 400 and 300 units with ReLU activations.
    Output is scaled by max_action so the policy respects the
    environment's action bounds without hard clipping.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))
