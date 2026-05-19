import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Action-value network Q(s, a | θQ) from Lillicrap et al. (2015).

    The state enters at layer 1; the action enters at layer 2 after
    concatenation with the 400-unit state embedding. This matches the
    paper's architecture and gives the first layer time to build a useful
    state representation before conditioning on the action.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.l2(x))
        return self.l3(x)
