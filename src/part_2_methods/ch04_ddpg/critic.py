import torch
import torch.nn as nn


class Critic(nn.Module):
    """Action-value network Q(s, a | theta^Q) from Lillicrap et al. (2015).

    The state enters at layer 1; the action is concatenated with the resulting
    400-unit state embedding before layer 2. This is the paper's architecture
    and the reason for it is in the gradient: the actor is trained through
    grad_a Q, so the action needs to pass through a layer that has already seen
    state-derived features rather than being appended to the raw observation.

    Mirrors listing 4.3 in the chapter.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Applies the paper's uniform initialization to the output layer.

        Same U(-3e-3, 3e-3) treatment as the actor, for a different reason: it
        starts every Q-value near zero. A critic whose initial outputs are
        large hands the actor a strong, meaningless gradient on the very first
        update, before a single Bellman backup has been applied.
        """
        nn.init.uniform_(self.l3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.l3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.l2(x))
        return self.l3(x)
