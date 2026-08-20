import torch
import torch.nn as nn


class Actor(nn.Module):
    """Deterministic policy network mu(s | theta^mu) from Lillicrap et al. (2015).

    Two hidden layers of 400 and 300 units with ReLU activations, then a tanh
    output scaled by ``max_action`` so the policy respects the environment's
    action bounds without hard clipping. This mirrors listing 4.2 in the
    chapter.

    A note on the layer widths: 400/300 is the paper's specification, kept here
    for fidelity. Modern reference implementations (CleanRL, Stable-Baselines3)
    have standardized on 256/256, which trains slightly faster and performs
    comparably on the standard benchmarks. Neither choice is load-bearing for
    the ideas in this chapter.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Applies the paper's uniform initialization to the output layer.

        Section 7 of the paper initializes the final layer from
        U(-3e-3, 3e-3), several orders of magnitude smaller than the default,
        so the pre-tanh activations start near zero. Without it a freshly
        initialized actor can emit saturated actions at +/-max_action, where
        tanh's gradient is nearly flat -- the policy then barely moves for the
        first few thousand updates.

        The hidden layers need no override: the paper initializes them from
        U(-1/sqrt(f), 1/sqrt(f)) with f the layer's fan-in, which is already
        what ``nn.Linear`` does by default.
        """
        nn.init.uniform_(self.l3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.l3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(state))
        x = torch.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))
