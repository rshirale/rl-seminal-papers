import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Twin soft Q-networks from Haarnoja et al. (2018), "Soft Actor-Critic."

    Two independently initialized Q-networks Q1 and Q2 map (state, action)
    pairs to scalar soft Q-values. SAC uses the minimum of the two as the
    Bellman target — the clipped double-Q trick of Fujimoto et al. (2018) —
    to counteract the Q-value overestimation that would otherwise let the
    actor exploit inflated estimates.

    forward() returns both Q-values so the caller can apply min(Q1, Q2)
    without crossing the class boundary twice.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = mlp()
        self.q2 = mlp()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


if __name__ == "__main__":
    torch.manual_seed(42)
    state_dim = 3
    action_dim = 1
    critic = Critic(state_dim, action_dim)

    s = torch.randn(1, state_dim)
    a = torch.tensor([[1.0]])
    q1, q2 = critic(s, a)

    print("--- Critic Smoke Test ---")
    print("Q1 shape:", q1.shape, "| Q2 shape:", q2.shape)
    print(f"Q1: {q1.item():.4f} | Q2: {q2.item():.4f} | Min Q: {torch.min(q1, q2).item():.4f}")

