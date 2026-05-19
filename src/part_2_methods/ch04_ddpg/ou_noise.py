import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck noise process for temporally correlated exploration.

    Discrete-time update:
        X_{t+1} = X_t + θ(μ - X_t) + σ ε_t,  ε_t ~ N(0, 1)

    Default parameters match Lillicrap et al. (2015):
        θ = 0.15  →  correlation timescale of ~7 steps
        σ = 0.2   →  excursions of roughly ±0.5 in action space
        μ = 0.0   →  zero long-run mean
    """

    def __init__(self, size: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset to the long-run mean at the start of each episode."""
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        x = self.state
        dx = (self.theta * (self.mu - x)
              + self.sigma * np.random.randn(len(x)))
        self.state = x + dx
        return self.state.copy()
