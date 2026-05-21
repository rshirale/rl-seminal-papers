import numpy as np

class GaussianNoise:
    """
    Standard i.i.d. Gaussian noise for action exploration.
    Matches the modern default in deep RL (e.g., TD3).
    """

    def __init__(self, size: int, sigma: float = 0.2):
        self.size = size
        self.sigma = sigma

    def reset(self):
        """Included for API consistency with stateful noise processes."""
        pass

    def sample(self) -> np.ndarray:
        """Sample from a standard normal distribution scaled by sigma."""
        return np.random.normal(0, self.sigma, size=self.size)
