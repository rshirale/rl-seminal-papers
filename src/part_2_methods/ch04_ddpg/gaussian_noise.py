import numpy as np


class GaussianNoise:
    """I.i.d. Gaussian exploration noise for a deterministic policy.

    A deterministic actor returns the same action for the same state, so
    exploration has to be injected from outside the network::

        a_explore = mu(s | theta^mu) + N,   N ~ Normal(0, sigma^2)

    The original paper used an Ornstein-Uhlenbeck process, whose samples are
    temporally correlated. Practice since has settled on plain Gaussian noise:
    it drops two hyperparameters and performs comparably on the standard
    continuous-control benchmarks, which is why TD3, Stable-Baselines3, and
    OpenAI Spinning Up all default to it. Note what is and is not being claimed
    about i.i.d. draws -- successive samples are *independent*, which means the
    sign of one says nothing about the sign of the next. They do not alternate.

    ``sigma=0.2`` matches the scale used in the paper. Annealing is off by
    default; pass ``sigma_final`` and ``decay_steps`` to taper exploration
    linearly as the policy matures, which is common in long production runs but
    is not needed to converge on Pendulum-v1.

    Mirrors listing 4.6 in the chapter (with annealing and ``rng`` added).
    """

    def __init__(self, size: int, sigma: float = 0.2,
                 sigma_final: float | None = None,
                 decay_steps: int = 100_000,
                 rng: np.random.Generator | None = None):
        if sigma_final is not None and sigma_final > sigma:
            raise ValueError(
                f"sigma_final ({sigma_final}) must not exceed sigma ({sigma}); "
                "annealing lowers the noise scale over training."
            )
        self.size = size
        self.sigma_start = sigma
        self.sigma_final = sigma if sigma_final is None else sigma_final
        self.decay_steps = max(1, decay_steps)
        self.steps = 0
        # Defaults to the legacy global RNG so ``seeding.set_seed`` covers it;
        # pass a Generator to isolate exploration from every other np.random
        # consumer.
        self._rng = rng

    @property
    def sigma(self) -> float:
        """Current noise scale, linearly interpolated toward ``sigma_final``."""
        fraction = min(1.0, self.steps / self.decay_steps)
        return self.sigma_start + fraction * (self.sigma_final - self.sigma_start)

    def reset(self) -> None:
        """No-op, kept so the training loop can treat noise processes alike.

        I.i.d. noise carries no state to clear between episodes. A stateful
        process such as Ornstein-Uhlenbeck would zero its running value here,
        and the loop should not have to know which kind it holds.

        Note that this deliberately does not reset the annealing counter --
        exploration decays over the whole run, not within an episode.
        """

    def sample(self) -> np.ndarray:
        """Draws one noise vector and advances the annealing schedule."""
        sigma = self.sigma
        self.steps += 1
        if self._rng is None:
            return np.random.normal(0.0, sigma, size=self.size)
        return self._rng.normal(0.0, sigma, size=self.size)
