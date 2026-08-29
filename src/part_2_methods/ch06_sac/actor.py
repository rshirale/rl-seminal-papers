import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Stochastic squashed-Gaussian policy, Appendix C of Haarnoja et al. (2018a).

    A shared two-layer backbone feeds separate mean and log-standard-deviation
    heads. The requirement that the actor be *stochastic* is mathematical, not
    a convenience: a deterministic policy is a Dirac delta, its entropy is
    negative infinity, and the entropy term in the maximum entropy objective is
    then undefined. SAC cannot use a deterministic actor.

    Note that ``log_std`` is a head, so it depends on the state. Chapter 5's
    PPO actor carried a state-independent learned scalar instead. A
    state-dependent spread lets SAC be exploratory in unfamiliar regions of the
    state space and precise in well-understood ones.

    Sampling uses the reparameterization trick -- ``rsample`` draws
    ``u = mu + sigma * eps`` with ``eps ~ N(0, I)`` treated as a constant, so
    the path from the action back to the weights stays differentiable -- and
    ``tanh`` bounds the result without the hard clipping that would kill the
    gradient at the boundary.

    The returned log-probability carries the tanh change-of-variables
    correction. Omitting it is the most common SAC implementation bug: the
    entropy signal that drives both the actor loss and the temperature update
    is then biased, and the policy may fail to explore or diverge outright.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256,
                 max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        """Returns ``(action, log_prob)``.

        With ``deterministic=True`` the mean action ``tanh(mu) * max_action``
        is returned and ``log_prob`` is ``None``. That is the evaluation path:
        the policy is trained against the entropy-augmented objective but
        scored against the plain reward, so the mean action typically returns
        slightly more than a sample does (Haarnoja et al., 2018a).
        """
        x = self.net(state)
        mean = self.mean_head(x)
        # Clamped for numerical stability: an unclamped head can drive sigma to
        # zero (log-probabilities to infinity) or explode it.
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mean) * self.max_action, None

        dist = Normal(mean, std)
        u = dist.rsample()                 # reparameterized: u = mu + sigma*eps
        action = torch.tanh(u)             # squashed to (-1, 1)

        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        # log pi(a|s) = log p(u|s) - sum_i log(1 - tanh^2(u_i)).
        #
        # The second term is the log absolute Jacobian determinant of tanh.
        # `2 * (log 2 - u - softplus(-2u))` is an algebraically identical but
        # numerically stable form of `log(1 - tanh^2(u))`, which underflows to
        # log(0) once |u| is large; it is the form the authors' own
        # implementation uses. `device=u.device` keeps the constant on the same
        # hardware as the network, so this runs unchanged on CPU or GPU.
        correction = (
            2 * (torch.log(torch.tensor(2.0, device=u.device))
                 - u - F.softplus(-2 * u))
        ).sum(-1, keepdim=True)
        log_prob = log_prob - correction

        return action * self.max_action, log_prob
