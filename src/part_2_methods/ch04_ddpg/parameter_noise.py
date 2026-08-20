"""Parameter-space exploration noise (Plappert et al., ICLR 2018).

Action-space noise perturbs what the policy *does*; parameter-space noise
perturbs what the policy *is*. Instead of adding a fresh sample to the actor's
output at every step, you perturb a copy of the actor's weights once per
episode and roll out that perturbed policy::

    theta_hat = theta + Normal(0, sigma^2 I),   a_t = mu(s_t | theta_hat)

The difference that matters is consistency. Action noise gives an agent that
twitches: two visits to the same state produce two different actions, so an
exploratory choice is never followed through. A weight perturbation is a
*different policy* -- deterministic, state-dependent, and coherent for the
whole episode. That is what gets a manipulator to try a genuinely different
grasp rather than jittering around the one it already knows.

The catch is that sigma has no interpretable scale: the same perturbation
magnitude changes behaviour wildly depending on the network's current weights,
and how much it changes them drifts as training proceeds. Plappert et al. solve
this with an adaptive rule -- measure how far the perturbed policy's actions
actually moved, and scale sigma up or down to hold that distance at a target
you *can* reason about, namely the action-space sigma you would otherwise have
used.

Two engineering constraints, both from the paper and the literature the
chapter's research notes cite:

  * Perturb the actor only. Applying learnable parameter noise to the critic
    has been reported to destabilize training.
  * Perturb a copy. The perturbed weights drive the rollout; the clean weights
    are what the optimizer updates.

This module is optional -- ``DDPGAgent`` uses ``GaussianNoise`` by default, and
Pendulum-v1 does not need anything stronger. It is here so the chapter's
parameter-noise discussion points at code a reader can run, and for the harder
exploration problems where action noise genuinely stalls.
"""

import copy

import numpy as np
import torch
import torch.nn as nn


class AdaptiveParameterNoise:
    """Episode-level weight perturbation with an adaptive noise scale.

    Usage inside a training loop::

        param_noise = AdaptiveParameterNoise()
        for episode in ...:
            perturbed = param_noise.perturb(agent.actor)   # once per episode
            ...                                            # roll out `perturbed`
            param_noise.adapt(
                action_distance(agent.actor, perturbed, states)
            )

    Args:
        sigma: initial standard deviation of the weight perturbation.
        target_action_stddev: the action-space noise scale this perturbation
            should be equivalent to. Defaults to 0.2, matching
            ``GaussianNoise``'s default, so switching between the two keeps
            exploration at a comparable magnitude.
        adaptation_coefficient: multiplicative step for sigma, applied once per
            ``adapt`` call. The paper's value is 1.01 -- deliberately slow, so
            the scale tracks the trend rather than the last episode's luck.
    """

    def __init__(self, sigma: float = 0.05,
                 target_action_stddev: float = 0.2,
                 adaptation_coefficient: float = 1.01):
        if adaptation_coefficient <= 1.0:
            raise ValueError(
                "adaptation_coefficient must exceed 1.0; it is applied as a "
                "multiplier in one direction and a divisor in the other."
            )
        self.sigma = sigma
        self.target_action_stddev = target_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

    def perturb(self, actor: nn.Module) -> nn.Module:
        """Returns a perturbed copy of ``actor``; the original is untouched.

        The copy is returned in eval mode with gradients disabled: it exists to
        act, never to learn. Gradient updates belong to the clean actor the
        optimizer holds.
        """
        perturbed = copy.deepcopy(actor)
        with torch.no_grad():
            for param in perturbed.parameters():
                param.add_(torch.randn_like(param) * self.sigma)
        for param in perturbed.parameters():
            param.requires_grad = False
        perturbed.eval()
        return perturbed

    def adapt(self, measured_distance: float) -> float:
        """Scales sigma toward the target action distance. Returns the new sigma.

        Under-perturbing (the perturbed policy acts almost like the clean one)
        raises sigma; over-perturbing lowers it. Because the relationship
        between weight-space and action-space distance changes as the network
        trains, this has to run continuously, not once at setup.
        """
        if measured_distance < self.target_action_stddev:
            self.sigma *= self.adaptation_coefficient
        else:
            self.sigma /= self.adaptation_coefficient
        return self.sigma


def action_distance(actor: nn.Module, perturbed_actor: nn.Module,
                    states: torch.Tensor) -> float:
    """Root-mean-square action difference between two actors on the same states.

    This is the distance measure Plappert et al. adapt against. Feed it a batch
    of recently visited states -- the replay buffer's own sample is the natural
    source -- because the quantity of interest is how differently the perturbed
    policy behaves *where the agent actually is*, not on the state space at
    large.
    """
    with torch.no_grad():
        clean = actor(states)
        noisy = perturbed_actor(states)
        return float(torch.sqrt(torch.mean((clean - noisy) ** 2)))
