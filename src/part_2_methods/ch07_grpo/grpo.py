"""The GRPO objective -- listing 7.4, factored so each term can be tested.

GRPO alters exactly one thing about PPO: where the advantage comes from.
Everything else in this file -- the ratio, the clip, the pessimistic min -- is
chapter 5's, unchanged. The substitution is in :func:`group_advantages`, which
is fifteen lines and is the entire replacement for a value network.

Three things in here look like bugs to a reader coming from chapter 5, and
none of them are:

* **The loss is exactly zero on every step of the chapter's run.** With a
  single update per exploration stage -- the setting the DeepSeek team used --
  the behavior policy and the current policy are identical, so the ratio is
  exactly 1 and the surrogate collapses to the negative mean of the
  advantages, which are standardized within the group and therefore sum to
  zero by construction. The *gradient* is emphatically not zero: differentiate
  through the ratio and REINFORCE-with-baseline falls out. This is why
  ``train_json.py`` reports the gradient norm rather than the loss.

* **The KL term is not a mean log-ratio.** It is Schulman's estimator,
  ``e^u - u - 1``, which is zero when the policies agree and strictly positive
  otherwise. A naive mean log-ratio is signed, and when it goes negative the
  penalty *subtracts* from the loss -- rewarding exactly the drift it was
  added to prevent. The KL term is the only thing holding the policy near a
  model that still writes readable text, so getting it wrong is not a small
  error. :func:`kl_penalty` is tested for non-negativity for that reason.

* **The standard deviation has ``1e-4`` added to it.** That guards the group
  whose members all score identically, which would otherwise divide by zero.
  It is a real occurrence once a model becomes reliable on easy prompts -- the
  condition ``group_size.py`` quantifies and DAPO's dynamic sampling attacks
  directly.

Requires torch. The reward function, which is the part most readers will
replace, does not -- see ``rewards.py``.
"""

import torch

#: Added to the group's standard deviation before dividing. Small enough not
#: to bias a healthy group's advantages, large enough that a degenerate one
#: yields zeros rather than infinities. The value is the chapter's.
STD_EPS = 1e-4

#: PPO's clipping range, inherited unchanged. Inert in the single-update
#: setting -- the ratio is exactly 1 -- and load-bearing the moment a reader
#: takes more than one gradient step per group.
CLIP_EPS = 0.2

#: The KL coefficient. 0.04 is DeepSeekMath's own setting; open-source
#: reimplementations run anywhere from 0.001 to 0.04. Its effect is *not*
#: monotonic -- raising it does not reliably buy stability -- so a good value
#: has to be found empirically rather than reasoned to.
KL_BETA = 0.04


def group_advantages(rewards, group_size, std_eps=STD_EPS):
    """Standardizes rewards within each group of ``group_size``.

        A_i = (s_i - mu_G) / (sigma_G + eps)

    ``rewards`` is a flat tensor of ``K * group_size`` scores laid out group by
    group; the return has the same shape. Reshaping to ``(K, group_size)`` and
    taking both statistics along the group dimension is what makes the baseline
    *relative*: a candidate is good or bad only compared to the others sampled
    beside it for the same prompt, which is why no critic is needed to say what
    "good" would have been.

    Note ``std``'s default of Bessel's correction (``unbiased=True``), which is
    what ``Tensor.std`` does and what listing 7.4 writes. With G = 8 the
    difference against the biased estimator is a factor of about 1.07 on the
    advantage scale -- immaterial next to the learning rate, and not worth
    diverging from the listing for.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if rewards.shape[0] % group_size:
        raise ValueError(
            f"{rewards.shape[0]} rewards is not a whole number of groups of "
            f"{group_size}; every prompt must contribute the same G.")

    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    return ((grouped - mean) / (std + std_eps)).view(-1)


def kl_penalty(logp, ref_logp):
    """Schulman's unbiased KL estimator, per sequence.

        D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1 = e^u - u - 1

    Writing ``u`` for the log-ratio ``ref_logp - logp``. Convex in ``u`` with
    its minimum of exactly 0 at ``u = 0``, so it is never negative however far
    the policies have drifted -- which is the property a penalty term needs and
    the mean log-ratio does not have.
    """
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(logp, old_logp, ref_logp, rewards, group_size,
              clip_eps=CLIP_EPS, kl_beta=KL_BETA):
    """The GRPO objective of listing 7.4, negated into a loss to minimize.

    All four tensors are one-dimensional and aligned: element ``i`` is the
    same sampled completion in each. ``logp`` carries gradient; ``old_logp``
    and ``ref_logp`` must not.

    Returns a scalar. Read it as a diagnostic with care -- see this module's
    docstring on why it sits at zero in the single-update setting.
    """
    advantages = group_advantages(rewards, group_size)

    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    # The pessimistic branch, exactly as PPO takes it: whichever of the two
    # products is smaller, so an update that would move the ratio far in the
    # advantageous direction earns nothing extra for the excess.
    loss_pg = -torch.min(ratio * advantages, clipped * advantages).mean()

    return loss_pg + kl_beta * kl_penalty(logp, ref_logp).mean()
