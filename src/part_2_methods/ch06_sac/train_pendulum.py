"""Train SAC-v2 on Pendulum-v1 (Haarnoja et al., 2018b, Algorithm 1).

Usage:
    python -m src.part_2_methods.ch06_sac.train_pendulum
    python -m src.part_2_methods.ch06_sac.train_pendulum --seed 42 --steps 50000

    # ...or directly, from this directory
    python train_pendulum.py --seed 42

Pendulum-v1 is a three-dimensional state (cos theta, sin theta, theta-dot) and
one continuous action, the torque at the pivot, bounded to [-2, 2]. A random
policy scores about -1200 per episode; a converged SAC agent has a median
episode return near -125.

The loop alternates one environment step with one gradient update -- the
one-to-one ratio of Appendix D of the paper. The first 10,000 steps are uniform
random actions, filling the buffer with diverse experience before any gradient
is taken, so the first updates do not train on a trivially unrepresentative
sample. That warmup is a fifth of the default budget, and it is why the first
50 episodes of the transcript sit flat at random-policy returns.

Seed 42 reproduces the run the chapter's "Expected training output" section and
figure 6.6 are drawn from: flat at the random-policy baseline with alpha pinned
at 1.000 for the first 50 episodes, a sharp climb once updates begin, and a
final 50-episode median around -126 with alpha near 0.04. Expect the alpha
trajectory to land within a few thousandths of the printed one and individual
episode returns to differ -- see the README on what exact reproduction does and
does not cover.
"""

import argparse
import statistics
from collections import namedtuple

import gymnasium as gym
import torch

if __package__:
    from .actor import LOG_STD_MAX, LOG_STD_MIN
    from .sac_agent import SACAgent
    from .seeding import seed_env, set_seed
else:  # pragma: no cover - only used by direct script execution.
    from actor import LOG_STD_MAX, LOG_STD_MIN
    from sac_agent import SACAgent
    from seeding import seed_env, set_seed

ENV_ID = "Pendulum-v1"
SEED = 42
TOTAL_STEPS = 50_000
WARMUP_STEPS = 10_000   # uniform random actions before the first gradient step
STEPS_PER_EPISODE = 200  # Pendulum-v1's time limit; it never terminates early
PRINT_EVERY = 5          # episodes

#: Episodes averaged for the reported final score. 50 episodes is 10,000
#: steps -- long enough that Pendulum's random start angle averages out.
SCORE_WINDOW = 50

#: What a uniform random policy scores, for context in every table here.
RANDOM_POLICY_BASELINE = -1200.0

#: States drawn from the replay buffer to characterize the trained policy.
POLICY_PROBE_STATES = 512

#: What one run reports back.
#:
#: ``returns`` alone is not enough to read this chapter's ablations. On
#: Pendulum-v1 a near-deterministic policy and a broadly stochastic one earn
#: the same return, so the return column cannot distinguish the variants the
#: exercises ask you to compare -- but ``sigma`` and ``entropy`` separate them
#: by two orders of magnitude. Kept in the shape of chapter 5's ``RunResult``,
#: which carries its own two diagnostics for the same reason.
RunResult = namedtuple("RunResult", "returns sigma entropy alpha")


def measure_policy(agent, n_states=POLICY_PROBE_STATES):
    """Characterizes a trained policy: mean sigma, and mean entropy estimate.

    Probed on states sampled from the agent's own replay buffer rather than
    from ``observation_space.sample()``. The observation space contains states
    the pendulum never actually visits, and a policy is only meaningfully
    described on the distribution it was trained on.

    ``entropy`` is the single-sample estimate H = E[-log pi(a|s)], which is
    what the temperature is regulating: with the SAC-v2 heuristic the target
    is H-bar = -dim(A), so a healthy learned-alpha run reports a number near
    -1 on Pendulum. Returns ``(nan, nan)`` if the buffer is too small to probe,
    which happens only on runs shorter than one minibatch.
    """
    if len(agent.replay) < 2:
        return float("nan"), float("nan")

    n = min(n_states, len(agent.replay))
    states, _, _, _, _ = agent.replay.sample(n)
    states = states.to(agent.device)
    with torch.no_grad():
        x = agent.actor.net(states)
        log_std = agent.actor.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        _, log_prob = agent.actor(states)
    return float(log_std.exp().mean()), float(-log_prob.mean())


def main(seed: int | None = None, total_steps: int = TOTAL_STEPS,
         warmup_steps: int = WARMUP_STEPS, gamma: float = 0.99,
         tau: float = 0.005, actor_lr: float = 3e-4, critic_lr: float = 3e-4,
         alpha_lr: float = 3e-4, batch_size: int = 256,
         buffer_size: int = 1_000_000, target_entropy: float | None = None,
         auto_alpha: bool = True, init_alpha: float = 1.0,
         reward_scale: float = 1.0, print_every: int = PRINT_EVERY,
         verbose: bool = True):
    """Trains on Pendulum-v1 and returns a :class:`RunResult`.

    Every knob in the chapter's table 6.2 is a keyword here and a flag on the
    CLI below, and the ablation switches (``auto_alpha``, ``init_alpha``,
    ``reward_scale``) pass straight through to the agent, so each row of
    ``ablation.py`` is one call to this function rather than a second copy of
    the loop.

    Note that the budget is in environment *steps*, not episodes. SAC updates
    per step, so steps are the unit its sample-efficiency claim is made in --
    which is also why Chapter 5's ``plot_efficiency.py`` converts before
    calling this.
    """
    env = gym.make(ENV_ID)

    # Seed before building the agent, so weight initialization is covered.
    if seed is not None:
        set_seed(seed)
        seed_env(env, seed)

    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        gamma=gamma,
        tau=tau,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_entropy=target_entropy,
        auto_alpha=auto_alpha,
        init_alpha=init_alpha,
        reward_scale=reward_scale,
    )

    state, _ = env.reset()
    ep_return = 0.0
    returns = []
    episode = 0

    for _ in range(1, total_steps + 1):
        # Warmup draws uniformly from the action space rather than running the
        # untrained policy. Unlike DDPG's, SAC's untrained actor is already
        # broadly random -- but a uniform draw covers the action bounds evenly,
        # where a freshly initialized Gaussian concentrates near tanh(0) = 0.
        if len(agent.replay) < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        # Store `terminated`, not `terminated or truncated`. Pendulum-v1 never
        # terminates -- it only truncates at its 200-step limit -- so
        # collapsing the two would zero the bootstrap on the last transition of
        # every episode, telling the critics that states at t=200 are worth
        # nothing. The value of a state does not depend on how much clock is
        # left.
        agent.store(state, action, reward, next_state, float(terminated))

        state = next_state
        ep_return += reward

        if len(agent.replay) >= warmup_steps:
            agent.train()

        if terminated or truncated:
            episode += 1
            returns.append(ep_return)
            if verbose and (episode == 1 or episode % print_every == 0):
                print(f"Ep {episode:4d} | R: {ep_return:8.1f} | "
                      f"alpha: {agent.alpha:.3f}")
            state, _ = env.reset()
            ep_return = 0.0

    # Probe the policy before closing the environment: this is the
    # measurement that separates the chapter's ablation variants, which their
    # returns do not.
    sigma, entropy = measure_policy(agent)
    env.close()

    if verbose and returns:
        window = min(SCORE_WINDOW, len(returns))
        tail = returns[-window:]
        # Median, not mean. Pendulum-v1 resets the pole to a uniformly random
        # angle, so roughly one episode in ten starts near upright and scores
        # close to zero for a converged policy. Those outliers drag a mean
        # around; the chapter judges convergence by the median for exactly
        # this reason.
        print(f"\nTraining complete. Final {window}-episode median: "
              f"{statistics.median(tail):.1f} "
              f"(mean {statistics.mean(tail):.1f}, "
              f"best {max(tail):.1f}, worst {min(tail):.1f})")
        print(f"Final policy: sigma {sigma:.3f}, "
              f"entropy {entropy:+.3f} (target {agent.target_entropy:+.1f}), "
              f"alpha {agent.alpha:.3f}")

    return RunResult(returns, sigma, entropy, agent.alpha)


if __name__ == "__main__":
    # Parsed here rather than inside main() so importing this module (as the
    # tests, ablation.py, and chapter 5's plot_efficiency.py all do) never
    # touches sys.argv.
    parser = argparse.ArgumentParser(
        description="Train SAC-v2 on Pendulum-v1.")
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Fix every RNG for a reproducible run. Seed 42 reproduces the "
             "chapter's transcript. Pass --seed -1 for a non-deterministic run.",
    )
    parser.add_argument(
        "--steps", type=int, default=TOTAL_STEPS,
        help="Environment steps to train for. SAC budgets in steps, not "
             "episodes, because it updates once per step.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=WARMUP_STEPS,
        help="Uniform random actions before the first gradient update.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor. Table 6.2's standard for continuous control.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Soft target update rate, applied after every gradient step.",
    )
    parser.add_argument(
        "--actor-lr", type=float, default=3e-4,
        help="Actor learning rate. Usually identical to the other two.",
    )
    parser.add_argument(
        "--critic-lr", type=float, default=3e-4,
        help="Critic learning rate. Unlike DDPG, it does not lead the actor.",
    )
    parser.add_argument(
        "--alpha-lr", type=float, default=3e-4,
        help="Temperature learning rate.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Transitions per gradient update.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1_000_000,
        help="Replay capacity. The paper's 1e6.",
    )
    parser.add_argument(
        "--target-entropy", type=float, default=None,
        help="Entropy floor H-bar. Defaults to the SAC-v2 heuristic -dim(A), "
             "which is -1 here.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, dest="init_alpha",
        help="Starting temperature; the fixed value under --fixed-alpha.",
    )
    parser.add_argument(
        "--fixed-alpha", action="store_true",
        help="Exercise 2: hold alpha at --alpha instead of learning it, "
             "reverting to SAC-v1's hand-set temperature.",
    )
    parser.add_argument(
        "--no-entropy", action="store_true",
        help="Exercise 1: alpha = 0 with no temperature update, removing the "
             "entropy bonus entirely.",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1.0,
        help="Exercise 3: multiply rewards before they reach the Bellman "
             "target. Reward scale is an implicit inverse temperature.",
    )
    parser.add_argument(
        "--print-every", type=int, default=PRINT_EVERY,
        help="Episodes between log lines.",
    )
    args = parser.parse_args()

    main(
        seed=None if args.seed < 0 else args.seed,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_entropy=args.target_entropy,
        auto_alpha=not (args.fixed_alpha or args.no_entropy),
        init_alpha=0.0 if args.no_entropy else args.init_alpha,
        reward_scale=args.reward_scale,
        print_every=args.print_every,
    )
