"""Train DDPG on Pendulum-v1 (Lillicrap et al., 2015, Algorithm 1).

Usage:
    python -m src.part_2_methods.ch04_ddpg.train_pendulum
    python -m src.part_2_methods.ch04_ddpg.train_pendulum --seed 42 --episodes 200

    # ...or directly, from this directory
    python train_pendulum.py --seed 42

Pendulum-v1 is a three-dimensional state (cos theta, sin theta, theta-dot) and
one continuous action, the torque applied at the pivot, bounded to [-2, 2]. The
reward is roughly -(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2), so the
agent is pushed to hold the pole upright, still, and with as little effort as
it can manage. A random policy scores about -1200 per episode; a converged
agent scores above -200.
"""

import argparse

import gymnasium as gym
import numpy as np

if __package__:
    from .ddpg_agent import DDPGAgent
    from .seeding import seed_env, set_seed
else:  # pragma: no cover - direct script execution fallback.
    from ddpg_agent import DDPGAgent
    from seeding import seed_env, set_seed

EPISODES = 200
MAX_STEPS = 200      # Pendulum-v1's time limit
WARMUP_STEPS = 1_000  # uniform random actions before the policy takes over
PRINT_EVERY = 10
SCORE_WINDOW = 20    # episodes averaged for the reported final score

ENV_ID = "Pendulum-v1"


def main(seed: int | None = None, episodes: int = EPISODES,
         warmup_steps: int = WARMUP_STEPS, sigma: float = 0.2,
         tau: float = 0.001, use_target_networks: bool = True,
         target_update: str = "soft", verbose: bool = True):
    """Trains on Pendulum-v1 and returns the per-episode returns.

    The ablation switches pass straight through to the agent, so each row of
    the chapter's ablation figure is one call to this function.
    """
    env = gym.make(ENV_ID)

    # Seed before building the agent, so weight initialization is covered.
    if seed is not None:
        set_seed(seed)
        seed_env(env, seed)

    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        tau=tau,
        sigma=sigma,
        use_target_networks=use_target_networks,
        target_update=target_update,
    )

    returns = []
    total_steps = 0

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        agent.reset_noise()
        ep_return = 0.0

        for _ in range(MAX_STEPS):
            # Warmup draws uniformly from the action space rather than running
            # the untrained policy. An untrained deterministic actor emits
            # nearly the same action everywhere -- its output layer starts at
            # U(-3e-3, 3e-3) by design -- so policy-driven warmup would fill
            # the buffer with a thousand near-identical torques and teach the
            # critic almost nothing about the action space it has to evaluate.
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store `terminated`, not `terminated or truncated`. Pendulum-v1
            # never terminates -- it only truncates at its 200-step limit -- so
            # collapsing the two would zero the bootstrap on the last
            # transition of every single episode, telling the critic that
            # states at t=200 are worth nothing. The value of a state does not
            # depend on how much clock is left; this is exactly the distinction
            # Gymnasium split `done` apart to express.
            agent.store(state, action, reward, next_state, float(terminated))

            if total_steps >= warmup_steps:
                agent.train()

            state = next_state
            ep_return += reward
            total_steps += 1

            if terminated or truncated:
                break

        returns.append(ep_return)

        if verbose and ep % PRINT_EVERY == 0:
            avg = np.mean(returns[-PRINT_EVERY:])
            print(f"Episode {ep:4d} | Return: {ep_return:8.1f} | "
                  f"Avg-{PRINT_EVERY}: {avg:8.1f}")

    env.close()

    if verbose:
        window = min(SCORE_WINDOW, len(returns))
        print(f"\nTraining complete. Final {window}-episode average: "
              f"{np.mean(returns[-window:]):.1f}")

    return returns


if __name__ == "__main__":
    # Parsed here rather than inside main() so importing this module (as the
    # tests and ablation.py do) never touches sys.argv.
    parser = argparse.ArgumentParser(
        description="Train DDPG on Pendulum-v1.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fix every RNG for a reproducible run. Omitted by default, which "
             "keeps the original non-deterministic behaviour.",
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES,
        help="Episodes to train for.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=WARMUP_STEPS,
        help="Uniform random actions before the policy takes over.",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.2,
        help="Gaussian exploration noise scale.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.001,
        help="Soft target update rate. Exercise 3 in the chapter sweeps this.",
    )
    parser.add_argument(
        "--no-target-networks", action="store_true",
        help="Ablation: let the online networks be their own targets, "
             "reproducing the moving-target instability.",
    )
    parser.add_argument(
        "--hard-target-updates", action="store_true",
        help="Ablation: copy the targets wholesale on a DQN-style schedule "
             "instead of drifting them every step.",
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        episodes=args.episodes,
        warmup_steps=args.warmup_steps,
        sigma=args.sigma,
        tau=args.tau,
        use_target_networks=not args.no_target_networks,
        target_update="hard" if args.hard_target_updates else "soft",
    )
