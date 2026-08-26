"""Train PPO on Pendulum-v1.

``main`` is importable and returns what it measured, so ``ablation.py`` can
drive one configuration per call rather than re-implementing the loop. That is
the same seam Chapter 4 uses; the difference is what comes back. DDPG's
ablation only needs the return curve, whereas PPO's chapter teaches readers to
watch ``approx_kl`` and ``clip_frac`` while training, so a run has to hand back
those traces too or the ablation cannot plot the diagnostics the text argues
for.

Usage:
    python -m src.part_2_methods.ch05_ppo.train_pendulum
    python -m src.part_2_methods.ch05_ppo.train_pendulum --seed 7 --episodes 200
"""

import argparse
from collections import namedtuple

import gymnasium as gym
import numpy as np

if __package__:
    from .ppo_agent import PPOAgent
    from .seeding import episode_seed, set_seed
else:  # pragma: no cover - only used by direct script execution.
    from ppo_agent import PPOAgent
    from seeding import episode_seed, set_seed

ENV_ID = "Pendulum-v1"
EPISODES = 400
UPDATE_EVERY = 2048
LOG_EVERY = 50

# The chapter's configuration. gamma = 0.9 is deliberate and not a typo for the
# conventional 0.99: Pendulum episodes are 200 steps of dense reward, and the
# shorter effective horizon learns markedly faster here. At 400 episodes,
# gamma = 0.9 finishes near -354 where gamma = 0.99 is still around -1074.
GAMMA = 0.9
LR = 1e-3
LAM = 0.95
EPS_CLIP = 0.2
K_EPOCHS = 10
BATCH_SIZE = 64

RANDOM_POLICY_BASELINE = -1200.0

#: ``returns`` is per episode; ``approx_kls`` and ``clip_fracs`` are per update,
#: so they are much shorter than ``returns`` and are not index-aligned with it.
RunResult = namedtuple("RunResult", "returns approx_kls clip_fracs")


def parse_args():
    """Parse options for a reproducible, configurable training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lam", type=float, default=LAM)
    parser.add_argument("--eps-clip", type=float, default=EPS_CLIP)
    parser.add_argument("--k-epochs", type=int, default=K_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main(seed=42, episodes=EPISODES, update_every=UPDATE_EVERY,
         log_every=LOG_EVERY, gamma=GAMMA, lr=LR, lam=LAM,
         eps_clip=EPS_CLIP, k_epochs=K_EPOCHS, batch_size=BATCH_SIZE,
         verbose=True):
    """Trains on Pendulum-v1 and returns a :class:`RunResult`.

    The hyperparameters pass straight through to the agent, so each row of the
    ablation is one call to this function.
    """
    # Covers weight init, the policy's action sampling, and the minibatch
    # permutation, and pins the thread count. See seeding.py for why the thread
    # count belongs here rather than in a performance note.
    set_seed(seed)

    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = PPOAgent(
        state_dim, action_dim, max_action,
        lr=lr, gamma=gamma, lam=lam,
        eps_clip=eps_clip, k_epochs=k_epochs, batch_size=batch_size
    )

    rollouts = []
    timestep = 0
    ep_rewards = []
    approx_kls = []
    clip_fracs = []
    approx_kl = 0.0
    clip_frac = 0.0

    try:
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=episode_seed(seed, episode))
            ep_reward = 0.0

            for _ in range(env.spec.max_episode_steps):
                timestep += 1
                action, logprob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # True termination stops bootstrapping; time-limit truncation does not.
                rollouts.append(
                    (state, action, reward, next_state,
                     float(terminated), logprob, value)
                )
                state = next_state
                ep_reward += reward

                if timestep % update_every == 0:
                    approx_kl, clip_frac = agent.update(rollouts)
                    rollouts = []
                    approx_kls.append(approx_kl)
                    clip_fracs.append(clip_frac)

                if terminated or truncated:
                    break

            ep_rewards.append(ep_reward)
            if verbose and episode % log_every == 0:
                avg_reward = np.mean(ep_rewards[-log_every:])
                print(
                    f"Episode {episode:3d} | Reward: {avg_reward:8.1f} | "
                    f"approx_kl: {approx_kl:.3f} | clip_frac: {clip_frac:.2f}"
                )

        if rollouts:
            approx_kl, clip_frac = agent.update(rollouts)
            approx_kls.append(approx_kl)
            clip_fracs.append(clip_frac)
    finally:
        env.close()

    return RunResult(ep_rewards, approx_kls, clip_fracs)


if __name__ == "__main__":
    args = parse_args()
    main(
        seed=args.seed, episodes=args.episodes,
        update_every=args.update_every, log_every=args.log_every,
        gamma=args.gamma, lr=args.lr, lam=args.lam,
        eps_clip=args.eps_clip, k_epochs=args.k_epochs,
        batch_size=args.batch_size,
    )
