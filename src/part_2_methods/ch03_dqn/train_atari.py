import gymnasium as gym
import numpy as np
import torch
import argparse
from collections import deque
import cv2

try:
    import ale_py
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise SystemExit(
        "The Atari environments require ale-py, which is not installed.\n"
        "Install the optional Atari stack with:  make install-atari"
    ) from exc

if __package__:
    from .dqn_agent import DQNAgent
    from .seeding import seed_env, set_seed
else:  # pragma: no cover - direct script execution fallback.
    from dqn_agent import DQNAgent
    from seeding import seed_env, set_seed

# --- Atari Preprocessing Wrappers ---
# (These mirror the preprocessing pipeline described in Mnih et al. 2015, Methods)

class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # A seed in kwargs would replay the identical failing episode, so the
        # recovery resets below drop it.
        recovery = {k: v for k, v in kwargs.items() if k != "seed"}
        for fire_action in (1, 2):
            obs, _, terminated, truncated, info = self.env.step(fire_action)
            if terminated or truncated:
                # Bind the reset's return value: the old code discarded it and
                # handed back the terminal frame from the episode that just died.
                obs, info = self.env.reset(**recovery)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (Frame Skipping). Max-pool over last 2 frames to fix sprite flicker."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Rolling window over the last two frames actually observed in this
            # skip block. Writing only at i == skip-2 / skip-1 meant an episode
            # ending at i == 0 or 1 max-pooled the *previous* step's frames and
            # never returned the terminal observation at all.
            self._obs_buffer[0] = self._obs_buffer[1] if i else obs
            self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info

class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 and convert to grayscale."""
    def __init__(self, env):
        super().__init__(env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    """Stack the last `k` frames to provide temporal context to the CNN."""
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k, shp[0], shp[1]), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=0).squeeze() # Shape (4, 84, 84)

class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to [-1, 1] as described in Mnih et al. 2015."""
    def reward(self, reward):
        return np.sign(reward)

def make_atari_env(env_id):
    """Creates the fully wrapped Atari environment."""
    # Gymnasium 1.0 dropped ale-py's entry-point auto-registration, so the ALE
    # environments have to be registered explicitly. Without this, gym.make()
    # below fails with NameNotFound even when ale-py is installed correctly.
    gym.register_envs(ale_py)

    # We use NoFrameskip-v4 to implement our own frame skipping and max pooling
    env = gym.make(env_id + "NoFrameskip-v4")
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    return env

# --- Main Training Loop ---

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Atari Pong")
    parser.add_argument("--env", type=str, default="Pong", help="Atari environment ID")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=100000,
        help="Replay buffer size in transitions. Costs ~55 KB each "
             "(two uint8 84x84x4 stacks), so the default needs ~5.3 GB of RAM. "
             "Lower it if you have less to spare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix every RNG for a reproducible run. Omitted by default, which "
             "keeps the original non-deterministic behaviour.",
    )
    args = parser.parse_args()

    # 1. Setup Environment
    env = make_atari_env(args.env)
    num_actions = env.action_space.n
    input_channels = 4 # Due to frame stacking

    # Seed before the agent is built, so weight initialization is covered too.
    if args.seed is not None:
        set_seed(args.seed)
        seed_env(env, args.seed)

    # 2. Setup Agent
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    warmup_steps = 1000 # Wait before learning starts
    
    agent = DQNAgent(
        input_channels=input_channels, 
        num_actions=num_actions, 
        device=device,
        buffer_capacity=args.buffer_capacity, # Scaled down for local testing (Paper used 1M)
        batch_size=32,
        warmup_steps=warmup_steps
    )

    # 3. Epsilon Annealing Schedule
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay_frames = 100000 # Paper used 1M frames, scaled down for faster local observation
    
    frame_idx = 0

    # 4. Training Loop (Algorithm 1)
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()

        # Frames stay as raw uint8 all the way into the replay buffer, which is
        # what keeps it at ~5.3 GB instead of ~21 GB. The agent normalizes to
        # [0, 1] internally, after sampling.
        episode_reward = 0
        done = False

        while not done:
            # Calculate current epsilon (wait for warmup to finish before decaying)
            if frame_idx < warmup_steps:
                epsilon = epsilon_start
            else:
                epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                          np.exp(-1. * (frame_idx - warmup_steps) / epsilon_decay_frames)
            
            # Select action using epsilon-greedy policy
            action = agent.select_action(state, epsilon)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store and learn. `terminated`, not `done`: a time-limit
            # truncation is not a real terminal state, so its bootstrap target
            # must not be zeroed. train_cartpole.py makes the same distinction.
            agent.step(state, action, reward, next_state, terminated)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1

        if episode % 10 == 0:
            print(f"Episode {episode} | Frames: {frame_idx} | Epsilon: {epsilon:.3f} | Clipped Reward: {episode_reward}")

    env.close()
    print("Training completed.")

if __name__ == "__main__":
    main()
