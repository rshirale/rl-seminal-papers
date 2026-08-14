# Chapter 3: Deep Q-Networks (DQN)

This directory contains the Python implementations for Chapter 3 of **"RL: The Seminal Papers"**. It implements the Deep Q-Network from Mnih et al. (2013, 2015) and the two innovations that made deep RL stable: **experience replay** and a **target network**. The runnable experiment verifies the full pipeline on `CartPole-v1` in about two minutes on a CPU.

## File Structure

- `replay_buffer.py`: `ReplayBuffer` — experience replay backed by **pre-allocated, contiguous NumPy arrays** (one per field). The constructor takes `capacity`, `state_shape`, and `state_dtype` (defaults to `float32`; pass `np.uint8` for raw Atari frames). Sampling is a fast index gather.
- `dqn_network.py`: `DQN` (three-convolutional-layer CNN matching the Mnih et al. 2015 spec, for 84×84 Atari frames) and `SimpleDQN` (two-hidden-layer MLP for vector states like CartPole).
- `dqn_agent.py`: `DQNAgent` — the Atari agent (Algorithm 1 of Mnih et al. 2015), composing the CNN, replay buffer, and target network with RMSprop and Huber loss.
- `train_cartpole.py`: the runnable CartPole-v1 experiment, using `SimpleDQN` to isolate and verify the core DQN logic (replay buffer, target network, Huber loss).
- `train_atari.py`: Atari training with the max-and-skip and frame-stacking wrappers (GPU recommended).
- `Chapter3_DQN.ipynb`: the interactive companion notebook (CartPole; Colab-ready).
- `__init__.py`: exposes `ReplayBuffer`, `DQN`, `SimpleDQN`, and `DQNAgent`.

## Installation

Chapter 3 is the first Deep RL chapter, so it needs the full stack (PyTorch + Gymnasium). From the project root:
```bash
make install-full
```

## Running the Experiments

CartPole-v1 (the verified, CPU-friendly path):
```bash
# From the project root
make run-ch3-cartpole

# ...or directly from this directory
python train_cartpole.py
```

Interactive notebook: open `Chapter3_DQN.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch03_dqn/Chapter3_DQN.ipynb).

Atari (requires a GPU; long-running):
```bash
python train_atari.py --env Pong --episodes 1000
```

## Implementation Notes

- **Replay buffer memory.** The buffer pre-allocates fixed-size arrays at construction, so memory stays close to the theoretical minimum and each batch is a contiguous gather rather than a walk over Python objects. For a full 1,000,000-frame Atari buffer, construct it with `state_dtype=np.uint8` so the frames fit in roughly 6.6 GB (versus ~26 GB as `float32`), and apply the `/255` normalization after sampling. CartPole stores its small float32 state vectors directly.
- **Two networks for two regimes.** CartPole uses `SimpleDQN` with the Adam optimizer for fast convergence on a CPU; the paper's Atari configuration uses the `DQN` CNN with RMSprop (see `dqn_agent.py`).
