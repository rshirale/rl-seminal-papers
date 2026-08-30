# Chapter 3: Deep Q-Networks (DQN)

This directory contains the Python implementations for Chapter 3 of **"RL: The Seminal Papers"**. It implements the Deep Q-Network from Mnih et al. (2013, 2015) and the two innovations that made deep RL stable: **experience replay** and a **target network**. The runnable experiment verifies the full pipeline on `CartPole-v1` in about three minutes on a CPU.

Q-learning with a neural network had been tried before and it diverged. The 2015 paper's contribution is less a new algorithm than a diagnosis of *why*: a network trained on consecutive frames sees a stream of nearly identical, highly correlated inputs, and a network that bootstraps from itself is chasing a target that moves every time it takes a step. Replay fixes the first by sampling the past uniformly. The target network fixes the second by freezing the thing being chased for `C` steps at a time. Everything else — the CNN, the Huber loss, RMSprop — is scaffolding around those two ideas.

This is the first Deep RL chapter, and the first place the book's structure appears: a small CPU-friendly task that verifies the logic (`CartPole-v1`), a faithful reproduction of the paper's real configuration (Atari), and an ablation that turns each idea off to show what breaks.

## File Structure

- `replay_buffer.py`: `ReplayBuffer` — experience replay backed by **pre-allocated, contiguous NumPy arrays** (one per field). The constructor takes `capacity`, `state_shape`, and `state_dtype` (defaults to `float32`; pass `np.uint8` for raw Atari frames). Sampling is a fast index gather. Also provides `sample_recent`, which only the ablation uses.
- `dqn_network.py`: `DQN` (the three-convolutional-layer CNN matching the Mnih et al. 2015 spec, for 84×84 frame stacks) and `SimpleDQN` (a two-hidden-layer MLP for vector states like CartPole).
- `dqn_agent.py`: `AtariDQNAgent` — the Atari agent (Algorithm 1 of Mnih et al. 2015), composing the CNN, replay buffer, and target network with RMSprop and Huber loss. Note the name: the CartPole agent is a **different, lighter class** called `DQNAgent`, and it lives in `train_cartpole.py`.
- `seeding.py`: `set_seed` and `seed_env` cover every RNG a run draws from, in one place. Shared by both training scripts.
- `train_cartpole.py`: the runnable CartPole-v1 experiment, plus the `DQNAgent` it drives. Uses `SimpleDQN` to isolate and verify the core DQN logic. `main()` is importable and returns the per-episode rewards, so `ablation.py` drives one configuration per call instead of re-implementing the loop.
- `ablation.py`: the paper's four-way ablation — full DQN, no target network, no replay, neither — run on CartPole across several seeds.
- `train_atari.py`: Atari training, with the preprocessing wrappers from the paper's Methods section (FIRE-on-reset, max-and-skip, grayscale warp to 84×84, four-frame stacking, reward clipping). GPU recommended.
- `Chapter3_DQN.ipynb`: the interactive companion notebook (CartPole; Colab-ready).
- `__init__.py`: exposes `ReplayBuffer`, `DQN`, `SimpleDQN`, `AtariDQNAgent`, and `DQNAgent`.

## Installation

Chapter 3 is the first Deep RL chapter, so it needs the full stack (PyTorch + Gymnasium). From the project root:
```bash
make install-full
```

Atari is a separate, optional install — it pulls `ale-py` and OpenCV:
```bash
make install-atari
```

## Running the Experiments

CartPole-v1 (the verified, CPU-friendly path):
```bash
# From the project root
make run-ch3-cartpole

# ...or directly, with a fixed seed
python -m src.part_2_methods.ch03_dqn.train_cartpole --seed 42 --episodes 600
```

The ablation turns each of the paper's two ideas off in turn, holding everything else fixed, so the whole gap is attributable to the missing component:
```bash
make run-ch3-ablation

# ...or with your own seeds
python -m src.part_2_methods.ch03_dqn.ablation --seeds 1 2 3 4 5 --episodes 400
```

Four variants across three seeds; budget about twenty minutes. Read the spread column before the mean — see the notes below on why the no-replay row in particular cannot be judged from one run.

Atari (requires a GPU; long-running):
```bash
make run-ch3-atari

# ...or directly, with the flags
python -m src.part_2_methods.ch03_dqn.train_atari --env Pong --episodes 1000

# ...on a machine with less RAM
python -m src.part_2_methods.ch03_dqn.train_atari --env Pong --buffer-capacity 50000
```

Interactive notebook: open `Chapter3_DQN.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch03_dqn/Chapter3_DQN.ipynb).

## Implementation Notes

- **The ablation does not reproduce the paper's ordering, and that is the interesting part.** On Atari, removing replay is catastrophic. On CartPole, removing the **target network** hurts far more. That is not a contradiction — replay's value scales with how correlated consecutive observations are, and a four-element control vector is far less correlated than a stream of 84×84 frames. The chapter's job is to be honest that a small task does not reproduce every result from a large one.

- **The no-replay variant is the most seed-sensitive thing in the chapter.** Measured across three seeds its scores ranged from 33.5 to 155.6, against a spread of 2.3 for the no-target-network variant. That spread *is* the lesson: without replay, what the agent learns depends on which trajectory it happened to wander down. This is why `ablation.py` averages over seeds rather than taking a flag for one, and why a single run of it proves nothing in either direction.

- **Frames are stored as `uint8`, and normalization is the agent's job.** The replay buffer holds raw bytes at one per pixel; `_states_to_tensor` does the `/255` after sampling. That is a 4× saving on by far the largest allocation in the agent: at `capacity=100_000` the two state arrays total roughly 5.3 GB instead of about 21 GB, and a full 1,000,000-transition buffer would need ~53 GB rather than ~210 GB. The trade-off is that the agent, not the caller, owns normalization — push raw environment frames and let it convert. The `.to(float32)` copies before the in-place `div_`, so the buffer's storage is never mutated; there is a test for exactly that, because getting it wrong silently corrupts the replay memory.

- **Sampling uses rejection, not `np.random.choice`.** The obvious implementation is `np.random.choice(size, batch_size, replace=False)`, which is what the buffer used to do. NumPy implements that by permuting all `size` elements: about **26 ms** per call on a 1M-transition buffer, paid on every gradient step, against about **9 µs** for the draw itself. Rejection sampling is O(batch_size) and draws from the same distribution. Below a threshold of `100 × batch_size` collisions get frequent enough that resampling stops paying — and the permutation is cheap anyway — so the simple call still runs there, which is also what makes it terminate when `batch_size == size`.

- **Two agents, two optimizers, on purpose.** CartPole uses `SimpleDQN` with Adam at `1e-4`, because it converges fast on a CPU. Atari uses the `DQN` CNN with RMSprop at `2.5e-4`, because that is what the paper used. Extended Data Table 1 lists gradient momentum 0.95 *and* squared gradient momentum 0.95; in PyTorch those are `momentum` and `alpha`, two separate arguments, so `dqn_agent.py` sets both. Reference implementations frequently set only `alpha` and quietly diverge from the paper.

- **Warmup and batch size are two different guards.** `AtariDQNAgent.step` checks both `steps_done >= warmup_steps` *and* `len(memory) >= batch_size` before learning. They are not redundant: `sample()` draws without replacement, so a `warmup_steps` set below `batch_size` would raise rather than simply train early. A regression test covers it.

- **`select_action` flips the network to `eval()` and must flip it back.** It does, and a test asserts it. Leaving the online network in eval mode after action selection would silently disable dropout/batch-norm behaviour during the subsequent gradient step. This architecture has neither, so nothing would break today — which is precisely why it would go unnoticed until someone added a layer that cares.

- **The Atari wrappers contain the two subtlest bugs in the chapter.** `MaxAndSkipEnv` max-pools over the **last two frames actually observed** in a skip block, to cancel the sprite flicker the hardware produced. Writing the buffer only at `i == skip-2` and `skip-1` looks equivalent and is not: when an episode ends early inside the block, those indices are never reached and the wrapper returns a stale frame. Terminal frames are exactly the ones the Bellman target treats specially, so corrupting them is not a harmless edge case. Separately, `FireResetEnv` strips `seed` from its recovery resets — passing it through would replay the identical failing episode forever.

- **Seeding covers four generators.** `random` for the epsilon-greedy coin flip, `np.random` for replay sampling, `torch` for weight initialization, and the environment's own RNG for episode starts and `action_space.sample()`. Miss one and the run is not reproducible. `set_seed` must be called *before* the agent is constructed, or weight initialization falls outside the seeded region. Note that unlike Chapters 5 and 6, this chapter does **not** pin the thread count; runs are reproducible on one machine, not across machines with different core counts.

## Troubleshooting

- **Every row of the ablation table is identical.** Your run was too short to outlast the warmup. A CartPole episode lasts about 20 steps under a random policy, so at fewer than ~100 episodes the 1,000-step warmup consumes the entire run and no variant ever trains — all four rows are the same random play. `ablation.py` prints a warning when it detects this. Use `--episodes 300` or more.

- **CartPole reward climbs, then collapses.** This is normal and worth watching rather than fixing. DQN on CartPole is famously non-monotonic: the policy improves, falls back, and recovers, sometimes repeatedly. Judge it on the final-window average across seeds, which is what `ablation.py` reports, not on the peak. For scale, three unseeded 200-episode runs recorded last-50 averages of 69.6, 76.2 and 92.1 — a 32% spread from RNG state alone.

- **Results differ from run to run even with `--seed`.** Check that you are on the same machine and PyTorch version. This chapter seeds four generators but does not pin `torch.set_num_threads`, so a different core count changes floating-point reduction order and therefore the run. Chapters 5 and 6 pin it; this one does not, deliberately, because no transcript in Chapter 3 is quoted to the digit.

- **`ModuleNotFoundError: No module named 'ale_py'`** (or `cv2`). `train_atari.py` imports both at module scope and raises a message pointing at `make install-atari`. The CartPole path needs neither, so this never blocks the chapter's main experiment.

- **The Atari run is killed by the OS, or the machine starts swapping.** The replay buffer is the problem. At the default `--buffer-capacity 100000` the two `uint8` state arrays need about 5.3 GB; each transition costs roughly 55 KB, being two 84×84×4 frame stacks. Lower the capacity: `--buffer-capacity 50000` halves it. The paper's own 1,000,000 needs about 53 GB and is not a laptop configuration.

- **Atari learns nothing for a long time.** That is expected — Pong needs on the order of a million frames before the score moves off −21, and the default `--episodes 1000` is a starting point, not a converged run. Verify your pipeline on CartPole first; it exercises the same replay buffer, target network, and Huber loss in three minutes.

- **`ModuleNotFoundError: No module named 'src'`.** Run the `-m` form from the project root, not from this directory. The scripts also work when run directly (`python train_cartpole.py`) from inside this directory, which is the fallback the `if __package__` branches exist for — but the two are not interchangeable from the same working directory.
