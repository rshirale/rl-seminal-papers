# Chapter 4: Deep Deterministic Policy Gradient (DDPG)

This directory contains the Python implementations for Chapter 4 of **"RL: The Seminal Papers"**. It implements DDPG from Lillicrap et al. (2015), *Continuous control with deep reinforcement learning* — the algorithm that carried Chapter 3's replay buffer and target networks across from discrete action menus to continuous control. The runnable experiment verifies the full pipeline on `Pendulum-v1` in a few minutes on a CPU.

DQN picks actions with `argmax_a Q(s, a)`. That is a table lookup over two actions on CartPole and an intractable optimisation over ℝᴺ on a robot arm. DDPG's answer is to make the argmax a network: a deterministic actor μ(s) proposes the action, a critic Q(s, a) scores it, and the actor is trained by pushing the critic's gradient back through the action it was given.

## File Structure

- `actor.py`: `Actor` — the deterministic policy μ(s | θᵘ). Two hidden layers of 400 and 300 units, tanh output scaled by `max_action`. `reset_parameters()` applies the paper's U(−3e-3, 3e-3) output-layer initialization.
- `critic.py`: `Critic` — the action-value network Q(s, a | θQ). The state enters at layer 1; the action is concatenated with the 400-unit state embedding before layer 2, as in the paper.
- `gaussian_noise.py`: `GaussianNoise` — i.i.d. exploration noise, with optional linear annealing of σ.
- `parameter_noise.py`: `AdaptiveParameterNoise` and `action_distance` — optional parameter-space exploration (Plappert et al., 2018), for the tasks where action noise stalls.
- `replay_buffer.py`: `ReplayBuffer` — a `deque`-backed circular buffer of (s, a, r, s', done) transitions, default capacity 1,000,000.
- `ddpg_agent.py`: `DDPGAgent` — Algorithm 1, composing all of the above, with ablation switches for the target networks and the update rule.
- `seeding.py`: `set_seed` and `seed_env` — every RNG a run draws from, in one place.
- `train_pendulum.py`: the runnable Pendulum-v1 experiment.
- `ablation.py`: the component ablation behind the chapter's ablation figure.
- `Chapter4_DDPG.ipynb`: the interactive companion notebook (Colab-ready).
- `__init__.py`: exposes `Actor`, `Critic`, `DDPGAgent`, `GaussianNoise`, `AdaptiveParameterNoise`, `action_distance`, and `ReplayBuffer`.

## Installation

Chapter 4 needs the full stack (PyTorch + Gymnasium). From the project root:
```bash
make install-full
```

## Running the Experiments

Train on Pendulum-v1:
```bash
# From the project root
make run-ch4-pendulum

# ...or directly, with a fixed seed
python -m src.part_2_methods.ch04_ddpg.train_pendulum --seed 0
```

A converged agent scores above −200 per episode against a random policy's −1200. Expect roughly three minutes for the default 200 episodes on a CPU.

Component ablation — target networks, and soft versus hard target updates:
```bash
make run-ch4-ablation

# ...or with your own sweep
python -m src.part_2_methods.ch04_ddpg.ablation --seeds 0 1 2 --episodes 200

# regenerate the book's figure
python -m src.part_2_methods.ch04_ddpg.ablation \
    --figure ../potential-eureka/Books/RL_Seminal_Papers/Chapter4/media
```

Interactive notebook: open `Chapter4_DDPG.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch04_ddpg/Chapter4_DDPG.ipynb).

## Implementation Notes

- **Truncation is not termination.** The loop stores `float(terminated)`, never `float(terminated or truncated)`. Pendulum-v1 never terminates — it only truncates at its 200-step limit — so collapsing the two zeroes the bootstrap term on the last transition of *every* episode, teaching the critic that states at t=200 are worthless. This is precisely the distinction Gymnasium split the old `done` flag apart to express.

  Worth knowing before you assume this is a performance fix: it mostly is not, here. Measured over three seeds on Pendulum-v1, the two versions score within noise of each other. One in 200 transitions carries the wrong target, the critic sees the same states again from a hundred other episodes, and a discount of 0.99 keeps the damage local. The bug is real and the fix is correct — but the honest lesson is that a defect can be genuinely wrong and still not show up in the headline metric, which is why you read the code as well as the reward curve. On an environment with a shorter horizon, or one where episodes genuinely terminate, the same mistake is not so forgiving.
- **Warmup uses uniform random actions.** A freshly initialised deterministic actor emits nearly the same action everywhere — its output layer starts at U(−3e-3, 3e-3) by design. Driving warmup with that policy fills the buffer with a thousand near-identical torques and teaches the critic almost nothing about the action space it must evaluate. The first `--warmup-steps` steps therefore sample the action space directly.
- **The critic is frozen during the actor update.** The actor's loss backpropagates *through* the critic to reach the actor's weights, so autograd would otherwise populate a full set of critic gradients that nothing consumes — only `actor_opt.step()` runs. Freezing skips that half of the graph; behaviour is identical, the work is not. The flag is restored immediately afterwards, and a test guards that, because leaving it off silently stops the critic learning.
- **Soft updates use `no_grad` + `copy_`, not `.data`.** Reaching through `.data` bypasses autograd's bookkeeping rather than satisfying it, which is why current PyTorch discourages it.
- **Layer widths are the paper's.** 400/300 is kept for fidelity to Lillicrap et al. Modern reference implementations (CleanRL, Stable-Baselines3) have standardized on 256/256, which trains slightly faster and performs comparably. Neither choice is load-bearing for the chapter's ideas.
- **Optional knobs the paper used and the chapter's listings omit.** `critic_weight_decay` defaults to 0.0; the paper uses 1e-2. `sigma_final` / `sigma_decay_steps` are off by default; production runs commonly anneal exploration. Both are available on `DDPGAgent` without changing the default behaviour the chapter teaches.
