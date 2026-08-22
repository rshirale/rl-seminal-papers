# Chapter 4: Deep Deterministic Policy Gradient (DDPG)

This directory contains the Python implementations for Chapter 4 of **"RL: The Seminal Papers"**. It implements DDPG from Lillicrap et al. (2015), *Continuous control with deep reinforcement learning* — the algorithm that carried Chapter 3's replay buffer and target networks across from discrete action menus to continuous control. The runnable experiment verifies the full pipeline on `Pendulum-v1` in a few minutes on a CPU.

DQN picks actions with `argmax_a Q(s, a)`. That is a table lookup over two actions on CartPole and an intractable optimisation over ℝᴺ on a robot arm. DDPG's answer is to make the argmax a network: a deterministic actor μ(s) proposes the action, a critic Q(s, a) scores it, and the actor is trained by pushing the critic's gradient back through the action it was given.

## What to run, and what to read

Only **two** files in this directory are meant to be executed, plus the notebook. Everything else is a library module you read alongside the chapter's listings — importing them is how `train_pendulum.py` and `ablation.py` stay short enough to follow.

If you are starting from scratch: `make install-full`, then `make run-ch4-pendulum`, then open the notebook.

### Files you run

| File | What it does | Time |
| --- | --- | --- |
| `train_pendulum.py` | The Pendulum-v1 experiment behind figure 4.9. Every knob in table 4.2 is a flag. | ~3 min |
| `ablation.py` | The three-variant component ablation behind figure 4.10, and it can regenerate the figure itself. | ~92 min |
| `Chapter4_DDPG.ipynb` | The whole chapter re-derived inline, cell by cell. Colab-ready. | ~3 min |

### Files you read

In the order the chapter builds them, so this list doubles as a reading path:

| File | Chapter | What it is |
| --- | --- | --- |
| `actor.py` | Listing 4.2 | `Actor` — the deterministic policy μ(s \| θᵘ). Two hidden layers of 400 and 300 units, tanh output scaled by `max_action`. `reset_parameters()` applies the paper's U(−3e-3, 3e-3) output-layer initialization. |
| `critic.py` | Listing 4.3 | `Critic` — the action-value network Q(s, a \| θQ). The state enters at layer 1; the action is concatenated with the 400-unit state embedding before layer 2, as in the paper. |
| `gaussian_noise.py` | Listing 4.6 | `GaussianNoise` — i.i.d. exploration noise, with optional linear annealing of σ. |
| `replay_buffer.py` | Listing 4.7 | `ReplayBuffer` — a `deque`-backed circular buffer of (s, a, r, s', done) transitions, default capacity 1,000,000. |
| `ddpg_agent.py` | Listings 4.4, 4.5, 4.8, 4.9 | `DDPGAgent` — Algorithm 1, composing all of the above: four-network setup, the soft update, the critic step, and the actor step. Carries the ablation switches. |

### Supporting files

| File | Why it exists |
| --- | --- |
| `seeding.py` | `set_seed` and `seed_env` — every RNG a run draws from, in one place. Deliberately the same shape as Chapter 3's, so the two chapters teach one seeding habit. |
| `parameter_noise.py` | `AdaptiveParameterNoise` and `action_distance` — the parameter-space exploration behind the §4.6 "Pro tip" sidebar and the notebook's section 12. Optional: nothing in the default training path imports it. |
| `__init__.py` | Package exports, so `from src.part_2_methods.ch04_ddpg import DDPGAgent` works from the repo root. |

Tests live outside this directory, in `tests/test_ddpg.py` and `tests/test_ch04_notebook.py` (55 tests). Run them with `make test`; `make test-all` adds the notebook execution test, which is slow because it trains for real.

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

Every hyperparameter in the chapter's tuning cheat sheet (table 4.2) is a flag, so the table can be worked through without editing a file:

```bash
python -m src.part_2_methods.ch04_ddpg.train_pendulum --help
```

`--seed`, `--episodes`, `--warmup-steps`, `--sigma`, `--tau`, `--gamma`, `--actor-lr`, `--critic-lr`, `--batch-size`, `--buffer-size`, `--critic-weight-decay`, `--no-target-networks`, `--hard-target-updates`.

The section 4.9 exercises map onto them directly:

| Exercise | Command |
| --- | --- |
| 1 — sweep the noise scale | `--sigma 0.05` / `0.2` / `0.4` |
| 2 — remove the target networks | `--no-target-networks` (drops *both* targets, not only the actor's) |
| 3 — sweep the soft update rate | `--tau 0.1` / `0.01` / `0.001` |
| 6 — break the bootstrap mask | edit `float(terminated)` to `float(terminated or truncated)` in `train_pendulum.py`, then re-run across three seeds |

Component ablation — the three variants behind the chapter's figure 4.10, with exploration noise held identical across all of them so the whole gap is attributable to the target networks rather than to the noise process:

1. **No target networks** — the online critic computes the target it is trained against.
2. **Full DDPG** — soft targets at τ = 0.001, the published configuration.
3. **Hard target copy** — targets overwritten wholesale on a DQN-style schedule.

```bash
make run-ch4-ablation

# ...or with your own sweep
python -m src.part_2_methods.ch04_ddpg.ablation --seeds 0 1 2 --episodes 200

# drop the hard copy for a quicker two-curve sweep while iterating
python -m src.part_2_methods.ch04_ddpg.ablation --no-hard-copy

# regenerate the book's figure
python -m src.part_2_methods.ch04_ddpg.ablation \
    --figure ../potential-eureka/Books/RL_Seminal_Papers/Chapter4/media
```

The full sweep is nine 200-episode training runs and it is genuinely slow — measured at **92 minutes** on an 8-core Intel MacBook Pro, and that will move with your core count. For a quicker look that still separates the variants, `--episodes 100 --seeds 0 1` costs about a fifth of that.

The spread column is the one to read first: an unstable variant is one whose outcome depends on the seed.

Reproduced on 2026-08-21 with the defaults above:

| Variant | seed 0 | seed 1 | seed 2 | mean | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| No target networks | −887.0 | −880.5 | −1266.0 | −1011.2 | 385.5 |
| Full DDPG (soft targets) | −140.9 | −132.0 | −175.5 | −149.5 | 43.4 |
| Hard target copy | −149.7 | −128.4 | −166.1 | −148.1 | 37.7 |

Target networks are worth about 860 points of return; how they are updated is worth 1.4, which is well inside seed noise on this task.

Interactive notebook: open `Chapter4_DDPG.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch04_ddpg/Chapter4_DDPG.ipynb). It runs top to bottom in about **three minutes** at its default of 200 episodes (measured: 190s on an 8-core Intel MacBook Pro), and re-derives every class inline so the chapter can be followed cell by cell. On Colab, uncomment the `%pip install` line in the first code cell. Set `CH4_NUM_EPISODES` to shorten the training loop while exploring.

## Implementation Notes

- **Truncation is not termination.** The loop stores `float(terminated)`, never `float(terminated or truncated)`. Pendulum-v1 never terminates — it only truncates at its 200-step limit — so collapsing the two zeroes the bootstrap term on the last transition of *every* episode, teaching the critic that states at t=200 are worthless. This is precisely the distinction Gymnasium split the old `done` flag apart to express.

  Worth knowing before you assume this is a performance fix: it mostly is not, here. Measured over three seeds on Pendulum-v1, the two versions score within noise of each other. One in 200 transitions carries the wrong target, the critic sees the same states again from a hundred other episodes, and a discount of 0.99 keeps the damage local. The bug is real and the fix is correct — but the honest lesson is that a defect can be genuinely wrong and still not show up in the headline metric, which is why you read the code as well as the reward curve. On an environment with a shorter horizon, or one where episodes genuinely terminate, the same mistake is not so forgiving.
- **Warmup uses uniform random actions.** A freshly initialised deterministic actor emits nearly the same action everywhere — its output layer starts at U(−3e-3, 3e-3) by design. Driving warmup with that policy fills the buffer with a thousand near-identical torques and teaches the critic almost nothing about the action space it must evaluate. The first `--warmup-steps` steps therefore sample the action space directly.
- **The critic is frozen during the actor update.** The actor's loss backpropagates *through* the critic to reach the actor's weights, so autograd would otherwise populate a full set of critic gradients that nothing consumes — only `actor_opt.step()` runs. Freezing skips that half of the graph; behaviour is identical, the work is not. The flag is restored immediately afterwards, and a test guards that, because leaving it off silently stops the critic learning.
- **Soft updates use `no_grad` + `copy_`, not `.data`.** Reaching through `.data` bypasses autograd's bookkeeping rather than satisfying it, which is why current PyTorch discourages it.
- **Layer widths are the paper's.** 400/300 is kept for fidelity to Lillicrap et al. Modern reference implementations (CleanRL, Stable-Baselines3) have standardized on 256/256, which trains slightly faster and performs comparably. Neither choice is load-bearing for the chapter's ideas.
- **Optional knobs the paper used and the chapter's listings omit.** `critic_weight_decay` defaults to 0.0; the paper uses 1e-2. `sigma_final` / `sigma_decay_steps` are off by default; production runs commonly anneal exploration. Both are available on `DDPGAgent` without changing the default behaviour the chapter teaches.
