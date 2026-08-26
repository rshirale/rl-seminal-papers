# Chapter 5: Proximal Policy Optimization (PPO)

This directory contains the Python implementations for Chapter 5 of **"RL: The Seminal Papers"**. It implements PPO from Schulman et al. (2017), *Proximal Policy Optimization Algorithms*. The runnable experiment verifies the full pipeline on `Pendulum-v1` in about four minutes on a CPU.

TRPO got its stability by solving a constrained optimisation problem: maximise the surrogate objective subject to a hard KL bound, which needs a conjugate-gradient solve and a Fisher-vector product on every update. PPO's claim is that you can keep almost all of that stability and throw away all of that machinery. Clip the probability ratio to `[1 − ε, 1 + ε]` and take the pessimistic minimum, and an ordinary first-order optimiser stops taking the step that would have destroyed the policy — because past the clip the objective goes flat and the gradient goes to zero.

Chapter 4's DDPG was off-policy and deterministic. PPO is on-policy and stochastic: it collects a rollout, updates on it for several epochs, and throws it away. That costs sample efficiency and buys a much more forgiving optimisation problem.

## File Structure

- `actor_critic.py`: `Actor` and `Critic`. The actor emits a `torch.distributions.Normal` whose **mean** is squashed into the action bounds with `tanh`; the standard deviation is a state-independent `log_std` parameter, a modern convention the 2017 paper does not specify. The critic is a separate backbone — the two networks share no parameters.
- `ppo_agent.py`: `PPOAgent`. Computes GAE(λ) advantages, then optimises the clipped surrogate for `k_epochs` passes of minibatches. `update()` returns `(approx_kl, clip_frac)` rather than nothing, because those are the two numbers the chapter teaches you to watch.
- `seeding.py`: `set_seed` and `episode_seed` cover every RNG a run draws from, in one place. Also runnable — it trains one seed at a time and prints the spread.
- `train_pendulum.py`: the runnable Pendulum-v1 experiment. `main()` is importable and returns a `RunResult` of `(returns, approx_kls, clip_fracs)`, so `ablation.py` drives one configuration per call instead of re-implementing the loop.
- `ablation.py`: the clipping ablation behind the chapter's figure 5.9, plus the one-at-a-time sensitivity sweeps behind figure 5.8.
- `plot_efficiency.py`: a DDPG/PPO/SAC sample-efficiency comparison. **Currently orphaned** — see Implementation Notes.
- `Chapter5_PPO.ipynb`: the interactive companion notebook (Colab-ready).
- `__init__.py`: exposes `Actor`, `Critic`, and `PPOAgent`.

## Installation

Chapter 5 needs the full stack (PyTorch + Gymnasium). From the project root:
```bash
make install-full
```

## Running the Experiments

Train on Pendulum-v1:
```bash
# From the project root
make run-ch5-pendulum

# ...or directly, with a fixed seed
python -m src.part_2_methods.ch05_ppo.train_pendulum --seed 0 --episodes 200
```

A converged agent scores around −310 per episode against a random policy's −1200. Expect roughly four minutes for the default 400 episodes on a CPU. Seed 42 reproduces the transcript printed in section 7 of the chapter, line for line.

See what the seed alone is worth, before trusting any comparison:
```bash
make run-ch5-seeding
```

Three runs of the identical configuration score −535.5, −618.8, and −772.7: a **237-point spread** with nothing changed but the RNG. That number is the threshold the chapter's "resolved by 3 seeds?" table is built on, and it is why every experiment here averages over seeds and prints a spread column.

The clipping ablation holds everything but `eps_clip` fixed, so the whole gap is attributable to the clip:
```bash
make run-ch5-ablation

# ...or with your own seeds
python -m src.part_2_methods.ch05_ppo.ablation --seeds 0 1 2 --episodes 200

# add the over-tight eps = 0.05 curve (the chapter's figure 5.9 has all three)
make run-ch5-ablation EXTRA="--include-tight"
```

The sensitivity sweeps vary one hyperparameter at a time around the published configuration, so every curve passes through the same baseline point. Budget about thirty minutes — twelve distinct configurations across three seeds, with the shared baseline run cached rather than retrained:
```bash
make run-ch5-sweep
```

Both write figures only if you ask. `FIGURE_DIR` emits PNG + SVG:
```bash
make run-ch5-ablation FIGURE_DIR=figures EXTRA="--include-tight"  # ch05-figure-clipping
make run-ch5-sweep    FIGURE_DIR=figures                          # ch05-figure-sensitivity
```

Interactive notebook: open `Chapter5_PPO.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch05_ppo/Chapter5_PPO.ipynb).

## Implementation Notes

- **Samples are unbounded; only the mean is squashed.** `Actor.forward` applies `tanh` to the mean and hands back a plain `Normal`. A sample from that distribution can land outside `[−max_action, max_action]`, and `Pendulum-v1` clips it. The alternative — a tanh-squashed distribution with the change-of-variables correction on the log-probability — is what Chapter 6's SAC uses, and it is a different algorithm's design choice, not an omission here. Assert on the mean, not on samples, if you are testing the bound.

- **Truncation is not termination, and here it is not a rounding error.** The loop stores `float(terminated)`, never `float(terminated or truncated)`. Pendulum-v1 never terminates — it only truncates at its 200-step limit — so collapsing the two zeroes the bootstrap at the end of every episode.

  Chapter 4 makes the same distinction and then admits that for DDPG it barely moves the metric: one transition in 200 carries a wrong target, and the critic sees those states again from a hundred other episodes. **PPO is not so forgiving.** GAE accumulates backwards through the rollout, so `(1 − done)` at the final transition does not corrupt one advantage — it truncates the recursion, and every advantage in the episode is computed from a discounted sum that stops early. The bug is the same shape as Chapter 4's; the blast radius is the whole episode.

- **`gamma = 0.9` is deliberate, not a typo for 0.99.** Pendulum episodes are 200 steps of dense reward, and the shorter effective horizon learns markedly faster here. At 400 episodes, `gamma = 0.9` finishes near −354 where `gamma = 0.99` is still around −1074. The sweep in `ablation.py` shows the whole curve.

- **`lr = 1e-3`, not the 3e-4 you will see quoted everywhere.** 3e-4 is the MuJoCo default, and it is the right default for MuJoCo. On a task this small it converges roughly twice as slowly, so the chapter's configuration uses 1e-3. Table 5.4 in the book lists 3e-4 as the standard and flags the difference.

- **One permutation per epoch, not per minibatch.** `np.random.permutation(n)` is drawn once per epoch and then sliced, so an epoch partitions the rollout. Drawing it inside the minibatch loop instead resamples the whole rollout every minibatch — some transitions get used several times per epoch, others not at all. It still trains, which is what makes it a nasty bug.

- **The thread count is part of seeding.** `set_seed` pins `torch.set_num_threads(1)`. Torch's intra-op parallelism changes the order floating-point work is reduced in, so the same seed on an 8-core machine and a 4-core one produces different returns. A chapter that prints an exact transcript cannot reproduce without this. The networks are 64 units wide, so one thread costs nothing in wall time.

- **Seeds overlap between runs.** `episode_seed` is `seed + episode`, so runs 0 and 1 share 199 of their 200 environment start states, shifted by one episode. The runs still differ — weight init and the minibatch permutations differ — but the environment streams are near-copies rather than independent draws. This makes the measured 237-point spread a *lower* bound on the spread across genuinely independent seeds, which is the conservative direction for the claims built on it. Documented rather than fixed, because widening the stride would change every number the chapter prints.

- **Knobs the paper's listing omits.** The loss carries a 0.5 value coefficient and a 0.01 entropy bonus, and gradients are clipped to a global norm of 0.5. All three are standard in reference implementations and none is in the paper's pseudocode.

- **`plot_efficiency.py` is orphaned.** It compares DDPG, PPO, and SAC sample efficiency on one axis, which is a genuinely useful figure — but nothing currently consumes it. The chapter references no such figure (Table 5.3 makes that comparison qualitatively instead), no make target or test covers it, and it is the one file here that does not follow the chapter's conventions: it manipulates `sys.path` directly, seeds by hand instead of calling `seeding.py`, and writes to a **hardcoded absolute path** rather than taking a `--figure` directory like `ablation.py` does. That path points into a second checkout of the manuscript repo, so on a machine with more than one it will silently write the figure somewhere other than the copy being edited. Treat it as unmaintained until it is wired up or removed.

## Troubleshooting

- **My numbers do not match the book.** Check the thread pin first: run `python -c "import torch; print(torch.get_num_threads())"` inside the training process' environment. `train_pendulum.py` pins it to 1, but the notebook is standalone by design and does not. Beyond that, exact reproduction is guaranteed only within a platform and PyTorch version — the trend, not the digits, is what transfers.

- **The agent is not learning.** Read `approx_kl` and `clip_frac` before touching anything. Healthy runs sit around `approx_kl` 0.005–0.03 and `clip_frac` 0.08–0.25. `approx_kl` above 0.05 means the policy is jumping too far — lower `--lr` or `--eps-clip`. `clip_frac` near 0 means the clip never binds and you have no trust region; near 1 means it binds on everything and throttles learning.

- **A hyperparameter change "helped" but I only ran one seed.** It probably did not. Run `make run-ch5-seeding` and compare your gap against the 237-point spread. Anything smaller is a draw.

- **`ModuleNotFoundError: No module named 'src'`.** Run the `-m` form from the project root, not from this directory. The scripts also work when run directly (`python train_pendulum.py`) from inside this directory, which is the fallback the `if __package__` branches exist for — but the two are not interchangeable from the same working directory.

- **The sweep is taking forever.** It is twelve configurations across three seeds. Shrink it while iterating: `python -m src.part_2_methods.ch05_ppo.ablation --sweep --episodes 50 --seeds 0`. The published numbers need the defaults.

- **The Colab install line fails.** The specifiers must stay quoted — an unquoted `>=` is read by the shell as a redirection. The notebook's setup cell already quotes them; keep them that way if you edit it.
