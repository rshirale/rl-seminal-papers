# Chapter 6: Soft Actor-Critic (SAC)

This directory contains the Python implementations for Chapter 6 of **"RL: The Seminal Papers"**. It implements SAC-v2 from Haarnoja et al. (2018b), *Soft Actor-Critic Algorithms and Applications*, with the stochastic actor and twin critics of Haarnoja et al. (2018a). The runnable experiment converges on `Pendulum-v1` in about ten minutes on a CPU.

Chapter 4's DDPG was data-efficient and brittle: a deterministic actor with noise bolted on the outside, prone to committing to one behavior and never finding a better one. Chapter 5's PPO was stable and wasteful: every gradient step burned a fresh batch of experience. SAC's claim is that you do not have to choose. Keep the replay buffer, make the actor stochastic, and then — this is the part that matters — put the policy's own entropy *into the objective*, so the agent is paid to keep its options open rather than talked into it by an external noise schedule.

Everything else follows from that one change. The actor has to be a distribution, because a Dirac delta has entropy of negative infinity and the objective would be undefined. The entropy term rides along inside the Bellman backup, so a state is valuable partly because many good actions are available from it. And the temperature that weighs entropy against reward turns out to be the Lagrange multiplier of an entropy constraint, so SAC-v2 learns it instead of asking you to tune it.

## File Structure

- `actor.py`: `Actor`. A shared two-layer backbone feeding separate mean and log-std heads. Sampling is reparameterized (`rsample`), squashed through `tanh`, and scaled by `max_action`; the returned log-probability carries the tanh change-of-variables correction. Note that `log_std` is a *head* — the spread depends on the state, unlike Chapter 5's state-independent scalar.
- `critic.py`: `Critic`. Two independently initialized Q-networks built by the same local factory and returned together from one `forward`, so the caller can take `min(Q1, Q2)` without crossing the class boundary twice.
- `replay_buffer.py`: `ReplayBuffer`. The same fixed-capacity deque as Chapters 3 and 4. Unchanged in purpose; what changes is that SAC's soft Bellman backup makes every stored transition valid training data forever.
- `sac_agent.py`: `SACAgent`. Owns the actor, the twin critics, their frozen targets, and the buffer. `train()` runs the three gradient steps of Algorithm 1's inner loop — critic, actor, temperature — then the soft target update. The `auto_alpha`, `init_alpha` and `reward_scale` switches exist to reproduce the chapter's exercises; all three default to the published configuration.
- `seeding.py`: `set_seed` and `seed_env` cover every RNG a run draws from, in one place. Also runnable — it trains one seed at a time and prints the spread.
- `train_pendulum.py`: the runnable Pendulum-v1 experiment. `main()` is importable and returns the per-episode returns, so `ablation.py` drives one configuration per call instead of re-implementing the loop. Budgets in environment **steps**, not episodes.
- `ablation.py`: the chapter's three exercises — the entropy ablation, the fixed-temperature comparison, and the reward-scale experiment — each as a table plus an optional figure.
- `Chapter6_SAC.ipynb`: the interactive companion notebook (Colab-ready).
- `__init__.py`: exposes `Actor`, `Critic`, `ReplayBuffer`, and `SACAgent`. Not `seeding` — it is runnable as a module, and importing it at package scope makes `python -m ...seeding` emit a double-import warning.

## Installation

Chapter 6 needs the full stack (PyTorch + Gymnasium). From the project root:
```bash
make install-full
```

## Running the Experiments

Train on Pendulum-v1:
```bash
# From the project root
make run-ch6-pendulum

# ...or directly, with the chapter's seed
python -m src.part_2_methods.ch06_sac.train_pendulum --seed 42 --steps 50000
```

Expect roughly ten minutes for the default 50,000 steps on a CPU — SAC takes a gradient step per environment step, which makes it the slowest runner in the book and the most sample-efficient one. Those two facts are not in tension; see the note on environment steps below.

Seed 42 is the run the chapter's "Expected training output" section and figure 6.6 are drawn from. What reproduces is the shape and the temperature: reward flat at the random-policy baseline with α pinned at exactly 1.000 for the first 50 episodes, a sharp climb once updates begin, and α decaying from 1.0 to roughly 0.04. On the machine this README was written on, α landed on 0.462 at episode 65 and 0.276 at episode 80 against the chapter's 0.460 and 0.276, and the final 50-episode median was −126.4 against the chapter's "near −125". Individual episode returns differ, and are supposed to — see the first troubleshooting entry.

See what the seed alone is worth, before trusting any comparison:
```bash
make run-ch6-seeding
```

The chapter's own caveat on figure 6.12 is that SAC's error bars on Hopper are roughly half its mean, and that "a single SAC run on that task tells you very little." This target puts a number on that for Pendulum, and it is the threshold every table below should be read against.

The three exercises at the end of the chapter each have a target. **Exercise 1** removes the entropy bonus entirely — `alpha = 0` with the temperature update skipped — while leaving the twin critics, the replay buffer, the soft target updates, and the stochastic actor in place, so the whole gap is attributable to the maximum entropy objective:
```bash
make run-ch6-ablation

# ...or with your own seeds
python -m src.part_2_methods.ch06_sac.ablation --seeds 0 1 2 --steps 30000
```

**Exercise 2** reverts to SAC-v1's hand-set temperature at the three values the chapter names, and measures all of them against the learned one:
```bash
make run-ch6-temperature
```

**Exercise 3** multiplies rewards by 10 before they reach the Bellman target and reruns both regimes. Reward scale is an implicit inverse temperature, so the fixed-temperature runs need retuning and the learned one retunes itself — which is the whole argument for SAC-v2:
```bash
make run-ch6-reward-scale
```

All three take an hour or more between them, so they also run on GitHub's machines: **Actions → Chapter 6 SAC ablations → Run workflow**, which executes the three experiments in parallel and attaches the tables and figures as artifacts. Absolute numbers from a CI runner will not match a local run — reduction order differs by CPU — but every variant inside one run shares a machine, so the comparison the ablation is making holds. The seed-42 transcript check stays local.

Every figure-producing target writes only if you ask. `FIGURE_DIR` emits PNG + SVG:
```bash
make run-ch6-ablation      FIGURE_DIR=figures  # ch06-figure-entropy
make run-ch6-temperature   FIGURE_DIR=figures  # ch06-figure-temperature
make run-ch6-reward-scale  FIGURE_DIR=figures  # ch06-figure-reward-scale
```

Compare SAC's sample efficiency against DDPG and PPO on one axis. That figure lives in Chapter 5 because it spans three chapters:
```bash
make run-ch5-efficiency
```

Interactive notebook: open `Chapter6_SAC.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch06_sac/Chapter6_SAC.ipynb).

## Implementation Notes

- **The tanh correction is the whole ballgame, and it is invisible when wrong.** `log π(a|s) = log p(u|s) − Σ log(1 − tanh²(uᵢ))`. Drop the second term and the agent still trains, still produces a plausible-looking curve, and is optimizing against a biased entropy estimate the entire time — which corrupts both the actor loss and the temperature update. The chapter calls this the most common implementation bug in SAC. `test_sac.py` checks it numerically against the change-of-variables identity rather than by grepping for the line, because a wrong-but-present correction would pass a grep.

  The code writes that term as `2 * (log 2 − u − softplus(−2u))`. It is algebraically identical to `log(1 − tanh²(u))` and does not underflow: once `|u|` is past about 9, `tanh²(u)` rounds to exactly 1.0 in float32 and the naive form returns `−inf`. This is the form the authors' own implementation uses.

- **The action rescaling is deliberately left out of that correction.** Pendulum actions span `[−2, 2]`, so the actor scales its squashed output by `max_action`, and a fully correct change of variables would subtract another `dim(A) · log(max_action)`. Listing 6.1 omits it and so does this module, because matching the book is the point. What the omission costs: the term is constant with respect to φ, so the actor's gradient is unchanged, but it does shift the log-probability by a fixed offset — which means the temperature settles at a different value than it would with the term restored, since α is chasing a fixed entropy floor `H̄ = −dim(A)`. On a one-dimensional action that is harmless. Restore it (Stable-Baselines3 and CleanRL both do) before trusting `H̄` in a high-dimensional action space.

- **`random.seed` is not optional here, and it was missing.** SAC draws its minibatch through `random.sample` inside the replay buffer. Seed torch and numpy only — which is what this chapter's trainer did before `seeding.py` existed — and every run starts from identical weights and then diverges at the first gradient step, which is the most confusing possible way for a run to be unreproducible. `set_seed` covers `random`, `numpy`, and `torch`; `seed_env` covers the environment stream and `action_space.sample()`, which matters more here than anywhere else in the book because the warmup is 10,000 uniform draws — a fifth of the default budget.

- **The thread pin is part of seeding.** `set_seed` pins `torch.set_num_threads(1)`, for the same reason Chapter 5 does: torch's intra-op parallelism changes the order floating-point work is reduced in, so the same seed on an 8-core machine and a 4-core one produces a different transcript. Chapter 5's `plot_efficiency.py` used to pin the thread count on this chapter's behalf because this module did not exist. It no longer has to.

- **Truncation is not termination.** The loop stores `float(terminated)`, never `float(terminated or truncated)`. Pendulum-v1 never terminates — it only truncates at its 200-step limit — so collapsing the two would zero the bootstrap on the last transition of every episode, telling the critics that states at t=200 are worth nothing. Chapters 4 and 5 make the same distinction, and Chapter 5's README explains why the blast radius differs by algorithm.

- **Return cannot tell the ablation variants apart; the policy can.** This is the most surprising thing in the chapter's exercises, and it is worth knowing before you run them. On Pendulum-v1 every variant here — entropy bonus off, α fixed at 0.01/0.2/1.0, rewards scaled by ten — converges to roughly the same episode return, within a few points of −122. The task is forgiving enough that a near-deterministic policy and a broadly stochastic one both hold the pole up.

  The mechanism is still there; it is just not in the return. Measured on short probe runs of 6,000 to 12,000 steps, σ moves from **0.008** with the bonus off to **0.74** at α = 1.0, and entropy from about **−7** to **0**. So `ablation.py` reports `sigma`, `entropy` and `alpha` beside the return, and `train_pendulum.main()` hands back a `RunResult` carrying them. Read those columns first. `entropy` is `E[−log π]`, the quantity the temperature regulates, so compare it directly against the SAC-v2 target of `−dim(A) = −1` — a fixed α = 0.2 run landed at −0.998 on a 6,000-step probe, which is why the auto-tuner converges near 0.2.

  Note also what the α = 0 run demonstrates: σ = 0.008 is the policy collapsing toward a Dirac delta, the exact failure the maximum entropy objective exists to prevent. Differential entropy has no floor at zero, so it plunges rather than bottoming out — the chapter's point about densities, visible in a table.

- **Judge convergence by the median, not the mean.** Pendulum-v1 resets the pole to a uniformly random angle, so roughly one episode in ten starts close enough to upright that a converged policy has almost nothing to correct and scores near zero. The chapter's transcript has one of those at episode 200 (`−1.6`). Both the trainer's final summary and `ablation.py`'s score column use the median for this reason. A mean over the same window looks better than the policy is.

- **The agent owns its networks; listing 6.4 is handed them.** `SACAgent.__init__` in the book takes a pre-built actor, critic and target critic, and `update()` takes the replay buffer as an argument. This module instead constructs all four from `state_dim`/`action_dim`/`max_action` and exposes `select_action`/`store`/`train`, which is the shape Chapters 3 and 4 already use. The listing's shape makes the composition visible on the page, which is what a listing is for; the module's shape means every trainer in `src/` is called the same way. The arithmetic inside `train()` is line-for-line the listing's.

- **`alpha = 0` is a real configuration, `auto_alpha=True` with `init_alpha=0` is not.** The temperature is optimized as `log α`, so a zero starting point has no finite gradient. Exercise 1's variant passes `auto_alpha=False` alongside it; the combination without that is rejected in the constructor rather than left to produce NaNs several hundred steps in.

- **The entropy ablation is not DDPG.** Turning off the entropy bonus leaves the actor stochastic — it still samples — so the run keeps a source of exploration. What it loses is any pressure to *maintain* that spread. This matches how the paper's own ablation reports the result: the deterministic variant's problem shows up as seed-to-seed variability before it shows up in the mean, which is why `ablation.py` prints a spread column and why `seeding.py` exists to calibrate it.

- **SAC needs no target actor.** DDPG and TD3 both keep one; TD3 additionally injects noise into the target action to smooth the target. SAC gets that for free, because the next-state action is freshly sampled from a policy that is stochastic by construction. There is no `actor_target` here, and its absence is asserted in the tests so nobody adds one back on autopilot.

- **Environment steps are not wall-clock.** SAC reaches a given return from fewer environment steps than DDPG or PPO and takes longer in wall-clock time to do it, because it runs a gradient update per step where PPO runs a few epochs per rollout. Both statements are true at once, and Chapter 5's `plot_efficiency.py` plots the first one. Do not read that figure as a speed comparison.

- **A GPU helps less than you would expect.** The networks are two-layer 256-unit MLPs. Wall-clock time on Pendulum is dominated by the simulator and the one-update-per-step schedule, not by matrix multiplication, and the per-step CPU↔GPU transfer can make a GPU run slower. The chapter says the same thing about the MuJoCo tasks.

## Troubleshooting

- **My numbers do not match the book.** They will not match line for line, and the chapter's transcript should be read as a shape rather than a checksum. The script that produced it ran on a different thread count from the one `set_seed` now pins, and floating-point reduction order alone is enough to separate two runs that agree on every seed. What should transfer: α holds at exactly 1.000 through the 50-episode warmup, then falls monotonically into the 0.03–0.06 range; reward is flat near −1200 during warmup and reaches the −120s within about 30 episodes after it; the final 50-episode median lands near −125.

  If the *shape* is wrong rather than the digits, check the thread pin: run `python -c "import torch; print(torch.get_num_threads())"` inside the training process' environment. It should be 1. `train_pendulum.py` pins it through `set_seed`, and the notebook pins it directly in its setup cell — the notebook is standalone by design and imports nothing from `src/`, so it duplicates the line rather than sharing it.

- **α collapses toward zero and the reward stops improving.** The learning rate is too high — lower `--actor-lr` and `--alpha-lr` together. α falling steadily is *normal*: the policy starts far more random than `H̄ = −1`, so the auto-tuner spends the whole run relaxing the entropy bonus, from 1.0 down to roughly 0.05. Falling and then flattening while reward climbs is a healthy run.

- **α grows continuously while reward stagnates.** The entropy target is too aggressive for the task. `--target-entropy` overrides the `−dim(A)` heuristic; try a more negative value, which asks the policy to be less random.

- **Nothing happens for the first 50 episodes.** That is the warmup, and it is supposed to look like that. The first 10,000 steps are uniform random actions with no gradient updates at all, which is 50 episodes of 200 steps. Reward sits near the random-policy baseline of −1200 and α holds at exactly 1.000 until the first update.

- **`ValueError: init_alpha must be positive when auto_alpha=True`.** You asked for a learned temperature starting at zero. Pass `--no-entropy` (which sets both switches) rather than `--alpha 0`.

- **A change "helped" but I only ran one seed.** It probably did not. Run `make run-ch6-seeding` and compare your gap against the spread it prints. Anything smaller is a draw. This chapter's own figure 6.12 makes the point with error bars.

- **`ModuleNotFoundError: No module named 'src'`.** Run the `-m` form from the project root, not from this directory. The scripts also work when run directly (`python train_pendulum.py`) from inside this directory, which is the fallback the `if __package__` branches exist for — but the two are not interchangeable from the same working directory.

- **The ablation is taking forever.** It is two to four full training runs per seed, and SAC updates once per environment step. Shrink it while iterating: `python -m src.part_2_methods.ch06_sac.ablation --steps 6000 --seeds 0`. The warmup shrinks with it, so short runs still take gradient steps. The published numbers need the defaults.

- **The Colab install line fails.** The specifiers must stay quoted — an unquoted `>=` is read by the shell as a redirection. The notebook's setup cell already quotes them; keep them that way if you edit it.
