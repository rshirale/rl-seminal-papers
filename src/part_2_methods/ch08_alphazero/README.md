# Chapter 8: AlphaGo and AlphaZero

This directory contains the Python implementations for Chapter 8 of **"RL: The Seminal Papers"**. It covers Silver et al. (2016), *Mastering the game of Go with deep neural networks and tree search*, and Silver et al. (2018), *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play*. The runnable experiment trains an AlphaZero agent from random weights, by self-play, on a board whose correct answer is a matter of published record.

Chapters 3 through 7 all learned control from an environment that hands back a reward. This chapter does something different: it learns to *search*. AlphaGo's move is not chosen by a policy network — it is chosen by a tree search that the networks make affordable. The policy network says which handful of the 250 legal moves are worth looking at, the value network says how a position is going without playing it out, and Monte Carlo tree search spends its budget where those two agree. Neither network plays Go. Together they turn an intractable search into a tractable one.

AlphaZero is what is left after you delete everything specific to that design. No human games, no hand-built features, no rollout policy, no second network — one residual tower with two heads, trained on nothing but its own play. It was stronger than AlphaGo within a day, and the same code with no changes learned chess and shogi.

That second result is the one this directory is built around, because it is the one a laptop can demonstrate. The agent here plays 6x6 Connect Four and 6x6 Othello through a single interface, with one set of hyperparameters and not a line of the agent differing between them. And both boards are *solved*: van den Herik, Uiterwijk and van Rijswijck (2002) published their game-theoretic values, so the question "did it work" has an answer rather than a loss curve.

## File Structure

**The runnable AlphaZero project**

- `game.py`: `Game` — the abstract interface every domain satisfies, and the seam the whole chapter turns on. Positions are canonical: the side to move is always `+1` and `apply_move` negates the board on the way out, so the agent never asks whose turn it is.
- `connect_four.py`: `ConnectFour` — any board size, with the published solved values. Rules in about forty lines, and a precomputed win-line table that makes the terminal test seventeen times cheaper than the obvious implementation.
- `othello.py`: `Othello` — 6x6, with pass handling and the eight symmetries of the square. The second domain, and the proof that the interface is doing real work.
- `network.py`: `AlphaZeroNet` and `ResBlock` — the residual tower with a policy head and a value head, at 456,647 parameters. `build` shapes it from the game; `pick_device` finds the accelerator.
- `mcts.py`: `BatchedMCTS` and `Node` — PUCT evaluated by the network, driven in lockstep across several hundred games so their leaves share one forward pass. Child boards are built lazily.
- `selfplay.py`: `Example` and `play_batch` — a fleet of games advancing together, with tree reuse between moves. The sign convention in the labelling comment is the most consequential dozen lines here.
- `arena.py`: `play_match`, `mirror_match` and `evaluate_agent` — the outside opinions, and a written argument about which of them to believe.
- `solver.py`: `solve`, `negamax` and `best_action` — exact alpha-beta with a transposition table. The oracle, and the depth-limited baseline opponent. Runnable on its own.
- `train.py`: `RunResult` and `main` — the budgeted training loop, and the chapter's entry point. Stops on wall clock, not on iteration count.
- `inspect_agent.py`: `load`, `report` and `play` — open a checkpoint, see what it believes about the opening against the proven value, and play it.
- `animate.py`: `animate_search` and `animate_game` — matplotlib animations of the search concentrating, for the notebook.

**The AlphaGo listings**

- `alphago.py`: `PolicyNetwork`, `ValueNetwork` and `PublishedAlphaZeroNet` — the published 19x19 architectures at their published sizes. Instantiate and measure these; do not try to train them. Runnable, and prints all four parameter counts.
- `benchmark.py`: `RolloutPolicy` and the timing measurement — the one AlphaGo experiment that needs no dataset, no training and no cluster.

**Supporting files**

- `Chapter8_AlphaZero.ipynb`: the interactive companion notebook, Colab-ready. Its code cells *are* the modules above, verbatim, and a test enforces that.
- `__init__.py`: re-exports `Game`, `ConnectFour` and `Othello` — the numpy-only half. The agent needs torch, and the runnable modules stay out so `python -m` does not warn about a double import.

## Installation

```bash
make install-full
```

That is torch, numpy and matplotlib — the same stack chapters 3 through 6 use, with nothing added. There is no dataset to download and no model weights to fetch; the agent starts from random weights and generates everything it learns from.

Three of the five targets below (`run-ch8-sizes`, `run-ch8-solve`, `run-ch8-benchmark`) need no accelerator and finish in under a minute. Run them before the trainer, in that order: they establish what the published system cost, what the right answer is, and why AlphaGo's search was shaped the way it was.

## Running the Experiments

**What the published networks cost, measured rather than quoted:**

```bash
make run-ch8-sizes
```

Instant. Instantiates all three published architectures and this project's, and counts their parameters. The published AlphaGo Zero tower is 22,837,864 parameters; the one that trains here in an hour is 456,647. Same architecture, two scale knobs, a factor of fifty.

**Reprove the small boards exactly:**

```bash
make run-ch8-solve
make run-ch8-solve EXTRA="--boards 4x4"      # just the instant one
```

Alpha-beta with a transposition table settles 4x4, 4x5 and 6x4 and checks each against van den Herik and colleagues (2002). About thirty seconds for the three; 6x5 and larger are hours of pure Python, which is why the chapter cites the table for the rest. Run this before the trainer — it is what makes the agent's convergence measurable instead of merely plausible.

**The AlphaGo timing measurement:**

```bash
make run-ch8-benchmark
make run-ch8-benchmark EXTRA="--device mps"     # or cuda
```

About fifteen seconds. Times the 19x19 policy network against a sparse linear rollout policy, reproducing the gap Silver and colleagues (2016) report — 3 ms against 2 microseconds — that forced AlphaGo to keep a rollout policy at all. On CPU, batching wins nothing, because one position already saturates every core; pass a GPU or MPS to see the number the project's batched self-play depends on.

**Train the agent:**

```bash
make run-ch8-train

# ...or directly, with the chapter's settings spelled out
python -m src.part_2_methods.ch08_alphazero.train --rows 6 --cols 6 --budget-seconds 3600 --seed 0
```

The default board is **6x6**, whose game-theoretic value is a proven draw. That choice is about what an hour can honestly demonstrate: on a drawn board, an agent approaching perfect play ends up drawing against itself every time, and `mirror_draw_rate` measures that directly. Pass `--cols 7` for the standard first-player-win board.

The loop stops on wall clock rather than after a fixed number of iterations, so the budget is the parameter and the iteration count is the outcome. One line prints per iteration; everything lands in `runs/<game>_<timestamp>/history.json` beside a `latest.pt` checkpoint.

**The same agent on a second game:**

```bash
make run-ch8-othello
```

Identical agent, identical hyperparameters, a different `Game`. 6x6 Othello is a proven *second*-player win, so a trained root value should go negative — the opposite sign from Connect Four, learned from nothing but self-play. This is AlphaZero's generality claim in the only form available to us.

**Look at what it learned:**

```bash
python -m src.part_2_methods.ch08_alphazero.inspect_agent runs/<run>/latest.pt
python -m src.part_2_methods.ch08_alphazero.inspect_agent runs/<run>/latest.pt --play
```

Any training run can be shortened with `EXTRA="--budget-seconds 900"`, and every flag is on the trainer's `--help`. The knobs that trade the most:

| Flag | Default | What it trades |
| --- | --- | --- |
| `--rows` / `--cols` | 6 / 6 | The board. `--cols 7` is the standard first-player win |
| `--games-per-iteration` | 192 | Fleet size. Larger fleets use the accelerator better but update the network less often |
| `--simulations` | 50 | Search depth per move. Better policy targets, linearly more time |
| `--channels` / `--blocks` | 64 / 4 | Network capacity. The published tower is 256 / 19 |
| `--budget-seconds` | 3600 | When to stop |
| `--device` | auto | `cpu`, `mps` or `cuda` |

Interactive notebook: open `Chapter8_AlphaZero.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch08_alphazero/Chapter8_AlphaZero.ipynb). It carries the same modules, runs a 30-minute budget by default, and animates the search.

## What a run actually produces

Reproduced on 2026-09-22 on an x86 Mac with Apple MPS and 16 cores, under
`caffeinate -i`, with:

```bash
python -m src.part_2_methods.ch08_alphazero.train --rows 6 --cols 6 --budget-seconds 900 --seed 0
```

| it | elapsed | loss | value loss | decisive | root value | vs random | vs depth-3 | mirror draws |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| 1 | 93 | 2.230 | 0.558 | 0.880 | +0.151 | — | — | — |
| 2 | 151 | 1.906 | 0.358 | 0.812 | +0.140 | 39/0/1 | 29/3/8 | 0.00 |
| 3 | 255 | 1.845 | 0.351 | 0.844 | +0.172 | — | — | — |
| 4 | 313 | 1.791 | 0.346 | 0.828 | +0.029 | 40/0/0 | 35/1/4 | 0.00 |
| 5 | 422 | 1.762 | 0.357 | 0.807 | +0.121 | — | — | — |
| 6 | 482 | 1.724 | 0.355 | 0.807 | +0.034 | 39/1/0 | 36/0/4 | 0.20 |
| 7 | 586 | 1.700 | 0.355 | 0.802 | +0.033 | — | — | — |
| 8 | 644 | 1.679 | 0.359 | 0.812 | +0.048 | 40/0/0 | 35/2/3 | 0.30 |
| 9 | 752 | 1.663 | 0.365 | 0.833 | +0.097 | — | — | — |
| 10 | 814 | 1.635 | 0.358 | 0.771 | +0.051 | 40/0/0 | 36/2/2 | 0.30 |
| 11 | 923 | 1.612 | 0.353 | 0.740 | +0.070 | — | — | — |

Eleven iterations in 923 seconds, which is roughly 43 in the default one-hour
budget. Read it in this order:

**Total loss falls monotonically**, 2.230 to 1.612, without a single bounce.
Value loss does *not* — it drops sharply in the first iteration and then wanders
between 0.346 and 0.365, which is what a value head looks like while the
distribution it is fitting keeps moving underneath it.

**The agent stops losing to a random player after iteration 2** — 39/0/1, then
40/0/0, 39/1/0, 40/0/0, 40/0/0 — and climbs from 0.762 to 0.925 against depth-3
exact search. Both saturate, `vs_random` almost immediately, which is exactly
why neither is the metric the chapter argues about.

**`decisive_fraction` drifts down**, 0.880 to 0.740. On a board that is a proven
draw, fewer games settled by a blunder is the right direction.

**`root_value` stays near zero and tells you almost nothing.** It is the correct
answer for this board, and it is also what the untrained network emitted at
iteration 0. This column is in the table as an illustration of a weak metric.

**`mirror_draw_rate` climbs from 0.00 to 0.30.** This is the one that means
something, and it is also the one that says the run is unfinished: perfect play
on 6x6 draws with itself 100% of the time.

That last number is worth measuring properly at the end rather than reading off
the last iteration. On the final checkpoint, 32 games each:

| Simulations per move | Draw rate |
| ---: | ---: |
| 50 | 0.188 |
| 200 | 0.375 |

Four times the search doubles the draw rate. That responsiveness is what a real
measurement looks like, and it is what the broken version of this metric — the
one that could only return 0.0 or 1.0 — could never have shown.

**The honest conclusion:** fifteen minutes buys an agent that dominates both
baselines and is nowhere near solving the board. That gap is not a failure. It
is the most useful thing here, because it gives a reader something concrete to
push on, and it is only visible because 6x6 has a published answer. An agent
measured by "the loss went down" would have looked finished at iteration 11.

## Implementation Notes

- **The metric to trust is `mirror_draw_rate`, not `root_value`.** The value head's opinion of the empty board is the obvious convergence signal and the weaker one. It averages over self-play in which both sides are still imperfect, so on 6x7 it hovers near zero for a dozen iterations while the agent becomes genuinely strong — and on a drawn board, near zero is also exactly what an untrained `tanh` head emits. It cannot distinguish a converged agent from an ignorant one. The mirror match can: on a proven draw, an agent approaching perfect play draws every game, an untrained agent scores exactly 0.0, and nothing about the number can be faked by a network that has learned to output zero. Choosing 6x6 over 6x7 was driven entirely by this.

- **The mirror match samples its opening plies, and the reason is a bug.** The first version played greedily with the search noise off from move one, which made all 32 "games" the same deterministic game repeated 32 times. The rate could only ever be exactly 0.0 or exactly 1.0. It reported 1.0, 0.0, 1.0, 0.0, 0.0 across a run — a coin flip dressed as a measurement — and that pattern is what gave it away. Sampling the first four plies from the search distribution fixed it, and the fixed version *responds to search depth*, which the broken one never could. A test pins both halves of this.

- **Search is batched across games, not within one game.** A laptop GPU answers 256 positions in about the same wall time as one — measured at 17.5 ms either way on Apple MPS — so playing a single game at a time wastes almost all of the hardware. Several hundred games advance in lockstep and share every forward pass. This is the same idea the chapter describes for AlphaGo's distributed search, doing real work rather than illustrating. On CUDA the balance between network time and Python time shifts, and the fleet buys less.

- **Positions are always canonical, and `apply_move` negates the board.** The side to move is `+1` and its opponent `-1`, in every module, at every ply. It removes an entire category of sign bug and is why `Game` has no `current_player` at all.

- **The self-play sign convention has a test because its failure is silent.** `outcome` reports the result for the player to move in the *terminal* position — the one who did not make the last move — and `play_batch` walks backwards from there flipping at every ply. Get it backwards and training still runs, the loss still falls, and the agent learns to lose. Nothing else in the loop notices.

- **Terminal tests are plain Python, deliberately.** The first implementation cost 87 microseconds per call, which is fatal when one self-play iteration asks the question millions of times. Precomputed win-line index tuples brought it to 5.1 microseconds — five times faster than the NumPy version, because these arrays are far too small to repay NumPy's per-call overhead.

- **Child boards are lazy and trees are reused.** Expanding a node creates one child per legal move, but a search visits only a few, so children store the move that makes them and compute the board on first visit. And once a move is played, the subtree beneath it already carries visit statistics, so it becomes the next root instead of being rebuilt. Dirichlet noise is applied to the root only, and only once per search, so reuse cannot compound it.

- **There is no arena gate, and that is historically accurate.** AlphaGo Zero (2017) kept a champion network and required a 55% win rate to promote a challenger. The generalized AlphaZero (2018) removed the evaluation step entirely and continuously updated one network, which is what happens here. If you extend this to much longer runs on harder games, a gate is worth adding back to prevent regression.

- **`network.py`'s `AlphaZeroNet` and the `PublishedAlphaZeroNet` in `alphago.py` are the same architecture at different budgets.** 19 blocks by 256 channels against 4 by 64; 22,837,864 parameters against 456,647. The naming is deliberate and the contrast is the chapter's argument: what a laptop cannot afford is the scale, not the algorithm.

- **The notebook's code cells are these modules, byte for byte.** Every earlier chapter's notebook re-implements its algorithm inline, and every one of them has drifted from its modules at least once. This one inlines the files instead, with only their relative imports removed, and `tests/test_ch08_notebook.py` compares them character for character. There is no "the notebook is a simplified version" escape hatch.

- **The constants are named for their games.** `CONNECT_FOUR_SOLVED_VALUES` and `OTHELLO_SOLVED_VALUES` rather than `SOLVED_VALUES` twice, because the notebook inlines both modules into one namespace and two constants of the same name silently become one — after which Connect Four reports Othello's answers.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`.** Run the `-m` form from the repository root, not from this directory. Every module here uses relative imports, so `python train.py` will not work; use `python -m src.part_2_methods.ch08_alphazero.train` or the `make` target.

- **The run is much slower than the table above.** Check `--device` first: the banner prints what it picked. On a machine with no accelerator the fleet buys much less, and `--games-per-iteration 64 --simulations 25` is a reasonable laptop-CPU setting — it makes each iteration worse but there are more of them. Most of the wall clock here is Python, not matrix multiplication, so a faster GPU helps less than you would expect.

- **One iteration takes wildly longer than the ones around it.** Almost certainly the machine sleeping or deprioritising the process while it runs unattended; a 35x jump with no algorithmic cause turned up exactly this way during development. On macOS, run under `caffeinate -i`. The budget check itself is not affected — it stops the loop correctly either way.

- **`root_value` is near zero and I cannot tell if anything is learning.** On 6x6 that is the *correct* answer, which is the problem — see the first implementation note. Read `mirror_draw_rate` and `score_depth3` instead. If both of those are also flat after four or five iterations, something is genuinely wrong; if the loss is falling and `vs_random` is not yet 40/0/0, it is just early.

- **`mirror_draw_rate` is exactly 0.0 or exactly 1.0 every time.** That is the coin-flip bug described above, and it means `opening_plies` has been set to 0 or the sampling has been removed from `mirror_match`. It is not a property of the agent.

- **The agent looks strong but `mirror_draw_rate` is stuck around 0.1.** That is the expected result of an hour, not a failure. Perfect play on 6x6 draws 100% of the time; twelve iterations buys something that dominates both baselines and is nowhere near solving the board. The gap is the chapter's most useful result, and it is only visible because the board has a proven answer.

- **`make run-ch8-solve` appears to hang.** It is solving a board it cannot finish. 4x4, 4x5 and 6x4 take seconds; 6x5 and 4x6 take hours in pure Python and 6x7 is out of reach entirely — about 10^14 positions. Stick to `--boards 4x4,4x5,6x4` unless you are prepared to wait.

- **`make run-ch8-benchmark` reports "batching wins back 1x".** Expected on CPU: a single 19x19 forward pass already saturates every core, so there is nothing for a batch to reclaim. Pass `--device mps` or `--device cuda` to see the real number, which is what the project's batched self-play depends on.

- **The notebook parity test fails after I edited a module.** That is the test working. The notebook cell is generated from the module, so edit the module and regenerate the cell rather than editing the notebook in place; the failure message names the file.
