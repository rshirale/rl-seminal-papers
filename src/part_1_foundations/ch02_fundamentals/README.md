# Chapter 2: Reinforcement Learning Fundamentals

This directory contains the Python implementations and benchmarks for Chapter 2 of **"RL: The Seminal Papers"**. It covers the tabular methods the rest of the book generalizes: TD(0) for value estimation, and Q-Learning and SARSA for control, on two classic gridworlds.

These are the last algorithms in the book with no neural network in them. A table of values, one entry per state, is enough when the state space is twelve squares — and holding onto that mental picture is what makes the function approximation of Chapter 3 legible, because a Q-network is doing the same job as this dictionary, only for a state space too large to enumerate.

The chapter's real payload is the Cliff Walking comparison, which shows two algorithms that differ by a single term in one line converging on visibly different personalities.

## File Structure

- `environments.py`: `GridWorld` (the 4×3 value-estimation grid) and `CliffWalking` (the 12×4 Sutton & Barto task). Both expose one method, `transition(state, action) -> (next_state, reward, done)`, deliberately simpler than the Gymnasium API used everywhere else in the book.
- `algorithms.py`: `td0` for prediction, and `run_agent` for control — one function covering both Q-Learning and SARSA, selected by `mode`, so the difference between them is visible as a two-line branch rather than two separate files.
- `run_td0_gridworld.py`: random-policy outcome statistics plus the converged TD(0) value table for the 4×3 grid.
- `run_cliff_benchmark.py`: the Q-Learning versus SARSA comparison, averaged over 100 runs, writing `cliff_results.png`.
- `Chapter2_Fundamentals.ipynb`: the interactive companion notebook (Colab-ready).
- `requirements.txt`: this chapter alone needs only NumPy, Matplotlib, and pandas — no PyTorch, no Gymnasium.

## Installation

Chapter 2 is pure NumPy, so it can be installed on its own:
```bash
python -m pip install -r requirements.txt
```

Or use the book-wide Foundation stack from the project root:
```bash
make install
```

## Running the Benchmarks

The TD(0) value estimation on the 4×3 grid — instant, and seeded, so it prints the same table every time:
```bash
# From the project root
make run-ch2-gridworld

# ...or directly, from this directory
python run_td0_gridworld.py
```

The Cliff Walking comparison — about **80 seconds**, since it averages Q-Learning and SARSA over 100 runs of 500 episodes each:
```bash
make run-ch2-cliff
```

It writes `cliff_results.png` into the working directory: a smoothed reward curve for both agents, and a bar chart of how often each fell off the cliff. `make clean` deletes it.

Interactive notebook: open `Chapter2_Fundamentals.ipynb` locally, or in [Google Colab](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_1_foundations/ch02_fundamentals/Chapter2_Fundamentals.ipynb).

## Implementation Notes

- **The goal state prints as `0.00`, and that is correct.** The converged value table shows `0.00` at the goal `(3,2)` and at both hazards, which looks wrong when the goal is worth +10. Terminal states are never updated: TD(0) updates the value of the state you are *leaving*, and once you enter a terminal the episode ends, so those entries keep their initialized zero forever. The +10 is not lost — it lives in the reward, and reaches the table through the neighbours. `(2,2)`, the square adjacent to the goal, converges to about −2.26, the highest value on the board, while the far corner sits near −14.

- **TD(0) here is prediction, not control.** `td0` picks actions with `np.random.choice(env.actions)` and never consults its own value estimates. It is answering "what is this random policy worth?", not "what should I do?". This is why the table is negative nearly everywhere: over 500 episodes the random policy reaches the goal **5.0%** of the time and falls into the `(1,1)` hazard **80.8%** of the time. Those numbers are what the value table is faithfully reporting.

- **Q-Learning and SARSA differ by one term, and the code shows exactly where.** Both call the same `choose_action` and follow the same ε-greedy behaviour. The only difference is the bootstrap: Q-Learning takes `max(Q[next_state, a])` over all actions, while SARSA takes `Q[next_state, next_action]` — the action it will actually take. Off-policy versus on-policy, in one line. The behavioural consequence is the whole point of the Cliff Walking figure: Q-Learning learns the optimal path along the cliff edge and keeps falling off it during training, because its ε-greedy exploration occasionally steps into the void that its *target* policy would never enter. SARSA's target includes the exploration, so it learns a safer path further from the edge. The gap is large and not subtle: over 20 runs here Q-Learning fell on **37.1%** of episodes against SARSA's **16.9%**, better than two to one.

- **`run_td0_gridworld.py` is seeded; `run_cliff_benchmark.py` is not.** The first calls `np.random.seed(42)` and reproduces the chapter's table exactly. The second averages over 100 runs and takes no seed, so its fall percentages move a little between invocations. Averaging over 100 runs is what makes that acceptable — but be aware the two scripts are not held to the same standard, and only one of them will match the book digit for digit.

- **`GridWorld.walls` is empty.** The classic Russell & Norvig 4×3 grid has an obstacle at `(1,1)`. Here that square is a −10 hazard instead, and `self.walls` is left as an empty list so the boundary logic has a place to look. If you want the textbook layout, put `(1,1)` in `walls` and remove it from `terminals` — the transition code already handles it.

- **Cliff falls are counted by sniffing for `reward == -100`.** It works, because −100 is unique to the cliff in this environment. It is also the kind of shortcut that breaks the moment someone adds a second penalty of the same size, so treat the counter as instrumentation for this one task rather than a pattern to carry forward.

- **There is no `__init__.py`, so these scripts must be run from this directory.** The modules import each other by bare name (`from environments import GridWorld`), which means `python run_td0_gridworld.py` works from here and `python -m src.part_1_foundations.ch02_fundamentals.run_td0_gridworld` does not. This is the one chapter where that is true; from Chapter 3 onward the packages carry `__init__.py` and the `-m` form is the primary path. The test suite handles it by adding this directory to `sys.path` in `tests/conftest.py`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'environments'`.** You are running from the project root. Chapter 2 has no `__init__.py` — `cd` into this directory first, or use `make run-ch2-gridworld` / `make run-ch2-cliff`, which do the `cd` for you.

- **My cliff fall percentages do not match the book.** `run_cliff_benchmark.py` takes no seed; see the note above. The gap between the two agents should be stable across runs even though the absolute numbers drift. If the *ordering* flips — SARSA falling more often than Q-Learning — that is worth investigating, because the gap is roughly two to one and should not close by chance.

- **`cliff_results.png` is not where I expected.** It is written to the current working directory, not to this one. `make run-ch2-cliff` cd's here first, so it lands beside the scripts; running the file some other way puts it wherever you were.

- **The benchmark seems to hang.** It is not hanging, it is doing 100 runs × 500 episodes × 2 algorithms with no progress output — about 80 seconds. Lower `runs` at the top of the file while iterating.
