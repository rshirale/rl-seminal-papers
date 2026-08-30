# Chapter 1: Why Reinforcement Learning, Why Now?

This directory contains the minimal agent-environment loop for Chapter 1 of **"RL: The Seminal Papers"**. It is the smallest complete RL program in the book: an agent, an environment, and the cycle between them, with no learning at all.

That omission is the point. Every algorithm in the rest of the book — DQN, DDPG, PPO, SAC — is a different answer to the question of what to put where the "choose an action" line currently picks at random. Getting the loop itself clear first means the later chapters are only ever changing one thing.

## File Structure

- `agent_loop_test.py`: a script using Gymnasium's `CartPole-v1` to run the observe–act–repeat cycle for up to 200 steps, taking uniformly random actions.

## Installation

Chapter 1 needs only the Foundation dependencies — NumPy, Matplotlib, and Gymnasium. No PyTorch. From the project root:
```bash
make install
```

## Running the Loop

```bash
# From the project root
make run-ch1

# ...or directly, from this directory
python agent_loop_test.py
```

The agent takes random actions and prints the reward it receives, then reports the step at which the episode ended.

## Implementation Notes

- **The episode ends long before step 200, and that is the lesson.** The loop is written to run 200 steps, but a random policy keeps the pole up for only about 12 to 30 of them. Three consecutive runs here finished at steps 29, 12, and 12. Because the script prints every 20 steps, you will often see the `Step 0` line and then the finish, with nothing in between — the run is over before the second print. A random agent is not a bad agent; it is not an agent at all in any interesting sense, and the gap between 12 steps and CartPole's 500-step maximum is the entire subject of the book.

- **Nothing is seeded, deliberately.** Run it three times and you will get three different answers. That is worth seeing once, at the start, because from Chapter 3 onward every script takes `--seed` and every chapter has a `seeding.py`, and the reason will be more convincing if you have already watched an unseeded run wobble.

- **`terminated` and `truncated` are two different things, and this is where they first appear.** The loop unpacks both from `env.step()` and treats them alike, which is correct here because the script only wants to know whether to stop. It is *not* correct once an algorithm bootstraps a value estimate: `terminated` means the pole actually fell, while `truncated` means the clock ran out on an episode that was going fine. Chapters 4, 5, and 6 each carry an implementation note about the bug that follows from collapsing the two, and this is the line where the distinction is introduced.

- **Headless by default, so it runs anywhere.** `gym.make("CartPole-v1")` opens no window, which is what lets the script work unchanged in Colab, over SSH, and in CI. The commented `render_mode="human"` line gives a pop-up window on a local machine; the `time.sleep(0.05)` guard exists only for that path, so a human can follow what is happening.

## Troubleshooting

- **No window appears.** That is the default. Uncomment the `render_mode="human"` line near the top of the script — and expect it to fail on a headless machine or in Colab, which have no display to draw into.

- **The output is different every time.** Expected; see the note on seeding above. There is nothing to fix.

- **`ModuleNotFoundError: No module named 'gymnasium'`.** Run `make install` from the project root. Chapter 1 needs only the lightweight Foundation stack, not the full PyTorch install that Chapter 3 onward requires.
