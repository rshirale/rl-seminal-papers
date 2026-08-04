# Chapter 2: Reinforcement Learning Fundamentals

This directory contains the Python implementations and benchmarks for Chapter 2 of **"RL: The Seminal Papers"**.

## File Structure

- `environments.py`: Core logic for the `GridWorld` and `CliffWalking` environments.
- `algorithms.py`: Implementation of TD(0), Q-Learning, and SARSA.
- `run_td0_gridworld.py`: Reproduces the value estimation results and outcome statistics for the 4x3 Grid World.
- `run_cliff_benchmark.py`: Compares the "Brave" (Q-Learning) and "Cautious" (SARSA) agents on the Cliff Walking task.

## Installation

You can install dependencies for just this chapter:
```bash
python -m pip install -r requirements.txt
```

Alternatively, if you are following the whole book, use the master list in the root directory:
```bash
# From the project root
make install
```

## Running the Benchmarks

To see the TD(0) random policy statistics and value table:
```bash
python run_td0_gridworld.py
```

To run the Cliff Walking comparison and generate performance plots:
```bash
python run_cliff_benchmark.py
```

Results and plots will be saved to the current directory.
