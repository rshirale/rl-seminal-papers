# Chapter 1: Why Reinforcement Learning, Why Now?

This directory contains the minimal Agent-Environment loop used to introduce the core concepts of RL.

## File Structure

- `agent_loop_test.py`: A simple script using Gymnasium's `CartPole-v1` to demonstrate the "observe, act, learn" cycle.

## Installation

Ensure you have the Foundation dependencies installed:
```bash
# From the project root
make install
```

## Running the Loop

To run the agent-environment simulation:
```bash
python agent_loop_test.py
```
Or from the root:
```bash
make run-ch1
```

The agent will take random actions in the CartPole environment, and the rewards will be printed to your terminal.
