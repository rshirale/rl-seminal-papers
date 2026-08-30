# Makefile for "RL: The Seminal Papers"
# ==========================================

.PHONY: help install install-full install-atari install-test test test-all doctor clean \
        run-ch1 run-ch2-gridworld run-ch2-cliff run-ch3-cartpole run-ch3-atari \
        run-ch3-ablation \
        run-ch4-pendulum run-ch4-ablation run-ch5-pendulum run-ch5-ablation \
        run-ch5-seeding run-ch5-sweep run-ch5-efficiency \
        run-ch6-pendulum run-ch6-ablation run-ch6-temperature \
        run-ch6-reward-scale run-ch6-seeding notebook

# Interpreter used by every target. Defaults to python3 because a bare `python`
# does not exist on most Linux distributions or on macOS 12.3+. Override it to
# target a specific virtualenv without activating it first:
#
#   make run-ch3-cartpole PYTHON=.venv/bin/python
#
PYTHON ?= python3

# Some targets cd into a chapter directory, which would break a relative
# interpreter path like .venv/bin/python. Absolutise anything that looks like a
# path; leave bare command names alone so they still resolve via PATH.
PYTHON_ABS := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

# Default command: show help
help:
	@echo "RL: The Seminal Papers - Command Menu"
	@echo "======================================"
	@echo "Setup Commands:"
	@echo "  make install             - Install Foundation stack (Chapters 1-2) - ~60MB"
	@echo "  make install-full        - Install Deep RL stack (Chapters 3-6)"
	@echo "  make install-atari       - Install optional Atari dependencies"
	@echo "  make install-test        - Install test dependencies"
	@echo "  make test                - Run fast tests (skips notebook execution)"
	@echo "  make test-all            - Run all tests, including notebook execution"
	@echo "  make doctor              - Report interpreter, platform, and package status"
	@echo ""
	@echo "Chapter 1: Introduction"
	@echo "  make run-ch1             - Run minimal Agent-Environment loop (CartPole)"
	@echo ""
	@echo "Chapter 2: Fundamentals"
	@echo "  make run-ch2-gridworld   - Run TD(0) value estimation on 4x3 Grid World"
	@echo "  make run-ch2-cliff       - Run Q-Learning vs SARSA on Cliff Walking"
	@echo ""
	@echo "Chapter 3: DQN"
	@echo "  make run-ch3-cartpole    - Train DQN on CartPole-v1 (~3 min on CPU)"
	@echo "  make run-ch3-atari       - Train DQN on Atari Pong (needs make install-atari)"
	@echo "  make run-ch3-ablation    - Ablate replay and the target network (~20 min)"
	@echo ""
	@echo "Chapter 4: DDPG"
	@echo "  make run-ch4-pendulum    - Train DDPG on Pendulum-v1 (~3 min on CPU)"
	@echo "  make run-ch4-ablation    - Ablate target networks (chapter figure 4.10)"
	@echo "     ...add FIGURE_DIR=dir to write PNG + SVG"
	@echo ""
	@echo "Chapter 5: PPO"
	@echo "  make run-ch5-pendulum    - Train PPO on Pendulum-v1 (~4 min on CPU)"
	@echo "  make run-ch5-ablation    - Ablate the clipped objective (~4 min)"
	@echo "  make run-ch5-seeding     - Show PPO's seed-to-seed spread (~3 min)"
	@echo "  make run-ch5-sweep       - Hyperparameter sensitivity bowls (~30 min)"
	@echo "  make run-ch5-efficiency  - DDPG vs PPO vs SAC sample efficiency (~30 min)"
	@echo "     ...add FIGURE_DIR=dir to any of these to write PNG + SVG"
	@echo ""
	@echo "Chapter 6: SAC"
	@echo "  make run-ch6-pendulum    - Train SAC on Pendulum-v1 (~10 min on CPU)"
	@echo "  make run-ch6-ablation    - Ablate the entropy bonus (exercise 1, ~30 min)"
	@echo "  make run-ch6-temperature - Fixed vs learned alpha (exercise 2, ~60 min)"
	@echo "  make run-ch6-reward-scale- Scale rewards by 10 (exercise 3, ~60 min)"
	@echo "  make run-ch6-seeding     - Show SAC's seed-to-seed spread (~30 min)"
	@echo "     ...add FIGURE_DIR=dir to write PNG + SVG"
	@echo ""
	@echo "Jupyter Notebooks:"
	@echo "  make notebook            - Launch Jupyter Lab to view interactive chapters"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean               - Remove temporary files and generated plots"

# Install Foundation dependencies (Includes base Gymnasium for Ch 1)
install:
	@echo "Installing Foundation dependencies (NumPy, Matplotlib, Gymnasium)..."
	$(PYTHON_ABS) -m pip install -r requirements.txt

# Install everything (Foundation + Deep RL stack)
install-full: install
	@echo "Installing Deep RL stack (PyTorch, classic-control environments)..."
	$(PYTHON_ABS) -m pip install -r requirements-deep.txt

install-atari:
	@echo "Installing optional Atari dependencies..."
	$(PYTHON_ABS) -m pip install -r requirements-atari.txt

install-test:
	$(PYTHON_ABS) -m pip install -r requirements-test.txt

# Fast suite: skips notebook execution and other end-to-end runs.
test:
	$(PYTHON_ABS) -m pytest -q -m "not slow"

# Everything, including executing the chapter notebooks top to bottom.
test-all:
	$(PYTHON_ABS) -m pytest -q

# Reports which interpreter and platform are actually in use, plus the status
# of every dependency. Start here when a chapter script does not run.
doctor:
	@$(PYTHON_ABS) tools/doctor.py

# --- Chapter 1 Commands ---

CH1_DIR = src/part_1_foundations/ch01_intro

run-ch1:
	@echo "Running Chapter 1: Agent-Environment Loop..."
	@cd $(CH1_DIR) && $(PYTHON_ABS) agent_loop_test.py

# --- Chapter 2 Commands ---

CH2_DIR = src/part_1_foundations/ch02_fundamentals

run-ch2-gridworld:
	@echo "Running Chapter 2: Grid World TD(0) Experiment..."
	@cd $(CH2_DIR) && $(PYTHON_ABS) run_td0_gridworld.py

run-ch2-cliff:
	@echo "Running Chapter 2: Cliff Walking Benchmark..."
	@cd $(CH2_DIR) && $(PYTHON_ABS) run_cliff_benchmark.py

# --- Chapter 3 Commands ---

CH3_DIR = src/part_2_methods/ch03_dqn

run-ch3-cartpole:
	@echo "Running Chapter 3: DQN on CartPole-v1..."
	@cd $(CH3_DIR) && $(PYTHON_ABS) train_cartpole.py

# Requires the optional Atari stack: make install-atari
run-ch3-atari:
	@echo "Running Chapter 3: DQN on Atari Pong..."
	@cd $(CH3_DIR) && $(PYTHON_ABS) train_atari.py

# The paper's four-way ablation -- full DQN, no target network, no replay,
# neither -- on CartPole, where a full sweep takes minutes rather than
# GPU-days. Four variants x three seeds; budget about twenty minutes.
# Pass EXTRA="--episodes 400 --seeds 1 2 3" to vary it.
#
# Read the spread column, not just the mean: the no-replay row swings from
# 33.5 to 155.6 across seeds, so one run of it proves nothing.
run-ch3-ablation:
	@echo "Running Chapter 3: DQN replay / target-network ablation..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch03_dqn.ablation $(EXTRA)

# --- Chapter 4 Commands ---

CH4_DIR = src/part_2_methods/ch04_ddpg

run-ch4-pendulum:
	@echo "Running Chapter 4: DDPG on Pendulum-v1..."
	@cd $(CH4_DIR) && $(PYTHON_ABS) train_pendulum.py --seed 0

# Prints its table to the terminal and writes nothing. Set FIGURE_DIR to also
# emit PNG + SVG, e.g.
#
#   make run-ch4-ablation FIGURE_DIR=figures
#
CH4_FIGURE_ARG = $(if $(FIGURE_DIR),--figure $(FIGURE_DIR))

# Three variants x three seeds -- nine 200-episode runs. Measured at 92 min
# on an 8-core Intel MacBook Pro; budget an hour and a half.
# Pass EXTRA="--no-hard-copy" for a quicker two-curve sweep while iterating.
run-ch4-ablation:
	@echo "Running Chapter 4: DDPG component ablation..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch04_ddpg.ablation $(CH4_FIGURE_ARG) $(EXTRA)

# --- Chapter 5 Commands ---

CH5_DIR = src/part_2_methods/ch05_ppo

run-ch5-pendulum:
	@echo "Running Chapter 5: PPO on Pendulum-v1..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch05_ppo.train_pendulum

# Both ablation targets print their table to the terminal and write nothing.
# Set FIGURE_DIR to also emit PNG + SVG, e.g.
#
#   make run-ch5-sweep FIGURE_DIR=figures
#
CH5_FIGURE_ARG = $(if $(FIGURE_DIR),--figure $(FIGURE_DIR))

# Two variants x three seeds; budget about four minutes on a CPU.
# Pass EXTRA="--include-tight" to add the over-tight eps = 0.05 curve.
run-ch5-ablation:
	@echo "Running Chapter 5: PPO clipping ablation..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch05_ppo.ablation $(CH5_FIGURE_ARG) $(EXTRA)

# Three seeds, one run each; budget about three minutes. Prints the
# seed-to-seed spread the chapter thresholds its "resolved by 3 seeds?" column
# on, so the number in the text has a source a reader can re-run.
run-ch5-seeding:
	@echo "Running Chapter 5: PPO seed variance..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch05_ppo.seeding $(EXTRA)

# Twelve distinct configurations x three seeds; budget about thirty minutes.
run-ch5-sweep:
	@echo "Running Chapter 5: PPO sensitivity sweeps..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch05_ppo.ablation --sweep $(CH5_FIGURE_ARG) $(EXTRA)

# Three algorithms x three seeds, spanning chapters 4, 5, and 6; budget
# about thirty minutes. SAC is the slow one: a gradient update per step.
run-ch5-efficiency:
	@echo "Running Chapter 5: DDPG vs PPO vs SAC sample efficiency..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch05_ppo.plot_efficiency $(CH5_FIGURE_ARG) $(EXTRA)

# --- Chapter 6 Commands ---

CH6_DIR = src/part_2_methods/ch06_sac

# Seed 42 reproduces the transcript in the chapter's "Expected training output"
# section. SAC takes a gradient step per environment step, which makes it the
# slowest runner here and the most sample-efficient one -- both at once.
run-ch6-pendulum:
	@echo "Running Chapter 6: SAC on Pendulum-v1..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch06_sac.train_pendulum --seed 42

# All three ablation targets print their table to the terminal and write
# nothing. Set FIGURE_DIR to also emit PNG + SVG, e.g.
#
#   make run-ch6-ablation FIGURE_DIR=figures
#
CH6_FIGURE_ARG = $(if $(FIGURE_DIR),--figure $(FIGURE_DIR))

# Exercise 1. Two variants x three seeds, 30,000 steps each; budget half an
# hour. Shrink it while iterating with EXTRA="--steps 6000 --seeds 0".
run-ch6-ablation:
	@echo "Running Chapter 6: SAC entropy ablation..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch06_sac.ablation $(CH6_FIGURE_ARG) $(EXTRA)

# Exercise 2. Four variants x three seeds; budget an hour.
run-ch6-temperature:
	@echo "Running Chapter 6: SAC fixed vs learned temperature..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch06_sac.ablation --temperature $(CH6_FIGURE_ARG) $(EXTRA)

# Exercise 3. Four variants x three seeds; budget an hour.
run-ch6-reward-scale:
	@echo "Running Chapter 6: SAC reward-scale sensitivity..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch06_sac.ablation --reward-scale $(CH6_FIGURE_ARG) $(EXTRA)

# Three seeds, one run each. Prints the seed-to-seed spread every table in
# this chapter should be read against, so the threshold has a source a reader
# can re-run rather than a figure they have to take on trust.
run-ch6-seeding:
	@echo "Running Chapter 6: SAC seed variance..."
	@$(PYTHON_ABS) -m src.part_2_methods.ch06_sac.seeding $(EXTRA)

# --- Notebooks ---

notebook:
	@echo "Launching Jupyter Lab..."
	jupyter lab

# --- Utilities ---

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "cliff_results.png" -delete
	@echo "✓ Done."
