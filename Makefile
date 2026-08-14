# Makefile for "RL: The Seminal Papers"
# ==========================================

.PHONY: help install install-full clean run-ch1 run-ch2-gridworld run-ch2-cliff run-ch3-cartpole run-ch4-pendulum run-ch6-pendulum notebook

# Default command: show help
help:
	@echo "RL: The Seminal Papers - Command Menu"
	@echo "======================================"
	@echo "Setup Commands:"
	@echo "  make install             - Install Foundation stack (Chapters 1-2) - ~60MB"
	@echo "  make install-full        - Install Deep RL stack (Chapters 3-5)"
	@echo "  make install-atari      - Install optional Atari dependencies"
	@echo "  make install-test        - Install test dependencies"
	@echo "  make test                - Run automated tests"
	@echo ""
	@echo "Chapter 1: Introduction"
	@echo "  make run-ch1             - Run minimal Agent-Environment loop (CartPole)"
	@echo ""
	@echo "Chapter 2: Fundamentals"
	@echo "  make run-ch2-gridworld   - Run TD(0) value estimation on 4x3 Grid World"
	@echo "  make run-ch2-cliff       - Run Q-Learning vs SARSA on Cliff Walking"
	@echo ""
	@echo "Chapter 3: DQN"
	@echo "  make run-ch3-cartpole    - Train DQN on CartPole-v1 (~2 min on CPU)"
	@echo ""
	@echo "Chapter 4: DDPG"
	@echo "  make run-ch4-pendulum    - Train DDPG on Pendulum-v1 (~10 min on CPU)"
	@echo ""
	@echo "Chapter 6: SAC"
	@echo "  make run-ch6-pendulum    - Train SAC on Pendulum-v1 (~5 min on CPU)"
	@echo ""
	@echo "Jupyter Notebooks:"
	@echo "  make notebook            - Launch Jupyter Lab to view interactive chapters"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean               - Remove temporary files and generated plots"

# Install Foundation dependencies (Includes base Gymnasium for Ch 1)
install:
	@echo "Installing Foundation dependencies (NumPy, Matplotlib, Gymnasium)..."
	python -m pip install -r requirements.txt

# Install everything (Foundation + Deep RL stack)
install-full: install
	@echo "Installing Deep RL stack (PyTorch, classic-control environments)..."
	python -m pip install -r requirements-deep.txt

install-atari:
	@echo "Installing optional Atari dependencies..."
	python -m pip install -r requirements-atari.txt

install-test:
	python -m pip install -r requirements-test.txt

test:
	python -m pytest -q

# --- Chapter 1 Commands ---

CH1_DIR = src/part_1_foundations/ch01_intro

run-ch1:
	@echo "Running Chapter 1: Agent-Environment Loop..."
	@cd $(CH1_DIR) && python agent_loop_test.py

# --- Chapter 2 Commands ---

CH2_DIR = src/part_1_foundations/ch02_fundamentals

run-ch2-gridworld:
	@echo "Running Chapter 2: Grid World TD(0) Experiment..."
	@cd $(CH2_DIR) && python run_td0_gridworld.py

run-ch2-cliff:
	@echo "Running Chapter 2: Cliff Walking Benchmark..."
	@cd $(CH2_DIR) && python run_cliff_benchmark.py

# --- Chapter 3 Commands ---

CH3_DIR = src/part_2_methods/ch03_dqn

run-ch3-cartpole:
	@echo "Running Chapter 3: DQN on CartPole-v1..."
	@cd $(CH3_DIR) && python train_cartpole.py

# --- Chapter 4 Commands ---

CH4_DIR = src/part_2_methods/ch04_ddpg

run-ch4-pendulum:
	@echo "Running Chapter 4: DDPG on Pendulum-v1..."
	@cd $(CH4_DIR) && python train_pendulum.py

# --- Chapter 6 Commands ---

CH6_DIR = src/part_2_methods/ch06_sac

run-ch6-pendulum:
	@echo "Running Chapter 6: SAC on Pendulum-v1..."
	@cd $(CH6_DIR) && python train_pendulum.py

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
