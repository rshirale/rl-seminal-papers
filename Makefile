# Makefile for "RL: The Seminal Papers"
# ==========================================

.PHONY: help install install-full clean run-ch2-gridworld run-ch2-cliff

# Default command: show help
help:
	@echo "RL: The Seminal Papers - Command Menu"
	@echo "======================================"
	@echo "Setup Commands:"
	@echo "  make install             - Install Foundation stack (Chapters 1-2) - ~50MB"
	@echo "  make install-full        - Install Full Deep RL stack (Chapter 3+) - ~1.5GB"
	@echo ""
	@echo "Chapter 2: Fundamentals"
	@echo "  make run-ch2-gridworld   - Run TD(0) value estimation on 4x3 Grid World"
	@echo "  make run-ch2-cliff       - Run Q-Learning vs SARSA on Cliff Walking"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean               - Remove temporary files and generated plots"

# Install Foundation dependencies
install:
	@echo "Installing Foundation dependencies (NumPy, Matplotlib, Pandas)..."
	pip install -r requirements.txt

# Install everything (Foundation + Deep RL)
install-full: install
	@echo "Installing Deep RL stack (PyTorch, Gymnasium, MuJoCo)..."
	pip install -r requirements-deep.txt

# --- Chapter 2 Commands ---

CH2_DIR = src/part_1_foundations/ch02_fundamentals

run-ch2-gridworld:
	@echo "Running Chapter 2: Grid World TD(0) Experiment..."
	@cd $(CH2_DIR) && python run_td0_gridworld.py

run-ch2-cliff:
	@echo "Running Chapter 2: Cliff Walking Benchmark..."
	@cd $(CH2_DIR) && python run_cliff_benchmark.py

# --- Utilities ---

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "cliff_results.png" -delete
	@echo "✓ Done."
