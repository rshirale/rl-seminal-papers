# Reinforcement Learning – The Seminal Papers
**Author: Rahul Vasant Shirale**

This is the official companion repository for the book **Reinforcement Learning – The Seminal Papers** (Manning Publications). Chapters 1–7 are currently implemented, providing a functional bridge between academic research and production-ready Python code. Additional algorithms and applications are planned as the book progresses.

## 🌐 Companion Website

Explore the interactive companion site, browser-based Q-Learning playground,
paper-to-code examples, roadmap, theme controls, and available Colab notebooks:

https://rshirale.github.io/rl-seminal-papers/

The site’s privacy notice is available at [docs/privacy.html](docs/privacy.html).

## 🚀 The Mission: "Papers-to-Code"
Academic papers are the blueprints of the AI revolution, but they are often written in a dialect of "Greek-symbol math" that can feel inaccessible to practitioners. This repository treats each implemented paper as a **technical specification**, translating its mathematics into modular Python so you don't just use these algorithms—you know them.

## ⚡ Quick Start
This repository includes a **Makefile** to simplify environment setup and running experiments.

### Prerequisites

- Python **3.10–3.13** is recommended.
- A virtual environment is strongly recommended.
- Chapters 3–7 require PyTorch. PyTorch availability depends on your operating system, CPU architecture, and Python version.
- Chapter 7 additionally needs `transformers` and `peft`, plus about a gigabyte of model weights on first run. Two of its three targets need neither.

Create and activate a virtual environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows PowerShell
python -m pip install --upgrade pip
```

### 1. Setup
Choose your installation depth based on which part of the book you are reading:

**Option A: Foundations (Chapters 1 & 2)**
Lightweight setup. Installs NumPy, Matplotlib, Pandas, Gymnasium, and Jupyter.
```bash
make install
```

**Option B: Deep RL (Chapters 3–6)**
Installs PyTorch, classic-control environments, OpenCV, and the foundation dependencies.
```bash
make install-full
```

**Option C: Reasoning models (Chapter 7)**
Installs the language-model stack — PyTorch, `transformers` and `peft`.
```bash
make install-llm
```

Chapter 7's reward function and its group-size analysis run without any of this,
on a bare interpreter. Only the trainer and the notebook need it.

The Atari training script is optional and has additional native dependencies.
Install it separately only if you plan to run Atari experiments:

```bash
make install-atari
```

Atari installation may require platform-specific tools such as SDL2 and a C++
toolchain. The CartPole and Pendulum chapters do not require Atari.

If anything fails to install or a chapter script will not start, run:

```bash
make doctor
```

It reports which interpreter is actually active, your platform and CPU
architecture, the status of every dependency, and any known-bad combination
(for example: Intel macOS has no PyTorch wheels past 2.2.x, so it is capped at
Python 3.12). Most "it doesn't run for me" reports come down to the wrong
virtual environment being active, which the first two lines of output make
obvious.

Every target takes a `PYTHON` override, so you can install into or run from a
specific interpreter without activating it first:

```bash
make install-full PYTHON=/path/to/.venv/bin/python
```

If PyTorch still reports `No matching distribution found`, the active Python
version or platform has no compatible wheel, or pip is using a mirror that does
not provide PyTorch. Use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for a platform-specific install command, then run `make install-full` again.

**Option D: Tests**
Install pytest and run the automated checks:
```bash
make install-test
make test        # fast suite
make test-all    # also executes the chapter notebooks top to bottom
```

### 2. Run Experiments
You can run specific chapter experiments directly from the root:
```bash
# Chapter 1: Agent-Environment Loop
make run-ch1

# Chapter 2: Grid World TD(0) value estimation
make run-ch2-gridworld

# Chapter 2: Cliff Walking Benchmark (Q-Learning vs SARSA)
make run-ch2-cliff

# Chapter 3: DQN on CartPole-v1 (~3 min on CPU)
make run-ch3-cartpole

# Chapter 3: DQN on Atari Pong (needs `make install-atari`; hours on CPU)
make run-ch3-atari

# Chapter 3: the paper's replay / target-network ablation, on CartPole
#   Four variants across several seeds -- about 20 min on a CPU.
make run-ch3-ablation

# Chapter 4: DDPG on Pendulum-v1 (~3 min on CPU)
make run-ch4-pendulum

# Chapter 4: DDPG component ablation (target networks; chapter figure 4.10)
#   Nine 200-episode runs -- about 92 min on an 8-core CPU.
make run-ch4-ablation

# Chapter 5: PPO on Pendulum-v1 (~4 min on CPU)
make run-ch5-pendulum

# Chapter 5: the clipped objective ablation (chapter figure 5.9)
make run-ch5-ablation

# Chapter 5: what the seed alone is worth, before trusting any comparison
make run-ch5-seeding

# Chapter 6: SAC on Pendulum-v1 (~10 min on CPU)
make run-ch6-pendulum

# Chapter 6: what the entropy bonus is worth (the chapter's exercise 1)
make run-ch6-ablation

# Chapter 6: SAC's seed-to-seed spread, the threshold its tables are read against
make run-ch6-seeding

# Chapter 7: what the rule-based reward pays for (instant, no install needed)
make run-ch7-reward

# Chapter 7: how often a group carries no gradient (the chapter's exercise 3)
make run-ch7-group-size

# Chapter 7: GRPO on strict JSON schema adherence, no critic and no
#   demonstrations. Needs `make install-llm`; about 36 min on a CPU.
make run-ch7-train
```

`make help` lists every target, including the longer hyperparameter sweeps.
Any target that draws a figure takes `FIGURE_DIR=dir` to write PNG + SVG.

### 3. Cleanup
```bash
# Remove temporary files and generated plots
make clean
```

## 🗺️ The Roadmap
The repository follows the book’s three parts, from mathematical foundations to recent work on LLMs and robotics.

### Part I: Foundations
| Chapter | Topic | Focus | Interactive |
| :--- | :--- | :--- | :--- |
| **Ch 1** | **Introduction** | The RL Loop and the "code-first" mental model | [🔗 Script](src/part_1_foundations/ch01_intro/agent_loop_test.py) |
| **Ch 2** | **RL Fundamentals** | MDPs, Bellman Equations, and Tabular Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_1_foundations/ch02_fundamentals/Chapter2_Fundamentals.ipynb) |

### Part II: Deep Reinforcement Learning (Methods)
| Chapter | Algorithm | Seminal Paper Reference | Interactive |
| :--- | :--- | :--- | :--- |
| **Ch 3** | **DQN** | *Playing Atari with Deep Reinforcement Learning* (2013) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch03_dqn/Chapter3_DQN.ipynb) |
| **Ch 4** | **DDPG** | *Continuous control with deep reinforcement learning* (2015) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch04_ddpg/Chapter4_DDPG.ipynb) |
| **Ch 5** | **PPO** | *Proximal Policy Optimization Algorithms* (2017) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch05_ppo/Chapter5_PPO.ipynb) |
| **Ch 6** | **SAC** | *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* (2018) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch06_sac/Chapter6_SAC.ipynb) |
| **Ch 7** | **GRPO** | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rshirale/rl-seminal-papers/blob/main/src/part_2_methods/ch07_grpo/Chapter7_GRPO.ipynb) |

### Part III: Real-World Applications (planned)
| Chapter | Application | Key Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Ch 8** | **AlphaGo** | Deep Learning + Monte Carlo Tree Search | 🚧 Coming Soon |
| **Ch 9** | **AlphaZero** | Self-play with Monte Carlo Tree Search | 🚧 Coming Soon |
| **Ch 10** | **RLHF** | Alignment via Human Feedback (2017) | 🚧 Coming Soon |
| **Ch 11** | **Dexterous Manipulation** | Robotics and Sim-to-Real transfer | 🚧 Coming Soon |
| **Ch 12** | **AlphaDev** | Reinforcement learning for algorithm discovery | 🚧 Coming Soon |
| **Ch 13** | **Humanoid Locomotion** | Learning-based control and transfer | 🚧 Coming Soon |
| **Ch 14** | **DeepSeek-R1** | RL-only pipelines for Incentivizing Reasoning | 🚧 Coming Soon |
| **Ch 15** | **Conclusion** | Building Your Own Experiments | 🚧 Coming Soon |

## 🌟 The Mathematical North Star: From Paper to Code
What makes this repository unique is the direct, line-by-line mapping from academic math to Python. We don't just implement the "vibe" of a paper—we implement the **math**.

### **Example: The TD(0) Value Update**
📄 **Seminal Paper**: *Sutton (1988)*
> $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

💻 **The Implementation** (`ch02_fundamentals/algorithms.py`):
```python
# Bellman expectation update
td_target = reward + gamma * V[next_state]
V[state] += alpha * (td_target - V[state])
```

---

## 🛠️ Practitioner's Stack
To run these experiments, you should be comfortable with:
* **Python**: Intermediate level (Object-oriented programming and modular code).
* **PyTorch**: Basic/Intermediate experience defining neural networks and training loops.
* **Gymnasium**: Standard interface for RL environments.
* **NumPy**: Array-based manipulation for data processing.

## 🎯 Key Takeaways
Upon finishing this book and exploring this code, you will be equipped to:
* **Translate Research to Code**: Convert mathematical objectives from papers into functional Python.
* **Master Core Engines**: Write foundational Deep RL algorithms (DQN, DDPG, PPO, SAC, and GRPO) from scratch.
* **Build Reasoning Pipelines**: Train a base language model with GRPO against a rule-based reward — the algorithm behind DeepSeek-R1, with no critic and no demonstrations.
* **Navigate Sim-to-Real**: Prepare agents for deployment on physical humanoid hardware.

## 🧪 Verification

The automated tests cover the Chapter 2 environments, algorithms and notebook,
the Chapter 3 DQN modules and notebook, the Chapter 4 DDPG modules and
notebook, the Chapter 5 PPO modules and notebook, the Chapter 6 SAC modules
and notebook, and the Chapter 7 GRPO modules and notebook. A separate suite
checks the chapter READMEs themselves — every
module listed, every `make` target documented, every class attributed to the
file that defines it. The PyTorch chapters' tests are skipped when PyTorch is
not installed, allowing the foundation tests to run with the lightweight
setup:

```bash
make test        # fast suite
make test-all    # also executes the chapter notebooks top to bottom
```

Chapter 7's suite is split by dependency rather than skipped wholesale: its
reward function and dataset import nothing outside the standard library and are
tested on the lightweight setup, its objective is skipped without PyTorch, and
its notebook-execution test skips itself unless the model weights are already
cached — so `make test-all` never triggers a gigabyte of downloads.

Every notebook suite executes its chapter's notebook cell by cell, which is
how a broken paste in a notebook gets caught before a reader hits it. Chapter 2
had no such suite until one was added, and shipped a `KeyError` in its first
experiment for exactly as long.
