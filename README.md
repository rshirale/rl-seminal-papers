# Reinforcement Learning – The Seminal Papers
**Author: Rahul Vasant Shirale**

This is the official companion repository for the book **Reinforcement Learning – The Seminal Papers** (Manning Publications). This project acts as a functional bridge between academic research and production-ready implementation, turning influential papers into clear, runnable Python code.

## 🚀 The Mission: "Papers-to-Code"
Academic papers are the blueprints of the AI revolution, but they are often written in a dialect of "Greek-symbol math" that can feel inaccessible to practitioners. This repository treats every seminal paper—from the birth of DQN to the reasoning leaps of DeepSeek-R1—as a **technical specification**. We refactor that math into modular Python so you don't just use these algorithms—you know them.

## ⚡ Quick Start
This repository includes a **Makefile** to simplify environment setup and running experiments.

### 1. Setup
Choose your installation depth based on which part of the book you are reading:

**Option A: Foundations (Chapters 1 & 2)**
Lightweight setup (~50MB). Only installs NumPy, Matplotlib, and Pandas.
```bash
make install
```

**Option B: Full Stack (Chapters 3–14)**
Complete Deep RL setup (~1.5GB). Installs PyTorch, Gymnasium (Atari/MuJoCo), and foundations.
```bash
make install-full
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

# Chapter 3: DQN on CartPole-v1 (~2 min on CPU)
make run-ch3-cartpole

# Chapter 4: DDPG on Pendulum-v1 (~10 min on CPU)
make run-ch4-pendulum
```

### 3. Cleanup
```bash
# Remove temporary files and generated plots
make clean
```

## 🗺️ The Roadmap
The repository follows the book’s three-part journey from mathematical foundations to the cutting edge of LLMs and robotics.

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
| **Ch 6** | **SAC** | *Soft Actor-Critic: Off-Policy RL with Entropy Regularization* (2018) | 🚧 Coming Soon |
| **Ch 7** | **GRPO** | RL for reasoning models (DeepSeek-related work) | 🚧 Coming Soon |

### Part III: Real-World Applications
| Chapter | Application | Key Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Ch 8** | **AlphaGo** | Deep Learning + Monte Carlo Tree Search | 🚧 Coming Soon |
| **Ch 10** | **RLHF** | Alignment via Human Feedback (2017) | 🚧 Coming Soon |
| **Ch 11** | **Robotics** | Dexterous Manipulation and Sim-to-Real transfer | 🚧 Coming Soon |
| **Ch 14** | **DeepSeek-R1** | RL-only pipelines for Incentivizing Reasoning | 🚧 Coming Soon |

## 🌟 The Mathematical North Star: From Paper to Code
What makes this repository unique is the direct, line-by-line mapping from academic math to Python. We don't just implement the "vibe" of a paper—we implement the **math**.

### **Example: The TD(0) Value Update**
📄 **Seminal Paper**: *Sutton (1988)*
> $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

💻 **The Implementation** (`ch02_fundamentals/algorithms.py`):
```python
# Bootstrapped TD Target minus the current belief
td_error = (reward + gamma * V[next_state]) - V[state]
V[state] += alpha * td_error
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
* **Master Core Engines**: Write foundational Deep RL algorithms (DQN, PPO, SAC) from scratch.
* **Build Reasoning Pipelines**: Implement GRPO to incentivize self-correction in models.
* **Navigate Sim-to-Real**: Prepare agents for deployment on physical humanoid hardware.