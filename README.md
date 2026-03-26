# Reinforcement Learning – The Seminal Papers
**Author: Rahul Vasant Shirale**

This is the official companion repository for the book **Reinforcement Learning – The Seminal Papers** (Manning Publications). This project acts as a functional bridge between academic research and production-ready implementation, turning influential papers into clear, runnable Python code.

## 🚀 The Mission: "Papers-to-Code"
Academic papers are the blueprints of the AI revolution, but they are often written in a dialect of "Greek-symbol math" that can feel inaccessible to practitioners. This repository treats every seminal paper—from the birth of DQN to the reasoning leaps of DeepSeek-R1—as a **technical specification**. We refactor that math into modular Python so you don't just use these algorithms—you know them.

## 🗺️ The Roadmap
The repository follows the book’s three-part journey from mathematical foundations to the cutting edge of LLMs and robotics.

### Part I: Foundations
| Chapter | Topic | Focus |
| :--- | :--- | :--- |
| **Ch 1** | **Introduction** | The RL Loop and the "code-first" mental model. |
| **Ch 2** | **RL Fundamentals** | MDPs, Bellman Equations, and Tabular Methods. |

### Part II: Deep Reinforcement Learning (Methods)
| Chapter | Algorithm | Seminal Paper Reference |
| :--- | :--- | :--- |
| **Ch 3** | **DQN** | *Playing Atari with Deep Reinforcement Learning* (Mnih et al., 2013). |
| **Ch 4** | **DDPG** | *Continuous control with deep reinforcement learning* (Lillicrap et al., 2015). |
| **Ch 5** | **PPO** | *Proximal Policy Optimization Algorithms* (Schulman et al., 2017). |
| **Ch 6** | **SAC** | *Soft Actor-Critic: Off-Policy RL with Entropy Regularization* (Haarnoja et al., 2018). |
| **Ch 7** | **GRPO** | RL for reasoning models (DeepSeek-related work). |

### Part III: Real-World Applications
| Chapter | Application | Key Implementation |
| :--- | :--- | :--- |
| **Ch 8** | **AlphaGo** | Deep Learning + Monte Carlo Tree Search. |
| **Ch 10** | **RLHF** | Alignment via Human Feedback (Christiano et al., 2017). |
| **Ch 11** | **Robotics** | Dexterous Manipulation and Sim-to-Real transfer. |
| **Ch 14** | **DeepSeek-R1** | RL-only pipelines for Incentivizing Reasoning. |

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