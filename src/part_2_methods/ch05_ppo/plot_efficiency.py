import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / "src" / "part_2_methods" / "ch06_sac"))
sys.path.insert(0, str(root_path / "src" / "part_2_methods" / "ch04_ddpg"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.part_2_methods.ch04_ddpg.train_pendulum import main as train_ddpg
from src.part_2_methods.ch05_ppo.train_pendulum import main as train_ppo
from src.part_2_methods.ch06_sac.train_pendulum import main as train_sac

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    seeds = 3
    episodes = 200
    steps_per_episode = 200
    x_axis = np.arange(episodes) * steps_per_episode

    # Run DDPG
    print("Training DDPG...")
    ddpg_returns = []
    for s in range(seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        print(f"  Seed {s}")
        ret = train_ddpg(seed=s, episodes=episodes)
        ddpg_returns.append(ret[:episodes])
    
    # Run PPO
    print("Training PPO...")
    ppo_returns = []
    for s in range(seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        print(f"  Seed {s}")
        # PPO usually takes 400 episodes to learn, but we will plot its first 200 to show sample efficiency
        res = train_ppo(seed=s, episodes=episodes)
        ppo_returns.append(res.returns[:episodes])

    # Run SAC
    print("Training SAC...")
    sac_returns = []
    for s in range(seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        print(f"  Seed {s}")
        # SAC trains for TOTAL_STEPS=50000 by default. Let's just collect the first 200 episodes.
        ret = train_sac(seed=s)
        sac_returns.append(ret[:episodes])

    ddpg_returns = np.array(ddpg_returns)
    ppo_returns = np.array(ppo_returns)
    sac_returns = np.array(sac_returns)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    def plot_curve(ax, x, returns, label, color, linestyle="-"):
        mean = np.mean(returns, axis=0)
        std = np.std(returns, axis=0)
        if len(mean) > 10:
            mean = smooth(mean, 10)
            std = smooth(std, 10)
            x_plot = x[9:]
        else:
            x_plot = x
        ax.plot(x_plot, mean, label=label, color=color, linewidth=2, linestyle=linestyle)
        ax.fill_between(x_plot, mean - std, mean + std, alpha=0.2, color=color)

    # Guidelines: Purple (Agents), Green (Environments), Orange (Learning)
    # But usually just distinctive colors. Let's use: SAC (Purple), PPO (Orange), DDPG (Green)
    plot_curve(ax, x_axis, sac_returns, "SAC", color="#5a4fcf") # Purple
    plot_curve(ax, x_axis, ppo_returns, "PPO", color="#f0a500", linestyle="--") # Orange
    plot_curve(ax, x_axis, ddpg_returns, "DDPG", color="#00b894", linestyle="-.") # Green

    ax.set_title("Sample Efficiency on Pendulum-v1\n(DDPG vs. PPO vs. SAC)", fontsize=12, pad=12)
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Episodic Return", fontsize=11)
    ax.set_ylim(-1600, 0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    # Save to the book's media folder!
    output_path = Path("/Users/rahulshirale/AntiGravity/potential-eureka/Books/RL_Seminal_Papers/Chapter5/media/ch05-figure-efficiency.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
