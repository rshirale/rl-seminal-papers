import numpy as np
import matplotlib.pyplot as plt
import os
from environments import CliffWalking
from algorithms import run_agent

def main():
    print("Running Cliff Walking Benchmark...")
    env = CliffWalking()
    runs = 100
    episodes = 500
    
    q_rewards_all = np.zeros((runs, episodes))
    sarsa_rewards_all = np.zeros((runs, episodes))
    
    q_falls_total = 0
    sarsa_falls_total = 0
    
    for r in range(runs):
        # Using fixed hyperparameters to match book stats
        # Q-Learning (Brave)
        _, q_rev, q_falls = run_agent(env, mode='qlearning', episodes=episodes, alpha=0.1, gamma=0.99)
        # SARSA (Cautious)
        _, sarsa_rev, sarsa_falls = run_agent(env, mode='sarsa', episodes=episodes, alpha=0.1, gamma=0.99)
        
        q_rewards_all[r] = q_rev
        sarsa_rewards_all[r] = sarsa_rev
        q_falls_total += q_falls
        sarsa_falls_total += sarsa_falls
        
    q_reward = np.mean(q_rewards_all, axis=0)
    sarsa_reward = np.mean(sarsa_rewards_all, axis=0)
    
    q_pct = (q_falls_total / runs) / episodes * 100
    sarsa_pct = (sarsa_falls_total / runs) / episodes * 100

    print(f"\nBenchmark Results (Average over {runs} runs):")
    print(f"Q-Learning Falls: {q_pct:.1f}%")
    print(f"SARSA Falls:      {sarsa_pct:.1f}%")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    window = 10
    q_smooth = np.convolve(q_reward, np.ones(window)/window, mode='valid')
    s_smooth = np.convolve(sarsa_reward, np.ones(window)/window, mode='valid')
    
    ax1.plot(q_smooth, label="Q-Learning (Brave)", color="#D97706")
    ax1.plot(s_smooth, label="SARSA (Cautious)", color="#3498db")
    ax1.set_title("Training Reward per Episode (Moving Average)")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(["SARSA", "Q-Learning"], [sarsa_pct, q_pct], color=['#3498db', '#D97706'], width=0.5)
    ax2.set_title("Average Cliff Fall Frequency (%)")
    ax2.set_ylabel("Fall %")

    plt.tight_layout()
    plt.savefig("cliff_results.png", dpi=300)
    print("\n✓ Results saved to 'cliff_results.png'")

if __name__ == '__main__':
    main()
