import numpy as np
from environments import GridWorld
from algorithms import td0

def main():
    print("Running TD(0) Value Estimation on Grid World...")
    env = GridWorld()
    np.random.seed(42)
    
    episodes = 500
    outcomes = {"Goal": 0, "Hazard (1,1)": 0, "Hazard (2,1)": 0}
    
    # 1. Track outcome statistics (Random Policy)
    for _ in range(episodes):
        state = (0, 0)
        done = False
        while not done:
            action = np.random.choice(env.actions)
            state, reward, done = env.transition(state, action)
        
        if state == (3, 2): outcomes["Goal"] += 1
        elif state == (1, 1): outcomes["Hazard (1,1)"] += 1
        elif state == (2, 1): outcomes["Hazard (2,1)"] += 1

    print(f"\nRandom Policy Outcomes over {episodes} episodes:")
    for key, count in outcomes.items():
        print(f"  - {key}: {count} ({count/episodes:.1%})")
    
    # 2. Run TD(0) learning
    print("\nRunning TD(0) learning (converging value table)...")
    V = td0(env, episodes=1000, alpha=0.1)
    
    print("\nConverged Value Table (State -> Value):")
    for y in range(env.height - 1, -1, -1):
        row = []
        for x in range(env.width):
            row.append(f"{V[(x, y)]:6.2f}")
        print("  " + " | ".join(row))

if __name__ == '__main__':
    main()
