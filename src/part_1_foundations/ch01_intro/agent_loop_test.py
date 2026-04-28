import gymnasium as gym
import time

# 1. Create the environment (the 'world')
# Default: headless mode (works everywhere, including Colab)
env = gym.make("CartPole-v1")

# UNCOMMENT THE LINE BELOW if you want to see the pop-up window (local machine only)
# env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()         # Initial state (S0)

print("Starting Agent-Environment Loop...")

for t in range(200):
    # 2. Agent chooses an action (A_t)
    action = env.action_space.sample()
    
    # 3. The Loop: Agent acts, Environment responds
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Update current state for the next iteration
    state = next_state
    
    # If rendering, slow down so a human can actually watch it!
    if env.render_mode == "human":
        time.sleep(0.05) 
    
    if t % 20 == 0:
        print(f"Step {t:3}: Reward received = {reward}")
    
    # 4. Handle end of episode
    if terminated or truncated:
        print(f"Episode finished at step {t}")
        break

env.close()
print("\nTest Completed Successfully!")
