import gymnasium as gym

# 1. Create the environment (the 'world')
env = gym.make("LunarLander-v3")
state, info = env.reset()         # Initial state (S0)

for t in range(1000):
    # 2. Agent chooses an action (A_t)
    action = env.action_space.sample()
    
    # 3. The Loop: Agent acts, Environment responds
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # 4. Handle end of episode (Reset if finished)
    if terminated or truncated:
        state, info = env.reset()
        print(f"Episode finished at step {t}")

print("Test Completed Successfully!")