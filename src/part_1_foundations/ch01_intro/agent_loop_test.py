import gymnasium as gym

# 1. Create the environment (the 'world')
# We use CartPole-v1 as a lightweight, built-in environment for Chapter 1.
env = gym.make("CartPole-v1")
state, info = env.reset()         # Initial state (S0)

print("Starting Agent-Environment Loop...")

for t in range(200):
    # 2. Agent chooses an action (A_t)
    # At this stage, our agent is choosing actions randomly to explore.
    action = env.action_space.sample()
    
    # 3. The Loop: Agent acts, Environment responds
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if t % 20 == 0:
        print(f"Step {t:3}: Reward received = {reward}")
    
    # 4. Handle end of episode (Reset if finished)
    if terminated or truncated:
        state, info = env.reset()
        print(f"Episode finished at step {t}")
        break

print("\nTest Completed Successfully!")
