import gym
import numpy as np
from ppo_agent import PPOAgent

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = PPOAgent(state_dim, action_dim, max_action)
    
    max_episodes = 500
    max_timesteps = 200
    update_timestep = 2000 # Update policy every 2000 timesteps
    
    timestep = 0
    rollouts = []
    
    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            timestep += 1
            
            # 1. Select action using current policy
            action, logprob, value = agent.select_action(state)
            
            # 2. Step environment
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            # 3. Store transition in rollout buffer
            rollouts.append((state, action, reward, next_state, done, logprob, value))
            
            state = next_state
            
            # 4. Perform PPO update if enough timesteps have passed
            if timestep % update_timestep == 0:
                agent.update(rollouts)
                rollouts = []
                
            if done or truncated:
                break
                
        if episode % 10 == 0:
            print(f"Episode {episode} \t Reward: {episode_reward:.2f}")

if __name__ == '__main__':
    main()
