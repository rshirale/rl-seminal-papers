import torch
import torch.nn as nn
import torch.optim as optim
from actor_critic import Actor, Critic

class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, k_epochs=10):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim)
        # Using a single optimizer for both actor and critic
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.max_action = max_action

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            dist = self.actor(state)
            action = dist.sample()
            # Calculate log probability of the sampled action
            action_logprob = dist.log_prob(action).sum(dim=-1)
            # Calculate state value using critic
            value = self.critic(state)
            
        return action.clamp(-self.max_action, self.max_action).numpy()[0], action_logprob.item(), value.item()

    def update(self, rollouts):
        # Unpack the collected trajectories
        states = torch.FloatTensor([r[0] for r in rollouts])
        actions = torch.FloatTensor([r[1] for r in rollouts])
        rewards = [r[2] for r in rollouts]
        next_states = torch.FloatTensor([r[3] for r in rollouts])
        dones = [r[4] for r in rollouts]
        old_logprobs = torch.FloatTensor([r[5] for r in rollouts])
        values = [r[6] for r in rollouts]
        
        # Calculate Returns and Advantages using GAE
        returns = []
        advantages = []
        gae = 0
        with torch.no_grad():
            next_value = self.critic(next_states[-1].unsqueeze(0)).item()
        
        # Iterate backwards to compute GAE
        for i in reversed(range(len(rollouts))):
            if i == len(rollouts) - 1:
                next_val = next_value
            else:
                next_val = values[i+1]
                
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            dist = self.actor(states)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(states).squeeze()
            
            # The probability ratio r_t(theta)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Clipped Surrogate Objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # PPO Loss: Policy Loss + Value Loss - Entropy Bonus
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
