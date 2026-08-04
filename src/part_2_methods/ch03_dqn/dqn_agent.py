import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if __package__:
    from .dqn_network import DQN
    from .replay_buffer import ReplayBuffer
else:  # pragma: no cover - direct script execution fallback.
    from dqn_network import DQN
    from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network Agent implementing Algorithm 1 from Mnih et al. (2015).
    Combines the Convolutional Q-Network, Experience Replay, and a Target Network.
    """
    def __init__(
        self, 
        input_channels: int, 
        num_actions: int, 
        device: torch.device,
        learning_rate: float = 0.00025,
        gamma: float = 0.99,
        buffer_capacity: int = 1000000,
        batch_size: int = 32,
        target_update_freq: int = 10000
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Step counter to trigger target network updates
        self.steps_done = 0

        # Initialize Online and Target Networks
        self.online_net = DQN(input_channels, num_actions).to(self.device)
        self.target_net = DQN(input_channels, num_actions).to(self.device)
        
        # Load identical weights initially and freeze target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # The 2015 paper used RMSprop. (Adam is also common today, but we stick to the original)
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=learning_rate, momentum=0.95, eps=0.01)
        
        # Initialize Replay Memory D
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
        # Huber Loss (Smooth L1 Loss)
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        """
        # Explore: select a random action
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
            
        # Exploit: select the action with max Q-value from the online network
        self.online_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        self.online_net.train()
        
        return action

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Stores transition in replay memory and potentially triggers a learning step.
        """
        # Store transition (s_t, a_t, r_t, s_{t+1}) in D
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
        
        # We only learn if the buffer has enough samples
        if len(self.memory) > self.batch_size: # In practice, wait for 50k 'warm up' steps
            self._learn()
            
        # Every C steps reset Q^ = Q (Update target network)
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_network()

    def _learn(self):
        """
        Samples a minibatch from memory and performs a gradient descent step.
        """
        # Sample random minibatch of transitions from D
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values: Q(s_j, a_j; theta)
        # .gather() picks the Q-value corresponding to the action that was actually taken
        current_q_values = self.online_net(states).gather(1, actions)

        # Compute next Q values using the Target Network: max_a' Q^(s_{j+1}, a'; theta^-)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            
            # If the state is terminal, the target is just the reward (no future return)
            # Mask out terminal next states
            next_q_values[dones] = 0.0
            
            # Compute target y_j
            target_q_values = rewards + (self.gamma * next_q_values)

        # Compute Huber Loss between current Q values and targets
        loss = self.criterion(current_q_values, target_q_values)

        # Perform gradient descent step on theta
        self.optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping for additional stability
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 1.0)
        
        self.optimizer.step()

    def _update_target_network(self):
        """
        Hard copy of weights from online network to target network.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())
