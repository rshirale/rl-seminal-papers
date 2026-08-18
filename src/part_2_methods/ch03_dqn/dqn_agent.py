import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    from .dqn_network import DQN
    from .replay_buffer import ReplayBuffer
except ImportError:
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
        target_update_freq: int = 10000,
        warmup_steps: int = 1000
    ):
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps = warmup_steps
        
        # Step counter to trigger target network updates
        self.steps_done = 0

        # Initialize Online and Target Networks
        self.online_net = DQN(input_channels, num_actions).to(self.device)
        self.target_net = DQN(input_channels, num_actions).to(self.device)
        
        # Load identical weights initially and freeze target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # The 2015 paper used RMSprop. (Adam is also common today, but we stick
        # to the original.) Extended Data Table 1 of Mnih et al. lists gradient
        # momentum 0.95 *and* squared gradient momentum 0.95 -- in PyTorch those
        # are `momentum` and `alpha`, two separate knobs, so both are set here.
        # `eps` is the paper's min squared gradient.
        self.optimizer = optim.RMSprop(
            self.online_net.parameters(),
            lr=learning_rate,
            alpha=0.95,
            momentum=0.95,
            eps=0.01,
        )
        
        # Initialize Replay Memory D. Frames are stacks of input_channels x 84 x 84.
        #
        # Frames are stored as raw uint8 (one byte per pixel) rather than as
        # normalized float32. That is a 4x saving on by far the largest
        # allocation in the agent: at capacity=100k the two state arrays total
        # ~5.3 GB instead of ~21 GB, and a full 1M-transition buffer would need
        # ~53 GB rather than ~210 GB. (Binary GB throughout, matching the
        # figures quoted in the chapter text.)
        #
        # The trade-off is that this agent, not the caller, owns pixel
        # normalization: push raw env frames, and /255 happens in
        # _states_to_tensor after sampling.
        self.memory = ReplayBuffer(
            capacity=buffer_capacity,
            state_shape=(input_channels, 84, 84),
            state_dtype=np.uint8,
        )
        
        # Huber Loss (Smooth L1 Loss)
        self.criterion = nn.SmoothL1Loss()

    def _states_to_tensor(self, states: np.ndarray) -> torch.Tensor:
        """
        Converts raw uint8 frames into the normalized float32 tensor the
        network expects. Kept in one place so the stored (uint8) and consumed
        (float in [0, 1]) representations can never drift apart.
        """
        # .to(float32) copies, so the in-place divide never touches the
        # caller's array or the replay buffer's storage.
        return torch.as_tensor(states, device=self.device).to(torch.float32).div_(255.0)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        Expects a raw uint8 frame stack; normalization happens internally.
        """
        # Explore: select a random action
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)

        # Exploit: select the action with max Q-value from the online network
        self.online_net.eval()
        with torch.no_grad():
            state_tensor = self._states_to_tensor(state).unsqueeze(0)
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
        
        # We only learn if the buffer has enough samples and warmup is complete.
        # Both checks matter: sample() draws without replacement, so a
        # warmup_steps below batch_size would otherwise raise here.
        if self.steps_done >= self.warmup_steps and len(self.memory) >= self.batch_size:
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
        
        # Convert to PyTorch tensors using as_tensor. States are stored as raw
        # uint8, so this is where the /255 normalization happens.
        states = self._states_to_tensor(states)
        next_states = self._states_to_tensor(next_states)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool).unsqueeze(1).to(self.device)

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
