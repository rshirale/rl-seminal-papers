import numpy as np

class ReplayBuffer:
    """
    Experience Replay Buffer as introduced by Lin (1992) and utilized in DQN (Mnih et al. 2013/2015).
    Stores transitions (state, action, reward, next_state, done) and samples uniform random minibatches
    to break temporal correlations in sequential data.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # We store as a tuple: (s, a, r, s', done)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # Circular buffer: overwrite oldest memory when capacity is reached
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Samples a random minibatch of transitions."""
        # Replace=False ensures we don't sample the same transition twice in one batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        """Returns current size of the buffer."""
        return len(self.buffer)
