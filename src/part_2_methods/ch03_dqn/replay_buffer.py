import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer as introduced by Lin (1992) and utilized in DQN
    (Mnih et al. 2013/2015). Stores transitions (state, action, reward,
    next_state, done) and samples uniform random minibatches to break temporal
    correlations in sequential data.

    Transitions are held in pre-allocated contiguous NumPy arrays (one per
    field) rather than a Python list of tuples. This keeps memory close to the
    theoretical minimum and makes sampling a fast index gather instead of a
    walk over scattered Python objects. Pass ``state_dtype=np.uint8`` to store
    raw Atari frames at one byte per pixel.
    """

    def __init__(self, capacity: int, state_shape, state_dtype=np.float32):
        shape = (capacity, *state_shape)
        self.capacity = capacity
        self.states = np.zeros(shape, dtype=state_dtype)
        self.next_states = np.zeros(shape, dtype=state_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.position = 0  # circular write head
        self.size = 0      # number of filled slots

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Saves a transition into the current slot (no object allocation)."""
        i = self.position
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done

        # Circular buffer: overwrite oldest memory when capacity is reached
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Samples a random minibatch of transitions (without replacement)."""
        # Sample distinct indices from the filled region only
        idx = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        """Returns current size of the buffer."""
        return self.size
