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
        return self._gather(self._random_indices(batch_size))

    def sample_recent(self, batch_size: int):
        """The most recent ``batch_size`` transitions, oldest first.

        Only the no-replay ablation uses this. It feeds the network the
        consecutive, highly correlated transitions that experience replay
        exists to break up, which is what makes the ablation a fair test: the
        batch size is unchanged, only *which* transitions are in it.
        """
        n = min(batch_size, self.size)
        idx = (self.position - np.arange(n, 0, -1)) % self.capacity
        return self._gather(idx)

    def _random_indices(self, batch_size: int):
        """Uniform sample of distinct indices from the filled region.

        ``np.random.choice(size, batch_size, replace=False)`` is the obvious
        way to write this, and it is what the buffer used to do. The catch is
        that NumPy implements it by permuting all ``size`` elements: ~26 ms per
        call on a 1M-transition Atari buffer, paid on every gradient step,
        against ~9 us for the draw itself. Rejection sampling is O(batch_size)
        and draws from the same distribution.

        Below the threshold, collisions are frequent enough that resampling
        stops paying and the permutation is cheap anyway - so the simple call
        still runs, and still terminates when batch_size == size.
        """
        if self.size < 100 * batch_size:
            return np.random.choice(self.size, batch_size, replace=False)

        idx = np.random.randint(0, self.size, batch_size)
        while len(np.unique(idx)) < batch_size:
            idx = np.random.randint(0, self.size, batch_size)
        return idx

    def _gather(self, idx):
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
