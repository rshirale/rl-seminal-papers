import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Uniform experience replay storing ``(s, a, r, s', done)`` transitions.

    Identical in purpose and structure to the buffers of Chapters 3 and 4: a
    fixed-capacity deque sampled uniformly at random. What earns it a place in
    SAC is the off-policy guarantee -- the soft Bellman backup is a Bellman
    equation, not a policy-ratio objective, so every stored transition is valid
    training data no matter which policy collected it. That is the whole
    difference between this chapter's data budget and Chapter 5's.

    The default capacity of 1,000,000 is the SAC-v2 value. On Pendulum-v1 the
    50,000-step run never comes close to filling it, so nothing is ever
    evicted; the number is the paper's, kept so the module is honest about what
    it would do on a MuJoCo-sized problem.
    """

    def __init__(self, capacity: int = 1_000_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        """Appends one transition. ``float(done)`` normalizes the termination
        flag to 0.0/1.0, so callers may pass a bool or a float."""
        self.buf.append((s, a, r, ns, float(done)))

    def _to_tensor(self, x):
        # np.array() first: stacking a tuple of arrays in NumPy and converting
        # once is markedly faster than letting torch walk the Python sequence.
        return torch.FloatTensor(np.array(x))

    def sample(self, batch_size: int):
        """A uniform sample without replacement, as five batched tensors."""
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            self._to_tensor(s),
            self._to_tensor(a),
            self._to_tensor(r).unsqueeze(1),
            self._to_tensor(ns),
            self._to_tensor(d).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)
