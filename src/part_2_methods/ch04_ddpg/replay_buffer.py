import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Circular experience replay buffer storing (s, a, r, s', done) transitions.

    Default capacity of 1,000,000 matches Lillicrap et al. (2015).
    Python's deque with maxlen discards the oldest entry automatically
    when the buffer is full — no manual pointer arithmetic required.
    """

    def __init__(self, max_size: int = 1_000_000):
        self.buf = deque(maxlen=max_size)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        to_t = lambda x: torch.FloatTensor(np.array(x))
        return (
            to_t(s), to_t(a),
            to_t(r).unsqueeze(1),
            to_t(ns),
            to_t(d).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)
