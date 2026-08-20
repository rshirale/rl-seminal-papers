import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Circular experience replay storing (s, a, r, s', done) transitions.

    Identical in purpose to Chapter 3's buffer; the differences are that
    actions are continuous vectors rather than integer indices, and that the
    default capacity of 1,000,000 matches Lillicrap et al. (2015).

    ``deque(maxlen=...)`` discards the oldest entry automatically once full, so
    there is no pointer arithmetic to get wrong. Chapter 3 used pre-allocated
    NumPy arrays instead, because an Atari buffer holds 1,000,000 84x84 frames
    and the difference between ``uint8`` arrays and Python objects is roughly
    6.6 GB against 26 GB. Pendulum's three-float states make that tradeoff
    irrelevant, and the deque is the clearer teaching object.

    Mirrors listing 4.7 in the chapter.
    """

    def __init__(self, max_size: int = 1_000_000):
        self.buf = deque(maxlen=max_size)

    def push(self, s, a, r, ns, done) -> None:
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size: int):
        """Returns one uniformly sampled mini-batch as float32 CPU tensors.

        Each field is stacked into a single contiguous float32 array first,
        then wrapped with ``torch.from_numpy``, which shares that array's memory
        rather than copying it. Building tensors straight from the list of
        tuples instead walks Python objects on every call -- on the critical
        path of a training step that runs once per environment step, which is
        the kind of overhead that quietly doubles a run's wall time.

        Rewards and done flags come back shaped (batch_size, 1) so they line up
        with the critic's (batch_size, 1) output. Left as (batch_size,) they
        would broadcast into a (batch_size, batch_size) target and the Bellman
        backup would be silently wrong rather than raise.
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)

        def stack(field):
            return torch.from_numpy(
                np.asarray(np.stack(field), dtype=np.float32)
            )

        return (
            stack(s),
            stack(a),
            torch.from_numpy(np.asarray(r, dtype=np.float32)).unsqueeze(1),
            stack(ns),
            torch.from_numpy(np.asarray(d, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)
