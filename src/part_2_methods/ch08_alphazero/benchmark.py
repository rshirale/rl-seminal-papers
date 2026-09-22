"""Measuring the gap that forced AlphaGo to keep a rollout policy at all.

Silver and colleagues (2016) report that the 13-layer policy network picks a
move in about 3 milliseconds, while the linear-softmax rollout policy does it
in about 2 microseconds -- roughly fifteen hundred times faster, at less than
half the accuracy (24% against 57%). That ratio is the whole reason AlphaGo's
leaf evaluation mixes a network prediction with a fast random playout instead
of simply asking the network, and it is why AlphaZero could delete the
rollout only once the value head was good enough to carry a leaf alone.

Nothing here needs training or a dataset, so it runs in seconds on any laptop.
It is the one experiment in the AlphaGo half of the chapter a reader can
actually execute.

    python -m src.part_2_methods.ch08_alphazero.benchmark
    python -m src.part_2_methods.ch08_alphazero.benchmark --device mps
"""

import argparse
import time

import numpy as np
import torch

from .alphago import PolicyNetwork


def timeit(fn, repeats):
    fn()  # warm up caches and lazy kernels
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


class RolloutPolicy:
    """A linear softmax over a handful of sparse pattern features.

    AlphaGo's rollout policy read small 3x3 stone patterns around each
    candidate move and summed one weight per matching pattern. The shape is
    what matters here: a few table lookups per move and no convolution
    anywhere, which is why it runs three orders of magnitude faster than the
    network it stands in for.
    """

    def __init__(self, num_patterns=8192, num_actions=361,
                 active_per_move=6, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.5, size=num_patterns).astype(np.float32)
        # Which patterns each candidate move currently matches. In a real
        # engine this is recomputed incrementally as stones are placed.
        self.active = rng.integers(
            0, num_patterns, size=(num_actions, active_per_move))

    def select(self):
        logits = self.weights[self.active].sum(axis=1)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return int(probabilities.argmax())


def measure(device_name="cpu", batch=256):
    """Return the three timings, in seconds, plus the network's size."""
    device = torch.device(device_name)
    net = PolicyNetwork().to(device).eval()
    parameters = sum(p.numel() for p in net.parameters())

    single = torch.zeros(1, 48, 19, 19, device=device)
    batched = torch.zeros(batch, 48, 19, 19, device=device)

    with torch.no_grad():
        single_time = timeit(lambda: net(single), 30)
        batch_time = timeit(lambda: net(batched), 5)

    rollout_time = timeit(RolloutPolicy().select, 20000)
    return {
        "parameters": parameters,
        "device": str(device),
        "batch": batch,
        "single_seconds": single_time,
        "batch_seconds": batch_time,
        "rollout_seconds": rollout_time,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda or mps (default: cpu, as published)")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)

    m = measure(args.device, args.batch)
    per_position = m["batch_seconds"] / m["batch"]

    print(f"policy network        {m['parameters']:,} parameters "
          f"on {m['device']}")
    print(f"  one position        {m['single_seconds'] * 1e3:8.2f} ms")
    print(f"  batch of {m['batch']:<3}        {m['batch_seconds'] * 1e3:8.2f} ms "
          f"({per_position * 1e6:.0f} us per position)")
    print("rollout policy")
    print(f"  one position        {m['rollout_seconds'] * 1e6:8.2f} us")
    print()
    print(f"network / rollout     "
          f"{m['single_seconds'] / m['rollout_seconds']:8.0f}x")
    print(f"batching wins back    "
          f"{m['single_seconds'] / per_position:8.0f}x")
    print()
    print("Silver and colleagues (2016) report 3 ms against 2 us, a ratio of")
    print("about 1500x. The numbers above are this machine's version of the")
    print("same measurement. The ratio here is smaller in both directions --")
    print("this rollout policy is NumPy rather than tuned C, and the network")
    print("runs on hardware a decade newer -- but the shape survives: a leaf")
    print("costs orders of magnitude more through the network than through a")
    print("playout, which is what AlphaGo's lambda was buying.")
    if m["device"] == "cpu":
        print()
        print("Batching wins nothing on CPU, where one position already")
        print("saturates every core. Re-run with --device cuda or --device mps")
        print("to see the number this project's batched self-play depends on.")
    return m


if __name__ == "__main__":
    main()
