"""The AlphaZero training loop, on a wall-clock budget.

One iteration is the whole cycle: play a fleet of games against the current
network, fold them into a replay buffer, take a few gradient steps, and
measure whether the agent got any better. The loop stops when the budget runs
out rather than after a fixed number of iterations, because the budget is what
this chapter's project is about -- the number of iterations is the outcome,
not the setting.

    python -m src.part_2_methods.ch08_alphazero.train --budget-seconds 3600

The default board is 6x6 Connect Four, whose game-theoretic value is a proven
draw. That matters for what the loop can honestly show: an agent approaching
perfect play on a drawn board ends up drawing against itself every single
time, and ``mirror_draw_rate`` measures exactly that. Pass ``--cols 7`` for the
standard board, a proven first-player win, or ``--game othello`` for the same
agent on a second domain with nothing else changed.

There is deliberately no arena gate. AlphaGo Zero (2017) kept a champion
network and promoted a challenger only on a 55% win rate; the generalized
AlphaZero (2018) removed that evaluation step entirely and continuously
updated one network, which is what happens here. On much longer runs against
harder games, a gate is worth adding back to prevent regression.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .arena import evaluate_agent
from .connect_four import ConnectFour
from .mcts import BatchedMCTS
from .network import build, pick_device
from .othello import Othello
from .selfplay import play_batch


class RunResult(NamedTuple):
    """What a run leaves behind, for a notebook or a test to pick up.

    ``history`` is the per-iteration log, also written to history.json. The
    network and game come back too because the obvious next thing to do with a
    finished run is measure it, and rebuilding them from the checkpoint to do
    that is a step with nothing to teach.
    """

    history: list
    net: object
    game: object
    device: object
    out_dir: Path


def build_game(name, rows, cols):
    if name == "connect4":
        return ConnectFour(rows or 6, cols or 7)
    if name == "othello":
        return Othello(rows or 6)
    raise ValueError(f"unknown game {name!r}")


def loss_fn(policy_logits, value_pred, target_policy, target_value):
    """AlphaZero's single objective, over a trunk shared by both heads.

    The value head is scored against the game's outcome, the policy head
    against the search's own visit counts -- the search is the teacher, and the
    network is trying to predict what the search would have concluded without
    having to run it. The two terms simply add, and both gradients reach the
    same residual trunk, which is what makes one network cheaper than AlphaGo's
    two rather than merely tidier.
    """
    value_loss = F.mse_loss(value_pred, target_value)
    policy_loss = -(target_policy
                    * F.log_softmax(policy_logits, dim=1)).sum(1).mean()
    return value_loss + policy_loss, value_loss.item(), policy_loss.item()


def train_steps(net, optimizer, buffer, device, batch_size, steps, rng):
    """A fixed number of gradient steps over the replay buffer."""
    net.train()
    totals = np.zeros(3)
    if len(buffer) < batch_size:
        return totals
    for _ in range(steps):
        idx = rng.integers(0, len(buffer), size=batch_size)
        planes = torch.from_numpy(
            np.stack([buffer[i][0] for i in idx])).to(device)
        target_policy = torch.from_numpy(
            np.stack([buffer[i][1] for i in idx])).to(device)
        target_value = torch.from_numpy(
            np.array([buffer[i][2] for i in idx], dtype=np.float32)).to(device)

        policy_logits, value_pred = net(planes)
        loss, value_loss, policy_loss = loss_fn(
            policy_logits, value_pred, target_policy, target_value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += (loss.item(), value_loss, policy_loss)
    net.eval()
    return totals / steps


def root_value(game, net, device):
    """What the network thinks of the opening position.

    The solved-game tables let this one be marked right or wrong. Read it as a
    sanity check rather than as evidence -- see ``arena.mirror_match`` for why.
    """
    planes = torch.from_numpy(
        game.encode(game.initial_board())[None]).to(device)
    with torch.no_grad():
        _, value = net(planes)
    return float(value.item())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="connect4",
                        choices=["connect4", "othello"])
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--budget-seconds", type=float, default=3600)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=192)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    game = build_game(args.game, args.rows, args.cols)
    device = pick_device(args.device)
    net = build(game, args.channels, args.blocks).to(device)
    net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    mcts = BatchedMCTS(game, net, device)
    buffer = deque(maxlen=args.buffer_size)

    out_dir = Path(args.out) / f"{game.name}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    solved = game.solved_value()
    print(f"game        {game.name}")
    print(f"device      {device}")
    print(f"parameters  {net.parameter_count():,}")
    print(f"budget      {args.budget_seconds:.0f}s")
    if solved is not None:
        print(f"solved value of the opening position: {solved:+d}")
    print()

    history = []
    started = time.time()
    for iteration in range(1, args.max_iterations + 1):
        elapsed = time.time() - started
        if elapsed >= args.budget_seconds:
            print(f"budget spent after {iteration - 1} iterations")
            break

        t0 = time.time()
        examples, results = play_batch(
            game, mcts, args.games_per_iteration, args.simulations, rng=rng)
        buffer.extend(examples)
        selfplay_time = time.time() - t0

        t0 = time.time()
        losses = train_steps(net, optimizer, buffer, device,
                             args.batch_size, args.train_steps, rng)
        train_time = time.time() - t0

        decisive = sum(1 for r in results if r != 0) / len(results)
        record = {
            "iteration": iteration,
            "elapsed": round(time.time() - started, 1),
            "selfplay_seconds": round(selfplay_time, 1),
            "train_seconds": round(train_time, 1),
            "buffer": len(buffer),
            # On a drawn board this should drift down: fewer games decided by
            # a blunder is what getting better looks like here.
            "decisive_fraction": round(decisive, 3),
            "loss": round(float(losses[0]), 4),
            "value_loss": round(float(losses[1]), 4),
            "policy_loss": round(float(losses[2]), 4),
            "root_value": round(root_value(game, net, device), 3),
        }

        if iteration % args.eval_every == 0:
            record.update(evaluate_agent(
                game, mcts, args.simulations, args.eval_games, rng))

        history.append(record)
        print(" ".join(f"{k}={v}" for k, v in record.items()), flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": net.state_dict(), "args": vars(args)},
                   out_dir / "latest.pt")

    total = time.time() - started
    print(f"\ntotal wall clock {total:.0f}s ({total / 60:.1f} min)")
    print(f"artifacts in {out_dir}")
    return RunResult(history, net, game, device, out_dir)


if __name__ == "__main__":
    main()
