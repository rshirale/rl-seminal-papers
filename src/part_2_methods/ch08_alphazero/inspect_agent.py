"""Look at what a trained agent believes, and play it.

    python -m src.part_2_methods.ch08_alphazero.inspect_agent runs/<run>/latest.pt
    python -m src.part_2_methods.ch08_alphazero.inspect_agent runs/<run>/latest.pt --play

The first form prints the value head's opinion of the opening position against
the proven game-theoretic value, and the search's preferred opening move. The
second drops into a game against it, which is the only way to find out what a
training curve actually bought.
"""

import argparse

import numpy as np
import torch

from .mcts import BatchedMCTS
from .network import build, pick_device
from .train import build_game


def load(path, device):
    """Rebuild the game and the network a checkpoint was trained with.

    The run's arguments travel inside the checkpoint, so a board size or a
    channel count never has to be remembered and retyped correctly.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    saved = blob["args"]
    game = build_game(saved["game"], saved["rows"], saved["cols"])
    net = build(game, saved["channels"], saved["blocks"]).to(device)
    net.load_state_dict(blob["model"])
    net.eval()
    return game, net, saved


def report(game, net, mcts, simulations):
    device = next(net.parameters()).device
    planes = torch.from_numpy(game.encode(game.initial_board())[None]).to(device)
    with torch.no_grad():
        _, value = net(planes)
    opinion = float(value.item())
    solved = game.solved_value()

    print(f"game                 {game.name}")
    print(f"value of the opening {opinion:+.3f}")
    if solved is not None:
        # A drawn board is the awkward case: "agrees" has to mean "close to
        # zero" rather than "same sign", since zero has no sign.
        agrees = (np.sign(opinion) == np.sign(solved)
                  or (solved == 0 and abs(opinion) < 0.25))
        print(f"proven value         {solved:+d}   "
              f"({'agrees' if agrees else 'DISAGREES'})")

    policies, _ = mcts.search([game.initial_board()], simulations,
                              add_noise=False)
    ranked = np.argsort(-policies[0])[:5]
    print("search's opening preferences:")
    for action in ranked:
        if policies[0][action] <= 0:
            continue
        print(f"  move {int(action):3d}   {policies[0][action]:.3f}")


def play(game, mcts, simulations):
    board = game.initial_board()
    human_turn = True
    while game.outcome(board) is None:
        print()
        print(game.render(board))
        legal = np.nonzero(game.legal_moves(board))[0].tolist()
        if human_turn:
            print(f"legal moves: {legal}")
            try:
                action = int(input("your move: ").strip())
            except (ValueError, EOFError):
                print("stopping")
                return
            if action not in legal:
                print("not a legal move")
                continue
        else:
            policies, _ = mcts.search([board], simulations, add_noise=False)
            action = int(np.argmax(policies[0]))
            print(f"agent plays {action}")
        board = game.apply_move(board, action)
        human_turn = not human_turn

    print()
    print(game.render(board))
    result = game.outcome(board)
    # `result` belongs to the player to move now, who did not make the last
    # move -- so a negative result means whoever just moved won.
    mover = "you" if human_turn else "the agent"
    print("draw" if result == 0 else
          (f"{mover} lost" if result < 0 else f"{mover} won"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    game, net, _ = load(args.checkpoint, device)
    mcts = BatchedMCTS(game, net, device)

    report(game, net, mcts, args.simulations)
    if args.play:
        play(game, mcts, args.simulations)


if __name__ == "__main__":
    main()
