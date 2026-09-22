"""The published 19x19 architectures -- read and profiled, not trained.

These are the networks the chapter prints: AlphaGo's separate policy and value
networks from Silver and colleagues (2016), and AlphaGo Zero's merged tower
from Silver and colleagues (2017). They are here to be instantiated,
measured and compared against the project's network, not to be trained.
Training them honestly needs the 30 million position KGS corpus, 50 GPUs for a
week for the value network alone, and in AlphaGo Zero's case thousands of
TPUs. Pretending otherwise is the contradiction the chapter's project exists
to avoid.

What *is* runnable from them is ``benchmark.py``, which times ``PolicyNetwork``
against a fast rollout policy and reproduces the millisecond-versus-microsecond
gap that forced AlphaGo to keep rollouts at all.

    python -m src.part_2_methods.ch08_alphazero.alphago

prints each network's parameter count beside the project's, which is the
comparison worth having in front of you before reading ``network.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """AlphaGo's 13-layer supervised-learning policy network.

    48 feature planes in -- stones, liberties, capture sizes, legality, move
    history -- a 5x5 convolution, eleven 3x3 convolutions at 192 filters, and a
    1x1 head that produces one logit per intersection. Trained on 30 million
    positions from the KGS server, it predicted the human expert's move 57% of
    the time.

    The softmax is inside ``forward`` here because that is how the chapter
    prints it. The project's network returns raw logits instead, so the legal
    mask can be applied before any probability mass is spent.
    """

    def __init__(self, num_input_channels=48, filters=192, hidden_layers=11):
        super().__init__()
        # The first layer uses a larger 5x5 kernel.
        self.conv1 = nn.Conv2d(num_input_channels, filters,
                               kernel_size=5, padding=2)
        # Subsequent layers use 3x3 kernels.
        self.hidden_layers = nn.ModuleList([
            nn.Conv2d(filters, filters, kernel_size=3, padding=1)
            for _ in range(hidden_layers)
        ])
        # The final layer outputs one channel for the 19x19 board.
        self.policy_head = nn.Conv2d(filters, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.policy_head(x)
        # Flatten the 19x19 output into a vector of 361 logits.
        x = x.view(x.size(0), -1)
        return F.softmax(x, dim=1)


class ValueNetwork(nn.Module):
    """AlphaGo's value network: the same feature extractor, a scalar out.

    Identical in shape to the policy network up to the head, and trained with
    mean squared error against the game's eventual outcome. The weights are
    *not* shared -- that is the whole point of the contrast with
    ``network.AlphaZeroNet``, which merges the two into one trunk.

    Trained first on the same human corpus, it memorised: 0.19 mean squared
    error on the training set against 0.37 on held-out positions, because
    successive positions within one game are almost the same position. The fix
    was 30 million self-play games contributing *one* position each.
    """

    def __init__(self, num_input_channels=48, filters=192, hidden_layers=11,
                 board_size=19):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input_channels, filters,
                               kernel_size=5, padding=2)
        self.hidden_layers = nn.ModuleList([
            nn.Conv2d(filters, filters, kernel_size=3, padding=1)
            for _ in range(hidden_layers)
        ])
        # The value head collapses features for scalar regression.
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = F.relu(self.value_conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # Tanh squashes the estimate into the [-1, 1] range.
        return torch.tanh(self.fc2(x))


class PublishedResBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # The skip connection allows deep gradient flow.
        x = x + residual
        return F.relu(x)


class PublishedAlphaZeroNet(nn.Module):
    """AlphaGo Zero's network at its published size: 19 blocks, 256 channels.

    Same architecture as ``network.AlphaZeroNet`` -- one residual trunk, a
    policy head and a value head -- with both scale knobs at the paper's
    settings. 362 actions is 361 intersections plus a pass.

    It carries roughly fifty times the project's parameters, which is the
    contrast the chapter draws: the algorithm a laptop runs is the published
    one, and the difference is budget rather than kind.
    """

    def __init__(self, input_channels=17, board_size=19, num_actions=362,
                 channels=256, blocks=19):
        super().__init__()
        self.conv_input = nn.Conv2d(input_channels, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        # The shared residual trunk extracts generalized board features.
        self.res_blocks = nn.Sequential(
            *[PublishedResBlock(channels) for _ in range(blocks)])

        # Policy head: collapses channels for action logits.
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size ** 2, num_actions)

        # Value head: collapses channels for scalar state evaluation.
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size ** 2, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        # Raw logits; the softmax is applied during MCTS expansion.
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def parameter_counts():
    """Every network in the chapter, measured rather than quoted."""
    from .connect_four import ConnectFour
    from .network import build

    project = build(ConnectFour(6, 6), channels=64, blocks=4)
    return [
        ("AlphaGo policy network (19x19, 48 planes, 192 filters)",
         sum(p.numel() for p in PolicyNetwork().parameters())),
        ("AlphaGo value network (same extractor, scalar head)",
         sum(p.numel() for p in ValueNetwork().parameters())),
        ("AlphaGo Zero network (19 blocks x 256 channels)",
         sum(p.numel() for p in PublishedAlphaZeroNet().parameters())),
        ("this chapter's project (6x6, 4 blocks x 64 channels)",
         project.parameter_count()),
    ]


def main():
    rows = parameter_counts()
    width = max(len(label) for label, _ in rows)
    print("Parameter counts, by instantiating each network:\n")
    for label, count in rows:
        print(f"  {label:<{width}}  {count:>12,}")
    published, project = rows[2][1], rows[3][1]
    print(f"\n  the published tower is {published / project:.0f}x the "
          f"project's, at the same architecture.")


if __name__ == "__main__":
    main()
