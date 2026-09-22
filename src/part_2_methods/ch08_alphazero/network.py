"""The unified AlphaZero network, sized for a board a laptop can train on.

The published network -- printed in ``alphago.py`` as the contrast -- is a
19-block, 256-channel residual tower with 22,837,864 parameters, trained on
thousands of TPUs. The architecture here is the same shape: a shared residual
trunk under a policy head and a value head. Two scale knobs are turned down,
``blocks`` from 19 to 4 and ``channels`` from 256 to 64, which brings it to
456,647 parameters on the 6x6 board and a full training run inside an hour.

Nothing about the algorithm changes. Only the budget does. That is the honest
version of "you can run AlphaZero at home", and it is worth being explicit
that the gap is a factor of fifty in parameters, not a difference in kind.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # The skip connection lets gradients reach the early layers.
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    """One trunk, two heads -- AlphaGo's two separate networks merged.

    AlphaGo trained a policy network and a value network with the same
    architecture and no shared weights, which meant learning the same board
    features twice. Merging them is most of what AlphaGo Zero changed, and it
    is why a leaf evaluation here costs one forward pass rather than two.
    """

    def __init__(self, input_planes, board_shape, action_size,
                 channels=64, blocks=4):
        super().__init__()
        rows, cols = board_shape
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # Policy head: collapse to a couple of planes, then score every action.
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * rows * cols, action_size)

        # Value head: collapse to one plane, then to a single number.
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32 * rows * cols, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.trunk(self.stem(x))
        policy = self.policy_fc(self.policy_conv(x).flatten(1))
        value = self.value_fc(self.value_conv(x).flatten(1))
        # Raw logits: the softmax happens where the legal mask is known, in
        # BatchedMCTS._expand. A softmax here would spend probability mass on
        # moves the rules forbid.
        return policy, value.squeeze(-1)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def build(game, channels=64, blocks=4):
    """A network shaped by the game, which is the only thing it reads."""
    return AlphaZeroNet(
        input_planes=game.input_planes,
        board_shape=game.board_shape,
        action_size=game.action_size,
        channels=channels,
        blocks=blocks,
    )


def pick_device(preference="auto"):
    """Whatever accelerator this laptop happens to have.

    MPS is chosen where it exists because the self-play fleet is batched: the
    thing that makes MPS a poor fit for chapter 7's token-by-token generation
    -- a fixed per-call latency of roughly 17 ms -- costs nothing here, where
    one call answers several hundred positions.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
