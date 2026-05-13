import torch
import torch.nn as nn


class SimpleDQN(nn.Module):
    """MLP Q-network for vector-based environments (e.g. CartPole)."""
    def __init__(self, input_dim: int, num_actions: int):
        super(SimpleDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) Architecture from Mnih et al. (2015) - Nature paper.
    Takes a stack of 4 grayscale frames (84x84) and outputs Q-values for each possible action.
    """
    def __init__(self, input_channels: int, num_actions: int):
        super(DQN, self).__init__()
        
        # The CNN Feature Extractor
        # Input shape: (Batch Size, 4, 84, 84)
        self.conv = nn.Sequential(
            # Layer 1: 32 filters of 8x8 with stride 4
            # 84x84 -> 20x20
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Layer 2: 64 filters of 4x4 with stride 2
            # 20x20 -> 9x9
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Layer 3: 64 filters of 3x3 with stride 1
            # 9x9 -> 7x7
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # The Fully Connected Action-Value Head
        # The output of conv3 is 64 channels * 7 height * 7 width = 3136 flattened features
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  # Output: One linear unit per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        x expected to be a normalized float tensor of shape (B, C, H, W).
        """
        # Pass through convolutional layers
        x = self.conv(x)
        
        # Flatten the spatial dimensions for the fully connected layers
        # x.size(0) is the batch size
        x = x.reshape(x.size(0), -1)
        
        # Pass through fully connected layers to get Q-values
        x = self.fc(x)
        
        return x
