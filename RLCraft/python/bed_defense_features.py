"""Structured feature extractor for the BedWars defense task."""

from __future__ import annotations

import math

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BedDefenseFeaturesExtractor(BaseFeaturesExtractor):
    """Legacy Conv3D encoder retained so existing checkpoints still load."""

    FEATURES_DIM = 192

    def __init__(self, observation_space: spaces.Dict) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("bed-defense features require a Dict observation space")
        super().__init__(observation_space, features_dim=self.FEATURES_DIM)

        state_space = observation_space.spaces["state"]
        grid_space = observation_space.spaces["grid"]
        if len(state_space.shape) != 1 or len(grid_space.shape) != 4:
            raise ValueError("unexpected bed-defense observation shapes")

        self.state_encoder = nn.Sequential(
            nn.Linear(state_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        grid_convolution = nn.Sequential(
            nn.Conv3d(grid_space.shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(
                16,
                24,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(grid_space.sample()[None]).float()
            flattened_grid = int(grid_convolution(sample).shape[1])
        self.grid_encoder = nn.Sequential(
            grid_convolution,
            nn.Linear(flattened_grid, 128),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"].float())
        grid_features = self.grid_encoder(observations["grid"].float())
        return torch.cat((state_features, grid_features), dim=1)


class BedDefenseFlatFeaturesExtractor(BaseFeaturesExtractor):
    """Fast MLP encoder for the small, fixed bed-defense occupancy grid."""

    FEATURES_DIM = 192

    def __init__(self, observation_space: spaces.Dict) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("bed-defense features require a Dict observation space")
        super().__init__(observation_space, features_dim=self.FEATURES_DIM)

        state_space = observation_space.spaces["state"]
        grid_space = observation_space.spaces["grid"]
        if len(state_space.shape) != 1 or len(grid_space.shape) != 4:
            raise ValueError("unexpected bed-defense observation shapes")

        self.state_encoder = nn.Sequential(
            nn.Linear(state_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        flattened_grid = math.prod(grid_space.shape)
        self.grid_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_grid, 128),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        state_features = self.state_encoder(observations["state"].float())
        grid_features = self.grid_encoder(observations["grid"].float())
        return torch.cat((state_features, grid_features), dim=1)


__all__ = ["BedDefenseFeaturesExtractor", "BedDefenseFlatFeaturesExtractor"]
