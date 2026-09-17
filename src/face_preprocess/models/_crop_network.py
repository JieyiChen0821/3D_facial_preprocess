from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    return points[batch, indices]


def _knn_indices(points: torch.Tensor, k: int, chunk_size: int) -> torch.Tensor:
    batch_size, point_count, _ = points.shape
    if point_count <= 1:
        return torch.zeros((batch_size, point_count, 1), dtype=torch.long, device=points.device)
    k_eff = min(k, point_count - 1)
    chunks = []
    for start in range(0, point_count, chunk_size):
        end = min(start + chunk_size, point_count)
        distances = torch.cdist(points[:, start:end, :], points)
        nearest = torch.topk(distances, k=k_eff + 1, dim=-1, largest=False).indices[:, :, 1:]
        chunks.append(nearest)
    return torch.cat(chunks, dim=1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.net(features), inplace=True)


class LocalAggregationBlock(nn.Module):
    def __init__(self, channels: int, k_neighbors: int, knn_chunk_size: int, dropout: float) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.knn_chunk_size = knn_chunk_size
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        indices = _knn_indices(points, self.k_neighbors, self.knn_chunk_size)
        features_bn = features.transpose(1, 2).contiguous()
        neighbor_features = _index_points(features_bn, indices)
        neighbor_points = _index_points(points, indices)
        center_features = features_bn.unsqueeze(2)
        center_points = points.unsqueeze(2)
        edge_features = torch.cat([neighbor_features - center_features, neighbor_points - center_points], dim=-1)
        aggregated = self.edge_mlp(edge_features).max(dim=2).values.transpose(1, 2).contiguous()
        return F.relu(features + self.norm(aggregated), inplace=True)


class PointNeXtS(nn.Module):
    """Compact PointNeXt-S-style binary point segmentation model.

    The model intentionally avoids external CUDA point-cloud extensions so it
    can run with a plain PyTorch installation on the A100 queue.
    """

    def __init__(self, k_neighbors: int = 16, width: int = 64, blocks: int = 4, dropout: float = 0.1, knn_chunk_size: int = 1024) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        for _ in range(blocks):
            layers.append(LocalAggregationBlock(width, k_neighbors, knn_chunk_size, dropout))
            layers.append(ResidualMLPBlock(width, dropout))
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(width, 2, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.stem(points.transpose(1, 2).contiguous())
        for block in self.blocks:
            if isinstance(block, LocalAggregationBlock):
                features = block(points, features)
            else:
                features = block(features)
        return self.head(features).transpose(1, 2).contiguous()
