from __future__ import annotations


def _require_torch():
    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on runtime environment.
        raise RuntimeError("torch is required for model training") from exc
    return torch, nn


torch, nn = _require_torch()


class HeatmapOffsetNet(nn.Module):
    def __init__(self, num_landmarks: int = 9, width: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.point_mlp = nn.Sequential(
            nn.Linear(3, width),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width * 2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(width * 2, width * 4),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
        )
        fused_width = width * 8
        self.head = nn.Sequential(
            nn.Linear(fused_width, width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Linear(width * 2, num_landmarks)
        self.offset_head = nn.Linear(width * 2, num_landmarks * 3)

    def forward(self, points):
        batch_size, num_points, _ = points.shape
        flat = points.reshape(batch_size * num_points, 3)
        feat = self.point_mlp(flat).reshape(batch_size, num_points, -1)
        global_feat = feat.max(dim=1, keepdim=True).values.expand(-1, num_points, -1)
        fused = torch.cat([feat, global_feat], dim=-1)
        hidden = self.head(fused)
        heatmap_logits = self.heatmap_head(hidden)
        offsets = self.offset_head(hidden).reshape(batch_size, num_points, self.num_landmarks, 3)
        return heatmap_logits, offsets
