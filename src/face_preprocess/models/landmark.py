from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from face_preprocess.errors import ArtifactError
from face_preprocess.models.checkpoint import (
    load_checkpoint,
    require_checkpoint_keys,
    require_torch,
)
from face_preprocess.models.common import normalize_points, sample_indices
from face_preprocess.types import Mesh


LANDMARK_NAMES = (
    "bijian",
    "bigen",
    "bixia",
    "wyjzuo",
    "nyjzuo",
    "nyjyou",
    "wyjyou",
    "kouzuo",
    "kouyou",
)


def snap_landmarks(
    continuous: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, indices = cKDTree(np.asarray(vertices, dtype=float)).query(
        np.asarray(continuous, dtype=float), k=1
    )
    indices = np.asarray(indices, dtype=np.int64)
    return (
        np.asarray(vertices, dtype=float)[indices].copy(),
        indices,
        np.asarray(distances, dtype=float),
    )


@dataclass(frozen=True)
class LandmarkResult:
    names: tuple[str, ...]
    continuous: np.ndarray
    snapped: np.ndarray
    snapped_vertex_indices: np.ndarray
    snap_distances: np.ndarray
    heatmap_peaks: np.ndarray
    topk_spreads: np.ndarray

    def coordinates(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in self.snapped[index]]
            for index, name in enumerate(self.names)
        }


class LandmarkPredictor:
    def __init__(self, model: Any, names: tuple[str, ...], device: str) -> None:
        self.model = model
        self.names = names
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str) -> "LandmarkPredictor":
        payload = load_checkpoint(path, device)
        require_checkpoint_keys(payload, {"model", "args", "landmarks"}, "landmark")
        names = tuple(str(item) for item in payload["landmarks"])
        if names != LANDMARK_NAMES:
            raise ArtifactError(
                f"landmark checkpoint names/order mismatch: expected {LANDMARK_NAMES}, got {names}"
            )
        args = payload["args"]
        if not isinstance(args, dict):
            raise ArtifactError("landmark checkpoint args must be a mapping")
        if "width" not in args or "dropout" not in args:
            raise ArtifactError("landmark checkpoint args must include width and dropout")
        from face_preprocess.models._landmark_network import HeatmapOffsetNet

        model = HeatmapOffsetNet(
            num_landmarks=len(names),
            width=int(args["width"]),
            dropout=float(args["dropout"]),
        )
        try:
            model.load_state_dict(payload["model"], strict=True)
        except Exception as exc:
            raise ArtifactError(f"landmark checkpoint state mismatch: {exc}") from exc
        model.to(device).eval()
        return cls(model, names, device)

    def predict(
        self, mesh: Mesh, *, num_points: int, top_k: int, seed: int
    ) -> LandmarkResult:
        torch = require_torch()
        normalized, center, scale = normalize_points(mesh.vertices)
        indices = sample_indices(len(mesh.vertices), num_points, seed)
        sampled = normalized[indices]
        points = torch.from_numpy(sampled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            heatmap_logits, offsets = self.model(points)
            probabilities = torch.sigmoid(heatmap_logits)
            actual_k = min(int(top_k), sampled.shape[0])
            predictions = []
            spreads = []
            peaks = []
            for landmark_index in range(len(self.names)):
                values, top_indices = probabilities[:, :, landmark_index].topk(
                    actual_k, dim=1
                )
                gathered_points = torch.gather(
                    points, 1, top_indices[:, :, None].expand(1, actual_k, 3)
                )
                landmark_offsets = offsets[:, :, landmark_index, :]
                gathered_offsets = torch.gather(
                    landmark_offsets,
                    1,
                    top_indices[:, :, None].expand(1, actual_k, 3),
                )
                candidates = gathered_points + gathered_offsets
                weights = values / values.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
                prediction = (candidates * weights[:, :, None]).sum(dim=1)
                predictions.append(prediction)
                spread = torch.sqrt(
                    (
                        torch.sum(
                            (candidates - prediction[:, None, :]) ** 2, dim=2
                        )
                        * weights
                    ).sum(dim=1)
                )
                spreads.append(spread)
                peaks.append(values[:, 0])
            continuous_normalized = (
                torch.stack(predictions, dim=1).squeeze(0).cpu().numpy()
            )
            topk_spreads = torch.stack(spreads, dim=1).squeeze(0).cpu().numpy()
            heatmap_peaks = torch.stack(peaks, dim=1).squeeze(0).cpu().numpy()
        continuous = continuous_normalized * scale + center
        snapped, snapped_indices, snap_distances = snap_landmarks(
            continuous, mesh.vertices
        )
        return LandmarkResult(
            names=self.names,
            continuous=continuous,
            snapped=snapped,
            snapped_vertex_indices=snapped_indices,
            snap_distances=snap_distances,
            heatmap_peaks=np.asarray(heatmap_peaks, dtype=float),
            topk_spreads=np.asarray(topk_spreads, dtype=float) * scale,
        )

