from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from face_preprocess.types import Mesh


@dataclass(frozen=True)
class CropAdapterResult:
    mesh: Mesh
    probabilities: np.ndarray | None = None
    initial_mask: np.ndarray | None = None
    final_mask: np.ndarray | None = None
    metrics: dict[str, float | int] | None = None


@dataclass(frozen=True)
class LandmarkAdapterResult:
    names: tuple[str, ...]
    snapped: np.ndarray
    continuous: np.ndarray | None = None
    snapped_vertex_indices: np.ndarray | None = None
    snap_distances: np.ndarray | None = None
    heatmap_peaks: np.ndarray | None = None
    topk_spreads: np.ndarray | None = None

    def coordinates(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in self.snapped[index]]
            for index, name in enumerate(self.names)
        }


class CropAdapter(Protocol):
    def crop(self, mesh: Mesh, *, seed: int) -> CropAdapterResult: ...


class LandmarkAdapter(Protocol):
    def predict(self, mesh: Mesh, *, seed: int) -> LandmarkAdapterResult: ...
