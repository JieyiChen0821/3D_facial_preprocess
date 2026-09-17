"""Built-in adapters for the checkpoints packaged with this pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from face_preprocess.adapters.contracts import CropAdapterResult, LandmarkAdapterResult
from face_preprocess.models.crop import CropPredictor
from face_preprocess.models.landmark import LandmarkPredictor
from face_preprocess.types import Mesh


@dataclass
class PointNextCropAdapter:
    predictor: CropPredictor
    parameters: dict[str, Any]

    @classmethod
    def from_config(cls, config: Any, device: str) -> "PointNextCropAdapter":
        return cls(
            predictor=CropPredictor.from_checkpoint(config.checkpoint, device),
            parameters=config.parameters,
        )

    def crop(self, mesh: Mesh, *, seed: int) -> CropAdapterResult:
        sampling = self.parameters["input"]["sampling"]
        inference = self.parameters["inference"]
        mapping = self.parameters["output_mapping"]
        postprocess = self.parameters["postprocess"]
        result = self.predictor.crop(
            mesh,
            num_points=int(sampling["num_points"]),
            extra_passes=int(inference["extra_passes"]),
            seed=int(seed),
            face_class_index=int(mapping["face_class_index"]),
            threshold=float(postprocess["probability_threshold"]),
            smooth_iterations=int(postprocess["smooth_iterations"]),
            smooth_alpha=float(postprocess["smooth_alpha"]),
            keep_largest_component=bool(postprocess["keep_largest_component"]),
            min_component_vertices=int(postprocess["min_component_vertices"]),
        )
        return CropAdapterResult(
            mesh=result.mesh,
            probabilities=result.probabilities,
            initial_mask=result.initial_mask,
            final_mask=result.final_mask,
            metrics=result.metrics,
        )


def _optional_reorder(value: np.ndarray | None, order: np.ndarray) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    return array[order].copy()


@dataclass
class HeatmapOffsetLandmarkAdapter:
    predictor: LandmarkPredictor
    parameters: dict[str, Any]

    @classmethod
    def from_config(cls, config: Any, device: str) -> "HeatmapOffsetLandmarkAdapter":
        return cls(
            predictor=LandmarkPredictor.from_checkpoint(config.checkpoint, device),
            parameters=config.parameters,
        )

    def predict(self, mesh: Mesh, *, seed: int) -> LandmarkAdapterResult:
        sampling = self.parameters["input"]["sampling"]
        inference = self.parameters["inference"]
        mapping = self.parameters["output_mapping"]
        result = self.predictor.predict(
            mesh,
            num_points=int(sampling["num_points"]),
            top_k=int(inference["top_k"]),
            seed=int(seed),
        )
        native = tuple(str(item) for item in mapping["native_order"])
        canonical = tuple(str(item) for item in mapping["canonical_order"])
        if tuple(result.names) != native:
            raise ValueError(
                f"landmark adapter native order mismatch: expected {native}, got {tuple(result.names)}"
            )
        if len(canonical) != len(native) or set(canonical) != set(native):
            raise ValueError("landmark adapter canonical order must be a permutation of native order")
        order = np.asarray([native.index(name) for name in canonical], dtype=np.int64)
        return LandmarkAdapterResult(
            names=canonical,
            snapped=np.asarray(result.snapped, dtype=float)[order].copy(),
            continuous=_optional_reorder(result.continuous, order),
            snapped_vertex_indices=_optional_reorder(result.snapped_vertex_indices, order),
            snap_distances=_optional_reorder(result.snap_distances, order),
            heatmap_peaks=_optional_reorder(result.heatmap_peaks, order),
            topk_spreads=_optional_reorder(result.topk_spreads, order),
        )
