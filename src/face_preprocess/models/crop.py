from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from face_preprocess.errors import ArtifactError
from face_preprocess.models.checkpoint import (
    load_checkpoint,
    require_checkpoint_keys,
    require_torch,
)
from face_preprocess.models.common import normalize_points
from face_preprocess.types import Mesh


def vote_indices(
    point_count: int, num_points: int, extra_passes: int, seed: int
) -> list[np.ndarray]:
    if point_count <= 0 or num_points <= 0 or extra_passes < 0:
        raise ValueError("invalid voting parameters")
    rng = np.random.default_rng(int(seed))
    shuffled = np.arange(point_count, dtype=np.int64)
    rng.shuffle(shuffled)
    batches: list[np.ndarray] = []
    for start in range(0, point_count, num_points):
        block = shuffled[start : start + num_points]
        if len(block) < num_points:
            block = np.concatenate(
                [
                    block,
                    rng.choice(
                        shuffled,
                        size=num_points - len(block),
                        replace=point_count < num_points - len(block),
                    ),
                ]
            )
        batches.append(block.astype(np.int64))
    for _ in range(extra_passes):
        batches.append(
            rng.choice(
                shuffled, size=num_points, replace=point_count < num_points
            ).astype(np.int64)
        )
    return batches


def _vertex_neighbors(mesh: Mesh) -> list[np.ndarray]:
    neighbors: list[set[int]] = [set() for _ in mesh.vertices]
    for a, b, c in mesh.faces:
        neighbors[int(a)].update((int(b), int(c)))
        neighbors[int(b)].update((int(a), int(c)))
        neighbors[int(c)].update((int(a), int(b)))
    return [np.asarray(sorted(item), dtype=np.int64) for item in neighbors]


def _components(mask: np.ndarray, neighbors: list[np.ndarray]) -> list[np.ndarray]:
    visited = np.zeros(len(mask), dtype=bool)
    output: list[np.ndarray] = []
    for raw_start in np.flatnonzero(mask):
        start = int(raw_start)
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for raw_neighbor in neighbors[current]:
                neighbor = int(raw_neighbor)
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        output.append(np.asarray(component, dtype=np.int64))
    return output


def probabilities_to_mask(
    mesh: Mesh,
    probabilities: np.ndarray,
    *,
    threshold: float,
    smooth_iterations: int,
    smooth_alpha: float = 0.5,
    keep_largest_component: bool = True,
    min_component_vertices: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float32).copy()
    if values.shape != (len(mesh.vertices),):
        raise ValueError("probability count does not match mesh vertex count")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    neighbors = _vertex_neighbors(mesh)
    alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
    for _ in range(max(0, int(smooth_iterations))):
        updated = values.copy()
        for index, adjacent in enumerate(neighbors):
            if len(adjacent):
                updated[index] = (1.0 - alpha) * values[index] + alpha * float(
                    values[adjacent].mean()
                )
        values = updated
    initial = values >= float(threshold)
    components = _components(initial, neighbors)
    final = np.zeros_like(initial)
    minimum = max(1, int(min_component_vertices))
    if components:
        selected = [max(components, key=len)] if keep_largest_component else components
        for component in selected:
            if len(component) >= minimum:
                final[component] = True
    return values, initial, final


def crop_mesh(mesh: Mesh, keep_mask: np.ndarray) -> Mesh:
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.shape != (len(mesh.vertices),):
        raise ValueError("keep_mask length does not match mesh vertex count")
    keep_faces = np.all(mask[mesh.faces], axis=1)
    old_faces = mesh.faces[keep_faces]
    if len(old_faces) == 0:
        return Mesh(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
    kept = np.unique(old_faces.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    return Mesh(mesh.vertices[kept], remap[old_faces])


@dataclass(frozen=True)
class CropResult:
    mesh: Mesh
    probabilities: np.ndarray
    initial_mask: np.ndarray
    final_mask: np.ndarray
    metrics: dict[str, float | int]


class CropPredictor:
    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str) -> "CropPredictor":
        payload = load_checkpoint(path, device)
        require_checkpoint_keys(payload, {"model_state", "config"}, "crop")
        config = payload["config"]
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            raise ArtifactError("crop checkpoint config.model must be a mapping")
        model_config = config["model"]
        required = {"k_neighbors", "width", "blocks", "dropout", "knn_chunk_size"}
        missing = sorted(required - set(model_config))
        if missing:
            raise ArtifactError(
                f"crop checkpoint model config is missing: {', '.join(missing)}"
            )
        from face_preprocess.models._crop_network import PointNeXtS

        model = PointNeXtS(
            k_neighbors=int(model_config["k_neighbors"]),
            width=int(model_config["width"]),
            blocks=int(model_config["blocks"]),
            dropout=float(model_config["dropout"]),
            knn_chunk_size=int(model_config["knn_chunk_size"]),
        )
        try:
            model.load_state_dict(payload["model_state"], strict=True)
        except Exception as exc:
            raise ArtifactError(f"crop checkpoint state mismatch: {exc}") from exc
        model.to(device).eval()
        return cls(model, device)

    def predict_probabilities(
        self,
        vertices: np.ndarray,
        *,
        num_points: int,
        extra_passes: int,
        seed: int,
        face_class_index: int = 1,
    ) -> np.ndarray:
        torch = require_torch()
        normalized, _, _ = normalize_points(vertices)
        sums = np.zeros(len(vertices), dtype=np.float64)
        counts = np.zeros(len(vertices), dtype=np.float64)
        with torch.no_grad():
            for choice in vote_indices(
                len(vertices), num_points=num_points, extra_passes=extra_passes, seed=seed
            ):
                points = torch.from_numpy(normalized[choice]).unsqueeze(0).to(self.device)
                logits = self.model(points)
                if int(face_class_index) < 0 or int(face_class_index) >= int(logits.shape[-1]):
                    raise ValueError(
                        f"face_class_index {face_class_index} is outside model output with "
                        f"{int(logits.shape[-1])} classes"
                    )
                probabilities = (
                    torch.softmax(logits, dim=-1)[0, :, int(face_class_index)]
                    .detach()
                    .cpu()
                    .numpy()
                )
                np.add.at(sums, choice, probabilities)
                np.add.at(counts, choice, 1.0)
        counts[counts == 0.0] = 1.0
        return (sums / counts).astype(np.float32)

    def crop(
        self,
        mesh: Mesh,
        *,
        num_points: int,
        extra_passes: int,
        seed: int,
        threshold: float,
        smooth_iterations: int,
        smooth_alpha: float,
        keep_largest_component: bool,
        min_component_vertices: int,
        face_class_index: int = 1,
    ) -> CropResult:
        probabilities = self.predict_probabilities(
            mesh.vertices,
            num_points=num_points,
            extra_passes=extra_passes,
            seed=seed,
            face_class_index=face_class_index,
        )
        smoothed, initial, final = probabilities_to_mask(
            mesh,
            probabilities,
            threshold=threshold,
            smooth_iterations=smooth_iterations,
            smooth_alpha=smooth_alpha,
            keep_largest_component=keep_largest_component,
            min_component_vertices=min_component_vertices,
        )
        cropped = crop_mesh(mesh, final)
        if len(cropped.vertices) < 3 or len(cropped.faces) == 0:
            raise ValueError("crop produced an empty or faceless mesh")
        components = _components(final, _vertex_neighbors(mesh))
        return CropResult(
            mesh=cropped,
            probabilities=smoothed,
            initial_mask=initial,
            final_mask=final,
            metrics={
                "retained_vertex_fraction": float(len(cropped.vertices) / len(mesh.vertices)),
                "retained_face_fraction": float(len(cropped.faces) / len(mesh.faces)),
                "probability_mean": float(np.mean(smoothed)),
                "probability_p05": float(np.percentile(smoothed, 5.0)),
                "probability_p95": float(np.percentile(smoothed, 95.0)),
                "postprocess_changed_vertices": int(np.count_nonzero(initial != final)),
                "postprocess_changed_fraction": float(
                    np.mean(initial != final)
                ),
                "kept_component_count": len(components),
                "largest_component_fraction": (
                    float(max(map(len, components)) / np.count_nonzero(final))
                    if components and np.count_nonzero(final)
                    else 0.0
                ),
            },
        )
