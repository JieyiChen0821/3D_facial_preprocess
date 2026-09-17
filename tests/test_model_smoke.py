from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from face_preprocess.models.crop import CropPredictor
from face_preprocess.models.landmark import LANDMARK_NAMES, LandmarkPredictor
from face_preprocess.types import Mesh


ROOT = Path(__file__).parents[1]
pytestmark = [
    pytest.mark.model_smoke,
    pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed"),
]


def test_real_crop_checkpoint_loads_strictly_and_emits_probabilities():
    predictor = CropPredictor.from_checkpoint(ROOT / "assets/models/crop_best.pt", "cpu")
    vertices = np.random.default_rng(1).normal(size=(32, 3))
    probabilities = predictor.predict_probabilities(vertices, num_points=32, extra_passes=0, seed=1)
    assert probabilities.shape == (32,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_real_crop_checkpoint_contains_inference_metadata_only():
    import torch

    payload = torch.load(
        ROOT / "assets/models/crop_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert set(payload) == {"model_state", "config"}
    assert set(payload["config"]) == {"model"}
    assert set(payload["config"]["model"]) == {
        "k_neighbors",
        "width",
        "blocks",
        "dropout",
        "knn_chunk_size",
    }


def test_real_landmark_checkpoint_loads_strictly_and_emits_nine_points():
    predictor = LandmarkPredictor.from_checkpoint(ROOT / "assets/models/landmark_best.pt", "cpu")
    vertices = np.random.default_rng(2).normal(size=(64, 3))
    faces = np.asarray([[0, index, index + 1] for index in range(1, 63)], dtype=np.int64)
    result = predictor.predict(Mesh(vertices, faces), num_points=64, top_k=16, seed=2)
    assert result.names == LANDMARK_NAMES
    assert result.snapped.shape == (9, 3)
