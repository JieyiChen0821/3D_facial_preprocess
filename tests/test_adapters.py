from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from face_preprocess.adapters import (
    AdapterSpec,
    build_adapter,
    get_adapter_spec,
    list_adapter_info,
    register_adapter,
    resolve_adapter_configuration,
)
from face_preprocess.adapters.contracts import CropAdapterResult, LandmarkAdapterResult
from face_preprocess.errors import ConfigError
from face_preprocess.models.crop import CropResult
from face_preprocess.models.landmark import LANDMARK_NAMES, LandmarkResult
from face_preprocess.types import Mesh


def _crop_parameters() -> dict[str, object]:
    return {
        "input": {
            "features": ["xyz"],
            "normalization": "centroid_max_radius",
            "sampling": {
                "strategy": "shuffled_cover_plus_random_votes",
                "num_points": 32768,
            },
        },
        "inference": {"extra_passes": 2},
        "output_mapping": {"face_class_index": 1},
        "postprocess": {
            "probability_threshold": 0.5,
            "smooth_iterations": 0,
            "smooth_alpha": 0.5,
            "keep_largest_component": True,
            "min_component_vertices": 100,
        },
    }


def test_builtin_adapter_registry_describes_current_models():
    crop = get_adapter_spec("crop", "pointnext_crop_v1")
    landmark = get_adapter_spec("landmark", "heatmap_offset_landmark_v1")

    assert crop.version == "1.0.0"
    assert crop.required_assets == ("checkpoint",)
    assert crop.supported_precisions == ("fp32",)
    assert landmark.required_assets == ("checkpoint",)
    assert "parameters.inference.top_k" in {
        item["path"] for item in list_adapter_info("landmark", landmark.name)[0]["parameters"]
    }


def test_adapter_parameters_are_strict_and_defaults_are_materialized():
    parameters = _crop_parameters()
    del parameters["postprocess"]["smooth_alpha"]  # type: ignore[index]

    resolved = resolve_adapter_configuration(
        stage="crop",
        adapter="pointnext_crop_v1",
        asset_names={"checkpoint"},
        precision="fp32",
        parameters=parameters,
    )

    assert resolved["postprocess"]["smooth_alpha"] == 0.5

    parameters["inference"]["mystery"] = 3  # type: ignore[index]
    with pytest.raises(ConfigError, match="unknown parameter"):
        resolve_adapter_configuration(
            stage="crop",
            adapter="pointnext_crop_v1",
            asset_names={"checkpoint"},
            precision="fp32",
            parameters=parameters,
        )


def test_builtin_precision_is_explicit_and_never_silently_falls_back():
    with pytest.raises(ConfigError, match="precision"):
        resolve_adapter_configuration(
            stage="crop",
            adapter="pointnext_crop_v1",
            asset_names={"checkpoint"},
            precision="fp16",
            parameters=_crop_parameters(),
        )


def test_public_registration_is_explicit_and_package_local():
    spec = AdapterSpec(
        stage="crop",
        name="test_crop_adapter_v1",
        version="1.0.0",
        factory=lambda config, device: (config, device),
        required_assets=("checkpoint",),
        supported_precisions=("fp32",),
        parameters=(),
        source_files=("adapters/contracts.py",),
    )
    register_adapter(spec)

    assert get_adapter_spec("crop", spec.name) is spec


def test_registered_adapter_must_declare_source_files_for_auditing():
    with pytest.raises(ConfigError, match="source_files"):
        register_adapter(
            AdapterSpec(
                stage="crop",
                name="missing_source_files_v1",
                version="1.0.0",
                factory=lambda config, device: None,
                required_assets=(),
                supported_precisions=("fp32",),
                parameters=(),
            )
        )


def test_adapter_info_cli_supports_machine_readable_output(run_cli):
    result = run_cli(
        [
            "--json",
            "adapter-info",
            "--stage",
            "landmark",
            "--adapter",
            "heatmap_offset_landmark_v1",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "adapter-info"
    assert payload["adapters"][0]["name"] == "heatmap_offset_landmark_v1"


def test_adapter_info_cli_has_readable_text_with_defaults_and_ranges(run_cli):
    result = run_cli(["adapter-info", "--stage", "crop", "--adapter", "pointnext_crop_v1"])

    assert result.returncode == 0, result.stderr
    assert "crop/pointnext_crop_v1 (version 1.0.0)" in result.stdout
    assert "parameters.input.sampling.num_points" in result.stdout
    assert "default=32768" in result.stdout
    assert "minimum=1" in result.stdout


def _triangle_mesh() -> Mesh:
    return Mesh(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )


def test_crop_builtin_adapter_owns_inference_and_postprocess_parameters(monkeypatch):
    mesh = _triangle_mesh()
    captured: dict[str, object] = {}

    class FakePredictor:
        def crop(self, value, **kwargs):
            captured.update(kwargs)
            return CropResult(
                mesh=value,
                probabilities=np.ones(3),
                initial_mask=np.ones(3, dtype=bool),
                final_mask=np.ones(3, dtype=bool),
                metrics={"retained_vertex_fraction": 1.0},
            )

    from face_preprocess.models.crop import CropPredictor

    monkeypatch.setattr(
        CropPredictor,
        "from_checkpoint",
        classmethod(lambda cls, path, device: FakePredictor()),
    )
    config = SimpleNamespace(
        adapter="pointnext_crop_v1",
        checkpoint=Path("crop.pt"),
        parameters=_crop_parameters(),
    )

    adapter = build_adapter("crop", config, "cpu")
    result = adapter.crop(mesh, seed=17)

    assert isinstance(result, CropAdapterResult)
    assert result.mesh is mesh
    assert captured == {
        "num_points": 32768,
        "extra_passes": 2,
        "seed": 17,
        "face_class_index": 1,
        "threshold": 0.5,
        "smooth_iterations": 0,
        "smooth_alpha": 0.5,
        "keep_largest_component": True,
        "min_component_vertices": 100,
    }


def test_landmark_builtin_adapter_returns_canonical_snapped_contract(monkeypatch):
    mesh = _triangle_mesh()
    snapped = np.arange(27, dtype=float).reshape(9, 3)
    captured: dict[str, object] = {}

    class FakePredictor:
        def predict(self, value, **kwargs):
            captured.update(kwargs)
            return LandmarkResult(
                names=LANDMARK_NAMES,
                continuous=snapped + 0.1,
                snapped=snapped,
                snapped_vertex_indices=np.arange(9),
                snap_distances=np.full(9, 0.1),
                heatmap_peaks=np.full(9, 0.9),
                topk_spreads=np.full(9, 0.2),
            )

    from face_preprocess.models.landmark import LandmarkPredictor

    monkeypatch.setattr(
        LandmarkPredictor,
        "from_checkpoint",
        classmethod(lambda cls, path, device: FakePredictor()),
    )
    config = SimpleNamespace(
        adapter="heatmap_offset_landmark_v1",
        checkpoint=Path("landmark.pt"),
        parameters={
            "input": {"sampling": {"num_points": 32768}},
            "inference": {"top_k": 16},
            "output_mapping": {
                "native_order": list(LANDMARK_NAMES),
                "canonical_order": list(LANDMARK_NAMES),
            },
        },
    )

    adapter = build_adapter("landmark", config, "cpu")
    result = adapter.predict(mesh, seed=23)

    assert isinstance(result, LandmarkAdapterResult)
    assert result.names == LANDMARK_NAMES
    np.testing.assert_array_equal(result.snapped, snapped)
    assert captured == {"num_points": 32768, "top_k": 16, "seed": 23}


def test_pipeline_dependency_builder_uses_registered_adapters(monkeypatch):
    from face_preprocess import pipeline

    crop_config = SimpleNamespace(adapter="crop_test")
    landmark_config = SimpleNamespace(adapter="landmark_test")
    config = SimpleNamespace(
        crop=crop_config,
        landmark=landmark_config,
        processing=SimpleNamespace(meshmonk={}),
    )
    calls: list[tuple[str, object, str]] = []

    def fake_build(stage, model_config, device):
        calls.append((stage, model_config, device))
        return f"{stage}-adapter"

    monkeypatch.setattr(pipeline, "build_adapter", fake_build, raising=False)

    dependencies = pipeline.build_dependencies(config, "cpu")

    assert dependencies.crop == "crop-adapter"
    assert dependencies.landmark == "landmark-adapter"
    assert calls == [
        ("crop", crop_config, "cpu"),
        ("landmark", landmark_config, "cpu"),
    ]
