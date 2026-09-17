from __future__ import annotations

import numpy as np

from face_preprocess.config import DeviceProfile, QCConfig, QCRule
from face_preprocess.qc.measurements import measure_input_mesh
from face_preprocess.qc.registry import (
    EMPIRICAL_GROUPS,
    HARD_CONTRACT_GROUPS,
    TELEMETRY_GROUPS,
)
from face_preprocess.qc.rules import evaluate_qc
from face_preprocess.types import Measurement, Mesh


def test_registry_has_locked_group_counts_and_enhanced_groups():
    assert len(EMPIRICAL_GROUPS) == 24
    assert len(HARD_CONTRACT_GROUPS) == 12
    assert len(TELEMETRY_GROUPS) == 4
    assert EMPIRICAL_GROUPS[-2:] == ("crop.stability", "landmark.stability")
    assert len(set(EMPIRICAL_GROUPS)) == 24


def test_rule_engine_is_typed_and_has_no_composite_score():
    config = QCConfig(
        schema_version=1,
        profile_id="x",
        profile_version="1",
        metric_groups={"crop.retention": "warning"},
        rules=(
            QCRule(
                rule_id="retention",
                metric="crop.retention.vertex_fraction",
                operator="outside",
                threshold=[0.4, 0.9],
                unit="ratio",
                severity="warning",
                minimum_qc_level="standard",
            ),
        ),
    )

    assessment = evaluate_qc(
        {"crop.retention.vertex_fraction": Measurement(0.2, "ratio")},
        config,
        level="standard",
    )

    assert assessment.status == "warning"
    assert assessment.results[0].triggered is True
    assert not hasattr(assessment, "score")


def test_enhanced_rule_cannot_fail_and_missing_metric_marks_qc_incomplete():
    config = QCConfig(
        schema_version=1,
        profile_id="x",
        profile_version="1",
        metric_groups={"crop.stability": "warning"},
        rules=(
            QCRule(
                rule_id="stability",
                metric="crop.stability.jaccard_min",
                operator="lt",
                threshold=0.8,
                unit="ratio",
                severity="warning",
                minimum_qc_level="enhanced",
            ),
        ),
    )

    assessment = evaluate_qc({}, config, level="enhanced")

    assert assessment.status == "warning"
    assert assessment.qc_incomplete is True


def test_input_measurements_keep_raw_edge_length_and_topology_values():
    mesh = Mesh(
        np.array([[0, 0, 0], [1.2, 0, 0], [0, 1.2, 0]], dtype=float),
        np.array([[0, 1, 2]], dtype=np.int64),
    )
    profile = DeviceProfile(1, "scanner", "1", "mm", 1.2, "scanner", "protocol")

    values = measure_input_mesh(mesh, profile)

    assert values["input.edge_length.median_mm"].unit == "mm"
    assert values["input.edge_length.expected_mm"].value == 1.2
    assert values["input.topology.boundary_edge_count"].value == 3
    assert values["input.degenerate_faces.count"].value == 0
