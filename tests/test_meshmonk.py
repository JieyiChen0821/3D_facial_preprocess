from __future__ import annotations

import numpy as np

from face_preprocess.geometry.meshmonk import PythonMeshMonkBackend
from face_preprocess.types import Mesh


def test_meshmonk_returns_diagnostic_result_without_changing_template_topology():
    template = Mesh(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
    )
    target = Mesh(template.vertices + np.array([0.5, -0.25, 0.75]), template.faces)
    backend = PythonMeshMonkBackend(
        rigid_iterations=1,
        nonrigid_iterations=0,
        transform_num_neighbours=3,
        flag_floating_boundary=False,
        flag_target_boundary=False,
        flag_target_badly_sized_triangles=False,
    )

    result = backend.map_mesh_diagnostic(target=target, template=template)

    np.testing.assert_array_equal(result.mesh.faces, template.faces)
    assert result.final_correspondences.shape == (4, 3)
    assert result.final_weights.shape == (4,)
    assert result.rigid_end_vertices.shape == (4, 3)
    assert result.rigid_scale > 0
    assert result.trajectory

