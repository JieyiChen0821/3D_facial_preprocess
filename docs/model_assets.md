# Model and Geometry Assets

## Inference checkpoints

| Stage | Path | SHA256 | Required metadata |
|---|---|---|---|
| Crop | `assets/models/crop_best.pt` | `7e124713ae409a39e83c2b859a59dafa2ed262c954c16f23f70d2306413668ff` | `model_state`, `config.model` |
| Nine-point landmarks | `assets/models/landmark_best.pt` | `88f3c0c23efbac8b2fcefa0c23a56e2ae1b6f82d4cc0a8e870732580445002f0` | `model`, `args`, `landmarks` |

The crop checkpoint was reduced to inference-only content. Its `config.model` mapping contains exactly `k_neighbors`, `width`, `blocks`, `dropout`, and `knn_chunk_size`. The landmark checkpoint already passed the public metadata audit without private paths or individual-level fields.

Both files are loaded with `torch.load(..., weights_only=True)` and strict state-dict matching. The selected model profiles under `configs/models/` pin their SHA256 values and all inference parameters.

## Nine-point order

The landmark model and coarse-alignment reference use this fixed order:

```text
bijian, bigen, bixia, wyjzuo, nyjzuo, nyjyou, wyjyou, kouzuo, kouyou
```

All nine snapped mesh vertices receive equal weight in a similarity transform with translation, rotation, and uniform scale. Reflection is forbidden.

## Geometry assets

- `assets/template/0000.obj`: fixed 7,906-vertex, 15,598-face template and MeshMonk topology.
- `assets/template/landmarks_9_reference.csv`: nine coarse-alignment reference points.
- `assets/template/Template.txt`: GPA reference coordinates.
- `assets/template/refindex.txt`: symmetry correspondence index.
- `assets/phenotype/lamda_ADNP.csv`: 32 template-face barycentric definitions.
- `assets/phenotype/landmark_names.pp`: names-only 32-landmark template with zero coordinates and no individual metadata.

The historical phenotype convention is preserved exactly:

```text
coordinate = lambda3 * vertex_a + lambda2 * vertex_b + lambda1 * vertex_c
```

Every asset hash is listed in `assets/manifest.sha256` and validated by tests or configuration loading.
