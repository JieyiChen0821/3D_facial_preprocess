# Output Contract

## Final geometry

A passed or warning sample produces `final_obj/<output_key>.obj` with exactly 7,906 vertices and 15,598 triangular faces. Coordinates are in the normalized template space and are unitless after GPA.

Failed samples do not produce a final OBJ. Their failure reason and hard-contract state remain in sample metadata.

## Optional phenotype landmarks

When `--phenotype-landmarks` is enabled, each successful sample also produces:

- `phenotype_landmarks/<output_key>.csv`
- `phenotype_landmarks/<output_key>.pp`

The CSV header is `name,x,y,z`. Both outputs contain the same 32 names and coordinates. Sample metadata records their relative paths, landmark count, and the SHA256 values of the topology, barycentric, and names assets.

Projection is transactional with the final OBJ. A requested phenotype projection that fails topology, finite-coordinate, face-index, weight, or name-count validation fails the sample and commits neither the final OBJ nor phenotype files.

## Intermediate outputs

`--output-mode full` adds:

- `cropped.obj`
- `landmark_aligned.obj`
- `mapped.obj`
- `gpa.obj`
- `diagnostic_arrays.npz`

Compact mode retains the final OBJ and metadata only, plus phenotype outputs when explicitly requested.

## Reproducibility

Run fingerprints separate the input set, scientific configuration, and execution environment. The optional phenotype flag and all three phenotype-asset hashes participate in the scientific fingerprint. Changing the flag or assets prevents an incompatible `--resume`.
