from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from face_preprocess.errors import ConfigError
from face_preprocess.obj_io import read_obj


def render_run(
    run_dir: str | Path,
    output: str | Path,
    *,
    seed: int = 20260730,
    limit: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ConfigError(f"render output directory must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted((root / "metadata" / "samples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("final_obj"):
            records.append(payload)
    records.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item.get('output_key')}".encode("utf-8")
        ).hexdigest()
    )
    if limit is not None:
        records = records[: max(0, int(limit))]
    manifest: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        blind_id = f"B{index:05d}"
        mesh_path = root / str(record["final_obj"])
        mesh, _ = read_obj(mesh_path)
        figure = plt.figure(figsize=(10, 5), constrained_layout=True)
        for plot_index, view in enumerate(((0, 90), (0, 0)), start=1):
            axis = figure.add_subplot(1, 2, plot_index, projection="3d")
            axis.plot_trisurf(
                mesh.vertices[:, 0],
                mesh.vertices[:, 1],
                mesh.vertices[:, 2],
                triangles=mesh.faces,
                color="#d8dde6",
                edgecolor="none",
                antialiased=True,
            )
            axis.view_init(elev=view[0], azim=view[1])
            axis.set_axis_off()
            axis.set_box_aspect((1, 1, 1))
        figure.suptitle(blind_id)
        figure.savefig(destination / f"{blind_id}.png", dpi=180)
        plt.close(figure)
        manifest.append(
            {
                "blind_id": blind_id,
                "output_key": str(record.get("output_key")),
                "sample_id": str(record.get("sample_id")),
            }
        )
    payload = {
        "schema": "face-preprocess-blinded-render-v1",
        "source_run": str(root),
        "seed": seed,
        "render_count": len(manifest),
        "mapping": manifest,
    }
    (destination / "blind_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload

