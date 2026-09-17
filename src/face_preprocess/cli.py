from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import signal
import sys
from datetime import datetime, timezone
import uuid
from typing import Any

from face_preprocess import __version__
from face_preprocess.errors import FacePreprocessError, UsageContractError


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_SAMPLE_FAILURE = 3
EXIT_RUN_FATAL = 4

COMMANDS = (
    "doctor",
    "adapter-info",
    "validate-run",
    "hash-assets",
    "phenotype-landmarks",
    "run",
    "qc-summarize",
    "qc-render",
    "qc-calibrate",
    "qc-evaluate",
    "rebuild-summary",
)


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input")
    source.add_argument("--input-manifest")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--device-profile")
    parser.add_argument("--qc-config")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qc-level", choices=("standard", "enhanced"), default="standard")
    parser.add_argument("--output-mode", choices=("compact", "full"), default="compact")
    parser.add_argument("--symmetry", action="store_true")
    parser.add_argument("--phenotype-landmarks", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face-preprocess",
        description="Automatic 3D facial OBJ preprocessing.",
    )
    parser.add_argument("--version", action="version", version=f"face-preprocess {__version__}")
    parser.add_argument("--json", action="store_true", dest="json_output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check runtime, assets, and optionally model loading.")
    doctor.add_argument("--config")
    doctor.add_argument("--device-profile")
    doctor.add_argument("--qc-config")
    doctor.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    doctor.add_argument("--qc-level", choices=("standard", "enhanced"), default="standard")


    adapter_info = subparsers.add_parser(
        "adapter-info", help="List registered model adapters and their supported configuration."
    )
    adapter_info.add_argument("--stage", choices=("crop", "landmark"))
    adapter_info.add_argument("--adapter")

    validate = subparsers.add_parser("validate-run", help="Validate inputs, configs, assets, and fingerprints.")
    _add_common_run_arguments(validate)

    hash_assets = subparsers.add_parser("hash-assets", help="Compute SHA256 for explicitly named assets.")
    hash_assets.add_argument("paths", nargs="*")

    phenotype = subparsers.add_parser(
        "phenotype-landmarks",
        help="Project configured phenotype landmarks onto template-topology OBJ files.",
    )
    phenotype.add_argument("--input", required=True)
    phenotype.add_argument("--output", required=True)
    phenotype.add_argument("--config", required=True)
    phenotype.add_argument("--no-pp", action="store_true")

    run = subparsers.add_parser("run", help="Run the complete automatic preprocessing pipeline.")
    _add_common_run_arguments(run)
    recovery = run.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true")
    recovery.add_argument("--overwrite", action="store_true")
    run.add_argument("--retry-failed", action="store_true")

    for command in ("qc-summarize", "qc-render", "qc-calibrate", "qc-evaluate", "rebuild-summary"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--run-dir", action="append", default=[])
        command_parser.add_argument("--output")
        command_parser.add_argument("--qc-config")
        command_parser.add_argument("--validation-plan")
        command_parser.add_argument("--labels")
        command_parser.add_argument("--qc-level", choices=("standard", "enhanced"), default="standard")
        command_parser.add_argument("--seed", type=int, default=20260730)
        command_parser.add_argument("--limit", type=int)

    return parser


def _adapter_info_payload(args: argparse.Namespace) -> dict[str, Any]:
    from face_preprocess.adapters import list_adapter_info

    return {
        "command": "adapter-info",
        "status": "ok",
        "adapters": list_adapter_info(args.stage, args.adapter),
    }


def _emit(payload: dict[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if payload.get("command") == "adapter-info" and payload.get("status") == "ok":
        for adapter in payload["adapters"]:
            print(f"{adapter['stage']}/{adapter['name']} (version {adapter['version']})")
            print(f"  source_sha256: {adapter['source_sha256']}")
            print(f"  required_assets: {', '.join(adapter['required_assets']) or 'none'}")
            print(
                "  supported_precisions: "
                + (", ".join(adapter["supported_precisions"]) or "none")
            )
            for parameter in adapter["parameters"]:
                details = [
                    str(parameter["type"]),
                    "default="
                    + json.dumps(parameter["default"], ensure_ascii=False, separators=(",", ":")),
                ]
                if "choices" in parameter:
                    details.append(
                        "choices="
                        + json.dumps(parameter["choices"], ensure_ascii=False, separators=(",", ":"))
                    )
                if "minimum" in parameter:
                    details.append(f"minimum={parameter['minimum']}")
                if "maximum" in parameter:
                    details.append(f"maximum={parameter['maximum']}")
                print(f"  {parameter['path']}: {', '.join(details)}")
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def _doctor_payload(args: argparse.Namespace) -> dict[str, Any]:
    torch_installed = importlib.util.find_spec("torch") is not None
    torch_loadable = torch_installed
    python_compatible = sys.version_info[:2] == (3, 10)
    payload: dict[str, Any] = {
        "command": "doctor",
        "status": "ok" if python_compatible else "setup_required",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compatible": python_compatible,
            "required": "3.10.x",
        },
        "torch": {
            "installed": torch_installed,
        },
        "platform": platform.platform(),
    }
    if torch_installed:
        try:
            import torch

            payload["torch"].update(
                {
                    "version": torch.__version__,
                    "compatible": str(torch.__version__).split("+", 1)[0] == "2.5.1",
                    "required": "2.5.1",
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_version": torch.version.cuda,
                    "device_count": int(torch.cuda.device_count()),
                }
            )
            if not payload["torch"]["compatible"]:
                payload["status"] = "setup_required"
        except (ImportError, OSError) as exc:
            payload["torch"].update(
                {"load_error": f"{type(exc).__name__}: {exc}", "compatible": False}
            )
            torch_loadable = False
            payload["status"] = "setup_required"
    provided = [args.config, args.device_profile, args.qc_config]
    if any(provided) and not all(provided):
        raise UsageContractError(
            "doctor requires --config, --device-profile, and --qc-config together"
        )
    if all(provided):
        from face_preprocess.config import load_run_config
        from face_preprocess.pipeline import _load_geometry_assets, build_dependencies, resolve_device

        config = load_run_config(
            args.config,
            args.device_profile,
            args.qc_config,
            args.qc_level,
            False,
        )
        template, landmarks, points, refindex = _load_geometry_assets(config)
        payload["assets"] = {
            "status": "ok",
            "template_vertices": len(template.vertices),
            "template_faces": len(template.faces),
            "template_points": len(points),
            "template_landmarks": len(landmarks),
            "refindex": len(refindex),
        }
        if torch_loadable:
            actual_device = resolve_device(args.device)
            build_dependencies(config, actual_device)
            payload["models"] = {"status": "ok", "device": actual_device}
        else:
            payload["models"] = {
                "status": "not_checked",
                "reason": "PyTorch is not installed",
            }
    return payload


def _required(args: argparse.Namespace, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if not getattr(args, name, None)]
    if missing:
        raise UsageContractError(f"missing required arguments: {', '.join(missing)}")


def _validate_payload(args: argparse.Namespace) -> dict[str, Any]:
    _required(args, "config", "device_profile", "qc_config")
    if not args.input and not args.input_manifest:
        raise UsageContractError("one of --input or --input-manifest is required")
    from face_preprocess.config import load_run_config
    from face_preprocess.inputs import discover_inputs
    from face_preprocess.pipeline import _load_geometry_assets

    records = discover_inputs(args.input, args.input_manifest)
    config = load_run_config(
        args.config,
        args.device_profile,
        args.qc_config,
        args.qc_level,
        args.symmetry,
    )
    template, landmarks, points, refindex = _load_geometry_assets(config)
    return {
        "command": "validate-run",
        "status": "ok",
        "sample_count": len(records),
        "input_geometry_hashes": [record.geometry_sha256 for record in records],
        "config_hashes": config.config_hashes,
        "template": {
            "vertices": len(template.vertices),
            "faces": len(template.faces),
            "points": len(points),
            "landmarks": len(landmarks),
            "refindex": len(refindex),
        },
        "note": "Model tensor loading is checked by doctor in an environment with PyTorch.",
    }


def _hash_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.paths:
        raise UsageContractError("hash-assets requires at least one path")
    values: list[dict[str, Any]] = []
    for raw in args.paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise UsageContractError(f"asset does not exist: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        values.append(
            {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
        )
    return {"command": "hash-assets", "status": "ok", "assets": values}


def _phenotype_payload(args: argparse.Namespace) -> dict[str, Any]:
    from face_preprocess.config import load_phenotype_config
    from face_preprocess.inputs import discover_inputs
    from face_preprocess.obj_io import read_obj
    from face_preprocess.phenotype_landmarks import (
        load_phenotype_assets,
        project_phenotype_landmarks,
        write_landmark_csv,
        write_picked_points,
    )

    config = load_phenotype_config(args.config)
    assets = load_phenotype_assets(
        config.template_obj,
        config.lamda_csv,
        config.names_pp,
    )
    records = discover_inputs(args.input, None)
    output = Path(args.output).resolve()
    outputs: list[dict[str, Any]] = []
    for record in records:
        if record.input_error:
            raise UsageContractError(f"invalid input OBJ {record.path}: {record.input_error}")
        mesh, _ = read_obj(record.path)
        result = project_phenotype_landmarks(mesh, assets)
        key = str(record.output_key)
        csv_path = output / f"{key}_landmarks.csv"
        pp_path = output / f"{key}_landmarks.pp"
        write_landmark_csv(csv_path, result)
        if not args.no_pp:
            write_picked_points(pp_path, result, record.path.name)
        outputs.append(
            {
                "input": str(record.path),
                "csv": str(csv_path),
                "picked_points": None if args.no_pp else str(pp_path),
                "landmark_count": len(result.names),
            }
        )
    return {
        "command": "phenotype-landmarks",
        "status": "ok",
        "processed": len(outputs),
        "outputs": outputs,
    }


def _run_payload(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    _required(args, "config", "device_profile", "qc_config")
    if not args.input and not args.input_manifest:
        raise UsageContractError("one of --input or --input-manifest is required")
    if args.resume and not args.output:
        raise UsageContractError("--resume requires an explicit --output run directory")
    output = Path(args.output) if args.output else Path("outputs") / (
        f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    from face_preprocess.pipeline import RunOptions, run_pipeline

    summary = run_pipeline(
        RunOptions(
            input_path=Path(args.input) if args.input else None,
            input_manifest=Path(args.input_manifest) if args.input_manifest else None,
            output=output,
            pipeline_config=Path(args.config),
            device_profile=Path(args.device_profile),
            qc_config=Path(args.qc_config),
            device=args.device,
            workers=args.workers,
            qc_level=args.qc_level,
            output_mode=args.output_mode,
            symmetry=args.symmetry,
            resume=args.resume,
            overwrite=args.overwrite,
            retry_failed=args.retry_failed,
            phenotype_landmarks=args.phenotype_landmarks,
        )
    )
    payload = {"command": "run", "status": "completed", **summary.as_dict()}
    return payload, EXIT_SAMPLE_FAILURE if summary.failed else EXIT_OK


def _qc_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "rebuild-summary":
        _required(args, "run_dir")
        from face_preprocess.pipeline import rebuild_run_summary

        values = [rebuild_run_summary(path) for path in args.run_dir]
        return {"command": args.command, "status": "ok", "summaries": values}
    _required(args, "run_dir", "output")
    if args.command == "qc-summarize":
        from face_preprocess.qc.tools import summarize_runs

        value = summarize_runs(args.run_dir, args.output)
    elif args.command == "qc-render":
        if len(args.run_dir) != 1:
            raise UsageContractError("qc-render accepts exactly one --run-dir")
        from face_preprocess.qc.render import render_run

        value = render_run(
            args.run_dir[0], args.output, seed=args.seed, limit=args.limit
        )
    elif args.command == "qc-calibrate":
        _required(args, "qc_config", "validation_plan")
        from face_preprocess.qc.tools import calibrate_qc

        value = calibrate_qc(
            args.run_dir,
            args.qc_config,
            args.validation_plan,
            args.output,
            labels=args.labels,
        )
    elif args.command == "qc-evaluate":
        _required(args, "qc_config")
        if len(args.run_dir) != 1:
            raise UsageContractError("qc-evaluate accepts exactly one --run-dir")
        from face_preprocess.qc.tools import evaluate_run_qc

        value = evaluate_run_qc(
            args.run_dir[0],
            args.qc_config,
            args.output,
            qc_level=args.qc_level,
        )
    else:  # pragma: no cover - parser constrains commands
        raise UsageContractError(f"unsupported command: {args.command}")
    return {"command": args.command, "status": "ok", **value}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run" and args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")

    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if args.command == "doctor":
            payload, exit_code = _doctor_payload(args), EXIT_OK
        elif args.command == "adapter-info":
            payload, exit_code = _adapter_info_payload(args), EXIT_OK
        elif args.command == "validate-run":
            payload, exit_code = _validate_payload(args), EXIT_OK
        elif args.command == "hash-assets":
            payload, exit_code = _hash_payload(args), EXIT_OK
        elif args.command == "phenotype-landmarks":
            payload, exit_code = _phenotype_payload(args), EXIT_OK
        elif args.command == "run":
            payload, exit_code = _run_payload(args)
        else:
            payload, exit_code = _qc_payload(args), EXIT_OK
        _emit(payload, bool(args.json_output))
        return exit_code
    except KeyboardInterrupt:
        payload = {
            "command": args.command,
            "status": "interrupted",
            "message": "Interrupted by user",
        }
        if args.json_output:
            _emit(payload, True)
        else:
            print("Interrupted by user", file=sys.stderr)
        return 130
    except FacePreprocessError as exc:
        payload = {
            "command": args.command,
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if args.json_output:
            _emit(payload, True)
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE if isinstance(exc, UsageContractError) else EXIT_RUN_FATAL
    except Exception as exc:
        payload = {
            "command": args.command,
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if args.json_output:
            _emit(payload, True)
        else:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUN_FATAL
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    sys.exit(main())
