#!/usr/bin/env python3
"""Run one frozen StateTriage3R Stage-0 capsule without altering ReCal3R."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]
PINNED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.run_recal3r_smoke as smoke
from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
from stateguard3r.state_triage_capsule_v2 import apply_stage0_transforms, load_stage0_capsule
from stateguard3r.state_triage_v2 import EvidenceConfig, EvidenceObserver, SCHEMA_VERSION as EVIDENCE_SCHEMA_VERSION


class Stage0RunnerError(RuntimeError):
    """Raised for a failed, non-retryable Stage-0 runner contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-capsule", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--size", type=int, choices=(512,), default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-base", type=float, default=0.1)
    parser.add_argument("--observer", choices=("off", "on"), required=True)
    return parser


def _sha256(path: Path) -> str:
    return smoke._sha256(path)


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _provenance() -> dict[str, Any]:
    try:
        commit = smoke._git_output(ROOT, "rev-parse", "HEAD")
        tracked = smoke._git_output(ROOT, "status", "--porcelain", "--untracked-files=no")
        tracked_script = smoke._git_output(ROOT, "ls-files", "--error-unmatch", str(Path(__file__).resolve().relative_to(ROOT)))
    except Exception as error:  # subprocess error type belongs to smoke's private API.
        raise Stage0RunnerError(f"cannot inspect Stage-0 runner provenance: {error}") from error
    if tracked:
        raise Stage0RunnerError("StateGuard3R has tracked changes; Stage-0 runner requires a commit")
    if tracked_script != str(Path(__file__).resolve().relative_to(ROOT)):
        raise Stage0RunnerError("Stage-0 runner is not tracked")
    return {
        "repository_root": str(ROOT),
        "commit": commit,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__)),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }


def _validate(args: argparse.Namespace) -> Any:
    args.baseline_root = args.baseline_root.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.input_capsule = args.input_capsule.resolve(strict=True)
    args.output_dir = args.output_dir.resolve(strict=False)
    expected_output_parent = ROOT / "outputs"
    if args.output_dir.parent != expected_output_parent or args.output_dir.exists():
        raise Stage0RunnerError("output-dir must be a new direct child of StateGuard3R/outputs")
    if not (args.baseline_root / "src").is_dir() or not (args.baseline_root / ".venv").is_dir():
        raise Stage0RunnerError("baseline root lacks src or isolated .venv")
    if smoke._git_output(args.baseline_root, "rev-parse", "HEAD") != PINNED_RECAL3R_COMMIT:
        raise Stage0RunnerError("ReCal3R commit differs from Stage-0 pin")
    if smoke._git_output(args.baseline_root, "status", "--porcelain", "--untracked-files=no"):
        raise Stage0RunnerError("ReCal3R has tracked changes")
    if args.checkpoint.name != "cut3r_512_dpt_4_64.pth" or _sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise Stage0RunnerError("checkpoint identity differs from Stage-0 pin")
    if Path(sys.prefix).resolve(strict=False) != (args.baseline_root / ".venv").resolve(strict=False):
        raise Stage0RunnerError("runner must use ReCal3R's isolated .venv")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or len([item for item in visible.split(",") if item.strip()]) != 1:
        raise Stage0RunnerError("CUDA_VISIBLE_DEVICES must name exactly one GPU")
    args.capsule = load_stage0_capsule(args.input_capsule)
    return args


def _load_upstream(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(args.baseline_root))
    import numpy as np
    import torch
    import add_ckpt_path as add_ckpt_path_module

    module_paths = {
        "add_ckpt_path": smoke._assert_module_source(add_ckpt_path_module, args.baseline_root / "add_ckpt_path.py", "add_ckpt_path")
    }
    add_ckpt_path_module.add_path_to_dust3r(str(args.baseline_root / "src" / args.checkpoint.name))
    import dust3r.inference as inference_module
    import dust3r.model as model_module
    import dust3r.utils.camera as camera_module
    import models.curope.curope2d as curope2d_module

    module_paths.update(
        {
            "dust3r.inference": smoke._assert_module_source(inference_module, args.baseline_root / "src" / "dust3r" / "inference.py", "dust3r.inference"),
            "dust3r.model": smoke._assert_module_source(model_module, args.baseline_root / "src" / "dust3r" / "model.py", "dust3r.model"),
            "dust3r.utils.camera": smoke._assert_module_source(camera_module, args.baseline_root / "src" / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera"),
            "models.curope.curope2d": smoke._assert_module_source(curope2d_module, args.baseline_root / "src" / "croco" / "models" / "curope" / "curope2d.py", "models.curope.curope2d"),
        }
    )
    kernel = smoke._binary_module_provenance(curope2d_module._kernels, expected_directory=args.baseline_root / "src" / "croco" / "models" / "curope", filename_prefix="curope.", label="cuRoPE CUDA extension")
    return np, torch, inference_module, model_module, camera_module, module_paths, kernel


def _observe(predictions: Sequence[dict[str, Any]], views: Sequence[dict[str, Any]], trajectory: Sequence[dict[str, Any]], health: Sequence[Any]) -> list[dict[str, Any]]:
    observer = EvidenceObserver(EvidenceConfig())
    ledger: list[dict[str, Any]] = []
    for frame_id, (prediction, view, pose, health_row) in enumerate(zip(predictions, views, trajectory, health, strict=True)):
        evidence = observer.observe(
            frame_id=frame_id,
            prediction=prediction,
            camera_to_reference=pose["camera_to_reference"],
            model_ready_rgb=view["img"],
            native_health=health_row.to_dict(),
        )
        ledger.append(evidence.to_dict())
    return ledger


def _write_json(path: Path, value: Any) -> None:
    smoke._write_json_atomic(path, value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args = _validate(args)
    provenance = _provenance()
    np, torch, inference_module, model_module, camera_module, module_paths, kernel = _load_upstream(args)
    if not torch.cuda.is_available():
        raise Stage0RunnerError("selected CUDA device is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    views_raw = smoke._prepare_views(args.capsule.frame_paths(), args.size, torch)
    raw_input_digests = [smoke._model_ready_rgb_sha256(view["img"]) for view in views_raw]
    views = apply_stage0_transforms(views_raw, args.capsule)
    if raw_input_digests != [smoke._model_ready_rgb_sha256(view["img"]) for view in views_raw]:
        raise Stage0RunnerError("deferred transform mutated a raw loaded image")
    model, checkpoint_audit = smoke._load_model_with_state_dict_audit(model_module.ARCroco3DStereo, args.checkpoint, torch)
    if checkpoint_audit["strict"] is not False or checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
        raise Stage0RunnerError("checkpoint load audit differs from pin")
    interface = smoke._validate_model_interface(model, args.checkpoint.name)
    model = model.to(args.device)
    runtime_config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base)
    model.eval()
    model.enable_u_calibration_trace(oracle_window=1)

    update_calls = 0
    native_update = model._compute_recal3r_update_mask

    def count_native_update(*call_args: Any, **kwargs: Any) -> Any:
        nonlocal update_calls
        update_calls += 1
        return native_update(*call_args, **kwargs)

    model._compute_recal3r_update_mask = count_native_update
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, state_args = inference_module.inference_recurrent_lighter(views, model, device, verbose=True)
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    predictions = outputs.get("pred")
    if not isinstance(predictions, list) or len(predictions) != len(views):
        raise Stage0RunnerError("official forward did not produce one prediction per frame")
    if update_calls != len(views) - 1:
        raise Stage0RunnerError("native ReCal3R update count differs from frame count")
    summary = smoke._prediction_summary(predictions, torch)
    pose_jumps, geometric_residuals, trajectory = smoke._output_health_signals(predictions, pose_encoding_to_camera=camera_module.pose_encoding_to_camera, np=np)
    trace = model.get_u_calibration_trace()
    health = adapt_recal3r_trace(trace, all_frame_ids=list(range(len(views))), timestamps=[frame.timestamp for frame in args.capsule.frames], batch_size=1)
    health = [replace(row, pose_jump=pose_jumps[index], geometric_residual=geometric_residuals[index]) for index, row in enumerate(health)]
    if len(health) != len(views) or any(health[index].global_state_delta is None for index in range(1, len(views))):
        raise Stage0RunnerError("native health trace is incomplete")
    observer_started = time.perf_counter()
    evidence = _observe(predictions, views, trajectory, health) if args.observer == "on" else None
    observer_seconds = time.perf_counter() - observer_started
    # Rehash the manifest and all RGB source files after the forward.  The
    # observer never receives the event object or final source index.
    load_stage0_capsule(args.input_capsule)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "checkpoint-load-audit.json", checkpoint_audit)
    _write_json(args.output_dir / "predictions-summary.json", summary)
    _write_json(args.output_dir / "trajectory.json", {"schema_version": "stateguard3r.trajectory.v0", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "frames": trajectory})
    write_health_jsonl_atomic(args.output_dir / "health.jsonl", health)
    if evidence is not None:
        _write_json(args.output_dir / "evidence.json", {"schema_version": EVIDENCE_SCHEMA_VERSION, "causality": "each row accepts current frame plus strict observer prefix only", "config": EvidenceConfig().to_dict(), "rows": evidence})
    metadata = {
        "schema_version": "stateguard3r.state-triage-stage0-run.v1",
        "status": "succeeded",
        "runner": provenance,
        "baseline_root": str(args.baseline_root),
        "baseline_commit": PINNED_RECAL3R_COMMIT,
        "checkpoint": _artifact(args.checkpoint),
        "input_capsule": _artifact(args.input_capsule),
        "frame_count": len(views),
        "seed": args.seed,
        "size": args.size,
        "beta_base": args.beta_base,
        "model_update_type": "recal3r",
        "loaded_model_interface": interface,
        "module_paths": module_paths,
        "curope_kernel": kernel,
        "observer": args.observer,
        "observer_reads": "prediction_t,current_model_ready_rgb_t,pose_t,native_health_t,strict_observer_prefix_only",
        "observer_wall_seconds": observer_seconds,
        "runtime_seconds": inference_seconds,
        "runtime_scope": "official_inference_recurrent_lighter_only",
        "peak_memory_allocated_mib": peak_memory_mib,
        "calibrated_update_calls": update_calls,
        "state_args_count": len(state_args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "started_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(args.output_dir / "run.json", metadata)
    smoke._freeze_output_tree(args.output_dir)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))
    except (Stage0RunnerError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
