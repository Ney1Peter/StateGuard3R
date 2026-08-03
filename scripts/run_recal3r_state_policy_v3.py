#!/usr/bin/env python3
"""Run the experimental external ReCal3R state-policy wrapper.

This is intentionally separate from the pinned smoke runner.  It neither
modifies ReCal3R nor claims recovery quality: it records whether a causal,
transactional hold/replay policy actually changes the complete recurrent state
closure while preserving an always-commit control path.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import sys
import time
from typing import Any, Sequence

# Direct ``python scripts/...`` execution places ``scripts`` rather than the
# repository root on sys.path.  Add only this repository root before importing
# shared runner helpers; no external path or baseline file is modified.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_recal3r_smoke as smoke


STATE_POLICY_SCHEMA_VERSION = "stateguard3r.recal3r-state-policy-v3.v1"


def _parser() -> argparse.ArgumentParser:
    parser = smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v2")
    parser.add_argument(
        "--state-policy",
        choices=("always-commit", "forced-prior-alarm", "detector-v3-prior-alarm"),
        required=True,
        help="always-commit is the equivalence control; detector-v3-prior-alarm consumes frozen v3 alarms",
    )
    parser.add_argument(
        "--alarm-frame",
        action="append",
        type=int,
        default=[],
        help="synthetic alarm position; a hold can begin only at the following frame",
    )
    parser.add_argument("--max-hold", type=int, default=3)
    parser.add_argument(
        "--detector-run-json",
        type=Path,
        help="frozen formal-v3 run.json whose input binding authorizes detector alarms",
    )
    parser.add_argument(
        "--detector-alarm-timeline",
        type=Path,
        help="frozen formal-v3 evaluation timeline; only its hybrid_alarm field is consumed",
    )
    return parser


def _self_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    try:
        commit = smoke._git_output(repository, "rev-parse", "HEAD")
        tracked_changes = smoke._git_output(
            repository, "status", "--porcelain", "--untracked-files=no"
        )
    except (OSError, smoke.subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect state-policy runner provenance: {error}") from error
    if tracked_changes:
        raise RuntimeError("StateGuard3R has tracked changes; state-policy runner refuses execution")
    return {
        "repository_root": str(repository),
        "commit": commit,
        "tracked_worktree_clean": True,
        "script_path": str(script_path),
        "script_sha256": smoke._sha256(script_path),
        "python_executable": sys.executable,
    }


def _validate_policy_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.health_profile != "v2":
        parser.error("state-policy runner requires --health-profile v2")
    if args.max_hold < 1 or args.max_hold > 3:
        parser.error("--max-hold must be in [1, 3]")
    detector_paths = (args.detector_run_json, args.detector_alarm_timeline)
    if args.state_policy == "always-commit":
        if args.alarm_frame:
            parser.error("--alarm-frame is valid only with an alarm-driven policy")
        if any(path is not None for path in detector_paths):
            parser.error("detector source arguments are valid only with --state-policy detector-v3-prior-alarm")
    if args.state_policy == "forced-prior-alarm":
        if not args.alarm_frame:
            parser.error("forced-prior-alarm requires at least one --alarm-frame")
        if any(frame < 0 for frame in args.alarm_frame):
            parser.error("--alarm-frame must be non-negative")
        if any(path is not None for path in detector_paths):
            parser.error("forced-prior-alarm may not use detector source arguments")
    if args.state_policy == "detector-v3-prior-alarm":
        if args.alarm_frame:
            parser.error("detector-v3-prior-alarm derives alarms from frozen evidence; do not pass --alarm-frame")
        if any(path is None for path in detector_paths):
            parser.error("detector-v3-prior-alarm requires --detector-run-json and --detector-alarm-timeline")


def _frozen_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    metadata = os.lstat(resolved)
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise RuntimeError(f"{label} must be frozen mode 0444")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value, {
        "path": str(resolved),
        "sha256": smoke._sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _detector_v3_alarms(args: argparse.Namespace) -> tuple[list[bool], dict[str, Any]]:
    """Bind causal policy alarms to frozen formal-v3 evidence, never labels."""
    assert args.detector_run_json is not None and args.detector_alarm_timeline is not None
    run, run_artifact = _frozen_json(args.detector_run_json, label="detector v3 run metadata")
    timeline, timeline_artifact = _frozen_json(
        args.detector_alarm_timeline, label="detector v3 evaluation timeline"
    )
    input_manifest = args.input_manifest.resolve(strict=True)
    bound_input = run.get("input_manifest")
    if not isinstance(bound_input, dict):
        raise RuntimeError("detector v3 run lacks input-manifest binding")
    if (
        bound_input.get("path") != str(input_manifest)
        or bound_input.get("sha256") != smoke._sha256(input_manifest)
        or (
            "size_bytes" in bound_input
            and bound_input.get("size_bytes") != input_manifest.stat().st_size
        )
    ):
        raise RuntimeError("detector v3 run is not bound to this state-policy input manifest")
    if run.get("health_profile") != "v3":
        raise RuntimeError("detector alarm source must be a v3 run")
    if timeline.get("run_id") != "blind-wrong-order":
        raise RuntimeError("detector alarm timeline must be the disclosed blind-wrong-order formal timeline")
    attribution = timeline.get("attribution")
    if not isinstance(attribution, list) or not attribution:
        raise RuntimeError("detector alarm timeline lacks attribution rows")
    alarms: list[bool] = []
    for frame_id, row in enumerate(attribution):
        if not isinstance(row, dict) or set(row) != {
            "frame_id", "continuous_alarm", "timestamp_order_alarm", "hybrid_alarm"
        }:
            raise RuntimeError(f"detector attribution row {frame_id} schema differs")
        if row.get("frame_id") != frame_id or not all(
            isinstance(row.get(name), bool)
            for name in ("continuous_alarm", "timestamp_order_alarm", "hybrid_alarm")
        ):
            raise RuntimeError(f"detector attribution row {frame_id} is invalid")
        # Labels and corruption fields in the evaluated timeline are deliberately
        # never referenced: the policy receives only the frozen hybrid decision.
        alarms.append(row["hybrid_alarm"])
    positions = [index for index, alarm in enumerate(alarms) if alarm]
    if not positions:
        raise RuntimeError("detector alarm source contains no hybrid alarms")
    return alarms, {
        "kind": "frozen_formal_v3_hybrid_alarm",
        "run": run_artifact,
        "timeline": timeline_artifact,
        "input_manifest": {
            "path": str(input_manifest),
            "sha256": smoke._sha256(input_manifest),
            "size_bytes": input_manifest.stat().st_size,
        },
        "alarm_positions": positions,
        "causality": "policy reads only alarm_t_minus_1; labels and event metadata are not consumed",
    }


def _freeze_output_tree(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    output_dir.chmod(0o555)


def _run_without_grad(torch: Any, callback: Any) -> Any:
    """Match the pinned ``@torch.no_grad`` lighter-inference contract."""

    with torch.no_grad():
        return callback()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_policy_args(args, parser)
    smoke._validate_args(args, parser)
    runner_provenance = _self_provenance()
    run_started_at = datetime.now().astimezone().isoformat()

    state_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(state_src))
    sys.path.insert(0, str(args.baseline_root))

    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_VISIBLE_DEVICES was set but PyTorch CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    import add_ckpt_path as add_ckpt_path_module

    module_paths = {
        "add_ckpt_path": smoke._assert_module_source(
            add_ckpt_path_module, args.baseline_root / "add_ckpt_path.py", "add_ckpt_path"
        )
    }
    add_ckpt_path_module.add_path_to_dust3r(str(args.baseline_src / args.checkpoint.name))
    import dust3r.model as dust3r_model_module
    import dust3r.utils.camera as dust3r_camera_module
    import dust3r.utils.image as dust3r_image_module
    import dust3r.utils.device as dust3r_device_module
    import models.curope.curope2d as curope2d_module

    module_paths.update(
        {
            "dust3r.model": smoke._assert_module_source(
                dust3r_model_module, args.baseline_src / "dust3r" / "model.py", "dust3r.model"
            ),
            "dust3r.utils.camera": smoke._assert_module_source(
                dust3r_camera_module, args.baseline_src / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera"
            ),
            "dust3r.utils.image": smoke._assert_module_source(
                dust3r_image_module, args.baseline_src / "dust3r" / "utils" / "image.py", "dust3r.utils.image"
            ),
            "dust3r.utils.device": smoke._assert_module_source(
                dust3r_device_module, args.baseline_src / "dust3r" / "utils" / "device.py", "dust3r.utils.device"
            ),
            "models.curope.curope2d": smoke._assert_module_source(
                curope2d_module,
                args.baseline_src / "croco" / "models" / "curope" / "curope2d.py",
                "models.curope.curope2d",
            ),
        }
    )
    curope_kernel_provenance = smoke._binary_module_provenance(
        curope2d_module._kernels,
        expected_directory=args.baseline_src / "croco" / "models" / "curope",
        filename_prefix="curope.",
        label="cuRoPE CUDA extension",
    )
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.recal3r_transactional_v3 import run_transactional_recurrent_lighter

    smoke._verify_preflight_inputs(args)
    views = smoke._prepare_input_views(args, torch)
    visual_started = time.perf_counter()
    v2_overlaps, v2_visual_metadata = smoke._v2_online_visual_overlap(views)
    v2_visual_metadata["runtime_seconds"] = time.perf_counter() - visual_started
    args.output_dir.mkdir(parents=True, exist_ok=False)

    model, checkpoint_state_dict = smoke._load_model_with_state_dict_audit(
        dust3r_model_module.ARCroco3DStereo, args.checkpoint, torch
    )
    smoke._write_json_atomic(args.output_dir / "checkpoint-load-audit.json", checkpoint_state_dict)
    if checkpoint_state_dict["strict"] is not False or checkpoint_state_dict["missing_keys"] or checkpoint_state_dict["unexpected_keys"]:
        raise RuntimeError("pinned ReCal3R checkpoint loader compatibility audit failed")
    loaded_model_interface = smoke._validate_model_interface(model, args.checkpoint.name)
    model = model.to(args.device)
    recal3r_runtime_config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base)
    runtime_config_source = args.baseline_root / "eval" / "relpose" / "launch.py"
    runtime_config_source_provenance = {
        "path": str(runtime_config_source),
        "sha256": smoke._sha256(runtime_config_source),
        "size_bytes": runtime_config_source.stat().st_size,
        "pinned_lines": "730-742",
    }
    model.eval()
    model.enable_u_calibration_trace(oracle_window=1)
    update_calls = 0
    original_update = model._compute_recal3r_update_mask

    def counted_update(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal update_calls
        update_calls += 1
        return original_update(*call_args, **call_kwargs)

    model._compute_recal3r_update_mask = counted_update
    alarms = [False] * len(views)
    alarm_source: dict[str, Any] | None = None
    if args.state_policy == "forced-prior-alarm":
        for frame in args.alarm_frame:
            if frame >= len(views):
                parser.error(f"--alarm-frame {frame} is outside {len(views)} frames")
            alarms[frame] = True
    elif args.state_policy == "detector-v3-prior-alarm":
        alarms, alarm_source = _detector_v3_alarms(args)
        if len(alarms) != len(views):
            raise RuntimeError("frozen detector alarm count does not equal state-policy input frame count")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = _run_without_grad(
        torch,
        lambda: run_transactional_recurrent_lighter(
            views,
            model,
            device,
            torch=torch,
            to_gpu=dust3r_device_module.to_gpu,
            to_cpu=dust3r_device_module.to_cpu,
            canonicalize_model_update_type=dust3r_model_module.canonicalize_model_update_type,
            alarms=alarms,
            max_hold=args.max_hold,
        ),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    expected_updates = len(views) - 1
    if update_calls != expected_updates:
        raise RuntimeError(f"expected {expected_updates} calibrated updates, observed {update_calls}")
    predictions = result.predictions
    if len(predictions) != len(views):
        raise RuntimeError("transactional ReCal3R did not return one prediction per input frame")
    prediction_summary = smoke._prediction_summary(predictions, torch)
    pose_jumps, geometric_residuals, trajectory = smoke._output_health_signals(
        predictions, pose_encoding_to_camera=dust3r_camera_module.pose_encoding_to_camera, np=np
    )
    trace = model.get_u_calibration_trace()
    if not isinstance(trace, dict):
        raise RuntimeError("ReCal3R calibration trace is unavailable")
    health = adapt_recal3r_trace(
        trace,
        all_frame_ids=list(range(len(views))),
        timestamps=smoke._input_timestamps(args),
        batch_size=1,
    )
    if len(health) != len(views):
        raise RuntimeError("health ledger does not align one-to-one with inputs")
    measured_steps = [record.frame_id for record in health if record.uncertainty_u is not None]
    expected_steps = list(range(1, len(views)))
    if measured_steps != expected_steps:
        raise RuntimeError(f"trace steps {measured_steps} do not match expected steps {expected_steps}")
    health = [
        replace(record, pose_jump=pose_jumps[index], geometric_residual=geometric_residuals[index])
        for index, record in enumerate(health)
    ]
    health = smoke._apply_v2_overlap_to_health(health, v2_overlaps)

    smoke._verify_preflight_inputs(args)
    write_health_jsonl_atomic(args.output_dir / "health.jsonl", health)
    smoke._write_json_atomic(args.output_dir / "predictions-summary.json", prediction_summary)
    input_frames = smoke._input_frame_metadata(args)
    reference_frame = input_frames[0]
    smoke._write_json_atomic(
        args.output_dir / "trajectory.json",
        {
            "schema_version": "stateguard3r.trajectory.v0",
            "pose_encoding": "absT_quaR",
            "matrix_convention": "camera_to_first_input_frame_reference",
            "reference_frame": {
                "frame_id": 0,
                "source_index": reference_frame["source_index"],
                "path": reference_frame["path"],
                "sha256": reference_frame["sha256"],
            },
            "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)",
            "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)",
            "frames": trajectory,
        },
    )
    timeline_payload = {
        "schema_version": STATE_POLICY_SCHEMA_VERSION,
        "policy": args.state_policy,
        "max_hold": args.max_hold,
        "alarm_positions": [index for index, value in enumerate(alarms) if value],
        "alarm_semantics": (
            "synthetic feasibility control; frame t alarm can only affect update t+1"
            if alarm_source is None
            else "frozen formal Detector v3 hybrid alarm; frame t alarm can only affect update t+1"
        ),
        "alarm_source": alarm_source,
        "source_provenance": result.source_provenance,
        "transactions": result.timeline,
        "dropped_transaction_frame_ids": result.dropped_transaction_frame_ids,
    }
    smoke._write_json_atomic(args.output_dir / "state-timeline.json", timeline_payload)
    run_finished_at = datetime.now().astimezone().isoformat()
    metadata = {
        "schema_version": STATE_POLICY_SCHEMA_VERSION,
        "status": "succeeded",
        "argv": sys.argv if argv is None else [str(value) for value in argv],
        "baseline_root": str(args.baseline_root),
        "baseline_commit": args.baseline_commit,
        "baseline_tracked_worktree_clean": True,
        "module_paths": module_paths,
        "curope_kernel": curope_kernel_provenance,
        "runner": runner_provenance,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_size_bytes": args.checkpoint_size_bytes,
        "input_manifest": smoke._input_manifest_metadata(args),
        "images": [{"path": frame["path"], "sha256": frame["sha256"]} for frame in input_frames],
        "input_frames": input_frames,
        "frame_count": len(views),
        "size": args.size,
        "seed": args.seed,
        "model_update_type": "recal3r",
        "loaded_model_interface": loaded_model_interface,
        "checkpoint_state_dict": checkpoint_state_dict,
        "beta_base": args.beta_base,
        "recal3r_runtime_config": recal3r_runtime_config,
        "recal3r_runtime_config_source": runtime_config_source_provenance,
        "model_eval": not model.training,
        "calibrated_update_calls": update_calls,
        "expected_calibrated_update_calls": expected_updates,
        "trace_frame_steps": measured_steps,
        "health_profile": "v2",
        "online_visual_correspondence": v2_visual_metadata,
        "state_policy": {
            "name": args.state_policy,
            "max_hold": args.max_hold,
            "alarm_positions": [index for index, value in enumerate(alarms) if value],
            "causality": "decision_for_frame_t_reads_only_alarm_t_minus_1",
            "timeline_path": str(args.output_dir / "state-timeline.json"),
            "timeline_sha256": smoke._sha256(args.output_dir / "state-timeline.json"),
            "alarm_source": alarm_source,
        },
        "runtime_seconds": elapsed,
        "runtime_scope": "external_transactional_recurrent_lighter_only",
        "fps": len(views) / elapsed if elapsed > 0 else None,
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "pid": os.getpid(),
        "output_dir": str(args.output_dir),
        "peak_memory_allocated_mib": peak_memory_mib,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    smoke._write_json_atomic(args.output_dir / "run.json", metadata)
    _freeze_output_tree(args.output_dir)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
