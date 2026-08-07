#!/usr/bin/env python3
"""Run the development-only Detector-v3 same-frame online quarantine candidate.

The runner is intentionally independent from the v1 hold/replay wrapper.  It
accepts only the disclosed legacy development manifests, uses the immutable
formal-v3 detector configuration, and rejects recovery-quality commitments or
alarm artifacts.  A current candidate output is retained; only its recurrent
post-state is rolled back when the prefix-only detector alarms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_recal3r_smoke as smoke


SCHEMA_VERSION = "stateguard3r.recal3r-online-quarantine-v2.v1"
DEVELOPMENT_INPUT_ROOT = REPOSITORY_ROOT / "outputs" / "formal-v1-inputs-0001" / "development"
FORMAL_V3_CONFIG = REPOSITORY_ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"


def _parser() -> argparse.ArgumentParser:
    parser = smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v3")
    parser.add_argument(
        "--state-policy",
        choices=("always-commit", "detector-v3-online-current-quarantine"),
        required=True,
    )
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--watchdog", type=int, default=8)
    return parser


def _self_provenance() -> dict[str, Any]:
    try:
        commit = smoke._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
        changes = smoke._git_output(REPOSITORY_ROOT, "status", "--porcelain", "--untracked-files=no")
    except (OSError, smoke.subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect v2 runner provenance: {error}") from error
    if changes:
        raise RuntimeError("StateGuard3R has tracked changes; online quarantine runner refuses execution")
    return {
        "repository_root": str(REPOSITORY_ROOT),
        "commit": commit,
        "tracked_worktree_clean": True,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": smoke._sha256(Path(__file__).resolve()),
        "python_executable": sys.executable,
    }


def _frozen_json(path: Path, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    metadata = os.lstat(resolved)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise RuntimeError(f"{label} must be a frozen regular 0444 file")
    try:
        value = json.loads(resolved.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value, {"path": str(resolved), "sha256": smoke._sha256(resolved), "size_bytes": resolved.stat().st_size, "mode_octal": "0444"}


def _validate_v2_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.health_profile != "v3":
        parser.error("online quarantine requires capture-timestamp health profile v3")
    if args.watchdog != 8:
        parser.error("online quarantine watchdog is fixed at 8")
    if args.recovery_quality_commitment is not None:
        parser.error("online quarantine development runner may not consume a recovery-quality commitment")
    if args.input_manifest is None:
        parser.error("online quarantine requires an input manifest")
    manifest = args.input_manifest.resolve(strict=True)
    root = DEVELOPMENT_INPUT_ROOT.resolve(strict=True)
    if root not in manifest.parents:
        parser.error(f"online quarantine accepts development manifests only under {root}")
    try:
        payload = json.loads(manifest.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        parser.error(f"cannot parse development input manifest: {error}")
    if not isinstance(payload, dict) or payload.get("schema_version") != "stateguard3r.corruption.v1":
        parser.error("online quarantine requires a legacy stateguard3r.corruption.v1 manifest")
    if payload.get("source_is_read_only") is not True:
        parser.error("online quarantine requires a read-only development source")
    requested_config = args.detector_config.resolve(strict=False)
    if requested_config != FORMAL_V3_CONFIG.resolve(strict=False):
        parser.error(f"online quarantine detector config is fixed at {FORMAL_V3_CONFIG}")


def _cleanup_staging(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


@dataclass(frozen=True)
class _Observation:
    record: Any
    decision: Any
    prediction: Mapping[str, Any]


class _OnlineOverlap:
    """Compute exactly one current/previous RGB overlap before each forward."""

    def __init__(self) -> None:
        from stateguard3r.visual_overlap import VisualOverlapConfig, opencv_runtime_provenance

        self._config = VisualOverlapConfig(**smoke.V2_VISUAL_CONFIG)
        self._previous_image: Any | None = None
        self.values: list[float | None] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._hashes: list[str] = []
        self._provenance = opencv_runtime_provenance()

    def before_frame(self, frame_id: int, view: Mapping[str, Any]) -> dict[str, Any]:
        from stateguard3r.visual_overlap import online_visual_correspondence_coverage

        if frame_id != len(self.values) or "img" not in view:
            raise RuntimeError("online visual overlap frame alignment is invalid")
        current = view["img"]
        current_hash = smoke._model_ready_rgb_sha256(current)
        if frame_id == 0:
            overlap = None
        else:
            previous_hash = smoke._model_ready_rgb_sha256(self._previous_image)
            result = online_visual_correspondence_coverage(
                self._previous_image,
                current,
                config=self._config,
                reference_frame_id=frame_id - 1,
                frame_id=frame_id,
            )
            if previous_hash != smoke._model_ready_rgb_sha256(self._previous_image) or current_hash != smoke._model_ready_rgb_sha256(current):
                raise RuntimeError("online visual overlap modified a model-ready RGB input")
            overlap = float(result.score)
            self._diagnostics.append(result.to_dict())
        self.values.append(overlap)
        self._hashes.append(current_hash)
        self._previous_image = current
        return {"overlap": overlap}

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "stateguard3r.visual-overlap.v1",
            "input": "post_official_loader_post_deferred_transform_normalized_rgb",
            "causal_reference": "immediately_previous_frame_only",
            "computed": "inside recurrent loop before current forward; no future RGB pair is scored",
            "model_ready_uint8_rgb_sha256": self._hashes,
            "frame_results": self._diagnostics,
            "opencv": self._provenance,
        }


class _HealthObserver:
    """Keep observed rollback health outside the detector's committed prefix."""

    def __init__(self, detector: Any, *, torch: Any, pose_encoding_to_camera: Any, np: Any) -> None:
        self._detector = detector
        self._torch = torch
        self._pose_encoding_to_camera = pose_encoding_to_camera
        self._np = np
        self._previous_safe_prediction: Mapping[str, Any] | None = None
        self.records: list[Any] = []
        self.observed_ledger: list[Any] = []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any]) -> _Observation:
        from stateguard3r.health import adapt_recal3r_trace

        trace = model.get_u_calibration_trace()
        if not isinstance(trace, Mapping):
            raise RuntimeError("online quarantine candidate lacks ReCal3R trace")
        trace_records = adapt_recal3r_trace(trace, all_frame_ids=list(range(frame_id + 1)), batch_size=1)
        if len(trace_records) != frame_id + 1:
            raise RuntimeError("online quarantine candidate trace is not prefix-aligned")
        signals = [prediction] if self._previous_safe_prediction is None else [self._previous_safe_prediction, prediction]
        pose_jumps, geometric_residuals, _ = smoke._output_health_signals(
            signals, pose_encoding_to_camera=self._pose_encoding_to_camera, np=self._np
        )
        overlap = frame_context.get("overlap")
        if frame_id == 0:
            if overlap is not None:
                raise RuntimeError("online overlap frame zero must be null")
        elif not isinstance(overlap, float) or not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise RuntimeError("online overlap must be finite in [0, 1] after frame zero")
        record = replace(trace_records[-1], overlap=overlap, pose_jump=pose_jumps[-1], geometric_residual=geometric_residuals[-1])
        decision = self._detector.observe(record)
        return _Observation(record=record, decision=decision, prediction=prediction)

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation, *, quarantined: bool) -> None:
        self.records.append(observation.record)
        if quarantined:
            self._detector.quarantine(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="quarantine_current_rollback"))
        else:
            self._detector.commit(observation.record)
            self._previous_safe_prediction = observation.prediction
            self.observed_ledger.append(replace(observation.record, decision="commit"))

    def timeline_evidence(self, observation: _Observation) -> Mapping[str, Any]:
        decision = observation.decision
        return {
            "online_detector": {
                "frame_id": decision.frame_id,
                "continuous_score": decision.continuous_score,
                "hybrid_score": decision.hybrid_score,
                "continuous_alarm": decision.continuous_alarm,
                "timestamp_order_alarm": decision.timestamp_order_alarm,
                "hybrid_alarm": decision.hybrid_alarm,
                "finite_component_count": decision.finite_component_count,
                "policy_input_sha256": decision.policy_input_sha256,
            }
        }


def _safe_input_frames(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = args.input_manifest_data
    if manifest is None:
        raise RuntimeError("online quarantine requires validated manifest data")
    return [
        {"frame_id": frame.frame_index, "path": str(frame.path), "sha256": args.image_sha256[frame.path]}
        for frame in manifest.frames
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_v2_args(args, parser)
    smoke._validate_args(args, parser)
    runner_provenance = _self_provenance()
    requested_output = args.output_dir.resolve(strict=False)
    if requested_output.exists() or requested_output.parent != REPOSITORY_ROOT / "outputs":
        raise RuntimeError("online quarantine output must be a new direct outputs child")
    config_payload, config_artifact = _frozen_json(args.detector_config, label="frozen formal-v3 detector config")
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    import numpy as np
    import torch

    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.online_detector_v2 import FrozenDetectorV3Config, OnlinePrefixDetector
    from stateguard3r.recal3r_online_runner_v2 import run_online_quarantine_recurrent_lighter
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar, validate_timestamp_order_sidecar

    frozen_detector = FrozenDetectorV3Config.from_mapping(config_payload)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_VISIBLE_DEVICES was set but PyTorch CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    smoke._verify_preflight_inputs(args)
    views = smoke._prepare_input_views(args, torch)
    captures, capture_provenance = capture_timestamp_records(
        [frame.path for frame in args.input_manifest_data.frames],
        rgb_txt=args.rgb_timestamp_listing,
        dataset_root=args.timestamp_dataset_root,
    )
    timestamp_sidecar = timestamp_order_sidecar(captures, provenance=capture_provenance)
    validate_timestamp_order_sidecar(timestamp_sidecar, require_available=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{requested_output.name}.", suffix=".staging", dir=requested_output.parent))
    args.output_dir = staging
    try:
        run_started_at = datetime.now().astimezone().isoformat()
        sys.path.insert(0, str(args.baseline_root))
        import add_ckpt_path as add_ckpt_path_module

        module_paths = {"add_ckpt_path": smoke._assert_module_source(add_ckpt_path_module, args.baseline_root / "add_ckpt_path.py", "add_ckpt_path")}
        add_ckpt_path_module.add_path_to_dust3r(str(args.baseline_src / args.checkpoint.name))
        import dust3r.model as dust3r_model_module
        import dust3r.utils.camera as dust3r_camera_module
        import dust3r.utils.device as dust3r_device_module
        import models.curope.curope2d as curope2d_module

        module_paths.update(
            {
                "dust3r.model": smoke._assert_module_source(dust3r_model_module, args.baseline_src / "dust3r" / "model.py", "dust3r.model"),
                "dust3r.utils.camera": smoke._assert_module_source(dust3r_camera_module, args.baseline_src / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera"),
                "dust3r.utils.device": smoke._assert_module_source(dust3r_device_module, args.baseline_src / "dust3r" / "utils" / "device.py", "dust3r.utils.device"),
                "models.curope.curope2d": smoke._assert_module_source(curope2d_module, args.baseline_src / "croco" / "models" / "curope" / "curope2d.py", "models.curope.curope2d"),
            }
        )
        kernel_provenance = smoke._binary_module_provenance(curope2d_module._kernels, expected_directory=args.baseline_src / "croco" / "models" / "curope", filename_prefix="curope.", label="cuRoPE CUDA extension")
        model, checkpoint_state_dict = smoke._load_model_with_state_dict_audit(dust3r_model_module.ARCroco3DStereo, args.checkpoint, torch)
        smoke._write_json_atomic(staging / "checkpoint-load-audit.json", checkpoint_state_dict)
        if checkpoint_state_dict["strict"] is not False or checkpoint_state_dict["missing_keys"] or checkpoint_state_dict["unexpected_keys"]:
            raise RuntimeError("pinned ReCal3R checkpoint loader compatibility audit failed")
        loaded_model_interface = smoke._validate_model_interface(model, args.checkpoint.name)
        model = model.to(args.device)
        recal3r_runtime_config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base)
        model.eval()
        model.enable_u_calibration_trace(oracle_window=1)
        update_calls = 0
        original_update = model._compute_recal3r_update_mask

        def counted_update(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal update_calls
            update_calls += 1
            return original_update(*call_args, **call_kwargs)

        model._compute_recal3r_update_mask = counted_update
        overlaps = _OnlineOverlap()
        observer = (
            _HealthObserver(
                OnlinePrefixDetector(frozen_detector, captures=captures, capture_provenance=capture_provenance),
                torch=torch,
                pose_encoding_to_camera=dust3r_camera_module.pose_encoding_to_camera,
                np=np,
            )
            if args.state_policy == "detector-v3-online-current-quarantine"
            else None
        )
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        result = smoke._run_without_grad(
            torch,
            lambda: run_online_quarantine_recurrent_lighter(
                views,
                model,
                device,
                torch=torch,
                to_gpu=dust3r_device_module.to_gpu,
                to_cpu=dust3r_device_module.to_cpu,
                canonicalize_model_update_type=dust3r_model_module.canonicalize_model_update_type,
                before_frame=overlaps.before_frame,
                observer=observer,
                watchdog_limit=args.watchdog,
            ),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
        if update_calls != len(views) - 1:
            raise RuntimeError(f"expected {len(views) - 1} calibrated updates, observed {update_calls}")
        if len(result.predictions) != len(views) or len(overlaps.values) != len(views):
            raise RuntimeError("online quarantine output or overlap count differs from input count")
        prediction_summary = smoke._prediction_summary(result.predictions, torch)
        pose_jumps, geometric_residuals, trajectory = smoke._output_health_signals(result.predictions, pose_encoding_to_camera=dust3r_camera_module.pose_encoding_to_camera, np=np)
        if observer is None:
            trace = model.get_u_calibration_trace()
            if not isinstance(trace, Mapping):
                raise RuntimeError("always-commit trace is unavailable")
            health = smoke._apply_v2_overlap_to_health(
                [replace(record, pose_jump=pose_jumps[index], geometric_residual=geometric_residuals[index]) for index, record in enumerate(adapt_recal3r_trace(trace, all_frame_ids=list(range(len(views))), batch_size=1))],
                overlaps.values,
            )
            observed_ledger = None
        else:
            health = observer.records
            observed_ledger = observer.observed_ledger
            if len(health) != len(views) or len(observed_ledger) != len(views):
                raise RuntimeError("online quarantine observed health ledger is incomplete")
        smoke._verify_preflight_inputs(args)
        write_health_jsonl_atomic(staging / "health.jsonl", health)
        if observed_ledger is not None:
            write_health_jsonl_atomic(staging / "observed-health-ledger.jsonl", observed_ledger)
        smoke._write_json_atomic(staging / "predictions-summary.json", prediction_summary)
        safe_frames = _safe_input_frames(args)
        smoke._write_json_atomic(
            staging / "trajectory.json",
            {
                "schema_version": "stateguard3r.trajectory.v0",
                "pose_encoding": "absT_quaR",
                "matrix_convention": "camera_to_first_input_frame_reference",
                "reference_frame": safe_frames[0],
                "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)",
                "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)",
                "frames": trajectory,
            },
        )
        smoke._write_json_atomic(staging / "timestamp-order-sidecar.json", timestamp_sidecar)
        smoke._write_json_atomic(
            staging / "state-timeline.json",
            {
                "schema_version": SCHEMA_VERSION,
                "policy": args.state_policy,
                "watchdog": args.watchdog,
                "causality": "current action reads only current/past online health, previous/current RGB overlap, and current/past capture timestamps",
                "pending_transaction_count": 0,
                "source_provenance": result.source_provenance,
                "transactions": result.timeline,
            },
        )
        finished_at = datetime.now().astimezone().isoformat()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "argv": sys.argv if argv is None else [str(value) for value in argv],
            "baseline_root": str(args.baseline_root),
            "baseline_commit": args.baseline_commit,
            "baseline_tracked_worktree_clean": True,
            "module_paths": module_paths,
            "curope_kernel": kernel_provenance,
            "runner": runner_provenance,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "input_manifest": smoke._input_manifest_metadata(args),
            "safe_input_frames": safe_frames,
            "frame_count": len(views),
            "size": args.size,
            "seed": args.seed,
            "model_update_type": "recal3r",
            "loaded_model_interface": loaded_model_interface,
            "checkpoint_state_dict": checkpoint_state_dict,
            "beta_base": args.beta_base,
            "recal3r_runtime_config": recal3r_runtime_config,
            "model_eval": not model.training,
            "calibrated_update_calls": update_calls,
            "health_profile": "v3-online-prefix-adapter",
            "online_visual_correspondence": overlaps.metadata(),
            "online_timestamp_order": {"sidecar_path": str(staging / "timestamp-order-sidecar.json"), "sidecar_sha256": smoke._sha256(staging / "timestamp-order-sidecar.json")},
            "online_detector": {"config": config_artifact, "allowed_inputs": ["candidate_model_health", "previous_current_model_ready_rgb_overlap", "raw_rgb_capture_timestamp"], "forbidden": ["GT", "depth", "label", "event", "source_index", "future_frame", "baseline_alarm_artifact"]},
            "state_policy": {"name": args.state_policy, "watchdog": args.watchdog, "timeline_path": str(staging / "state-timeline.json"), "timeline_sha256": smoke._sha256(staging / "state-timeline.json")},
            "runtime_seconds": elapsed,
            "runtime_scope": "external_structural_snapshot_only_on_actual_quarantine",
            "fps": len(views) / elapsed if elapsed > 0 else None,
            "started_at": run_started_at,
            "finished_at": finished_at,
            "pid": os.getpid(),
            "output_dir": str(requested_output),
            "peak_memory_allocated_mib": peak_memory_mib,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        smoke._write_json_atomic(staging / "run.json", metadata)
        smoke._freeze_output_tree(staging)
        os.replace(staging, requested_output)
        print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception:
        _cleanup_staging(staging)
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
