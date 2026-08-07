#!/usr/bin/env python3
"""Run the preregistered development-only Detector-v3 safe-anchor candidate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.run_recal3r_online_quarantine_v2 as v2

smoke = v2.smoke
SCHEMA_VERSION = "stateguard3r.recal3r-safe-anchor-export-v3.v1"
DEVELOPMENT_INPUT_ROOT = ROOT / "outputs" / "formal-v1-inputs-0001" / "development"
FORMAL_V3_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"


def _parser() -> argparse.ArgumentParser:
    parser = smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v3")
    parser.add_argument("--state-policy", choices=("always-commit", "detector-v3-incremental-safe-anchor-se3-export"), required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--watchdog", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.health_profile != "v3" or args.watchdog != 8:
        parser.error("safe-anchor recovery requires fixed v3 health and watchdog=8")
    if args.recovery_quality_commitment is not None:
        parser.error("safe-anchor development may not consume a recovery-quality commitment")
    if args.input_manifest is None:
        parser.error("safe-anchor recovery requires a development input manifest")
    manifest = args.input_manifest.resolve(strict=True)
    if DEVELOPMENT_INPUT_ROOT.resolve() not in manifest.parents:
        parser.error("safe-anchor recovery accepts only legacy development manifests")
    payload = json.loads(manifest.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema_version") != "stateguard3r.corruption.v1" or payload.get("source_is_read_only") is not True:
        parser.error("safe-anchor recovery requires a read-only legacy development manifest")
    if args.detector_config.resolve(strict=False) != FORMAL_V3_CONFIG.resolve(strict=False):
        parser.error("safe-anchor recovery detector config is fixed")


def _median_numpy_equivalent(torch: Any, values: Any) -> Any:
    flat = values.reshape(-1)
    ordered = torch.sort(flat).values
    length = int(ordered.numel())
    return ordered[length // 2] if length % 2 else (ordered[length // 2 - 1] + ordered[length // 2]) / 2


@dataclass(frozen=True)
class _Observation:
    frame_id: int
    record: Any
    decision: Any
    raw_pose: Any


class _V3Observer:
    """Raw candidate health is consumed before any export-only replacement."""

    def __init__(self, detector: Any, exporter: Any, *, timestamps: Sequence[Any], torch: Any, pose_encoding_to_camera: Any, camera_to_pose_encoding: Any) -> None:
        self._detector, self._exporter = detector, exporter
        self._timestamps, self._torch = list(timestamps), torch
        self._decode, self._encode = pose_encoding_to_camera, camera_to_pose_encoding
        self._previous_safe_camera: Any | None = None
        self.records: list[Any] = []
        self.observed_ledger: list[Any] = []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any]) -> _Observation:
        from stateguard3r.health import adapt_recal3r_trace

        trace = model.get_u_calibration_trace()
        records = adapt_recal3r_trace(trace, all_frame_ids=list(range(frame_id + 1)), timestamps=self._timestamps[: frame_id + 1], batch_size=1)
        if len(records) != frame_id + 1:
            raise RuntimeError("safe-anchor current health trace is not prefix aligned")
        overlap = frame_context.get("overlap")
        if frame_id == 0:
            if overlap is not None:
                raise RuntimeError("safe-anchor overlap frame zero must be null")
        elif not isinstance(overlap, float) or not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise RuntimeError("safe-anchor overlap must be finite in [0, 1]")
        pose = prediction["camera_pose"].detach().clone()
        camera = self._decode(pose)
        pose_jump = None
        if self._previous_safe_camera is not None:
            relative_rotation = self._previous_safe_camera[:, :3, :3].transpose(-1, -2) @ camera[:, :3, :3]
            cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            rotation = self._torch.acos(cosine)
            delta = self._previous_safe_camera[:, :3, :3].transpose(-1, -2) @ (camera[:, :3, 3] - self._previous_safe_camera[:, :3, 3]).unsqueeze(-1)
            pose_jump = float(self._torch.hypot(self._torch.linalg.vector_norm(delta.squeeze(-1), dim=-1), rotation).item())
        transformed = prediction["pts3d_in_self_view"] @ camera[:, :3, :3].transpose(-1, -2) + camera[:, None, None, :3, 3]
        errors = self._torch.linalg.vector_norm(transformed - prediction["pts3d_in_other_view"], dim=-1)
        scales = self._torch.linalg.vector_norm(prediction["pts3d_in_other_view"], dim=-1)
        geometric = float((_median_numpy_equivalent(self._torch, errors) / self._torch.clamp(_median_numpy_equivalent(self._torch, scales), min=1e-8)).item())
        record = replace(records[-1], overlap=overlap, pose_jump=pose_jump, geometric_residual=geometric)
        return _Observation(frame_id, record, self._detector.observe(record), pose)

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation, *, quarantined: bool, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.records.append(observation.record)
        if quarantined:
            self._detector.quarantine(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="quarantine_current_rollback"))
            exported, action = self._exporter.fallback(observation.frame_id, prediction)
        else:
            self._detector.commit(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="commit"))
            self._previous_safe_camera = observation.raw_pose.detach().clone()
            exported, action = self._exporter.commit_real(observation.frame_id, prediction)
        return exported, {"export_action": action.action, "anchor_frame_ids": list(action.anchor_frame_ids) if action.anchor_frame_ids is not None else None}

    def timeline_evidence(self, observation: _Observation) -> Mapping[str, Any]:
        decision = observation.decision
        return {"online_detector": {"frame_id": decision.frame_id, "continuous_score": decision.continuous_score, "hybrid_score": decision.hybrid_score, "continuous_alarm": decision.continuous_alarm, "timestamp_order_alarm": decision.timestamp_order_alarm, "hybrid_alarm": decision.hybrid_alarm, "finite_component_count": decision.finite_component_count, "policy_input_sha256": decision.policy_input_sha256}}


def _exported_trajectory(predictions: Sequence[Mapping[str, Any]], records: Sequence[Any], *, decode: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame_id, (prediction, raw) in enumerate(zip(predictions, records, strict=True)):
        pose = prediction["camera_pose"].detach().clone()
        camera = decode(pose).detach().cpu().numpy()[0]
        output.append({"frame_id": frame_id, "camera_pose_encoding_absT_quaR": pose.detach().cpu().numpy()[0].tolist(), "camera_to_reference": camera.tolist(), "pose_jump": None, "geometric_residual": raw.geometric_residual, "geometric_residual_semantics": "raw_candidate_pointmaps_before_export_replacement"})
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    smoke._validate_args(args, parser)
    runner_provenance = v2._self_provenance()
    requested = args.output_dir.resolve(strict=False)
    if requested.exists() or requested.parent != ROOT / "outputs":
        raise RuntimeError("safe-anchor output must be a new direct outputs child")
    config_payload, config_artifact = v2._frozen_json(args.detector_config, label="frozen Detector-v3 config")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from stateguard3r.online_detector_incremental_v3 import IncrementalOnlinePrefixDetector
    from stateguard3r.online_detector_v2 import FrozenDetectorV3Config
    from stateguard3r.recal3r_safe_anchor_runner_v3 import run_safe_anchor_recurrent_lighter
    from stateguard3r.safe_anchor_export_v3 import SafeAnchorSE3Export
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar, validate_timestamp_order_sidecar
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == "cuda": torch.cuda.manual_seed_all(args.seed)
    for source_root in (args.baseline_root, args.baseline_src):
        if str(source_root) not in sys.path: sys.path.insert(0, str(source_root))
    smoke._verify_preflight_inputs(args)
    views = smoke._prepare_input_views(args, torch)
    captures, provenance = capture_timestamp_records([frame.path for frame in args.input_manifest_data.frames], rgb_txt=args.rgb_timestamp_listing, dataset_root=args.timestamp_dataset_root)
    sidecar = timestamp_order_sidecar(captures, provenance=provenance); validate_timestamp_order_sidecar(sidecar, require_available=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{requested.name}.", suffix=".staging", dir=requested.parent)); args.output_dir = staging
    try:
        started_at = datetime.now().astimezone().isoformat()
        import add_ckpt_path as add_ckpt_path_module
        module_paths = {"add_ckpt_path": smoke._assert_module_source(add_ckpt_path_module, args.baseline_root / "add_ckpt_path.py", "add_ckpt_path")}
        add_ckpt_path_module.add_path_to_dust3r(str(args.baseline_src / args.checkpoint.name))
        import dust3r.model as dust3r_model; import dust3r.utils.camera as camera; import dust3r.utils.device as device_module; import models.curope.curope2d as curope
        module_paths.update({"dust3r.model": smoke._assert_module_source(dust3r_model, args.baseline_src / "dust3r" / "model.py", "dust3r.model"), "dust3r.utils.camera": smoke._assert_module_source(camera, args.baseline_src / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera")})
        model, audit = smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, args.checkpoint, torch); smoke._write_json_atomic(staging / "checkpoint-load-audit.json", audit)
        if audit["strict"] is not False or audit["missing_keys"] or audit["unexpected_keys"]: raise RuntimeError("checkpoint audit failed")
        interface = smoke._validate_model_interface(model, args.checkpoint.name); model = model.to(args.device); config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base); model.eval(); model.enable_u_calibration_trace(oracle_window=1)
        updates = 0; original_update = model._compute_recal3r_update_mask
        def counted(*a: Any, **k: Any) -> Any:
            nonlocal updates
            updates += 1; return original_update(*a, **k)
        model._compute_recal3r_update_mask = counted
        overlaps = v2._OnlineOverlap()
        observer = None
        if args.state_policy != "always-commit":
            observer = _V3Observer(IncrementalOnlinePrefixDetector(FrozenDetectorV3Config.from_mapping(config_payload), captures=captures, capture_provenance=provenance), SafeAnchorSE3Export(pose_encoding_to_camera=camera.pose_encoding_to_camera, camera_to_pose_encoding=camera.camera_to_pose_encoding, inverse=torch.linalg.inv), timestamps=smoke._input_timestamps(args), torch=torch, pose_encoding_to_camera=camera.pose_encoding_to_camera, camera_to_pose_encoding=camera.camera_to_pose_encoding)
        device = torch.device(args.device)
        if device.type == "cuda": torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad(): result = run_safe_anchor_recurrent_lighter(views, model, device, torch=torch, to_gpu=device_module.to_gpu, to_cpu=device_module.to_cpu, canonicalize_model_update_type=dust3r_model.canonicalize_model_update_type, before_frame=overlaps.before_frame, observer=observer, watchdog_limit=args.watchdog)
        if device.type == "cuda": torch.cuda.synchronize(device)
        synchronized_elapsed = time.perf_counter() - started
        if updates != len(views) - 1 or len(result.predictions) != len(views) or len(overlaps.values) != len(views): raise RuntimeError("safe-anchor recurrent accounting differs")
        summary = smoke._prediction_summary(result.predictions, torch)
        if observer is None:
            jumps, residuals, trajectory = smoke._output_health_signals(result.predictions, pose_encoding_to_camera=camera.pose_encoding_to_camera, np=np)
            trace = model.get_u_calibration_trace(); health = smoke._apply_v2_overlap_to_health([replace(record, pose_jump=jumps[index], geometric_residual=residuals[index]) for index, record in enumerate(adapt_recal3r_trace(trace, all_frame_ids=list(range(len(views))), timestamps=smoke._input_timestamps(args), batch_size=1))], overlaps.values); ledger = None
        else:
            health, ledger = observer.records, observer.observed_ledger; trajectory = _exported_trajectory(result.predictions, health, decode=camera.pose_encoding_to_camera)
            if len(health) != len(views) or len(ledger) != len(views): raise RuntimeError("safe-anchor observed ledger is incomplete")
        smoke._verify_preflight_inputs(args); write_health_jsonl_atomic(staging / "health.jsonl", health)
        if ledger is not None: write_health_jsonl_atomic(staging / "observed-health-ledger.jsonl", ledger)
        smoke._write_json_atomic(staging / "predictions-summary.json", summary)
        safe_frames = [{"frame_id": frame.frame_index, "path": str(frame.path), "sha256": args.image_sha256[frame.path]} for frame in args.input_manifest_data.frames]
        smoke._write_json_atomic(staging / "trajectory.json", {"schema_version": "stateguard3r.trajectory.safe-anchor-v3", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "camera_pose_semantics": "exported_camera_pose; candidate health remains raw", "frames": trajectory})
        smoke._write_json_atomic(staging / "timestamp-order-sidecar.json", sidecar)
        smoke._write_json_atomic(staging / "state-timeline.json", {"schema_version": SCHEMA_VERSION, "policy": args.state_policy, "watchdog": args.watchdog, "causality": "current/past raw candidate health, previous/current RGB overlap, and current/past captures only; safe export never feeds model or detector", "pending_transaction_count": 0, "source_provenance": result.source_provenance, "transactions": result.timeline})
        finished = datetime.now().astimezone().isoformat()
        metadata = {"schema_version": SCHEMA_VERSION, "status": "succeeded", "argv": sys.argv if argv is None else [str(x) for x in argv], "baseline_root": str(args.baseline_root), "baseline_commit": args.baseline_commit, "baseline_tracked_worktree_clean": True, "module_paths": module_paths, "runner": runner_provenance, "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256, "input_manifest": smoke._input_manifest_metadata(args), "safe_input_frames": safe_frames, "frame_count": len(views), "size": args.size, "seed": args.seed, "model_update_type": "recal3r", "loaded_model_interface": interface, "checkpoint_state_dict": audit, "beta_base": args.beta_base, "recal3r_runtime_config": config, "model_eval": not model.training, "calibrated_update_calls": updates, "health_profile": "v3-gpu-current-raw-candidate", "online_visual_correspondence": overlaps.metadata(), "online_detector": {"config": config_artifact, "allowed_inputs": ["candidate_model_health", "previous_current_model_ready_rgb_overlap", "raw_rgb_capture_timestamp"], "forbidden": ["GT", "depth", "label", "event", "source_index", "future_frame", "baseline_alarm_artifact"]}, "state_policy": {"name": args.state_policy, "watchdog": args.watchdog, "timeline_path": str(staging / "state-timeline.json"), "timeline_sha256": smoke._sha256(staging / "state-timeline.json")}, "runtime_seconds": synchronized_elapsed, "wall_runtime_seconds": synchronized_elapsed, "runtime_scope": "synchronized_v3_recurrent_policy_and_common_causal_overlap_loop", "started_at": started_at, "finished_at": finished, "pid": os.getpid(), "output_dir": str(requested), "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0, "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}
        smoke._write_json_atomic(staging / "run.json", metadata); smoke._freeze_output_tree(staging); os.replace(staging, requested); print(json.dumps(metadata, ensure_ascii=False, indent=2)); return 0
    except Exception:
        v2._cleanup_staging(staging); raise


if __name__ == "__main__":
    raise SystemExit(main())
