#!/usr/bin/env python3
"""Run preregistered v13 selective persistent state/memory repair.

The candidate never replaces an alarm-frame pose.  It uses the frozen current
Detector-v3 decision to repair only the largest native state/memory row deltas
before later frames consume the persistent tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
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
SCHEMA_VERSION = "stateguard3r.recal3r-selective-state-memory-repair-v13.v1"
DEVELOPMENT_INPUT_ROOT = ROOT / "outputs" / "formal-v1-inputs-0001" / "development"
DEVELOPMENT_MANIFESTS = frozenset(
    DEVELOPMENT_INPUT_ROOT / f"development-{condition}" / "input-manifest.json"
    for condition in ("dynamic", "wrong", "low")
)
FORMAL_V3_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
V13_COMPONENTS = (
    ROOT / "scripts" / "run_recal3r_selective_state_memory_repair_v13.py",
    ROOT / "src" / "stateguard3r" / "recal3r_selective_state_memory_repair_runner_v13.py",
    ROOT / "src" / "stateguard3r" / "selective_state_memory_repair_v13.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v3")
    parser.add_argument(
        "--state-policy",
        choices=("always-commit", "detector-v3-incremental-selective-state-memory-repair"),
        required=True,
    )
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--watchdog", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.health_profile != "v3" or args.watchdog != 8:
        parser.error("v13 selective repair requires fixed v3 health and watchdog=8")
    if args.recovery_quality_commitment is not None:
        parser.error("v13 development may not consume a recovery-quality commitment")
    if args.input_manifest is None:
        parser.error("v13 selective repair requires a development input manifest")
    manifest = args.input_manifest.resolve(strict=True)
    if manifest not in {path.resolve(strict=True) for path in DEVELOPMENT_MANIFESTS}:
        parser.error("v13 selective repair accepts only disclosed development manifests")
    payload = json.loads(manifest.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema_version") != "stateguard3r.corruption.v1" or payload.get("source_is_read_only") is not True:
        parser.error("v13 selective repair requires a read-only development manifest")
    if args.detector_config.resolve(strict=False) != FORMAL_V3_CONFIG.resolve(strict=False):
        parser.error("v13 selective repair detector config is fixed")


def _self_provenance() -> dict[str, Any]:
    try:
        commit = smoke._git_output(ROOT, "rev-parse", "HEAD")
        changes = smoke._git_output(ROOT, "status", "--porcelain", "--untracked-files=no")
    except Exception as error:
        raise RuntimeError(f"cannot inspect v13 runner provenance: {error}") from error
    if changes:
        raise RuntimeError("StateGuard3R has tracked changes; v13 runner refuses execution")
    for component in V13_COMPONENTS:
        try:
            relative = component.resolve(strict=True).relative_to(ROOT)
            smoke._git_output(ROOT, "ls-files", "--error-unmatch", "--", str(relative))
            if smoke._git_output(ROOT, "rev-parse", f"HEAD:{relative}") != smoke._git_output(ROOT, "hash-object", "--", str(relative)):
                raise RuntimeError("component differs from its HEAD blob")
        except Exception as error:
            raise RuntimeError(f"v13 component is not tracked at HEAD: {component}") from error
    return {
        "repository_root": str(ROOT), "commit": commit, "tracked_worktree_clean": True,
        "script_path": str(Path(__file__).resolve()), "script_sha256": smoke._sha256(Path(__file__).resolve()),
        "python_executable": sys.executable,
    }


def _component_provenance() -> list[dict[str, str]]:
    return [{"path": str(path.resolve(strict=True)), "sha256": smoke._sha256(path.resolve(strict=True))} for path in V13_COMPONENTS]


def _validate_new_output_dir(requested: Path) -> None:
    if requested.exists() or requested.parent != ROOT / "outputs":
        raise RuntimeError("v13 selective-repair output must be a new direct outputs child")


def _median(torch: Any, values: Any) -> Any:
    ordered = torch.sort(values.reshape(-1)).values
    return ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


def _pose_gpu_digest(value: Any, *, torch: Any) -> str:
    tensor = value.detach().contiguous()
    if getattr(tensor.device, "type", None) != "cuda" or not bool(tensor.isfinite().all()):
        raise RuntimeError("v13 raw-pose evidence must be finite and CUDA-resident")
    flat = tensor.reshape(-1).to(dtype=torch.float64)
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    moments = (flat.sum(), flat.abs().sum(), (flat * positions).sum(), flat.square().sum(), flat.min(), flat.max())
    payload = "|".join((str(tuple(tensor.shape)), str(tensor.dtype), *(float(item.item()).hex() for item in moments)))
    return "gpu-fingerprint-v1:" + hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class _Observation:
    frame_id: int
    record: Any
    decision: Any
    raw_pose_digest: str


class _V13Observer:
    """Consume raw health first; finalize an unchanged raw current prediction."""

    def __init__(self, detector: Any, *, timestamps: Sequence[Any], torch: Any, pose_encoding_to_camera: Any) -> None:
        self._detector, self._timestamps, self._torch, self._decode = detector, list(timestamps), torch, pose_encoding_to_camera
        self._previous_raw_camera = None
        self.records: list[Any] = []
        self.observed_ledger: list[Any] = []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any]) -> _Observation:
        from stateguard3r.health import adapt_recal3r_trace
        if not isinstance(frame_context, Mapping):
            raise RuntimeError("v13 candidate frame context is unavailable")
        records = adapt_recal3r_trace(model.get_u_calibration_trace(), all_frame_ids=list(range(frame_id + 1)), timestamps=self._timestamps[: frame_id + 1], batch_size=1)
        if len(records) != frame_id + 1:
            raise RuntimeError("v13 current health trace is not prefix aligned")
        overlap = frame_context.get("overlap")
        if (frame_id == 0 and overlap is not None) or (frame_id and (not isinstance(overlap, float) or not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0)):
            raise RuntimeError("v13 overlap is not a valid causal current/previous value")
        raw_pose = prediction["camera_pose"].detach().clone()
        raw_camera = self._decode(raw_pose)
        pose_jump = None
        if self._previous_raw_camera is not None:
            relative = self._previous_raw_camera[:, :3, :3].transpose(-1, -2) @ raw_camera[:, :3, :3]
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            delta = self._previous_raw_camera[:, :3, :3].transpose(-1, -2) @ (raw_camera[:, :3, 3] - self._previous_raw_camera[:, :3, 3]).unsqueeze(-1)
            pose_jump = float(self._torch.hypot(self._torch.linalg.vector_norm(delta.squeeze(-1), dim=-1), self._torch.acos(cosine)).item())
        transformed = prediction["pts3d_in_self_view"] @ raw_camera[:, :3, :3].transpose(-1, -2) + raw_camera[:, None, None, :3, 3]
        errors = self._torch.linalg.vector_norm(transformed - prediction["pts3d_in_other_view"], dim=-1)
        scales = self._torch.linalg.vector_norm(prediction["pts3d_in_other_view"], dim=-1)
        geometric = float((_median(self._torch, errors) / self._torch.clamp(_median(self._torch, scales), min=1e-8)).item())
        record = replace(records[-1], overlap=overlap, pose_jump=pose_jump, geometric_residual=geometric)
        return _Observation(frame_id, record, self._detector.observe(record), _pose_gpu_digest(raw_pose, torch=self._torch))

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation, *, repaired: bool, prediction: Mapping[str, Any], repair_evidence: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        exported_digest = _pose_gpu_digest(prediction["camera_pose"], torch=self._torch)
        if exported_digest != observation.raw_pose_digest:
            raise RuntimeError("v13 repair changed the raw current camera pose")
        self.records.append(observation.record)
        if repaired:
            self._detector.quarantine(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="selective_state_memory_repair"))
        else:
            self._detector.commit(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="commit"))
        self._previous_raw_camera = self._decode(prediction["camera_pose"].detach().clone()).detach().clone()
        return prediction, {
            "export_action": "export_real_camera_pose",
            "raw_candidate_pose_gpu_digest": observation.raw_pose_digest,
            "exported_camera_pose_gpu_digest": exported_digest,
            "raw_pose_unchanged": True,
            "selective_state_memory_repair": repair_evidence,
        }

    def timeline_evidence(self, observation: _Observation) -> Mapping[str, Any]:
        decision = observation.decision
        return {"online_detector": {"frame_id": decision.frame_id, "continuous_score": decision.continuous_score, "hybrid_score": decision.hybrid_score, "continuous_alarm": decision.continuous_alarm, "timestamp_order_alarm": decision.timestamp_order_alarm, "hybrid_alarm": decision.hybrid_alarm, "finite_component_count": decision.finite_component_count, "policy_input_sha256": decision.policy_input_sha256}}


def _raw_trajectory(predictions: Sequence[Mapping[str, Any]], records: Sequence[Any], *, decode: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_id, (prediction, record) in enumerate(zip(predictions, records, strict=True)):
        pose = prediction["camera_pose"].detach().clone()
        camera = decode(pose).detach().cpu().numpy()[0]
        rows.append({"frame_id": frame_id, "camera_pose_encoding_absT_quaR": pose.detach().cpu().numpy()[0].tolist(), "camera_to_reference": camera.tolist(), "pose_jump": None, "geometric_residual": record.geometric_residual, "geometric_residual_semantics": "raw_current_prediction_before_and_after_v13_state_memory_repair"})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    smoke._validate_args(args, parser)
    runner_provenance, component_provenance = _self_provenance(), _component_provenance()
    requested = args.output_dir.resolve(strict=False)
    _validate_new_output_dir(requested)
    config_payload, config_artifact = v2._frozen_json(args.detector_config, label="frozen Detector-v3 config")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.online_detector_incremental_v3 import IncrementalOnlinePrefixDetector
    from stateguard3r.online_detector_v2 import FrozenDetectorV3Config
    from stateguard3r.recal3r_selective_state_memory_repair_runner_v13 import audit_v13_runner_contract, run_selective_state_memory_repair_recurrent_lighter
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar, validate_timestamp_order_sidecar
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v13 production runner requires CUDA")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    for source_root in (args.baseline_root, args.baseline_src):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    smoke._verify_preflight_inputs(args)
    runner_audit = audit_v13_runner_contract(ROOT / "src" / "stateguard3r" / "recal3r_selective_state_memory_repair_runner_v13.py")
    views = smoke._prepare_input_views(args, torch)
    captures, capture_provenance = capture_timestamp_records([frame.path for frame in args.input_manifest_data.frames], rgb_txt=args.rgb_timestamp_listing, dataset_root=args.timestamp_dataset_root)
    sidecar = timestamp_order_sidecar(captures, provenance=capture_provenance)
    validate_timestamp_order_sidecar(sidecar, require_available=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{requested.name}.", suffix=".staging", dir=requested.parent))
    args.output_dir = staging
    try:
        started_at = datetime.now().astimezone().isoformat()
        import add_ckpt_path as add_ckpt_path_module
        module_paths = {"add_ckpt_path": smoke._assert_module_source(add_ckpt_path_module, args.baseline_root / "add_ckpt_path.py", "add_ckpt_path")}
        add_ckpt_path_module.add_path_to_dust3r(str(args.baseline_src / args.checkpoint.name))
        import dust3r.model as dust3r_model
        import dust3r.utils.camera as camera
        import dust3r.utils.device as device_module
        import models.curope.curope2d as curope  # noqa: F401
        module_paths.update({"dust3r.model": smoke._assert_module_source(dust3r_model, args.baseline_src / "dust3r" / "model.py", "dust3r.model"), "dust3r.utils.camera": smoke._assert_module_source(camera, args.baseline_src / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera")})
        model, checkpoint_audit = smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, args.checkpoint, torch)
        smoke._write_json_atomic(staging / "checkpoint-load-audit.json", checkpoint_audit)
        if checkpoint_audit["strict"] is not False or checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
            raise RuntimeError("checkpoint audit failed")
        interface = smoke._validate_model_interface(model, args.checkpoint.name)
        model = model.to(args.device)
        config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base)
        model.eval(); model.enable_u_calibration_trace(oracle_window=1)
        updates, original_update = 0, model._compute_recal3r_update_mask
        def counted(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal updates
            updates += 1
            return original_update(*call_args, **call_kwargs)
        model._compute_recal3r_update_mask = counted
        overlaps = v2._OnlineOverlap()
        observer = None
        if args.state_policy != "always-commit":
            observer = _V13Observer(IncrementalOnlinePrefixDetector(FrozenDetectorV3Config.from_mapping(config_payload), captures=captures, capture_provenance=capture_provenance), timestamps=smoke._input_timestamps(args), torch=torch, pose_encoding_to_camera=camera.pose_encoding_to_camera)
        device = torch.device(args.device)
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            result = run_selective_state_memory_repair_recurrent_lighter(views, model, device, torch=torch, to_gpu=device_module.to_gpu, to_cpu=device_module.to_cpu, canonicalize_model_update_type=dust3r_model.canonicalize_model_update_type, before_frame=overlaps.before_frame, observer=observer, watchdog_limit=args.watchdog, synchronize=lambda: torch.cuda.synchronize(device))
        torch.cuda.synchronize(device)
        wall_elapsed = time.perf_counter() - started
        if updates != len(views) - 1 or len(result.predictions) != len(views) or len(overlaps.values) != len(views):
            raise RuntimeError("v13 recurrent accounting differs")
        summary = smoke._prediction_summary(result.predictions, torch)
        if observer is None:
            jumps, residuals, trajectory = smoke._output_health_signals(result.predictions, pose_encoding_to_camera=camera.pose_encoding_to_camera, np=np)
            trace = model.get_u_calibration_trace()
            health = smoke._apply_v2_overlap_to_health([replace(record, pose_jump=jumps[index], geometric_residual=residuals[index]) for index, record in enumerate(adapt_recal3r_trace(trace, all_frame_ids=list(range(len(views))), timestamps=smoke._input_timestamps(args), batch_size=1))], overlaps.values)
            ledger = None
            reference = smoke._input_frame_metadata(args)[0]
            trajectory_payload = {"schema_version": "stateguard3r.trajectory.v0", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "reference_frame": {"frame_id": 0, "source_index": reference["source_index"], "path": reference["path"], "sha256": reference["sha256"]}, "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)", "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)", "frames": trajectory}
        else:
            health, ledger = observer.records, observer.observed_ledger
            if len(health) != len(views) or len(ledger) != len(views):
                raise RuntimeError("v13 observed health ledger is incomplete")
            trajectory_payload = {"schema_version": "stateguard3r.trajectory.selective-state-memory-repair-v13", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "camera_pose_semantics": "every frame exports its unmodified raw current camera pose; v13 repairs only future persistent state/memory", "repair_evidence_location": "state-timeline.json transactions", "frames": _raw_trajectory(result.predictions, health, decode=camera.pose_encoding_to_camera)}
        smoke._verify_preflight_inputs(args)
        write_health_jsonl_atomic(staging / "health.jsonl", health)
        if ledger is not None:
            write_health_jsonl_atomic(staging / "observed-health-ledger.jsonl", ledger)
        smoke._write_json_atomic(staging / "predictions-summary.json", summary)
        smoke._write_json_atomic(staging / "trajectory.json", trajectory_payload)
        smoke._write_json_atomic(staging / "timestamp-order-sidecar.json", sidecar)
        timeline = {"schema_version": SCHEMA_VERSION, "policy": args.state_policy, "watchdog": args.watchdog, "causality": "detector consumes raw current health before selective repair; repair consumes only pre/proposed persistent state and memory tensors; current raw camera pose is unchanged; no GT/future/anchor/pointmap/latent/pose fallback or export feedback", "selective_state_memory_runner_audit": runner_audit, "pending_transaction_count": 0, "source_provenance": result.source_provenance, "transactions": result.timeline}
        smoke._write_json_atomic(staging / "state-timeline.json", timeline)
        metadata = {"schema_version": SCHEMA_VERSION, "status": "succeeded", "argv": sys.argv if argv is None else [str(item) for item in argv], "baseline_root": str(args.baseline_root), "baseline_commit": args.baseline_commit, "baseline_tracked_worktree_clean": True, "module_paths": module_paths, "runner": runner_provenance, "v13_components": component_provenance, "selective_state_memory_runner_audit": runner_audit, "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha256, "input_manifest": smoke._input_manifest_metadata(args), "safe_input_frames": [{"frame_id": frame.frame_index, "path": str(frame.path), "sha256": args.image_sha256[frame.path]} for frame in args.input_manifest_data.frames], "frame_count": len(views), "size": args.size, "seed": args.seed, "model_update_type": "recal3r", "loaded_model_interface": interface, "checkpoint_state_dict": checkpoint_audit, "beta_base": args.beta_base, "recal3r_runtime_config": config, "model_eval": not model.training, "calibrated_update_calls": updates, "health_profile": "v13-gpu-current-raw-candidate", "online_visual_correspondence": overlaps.metadata(), "online_detector": {"config": config_artifact, "allowed_inputs": ["candidate_model_health", "previous_current_model_ready_rgb_overlap", "raw_rgb_capture_timestamp"], "forbidden": ["GT", "depth", "label", "event", "source_index", "future_frame", "baseline_alarm_artifact"]}, "state_policy": {"name": args.state_policy, "watchdog": args.watchdog, "timeline_path": str(requested / "state-timeline.json"), "timeline_sha256": smoke._sha256(staging / "state-timeline.json")}, "runtime_seconds": result.recurrent_policy_runtime_seconds, "wall_runtime_seconds": wall_elapsed, "runtime_scope": "per_frame_cuda_synchronized_v13_recurrent_policy_including_detector_and_selective_state_memory_repair_excluding_common_causal_rgb_overlap", "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(), "pid": os.getpid(), "output_dir": str(requested), "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2), "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda}
        smoke._write_json_atomic(staging / "run.json", metadata)
        smoke._freeze_output_tree(staging)
        os.replace(staging, requested)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        v2._cleanup_staging(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
