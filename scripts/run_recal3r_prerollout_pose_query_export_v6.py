#!/usr/bin/env python3
"""Run the preregistered v6 pre-rollout pose-query export candidate."""

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
SCHEMA_VERSION = "stateguard3r.recal3r-prerollout-pose-query-export-v6.v1"
DEVELOPMENT_INPUT_ROOT = ROOT / "outputs" / "formal-v1-inputs-0001" / "development"
FORMAL_V3_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
V6_COMPONENTS = (
    ROOT / "scripts" / "run_recal3r_prerollout_pose_query_export_v6.py",
    ROOT / "src" / "stateguard3r" / "recal3r_prerollout_pose_query_runner_v6.py",
    ROOT / "src" / "stateguard3r" / "prerollout_pose_query_export_v6.py",
    ROOT / "src" / "stateguard3r" / "prerollout_pose_query_v6.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v3")
    parser.add_argument(
        "--state-policy",
        choices=("always-commit", "detector-v3-incremental-prerollout-pose-query-export"),
        required=True,
    )
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--watchdog", type=int, default=8)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.health_profile != "v3" or args.watchdog != 8:
        parser.error("v6 pre-rollout pose query requires fixed v3 health and watchdog=8")
    if args.recovery_quality_commitment is not None:
        parser.error("v6 development may not consume a recovery-quality commitment")
    if args.input_manifest is None:
        parser.error("v6 pre-rollout pose query requires a development input manifest")
    manifest = args.input_manifest.resolve(strict=True)
    if DEVELOPMENT_INPUT_ROOT.resolve() not in manifest.parents:
        parser.error("v6 pre-rollout pose query accepts only disclosed development manifests")
    payload = json.loads(manifest.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema_version") != "stateguard3r.corruption.v1" or payload.get("source_is_read_only") is not True:
        parser.error("v6 pre-rollout pose query requires a read-only development manifest")
    if args.detector_config.resolve(strict=False) != FORMAL_V3_CONFIG.resolve(strict=False):
        parser.error("v6 pre-rollout pose query detector config is fixed")


def _self_provenance() -> dict[str, Any]:
    try:
        commit = smoke._git_output(ROOT, "rev-parse", "HEAD")
        changes = smoke._git_output(ROOT, "status", "--porcelain", "--untracked-files=no")
    except Exception as error:
        raise RuntimeError(f"cannot inspect v6 runner provenance: {error}") from error
    if changes:
        raise RuntimeError("StateGuard3R has tracked changes; v6 runner refuses execution")
    for component in V6_COMPONENTS:
        try:
            relative = component.resolve(strict=True).relative_to(ROOT)
            relative_text = str(relative)
            smoke._git_output(ROOT, "ls-files", "--error-unmatch", "--", relative_text)
            committed_blob = smoke._git_output(ROOT, "rev-parse", f"HEAD:{relative_text}")
            working_blob = smoke._git_output(ROOT, "hash-object", "--", relative_text)
        except Exception as error:
            raise RuntimeError(f"v6 component is not tracked at HEAD: {component}") from error
        if committed_blob != working_blob:
            raise RuntimeError(f"v6 component differs from its HEAD blob: {component}")
    return {
        "repository_root": str(ROOT),
        "commit": commit,
        "tracked_worktree_clean": True,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": smoke._sha256(Path(__file__).resolve()),
        "python_executable": sys.executable,
    }


def _component_provenance() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in V6_COMPONENTS:
        resolved = path.resolve(strict=True)
        result.append({"path": str(resolved), "sha256": smoke._sha256(resolved)})
    return result


def _pinned_pose_head_interface(model: Any, dpt_head_module: Any) -> tuple[Any, Any]:
    """Fail closed unless the loaded model exposes the audited official head."""
    expected = getattr(dpt_head_module, "DPTPts3dPose", None)
    downstream = getattr(model, "downstream_head", None)
    if expected is None or type(downstream) is not expected:
        raise RuntimeError("v6 model downstream head is not pinned DPTPts3dPose")
    pose_head, pose_mode = getattr(downstream, "pose_head", None), getattr(downstream, "pose_mode", None)
    if not callable(pose_head) or not isinstance(pose_mode, tuple) or len(pose_mode) != 3:
        raise RuntimeError("v6 pinned DPT pose-head interface is unavailable")
    return pose_head, pose_mode


def _median_numpy_equivalent(torch: Any, values: Any) -> Any:
    flat = values.reshape(-1)
    ordered = torch.sort(flat).values
    length = int(ordered.numel())
    return ordered[length // 2] if length % 2 else (ordered[length // 2 - 1] + ordered[length // 2]) / 2


def _pose_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    if not bool(tensor.isfinite().all()):
        raise RuntimeError("v6 pose audit tensor is nonfinite")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class _Observation:
    frame_id: int
    record: Any
    decision: Any
    raw_pose: Any
    pre_rollout_pose_query: Any
    query_source: str


class _V6Observer:
    """Keep health raw; decode a captured pre-rollout token only after rollback."""

    def __init__(self, detector: Any, exporter: Any, *, timestamps: Sequence[Any], torch: Any, pose_encoding_to_camera: Any) -> None:
        self._detector, self._exporter = detector, exporter
        self._timestamps, self._torch, self._decode = list(timestamps), torch, pose_encoding_to_camera
        self._previous_safe_camera: Any | None = None
        self.records: list[Any] = []
        self.observed_ledger: list[Any] = []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any], pre_rollout_pose_query: Any) -> _Observation:
        from stateguard3r.health import adapt_recal3r_trace

        trace = model.get_u_calibration_trace()
        records = adapt_recal3r_trace(trace, all_frame_ids=list(range(frame_id + 1)), timestamps=self._timestamps[: frame_id + 1], batch_size=1)
        if len(records) != frame_id + 1:
            raise RuntimeError("v6 current health trace is not prefix aligned")
        overlap = frame_context.get("overlap")
        if frame_id == 0:
            if overlap is not None:
                raise RuntimeError("v6 overlap frame zero must be null")
        elif not isinstance(overlap, float) or not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
            raise RuntimeError("v6 overlap must be finite in [0, 1]")
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
        source = "initial_pose_token" if frame_id == 0 else "pose_retriever_inquire_pre_rollout"
        return _Observation(frame_id, record, self._detector.observe(record), pose, pre_rollout_pose_query, source)

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation, *, quarantined: bool, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.records.append(observation.record)
        raw_digest = _pose_sha256(prediction["camera_pose"])
        if quarantined:
            self._detector.quarantine(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="quarantine_current_rollback"))
            exported, action = self._exporter.export_alarm(observation.frame_id, prediction, observation.pre_rollout_pose_query, query_source=observation.query_source)
        else:
            self._detector.commit(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="commit"))
            self._previous_safe_camera = self._decode(observation.raw_pose.detach().clone()).detach().clone()
            exported, action = self._exporter.commit_real(observation.frame_id, prediction)
        return exported, {
            "export_action": action.action,
            "pre_rollout_pose_query": action.evidence,
            "raw_candidate_pose_sha256": raw_digest,
            "exported_camera_pose_sha256": _pose_sha256(exported["camera_pose"]),
        }

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
    runner_provenance = _self_provenance()
    component_provenance = _component_provenance()
    requested = args.output_dir.resolve(strict=False)
    if requested.exists() or requested.parent != ROOT / "outputs":
        raise RuntimeError("v6 pre-rollout pose-query output must be a new direct outputs child")
    config_payload, config_artifact = v2._frozen_json(args.detector_config, label="frozen Detector-v3 config")
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from stateguard3r.prerollout_pose_query_export_v6 import PreRolloutPoseQueryExport
    from stateguard3r.prerollout_pose_query_v6 import audit_pinned_pose_query_sources
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.online_detector_incremental_v3 import IncrementalOnlinePrefixDetector
    from stateguard3r.online_detector_v2 import FrozenDetectorV3Config
    from stateguard3r.recal3r_prerollout_pose_query_runner_v6 import run_prerollout_pose_query_recurrent_lighter
    from stateguard3r.timestamp_order_v3 import capture_timestamp_records, timestamp_order_sidecar, validate_timestamp_order_sidecar
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    for source_root in (args.baseline_root, args.baseline_src):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    smoke._verify_preflight_inputs(args)
    pose_query_source_audit = audit_pinned_pose_query_sources(args.baseline_src / "dust3r" / "model.py", args.baseline_src / "dust3r" / "heads" / "dpt_head.py", args.baseline_src / "dust3r" / "heads" / "postprocess.py")
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
        import dust3r.heads.dpt_head as dpt_head
        import dust3r.utils.camera as camera
        import dust3r.utils.device as device_module
        import dust3r.heads.postprocess as pose_postprocess
        import models.curope.curope2d as curope
        module_paths.update({"dust3r.model": smoke._assert_module_source(dust3r_model, args.baseline_src / "dust3r" / "model.py", "dust3r.model"), "dust3r.heads.dpt_head": smoke._assert_module_source(dpt_head, args.baseline_src / "dust3r" / "heads" / "dpt_head.py", "dust3r.heads.dpt_head"), "dust3r.utils.camera": smoke._assert_module_source(camera, args.baseline_src / "dust3r" / "utils" / "camera.py", "dust3r.utils.camera")})
        model, checkpoint_audit = smoke._load_model_with_state_dict_audit(dust3r_model.ARCroco3DStereo, args.checkpoint, torch)
        smoke._write_json_atomic(staging / "checkpoint-load-audit.json", checkpoint_audit)
        if checkpoint_audit["strict"] is not False or checkpoint_audit["missing_keys"] or checkpoint_audit["unexpected_keys"]:
            raise RuntimeError("checkpoint audit failed")
        interface = smoke._validate_model_interface(model, args.checkpoint.name)
        model = model.to(args.device)
        pose_head, pose_mode = _pinned_pose_head_interface(model, dpt_head)
        config = smoke._configure_official_recal3r_runtime(model, beta_base=args.beta_base)
        model.eval()
        model.enable_u_calibration_trace(oracle_window=1)
        updates, original_update = 0, model._compute_recal3r_update_mask

        def counted(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal updates
            updates += 1
            return original_update(*call_args, **call_kwargs)

        model._compute_recal3r_update_mask = counted
        overlaps = v2._OnlineOverlap()
        observer = None
        if args.state_policy != "always-commit":
            observer = _V6Observer(
                IncrementalOnlinePrefixDetector(FrozenDetectorV3Config.from_mapping(config_payload), captures=captures, capture_provenance=capture_provenance),
                PreRolloutPoseQueryExport(
                    torch=torch,
                    pose_head=pose_head,
                    postprocess_pose=pose_postprocess.postprocess_pose,
                    pose_mode=pose_mode,
                    pose_encoding_to_camera=camera.pose_encoding_to_camera,
                ),
                timestamps=smoke._input_timestamps(args),
                torch=torch,
                pose_encoding_to_camera=camera.pose_encoding_to_camera,
            )
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
        with torch.no_grad():
            result = run_prerollout_pose_query_recurrent_lighter(
                views, model, device, torch=torch, to_gpu=device_module.to_gpu, to_cpu=device_module.to_cpu,
                canonicalize_model_update_type=dust3r_model.canonicalize_model_update_type,
                before_frame=overlaps.before_frame, observer=observer, watchdog_limit=args.watchdog, synchronize=synchronize,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_elapsed = time.perf_counter() - started
        if updates != len(views) - 1 or len(result.predictions) != len(views) or len(overlaps.values) != len(views):
            raise RuntimeError("v6 recurrent accounting differs")
        summary = smoke._prediction_summary(result.predictions, torch)
        if observer is None:
            jumps, residuals, trajectory = smoke._output_health_signals(result.predictions, pose_encoding_to_camera=camera.pose_encoding_to_camera, np=np)
            trace = model.get_u_calibration_trace()
            health = smoke._apply_v2_overlap_to_health([replace(record, pose_jump=jumps[index], geometric_residual=residuals[index]) for index, record in enumerate(adapt_recal3r_trace(trace, all_frame_ids=list(range(len(views))), timestamps=smoke._input_timestamps(args), batch_size=1))], overlaps.values)
            ledger = None
            legacy_reference = smoke._input_frame_metadata(args)[0]
            trajectory_payload = {"schema_version": "stateguard3r.trajectory.v0", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "reference_frame": {"frame_id": 0, "source_index": legacy_reference["source_index"], "path": legacy_reference["path"], "sha256": legacy_reference["sha256"]}, "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)", "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)", "frames": trajectory}
        else:
            health, ledger = observer.records, observer.observed_ledger
            trajectory = _exported_trajectory(result.predictions, health, decode=camera.pose_encoding_to_camera)
            if len(health) != len(views) or len(ledger) != len(views):
                raise RuntimeError("v6 observed health ledger is incomplete")
            trajectory_payload = {"schema_version": "stateguard3r.trajectory.pre-rollout-pose-query-v6", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "camera_pose_semantics": "clear frames export real raw pose; alarm frames export only captured pre-rollout pose-query decode; detector health remains raw", "query_evidence_location": "state-timeline.json transactions", "frames": trajectory}
        smoke._verify_preflight_inputs(args)
        write_health_jsonl_atomic(staging / "health.jsonl", health)
        if ledger is not None:
            write_health_jsonl_atomic(staging / "observed-health-ledger.jsonl", ledger)
        smoke._write_json_atomic(staging / "predictions-summary.json", summary)
        safe_frames = [{"frame_id": frame.frame_index, "path": str(frame.path), "sha256": args.image_sha256[frame.path]} for frame in args.input_manifest_data.frames]
        smoke._write_json_atomic(staging / "trajectory.json", trajectory_payload)
        smoke._write_json_atomic(staging / "timestamp-order-sidecar.json", sidecar)
        state_timeline = {
            "schema_version": SCHEMA_VERSION,
            "policy": args.state_policy,
            "watchdog": args.watchdog,
            "causality": "detector consumes current/past raw candidate health, adjacent model-ready RGB overlap, and current/past capture timestamps; alarm export consumes only a captured current pre-rollout pose-query token through frozen pose head/postprocess; raw camera_pose is never a numerical v6 export input; no export feeds model, detector, state, or later frame",
            "pose_query_source_audit": pose_query_source_audit,
            "pending_transaction_count": 0,
            "source_provenance": result.source_provenance,
            "transactions": result.timeline,
        }
        smoke._write_json_atomic(staging / "state-timeline.json", state_timeline)
        finished = datetime.now().astimezone().isoformat()
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "succeeded",
            "argv": sys.argv if argv is None else [str(item) for item in argv],
            "baseline_root": str(args.baseline_root),
            "baseline_commit": args.baseline_commit,
            "baseline_tracked_worktree_clean": True,
            "module_paths": module_paths,
            "runner": runner_provenance,
            "v6_components": component_provenance,
            "pose_query_source_audit": pose_query_source_audit,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "input_manifest": smoke._input_manifest_metadata(args),
            "safe_input_frames": safe_frames,
            "frame_count": len(views),
            "size": args.size,
            "seed": args.seed,
            "model_update_type": "recal3r",
            "loaded_model_interface": interface,
            "checkpoint_state_dict": checkpoint_audit,
            "beta_base": args.beta_base,
            "recal3r_runtime_config": config,
            "model_eval": not model.training,
            "calibrated_update_calls": updates,
            "health_profile": "v6-gpu-current-raw-candidate",
            "online_visual_correspondence": overlaps.metadata(),
            "online_detector": {"config": config_artifact, "allowed_inputs": ["candidate_model_health", "previous_current_model_ready_rgb_overlap", "raw_rgb_capture_timestamp"], "forbidden": ["GT", "depth", "label", "event", "source_index", "future_frame", "baseline_alarm_artifact"]},
            "state_policy": {"name": args.state_policy, "watchdog": args.watchdog, "timeline_path": str(requested / "state-timeline.json"), "timeline_sha256": smoke._sha256(staging / "state-timeline.json")},
            "runtime_seconds": result.recurrent_policy_runtime_seconds,
            "wall_runtime_seconds": wall_elapsed,
            "runtime_scope": "per_frame_cuda_synchronized_v6_recurrent_policy_including_pre_rollout_pose_query_export_excluding_common_causal_rgb_overlap",
            "started_at": started_at,
            "finished_at": finished,
            "pid": os.getpid(),
            "output_dir": str(requested),
            "peak_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
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
