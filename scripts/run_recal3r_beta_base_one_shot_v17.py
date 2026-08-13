#!/usr/bin/env python3
"""One-use v17 ReCal3R dynamic beta-base-floor runtime.

The always-commit branch invokes the pinned native lightweight loop directly;
therefore its protected outputs use the formal serializer without an observer
or candidate operation.  The candidate uses the separately audited v17 loop
and only arms a later scalar beta-base scope after current raw health commits.
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
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from stateguard3r.dynamic_rgb_capability_v17 import (
    DYNAMIC_CAPSULE_ID,
    dynamic_rgb_capability_provenance_v17,
    load_dynamic_rgb_capability_v17,
    prepare_dynamic_rgb_views_v17,
    sha256_file_v17,
    verify_dynamic_rgb_capability_v17,
)


SCHEMA = "stateguard3r.recal3r-beta-base-one-shot-v17.v1"
RECAL3R_ROOT = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT = RECAL3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
CAPSULE = ROOT / "outputs" / f"{DYNAMIC_CAPSULE_ID}.json"
DETECTOR_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
CONTROL_POLICY = "always-commit"
CANDIDATE_POLICY = "detector-v3-one-shot-beta-base-floor"
CONTROL_RUN_ID = "recovery-beta-floor-v17-dynamic-always-commit-0001"
CANDIDATE_RUN_ID = "recovery-beta-floor-v17-dynamic-candidate-0001"
FIXED_OUTPUTS = {
    CONTROL_POLICY: ROOT / "outputs" / CONTROL_RUN_ID,
    CANDIDATE_POLICY: ROOT / "outputs" / CANDIDATE_RUN_ID,
}


class RunBetaBaseOneShotV17Error(RuntimeError):
    """The preregistered v17 production boundary has been violated."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--size", type=int, choices=(512,), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-base", type=float, default=0.1)
    parser.add_argument("--state-policy", choices=(CONTROL_POLICY, CANDIDATE_POLICY), required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    return parser


def _regular_0444(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise RunBetaBaseOneShotV17Error(f"cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise RunBetaBaseOneShotV17Error(f"{label} must be a frozen non-symlink mode-0444 regular file")


def _frozen_config(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _regular_0444(path, label="frozen detector config")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunBetaBaseOneShotV17Error("cannot parse frozen detector config") from error
    if not isinstance(value, Mapping):
        raise RunBetaBaseOneShotV17Error("frozen detector config is not an object")
    return value, {"path": str(path.resolve(strict=True)), "sha256": sha256_file_v17(path), "size_bytes": path.stat().st_size, "mode_octal": "0444"}


def _checkpoint_audit(model_class: Any, *, torch: Any) -> tuple[Any, Mapping[str, Any]]:
    original, calls = torch.nn.Module.load_state_dict, []

    def audited(module: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(module, state, *args, **kwargs)
        strict = kwargs.get("strict", args[0] if args else True)
        calls.append({"module_class": type(module).__name__, "strict": bool(strict), "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)})
        return result

    torch.nn.Module.load_state_dict = audited
    try:
        model = model_class.from_pretrained(str(CHECKPOINT))
    finally:
        torch.nn.Module.load_state_dict = original
    expected = {"module_class": "ARCroco3DStereo", "strict": False, "missing_keys": [], "unexpected_keys": []}
    if calls != [expected]:
        raise RunBetaBaseOneShotV17Error("checkpoint compatibility audit differs")
    return model, expected


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _freeze_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RunBetaBaseOneShotV17Error("v17 output contains an unsafe child")
        child.chmod(0o444)
    path.chmod(0o555)


def _median(torch: Any, value: Any) -> Any:
    sorted_values = torch.sort(value.reshape(-1)).values
    return sorted_values[len(sorted_values) // 2] if len(sorted_values) % 2 else (sorted_values[len(sorted_values) // 2 - 1] + sorted_values[len(sorted_values) // 2]) / 2


def _tensor_summary(value: Any, *, torch: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    data = value.detach()
    finite = torch.isfinite(data) if data.is_floating_point() else None
    return {"shape": list(data.shape), "dtype": str(data.dtype), "finite": bool(finite.all().item()) if finite is not None else True, "min": float(data[finite].min().item()) if finite is not None and bool(finite.any().item()) else None, "max": float(data[finite].max().item()) if finite is not None and bool(finite.any().item()) else None}


def _prediction_summary(predictions: Sequence[Mapping[str, Any]], *, torch: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for frame_id, prediction in enumerate(predictions):
        missing = sorted({"camera_pose", "pts3d_in_self_view", "pts3d_in_other_view"} - set(prediction))
        if missing:
            raise RunBetaBaseOneShotV17Error(
                "prediction is missing required output(s): " + ", ".join(missing)
            )
        fields = {name: summary for name, value in sorted(prediction.items()) if (summary := _tensor_summary(value, torch=torch)) is not None}
        if not all(bool(item["finite"]) for item in fields.values()):
            raise RunBetaBaseOneShotV17Error("prediction summary differs")
        result.append({"frame_id": frame_id, "tensor_fields": fields})
    return result


def _numpy_value_v17(value: Any, *, label: str, np: Any) -> Any:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        array = np.asarray(current, dtype=np.float64)
    except Exception as error:
        raise RunBetaBaseOneShotV17Error(f"{label} cannot be converted to NumPy") from error
    if not bool(np.isfinite(array).all()):
        raise RunBetaBaseOneShotV17Error(f"{label} contains NaN/Inf")
    return array


def _output_health_signals_v17(
    predictions: Sequence[Mapping[str, Any]], *, decode: Any, np: Any
) -> tuple[list[float | None], list[float | None], list[Mapping[str, Any]]]:
    """Independently reproduce formal-v3's output-level scalar contract."""

    pose_jumps: list[float | None] = []
    geometric_residuals: list[float | None] = []
    frames: list[Mapping[str, Any]] = []
    prior = None
    for frame_id, prediction in enumerate(predictions):
        encoded = _numpy_value_v17(prediction["camera_pose"], label=f"prediction {frame_id} camera_pose", np=np)
        if encoded.shape != (1, 7):
            raise RunBetaBaseOneShotV17Error("camera pose interface differs")
        camera = _numpy_value_v17(decode(prediction["camera_pose"].clone()), label=f"prediction {frame_id} camera matrix", np=np)
        if camera.shape != (1, 4, 4) or not bool(np.allclose(camera[0, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)):
            raise RunBetaBaseOneShotV17Error("camera matrix interface differs")
        pose = camera[0]
        translation_jump = rotation_jump = pose_jump = None
        if prior is not None:
            relative = prior[:3, :3].T @ pose[:3, :3]
            rotation_jump = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            translation_jump = float(np.linalg.norm(prior[:3, :3].T @ (pose[:3, 3] - prior[:3, 3])))
            pose_jump = float(np.hypot(translation_jump, rotation_jump))
            if not math.isfinite(pose_jump):
                raise RunBetaBaseOneShotV17Error("pose jump is non-finite")
        points_self = _numpy_value_v17(prediction["pts3d_in_self_view"], label=f"prediction {frame_id} pts3d_in_self_view", np=np)
        points_other = _numpy_value_v17(prediction["pts3d_in_other_view"], label=f"prediction {frame_id} pts3d_in_other_view", np=np)
        if points_self.shape != points_other.shape or points_self.ndim != 4 or points_self.shape[0] != 1 or points_self.shape[-1] != 3:
            raise RunBetaBaseOneShotV17Error("pointmap interface differs")
        errors = np.linalg.norm(points_self @ pose[:3, :3].T + pose[:3, 3] - points_other, axis=-1)
        scales = np.linalg.norm(points_other, axis=-1)
        residual = float(np.median(errors) / max(float(np.median(scales)), 1e-8))
        if not math.isfinite(residual):
            raise RunBetaBaseOneShotV17Error("geometric residual is non-finite")
        pose_jumps.append(pose_jump)
        geometric_residuals.append(residual)
        frames.append({"frame_id": frame_id, "camera_pose_encoding_absT_quaR": encoded[0].tolist(), "camera_to_reference": pose.tolist(), "translation_jump": translation_jump, "rotation_jump_rad": rotation_jump, "pose_jump": pose_jump, "geometric_residual": residual})
        prior = pose
    return pose_jumps, geometric_residuals, frames


def _trajectory(predictions: Sequence[Mapping[str, Any]], *, capability: Any, decode: Any, np: Any) -> Mapping[str, Any]:
    _pose_jumps, _residuals, frames = _output_health_signals_v17(predictions, decode=decode, np=np)
    first = capability.frames[0]
    return {"schema_version": "stateguard3r.trajectory.v0", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "reference_frame": {"frame_id": 0, "source_index": 0, "path": str(capability.rgb_paths[0]), "sha256": first.sha256}, "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)", "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)", "frames": frames}


def _health_from_native_trace(model: Any, *, captures: Sequence[Mapping[str, Any]], overlaps: Sequence[float | None]) -> list[Any]:
    from stateguard3r.health import adapt_recal3r_trace

    records = adapt_recal3r_trace(model.get_u_calibration_trace(), all_frame_ids=list(range(30)), timestamps=[float(value["rgb_capture_timestamp"]) for value in captures], batch_size=1)
    if len(records) != 30 or len(overlaps) != len(records):
        raise RunBetaBaseOneShotV17Error("native calibration trace does not align with the fixed stream")
    measured = [record.frame_id for record in records if record.uncertainty_u is not None]
    if measured != list(range(1, 30)) or any(records[index].global_state_delta is None for index in range(1, 30)):
        raise RunBetaBaseOneShotV17Error("native calibration trace lacks every expected update")
    return [replace(record, overlap=overlaps[index]) for index, record in enumerate(records)]


class _CausalOverlapV17:
    def __init__(self) -> None:
        from stateguard3r.online_visual_overlap_v17 import OnlineVisualOverlapConfigV17, normalized_tensor_to_uint8_rgb_v17, online_visual_correspondence_coverage_v17, online_visual_overlap_provenance_v17

        self._convert, self._measure = normalized_tensor_to_uint8_rgb_v17, online_visual_correspondence_coverage_v17
        self._config = OnlineVisualOverlapConfigV17()
        self._previous, self.values, self._diagnostics, self._digests = None, [], [], []
        self._runtime = online_visual_overlap_provenance_v17()

    def before_frame(self, frame_id: int, view: Mapping[str, Any]) -> Mapping[str, Any]:
        if frame_id != len(self.values):
            raise RunBetaBaseOneShotV17Error("v17 RGB causal order differs")
        image = view["img"]
        current = self._convert(image)
        digest = hashlib.sha256(current.tobytes()).hexdigest()
        if frame_id == 0:
            value = None
        else:
            before = hashlib.sha256(self._convert(self._previous).tobytes()).hexdigest()
            measured = self._measure(self._previous, image, config=self._config, reference_frame_id=frame_id - 1, frame_id=frame_id)
            if before != hashlib.sha256(self._convert(self._previous).tobytes()).hexdigest() or digest != hashlib.sha256(self._convert(image).tobytes()).hexdigest():
                raise RunBetaBaseOneShotV17Error("v17 RGB overlap changed a model-ready input")
            value = float(measured.score)
            self._diagnostics.append(measured.to_dict())
        self.values.append(value)
        self._digests.append(digest)
        self._previous = image
        return {"overlap": value}

    def metadata(self) -> Mapping[str, Any]:
        return {"schema_version": "stateguard3r.online-visual-overlap-v17.v1", "input": "official_loader_then_v17_rectangle_clone", "causal_reference": "immediately_previous_frame_only", "computed": "before_current_native_rollout", "model_ready_uint8_rgb_sha256": self._digests, "frame_results": self._diagnostics, "opencv": self._runtime}


@dataclass(frozen=True)
class _Observation:
    health: Any
    decision: Any
    raw_camera: Any


class _RawObserverV17:
    def __init__(self, detector: Any, *, captures: Sequence[Mapping[str, Any]], torch: Any, decode: Any) -> None:
        self._detector, self._captures, self._torch, self._decode = detector, list(captures), torch, decode
        self._previous_camera, self.records = None, []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any] | None) -> _Observation:
        from stateguard3r.health import adapt_recal3r_trace

        if frame_context is None or frame_id >= len(self._captures):
            raise RunBetaBaseOneShotV17Error("v17 current health lacks causal RGB context")
        native = adapt_recal3r_trace(model.get_u_calibration_trace(), all_frame_ids=list(range(frame_id + 1)), timestamps=[float(item["rgb_capture_timestamp"]) for item in self._captures[:frame_id + 1]], batch_size=1)[-1]
        raw_camera = self._decode(prediction["camera_pose"])
        pose_jump = None
        if self._previous_camera is not None:
            relative = self._previous_camera[:, :3, :3].transpose(-1, -2) @ raw_camera[:, :3, :3]
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            delta = self._previous_camera[:, :3, :3].transpose(-1, -2) @ (raw_camera[:, :3, 3] - self._previous_camera[:, :3, 3]).unsqueeze(-1)
            pose_jump = float(self._torch.hypot(self._torch.linalg.vector_norm(delta.squeeze(-1), dim=-1), self._torch.acos(cosine)).item())
        transformed = prediction["pts3d_in_self_view"] @ raw_camera[:, :3, :3].transpose(-1, -2) + raw_camera[:, None, None, :3, 3]
        residual = float((_median(self._torch, self._torch.linalg.vector_norm(transformed - prediction["pts3d_in_other_view"], dim=-1)) / self._torch.clamp(_median(self._torch, self._torch.linalg.vector_norm(prediction["pts3d_in_other_view"], dim=-1)), min=1e-8)).item())
        health = replace(native, overlap=frame_context["overlap"], pose_jump=pose_jump, geometric_residual=residual)
        scalar = {"frame_id": frame_id, "overlap": health.overlap, "pose_jump": health.pose_jump, "geometric_residual": health.geometric_residual, "update_magnitude": health.global_state_delta, "reliability": health.reliability, "uncertainty_u": health.uncertainty_u, "timestamp_order_alarm": bool(frame_context["timestamp_order_alarm"])}
        return _Observation(health, self._detector.observe(scalar), raw_camera.detach().clone())

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation) -> Mapping[str, Any]:
        scalar = {"frame_id": observation.decision.frame_id, "overlap": observation.health.overlap, "pose_jump": observation.health.pose_jump, "geometric_residual": observation.health.geometric_residual, "update_magnitude": observation.health.global_state_delta, "reliability": observation.health.reliability, "uncertainty_u": observation.health.uncertainty_u, "timestamp_order_alarm": observation.decision.timestamp_order_alarm}
        self._detector.commit(scalar)
        self.records.append(observation.health)
        self._previous_camera = observation.raw_camera.detach().clone()
        return {"detector_history_action": "commit_raw_current_health"}

    def timeline_evidence(self, observation: _Observation) -> Mapping[str, Any]:
        item = observation.decision
        return {"online_detector": {"frame_id": item.frame_id, "continuous_score": item.continuous_score, "hybrid_score": item.hybrid_score, "continuous_alarm": item.continuous_alarm, "timestamp_order_alarm": item.timestamp_order_alarm, "hybrid_alarm": item.hybrid_alarm, "finite_component_count": item.finite_component_count, "policy_input_sha256": item.policy_input_sha256}}

def _fixed_command(args: argparse.Namespace) -> None:
    if (
        args.capsule != CAPSULE
        or args.output_dir != FIXED_OUTPUTS[args.state_policy]
        or args.output_dir.exists()
        or args.detector_config != DETECTOR_CONFIG
        or args.seed != 0
        or args.beta_base != 0.1
        or not math.isfinite(args.beta_base)
        or args.size != 512
        or args.device != "cuda"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
    ):
        raise RunBetaBaseOneShotV17Error("v17 production command differs from its fixed release")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _fixed_command(args)
    capability = load_dynamic_rgb_capability_v17(args.capsule)
    verify_dynamic_rgb_capability_v17(capability)
    config_payload, config_artifact = _frozen_config(args.detector_config)
    import numpy as np
    import torch
    from stateguard3r.health import write_health_jsonl_atomic
    from stateguard3r.timestamp_order_v17 import capture_timestamp_records_v17, timestamp_order_sidecar_v17
    from stateguard3r.v17_source_import_audit import audit_v17_native_runner_contract, audit_v17_production_script

    if not torch.cuda.is_available():
        raise RunBetaBaseOneShotV17Error("v17 CUDA is unavailable")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    sys.path[:0] = [str(RECAL3R_ROOT), str(RECAL3R_ROOT / "src")]
    import add_ckpt_path
    add_ckpt_path.add_path_to_dust3r(str(CHECKPOINT))
    import dust3r.inference as inference
    import dust3r.model as dust3r_model
    import dust3r.utils.camera as camera
    import dust3r.utils.device as device_module
    import models.curope.curope2d as curope  # noqa: F401
    if sha256_file_v17(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RunBetaBaseOneShotV17Error("v17 checkpoint hash differs")
    views = prepare_dynamic_rgb_views_v17(capability, torch=torch)
    captures, capture_provenance = capture_timestamp_records_v17(capability.rgb_paths, rgb_txt=capability.listing, dataset_root=capability.dataset_root)
    sidecar = timestamp_order_sidecar_v17(captures, provenance=capture_provenance)
    source_paths = tuple(ROOT / relative for relative in (
        "src/stateguard3r/beta_base_floor_v17.py", "src/stateguard3r/recal3r_beta_base_floor_runner_v17.py", "src/stateguard3r/online_detector_v17.py", "src/stateguard3r/dynamic_rgb_capability_v17.py", "src/stateguard3r/online_visual_overlap_v17.py", "src/stateguard3r/timestamp_order_v17.py", "src/stateguard3r/v17_source_import_audit.py"))
    source_audit = audit_v17_production_script(Path(__file__), source_paths)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", suffix=".staging", dir=args.output_dir.parent))
    try:
        model, checkpoint_audit = _checkpoint_audit(dust3r_model.ARCroco3DStereo, torch=torch)
        _write_json(staging / "checkpoint-load-audit.json", checkpoint_audit)
        model = model.to("cuda")
        for target in (model, model.config):
            target.model_update_type, target.beta_base, target.entropy_eps, target.entropy_head_reduce, target.uncertainty_clamp_max, target.decay = "recal3r", 0.1, 2e-14, "mean", 1.0, 0.95
        model.eval(); model.enable_u_calibration_trace(oracle_window=1)
        device = torch.device("cuda")
        torch.cuda.synchronize(device)
        if args.state_policy == CONTROL_POLICY:
            overlaps = _CausalOverlapV17()
            for frame_id, view in enumerate(views):
                overlaps.before_frame(frame_id, view)
        started = time.perf_counter()
        if args.state_policy == CONTROL_POLICY:
            output, _state_args = inference.inference_recurrent_lighter(views, model, device, verbose=True)
            predictions = output["pred"]
            timeline = [{"frame_id": index, "action": "native_commit", "current_alarm": False, "arm_pending": False, "beta_base_override": None, "detector_constructed": False, "operator_constructed": False} for index in range(30)]
            pose_jumps, residuals, _frames = _output_health_signals_v17(predictions, decode=camera.pose_encoding_to_camera, np=np)
            health = [
                replace(record, pose_jump=pose_jumps[index], geometric_residual=residuals[index])
                for index, record in enumerate(_health_from_native_trace(model, captures=captures, overlaps=overlaps.values))
            ]
            recurrent_runtime = None
            candidate_audit = None
        else:
            from stateguard3r.online_detector_v17 import FrozenDetectorV17Config, IncrementalOnlinePrefixDetectorV17
            from stateguard3r.recal3r_beta_base_floor_runner_v17 import run_recurrent_beta_floor_native_v17

            overlaps = _CausalOverlapV17()
            predicates = [record["timestamp_order_violation"] for record in sidecar["records"]]
            class _ContextOverlap:
                def __call__(self, frame_id: int, view: Mapping[str, Any]) -> Mapping[str, Any]:
                    return dict(overlaps.before_frame(frame_id, view)) | {"timestamp_order_alarm": bool(predicates[frame_id])}
            observer = _RawObserverV17(IncrementalOnlinePrefixDetectorV17(FrozenDetectorV17Config.from_mapping(config_payload)), captures=captures, torch=torch, decode=camera.pose_encoding_to_camera)
            result = run_recurrent_beta_floor_native_v17(views, model, device, torch=torch, to_gpu=device_module.to_gpu, to_cpu=device_module.to_cpu, canonicalize_model_update_type=dust3r_model.canonicalize_model_update_type, before_frame=_ContextOverlap(), observer=observer, synchronize=lambda: torch.cuda.synchronize(device))
            predictions, timeline, health, recurrent_runtime = result.predictions, result.timeline, observer.records, result.recurrent_policy_runtime_seconds
            candidate_audit = audit_v17_native_runner_contract(ROOT / "src" / "stateguard3r" / "recal3r_beta_base_floor_runner_v17.py")
        torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
        if len(predictions) != 30 or len(health) != 30 or len(timeline) != 30:
            raise RunBetaBaseOneShotV17Error("v17 runtime output count differs")
        write_health_jsonl_atomic(staging / "health.jsonl", health)
        _write_json(staging / "predictions-summary.json", _prediction_summary(predictions, torch=torch))
        _write_json(staging / "trajectory.json", _trajectory(predictions, capability=capability, decode=camera.pose_encoding_to_camera, np=np))
        _write_json(staging / "timestamp-order-sidecar.json", sidecar)
        _write_json(staging / "state-timeline.json", {"schema_version": SCHEMA, "policy": args.state_policy, "frames": timeline})
        _write_json(staging / "run.json", {"schema_version": SCHEMA, "status": "succeeded", "state_policy": {"name": args.state_policy}, "runtime_seconds": runtime, "recurrent_policy_runtime_seconds": recurrent_runtime, "capability": dynamic_rgb_capability_provenance_v17(capability), "detector_config": config_artifact, "source_import_audit": source_audit, "runner_contract": candidate_audit, "checkpoint_load_audit": checkpoint_audit, "online_visual_correspondence": None if overlaps is None else overlaps.metadata(), "timestamp_capture_provenance": capture_provenance, "started_at": datetime.now().astimezone().isoformat(), "finished_at": datetime.now().astimezone().isoformat()})
        _freeze_tree(staging)
        os.rename(staging, args.output_dir)
    except Exception:
        try:
            _write_json(staging / "failure.json", {"schema_version": SCHEMA, "status": "failed"})
            _freeze_tree(staging)
        except Exception:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
