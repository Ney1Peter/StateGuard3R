#!/usr/bin/env python3
"""Run v16's fixed pressure-only policy from its dynamic RGB capability."""

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
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from stateguard3r.dynamic_rgb_capability_v16 import DYNAMIC_CAPSULE_ID, dynamic_rgb_capability_provenance_v16, load_dynamic_rgb_capability_v16, prepare_dynamic_rgb_views_v16, sha256_file_v16, verify_dynamic_rgb_capability_v16


SCHEMA = "stateguard3r.recal3r-bounded-update-pressure-v16.v1"
RECAL3R_ROOT = ROOT.parent / "baselines" / "ReCal3R"
RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
CHECKPOINT = RECAL3R_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
CAPSULE = ROOT / "outputs" / f"{DYNAMIC_CAPSULE_ID}.json"
DETECTOR_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"


class RunBoundedPressureV16Error(RuntimeError):
    """The independently preregistered v16 production release was violated."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-policy", choices=("always-commit", "detector-v3-bounded-update-pressure"), required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta-base", type=float, default=0.1)
    parser.add_argument("--detector-config", type=Path, default=DETECTOR_CONFIG)
    return parser


def _regular_0444(path: Path, *, label: str) -> None:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444:
        raise RunBoundedPressureV16Error(f"{label} must be a frozen regular mode-0444 file")


def _frozen_config(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _regular_0444(path, label="frozen Detector-v3 config")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunBoundedPressureV16Error("cannot parse frozen Detector-v3 config") from error
    if not isinstance(value, Mapping):
        raise RunBoundedPressureV16Error("frozen Detector-v3 config is not an object")
    return value, {"path": str(path.resolve(strict=True)), "sha256": sha256_file_v16(path), "size_bytes": path.stat().st_size, "mode_octal": "0444"}


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
    if len(calls) != 1 or calls[0] != expected:
        raise RunBoundedPressureV16Error("checkpoint compatibility audit differs from protected control")
    return model, calls[0]


class _CausalOverlap:
    """Online RGB-only previous/current overlap, before the current rollout."""

    def __init__(self) -> None:
        from stateguard3r.online_visual_overlap_v16 import OnlineVisualOverlapConfigV16, online_visual_overlap_provenance_v16
        self._config = OnlineVisualOverlapConfigV16(nfeatures=2000, grid_columns=8, grid_rows=6, minimum_matches=12, ratio_threshold=0.80, ransac_pixel_limit=1.0, rng_seed=0)
        self._previous, self.values, self._diagnostics, self._hashes = None, [], [], []
        self._opencv = online_visual_overlap_provenance_v16()

    def before_frame(self, frame_id: int, view: Mapping[str, Any]) -> Mapping[str, Any]:
        from stateguard3r.online_visual_overlap_v16 import normalized_tensor_to_uint8_rgb_v16, online_visual_correspondence_coverage_v16
        if frame_id != len(self.values):
            raise RunBoundedPressureV16Error("v16 causal RGB overlap frame ordering differs")
        image = view["img"]
        current_digest = hashlib.sha256(normalized_tensor_to_uint8_rgb_v16(image).tobytes()).hexdigest()
        if frame_id == 0:
            overlap = None
        else:
            before_previous = hashlib.sha256(normalized_tensor_to_uint8_rgb_v16(self._previous).tobytes()).hexdigest()
            result = online_visual_correspondence_coverage_v16(self._previous, image, config=self._config, reference_frame_id=frame_id - 1, frame_id=frame_id)
            if before_previous != hashlib.sha256(normalized_tensor_to_uint8_rgb_v16(self._previous).tobytes()).hexdigest() or current_digest != hashlib.sha256(normalized_tensor_to_uint8_rgb_v16(image).tobytes()).hexdigest():
                raise RunBoundedPressureV16Error("v16 online RGB overlap mutated a model-ready tensor")
            overlap = float(result.score)
            self._diagnostics.append(result.to_dict())
        self.values.append(overlap)
        self._hashes.append(current_digest)
        self._previous = image
        return {"overlap": overlap}

    def metadata(self) -> Mapping[str, Any]:
        return {"schema_version": "stateguard3r.online-visual-overlap-v16.v1", "input": "post_official_loader_post_capability_rectangle_normalized_rgb", "causal_reference": "immediately_previous_frame_only", "computed": "inside_recurrent_loop_before_current_forward", "model_ready_uint8_rgb_sha256": self._hashes, "frame_results": self._diagnostics, "opencv": self._opencv}


def _median(torch: Any, value: Any) -> Any:
    ordered = torch.sort(value.reshape(-1)).values
    return ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2


@dataclass(frozen=True)
class _Observation:
    record: Any
    decision: Any
    raw_camera: Any
    pose_fingerprint: str


class _RawObserver:
    """Commit the current raw-health record before the bounded pressure write."""

    def __init__(self, detector: Any, *, captures: Sequence[Mapping[str, Any]], torch: Any, decode: Any) -> None:
        self._detector, self._captures, self._torch, self._decode = detector, list(captures), torch, decode
        self._previous_camera, self.records, self.ledger = None, [], []

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any] | None) -> _Observation:
        from stateguard3r.bounded_update_pressure_v16 import gpu_fingerprint_v16
        if frame_context is None or frame_id >= len(self._captures):
            raise RunBoundedPressureV16Error("v16 detector health lacks a causal RGB prefix")
        from stateguard3r.health import adapt_recal3r_trace
        records = adapt_recal3r_trace(model.get_u_calibration_trace(), all_frame_ids=list(range(frame_id + 1)), timestamps=[float(item["rgb_capture_timestamp"]) for item in self._captures[: frame_id + 1]], batch_size=1)
        if len(records) != frame_id + 1:
            raise RunBoundedPressureV16Error("v16 native calibration trace lacks a causal current record")
        raw_camera = self._decode(prediction["camera_pose"])
        pose_jump = None
        if self._previous_camera is not None:
            relative = self._previous_camera[:, :3, :3].transpose(-1, -2) @ raw_camera[:, :3, :3]
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
            delta = self._previous_camera[:, :3, :3].transpose(-1, -2) @ (raw_camera[:, :3, 3] - self._previous_camera[:, :3, 3]).unsqueeze(-1)
            pose_jump = float(self._torch.hypot(self._torch.linalg.vector_norm(delta.squeeze(-1), dim=-1), self._torch.acos(cosine)).item())
        transformed = prediction["pts3d_in_self_view"] @ raw_camera[:, :3, :3].transpose(-1, -2) + raw_camera[:, None, None, :3, 3]
        errors = self._torch.linalg.vector_norm(transformed - prediction["pts3d_in_other_view"], dim=-1)
        scales = self._torch.linalg.vector_norm(prediction["pts3d_in_other_view"], dim=-1)
        residual = float((_median(self._torch, errors) / self._torch.clamp(_median(self._torch, scales), min=1e-8)).item())
        record = replace(records[-1], overlap=frame_context.get("overlap"), pose_jump=pose_jump, geometric_residual=residual)
        return _Observation(record, self._detector.observe(record), raw_camera.detach().clone(), gpu_fingerprint_v16(prediction["camera_pose"], torch=self._torch, label="raw pose for detector"))

    def alarm(self, observation: _Observation) -> bool:
        return bool(observation.decision.hybrid_alarm)

    def finalize(self, observation: _Observation) -> Mapping[str, Any]:
        self._detector.commit(observation.record)
        self.records.append(observation.record)
        self.ledger.append(observation.record)
        self._previous_camera = observation.raw_camera.detach().clone()
        return {"detector_history_action": "commit_raw_current_health", "raw_pose_gpu_fingerprint_before_pressure": observation.pose_fingerprint}

    def timeline_evidence(self, observation: _Observation) -> Mapping[str, Any]:
        decision = observation.decision
        return {"online_detector": {"frame_id": decision.frame_id, "continuous_score": decision.continuous_score, "hybrid_score": decision.hybrid_score, "continuous_alarm": decision.continuous_alarm, "timestamp_order_alarm": decision.timestamp_order_alarm, "hybrid_alarm": decision.hybrid_alarm, "finite_component_count": decision.finite_component_count, "policy_input_sha256": decision.policy_input_sha256}}


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _tensor_summary(value: Any, *, torch: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    data, finite = value.detach(), (torch.isfinite(value.detach()) if value.is_floating_point() else None)
    return {"shape": list(data.shape), "dtype": str(data.dtype), "finite": bool(finite.all().item()) if finite is not None else True, "min": float(data[finite].min().item()) if finite is not None and bool(finite.any().item()) else None, "max": float(data[finite].max().item()) if finite is not None and bool(finite.any().item()) else None}


def _prediction_summary(predictions: Sequence[Mapping[str, Any]], *, torch: Any) -> list[Mapping[str, Any]]:
    output = []
    for frame_id, prediction in enumerate(predictions):
        fields = {name: summary for name, value in sorted(prediction.items()) if (summary := _tensor_summary(value, torch=torch)) is not None}
        if {"camera_pose", "pts3d_in_self_view", "pts3d_in_other_view"} - set(fields) or not all(row["finite"] for row in fields.values()):
            raise RunBoundedPressureV16Error("v16 prediction summary differs")
        output.append({"frame_id": frame_id, "tensor_fields": fields})
    return output


def _trajectory(predictions: Sequence[Mapping[str, Any]], *, capability: Any, decode: Any, np: Any) -> Mapping[str, Any]:
    prior, frames = None, []
    for frame_id, prediction in enumerate(predictions):
        encoding = prediction["camera_pose"].detach().cpu().numpy()[0]
        camera = decode(prediction["camera_pose"].clone()).detach().cpu().numpy()[0]
        translated = prediction["pts3d_in_self_view"].detach().cpu().numpy()[0] @ camera[:3, :3].T + camera[:3, 3]
        other = prediction["pts3d_in_other_view"].detach().cpu().numpy()[0]
        errors, scales = np.linalg.norm(translated - other, axis=-1), np.linalg.norm(other, axis=-1)
        translation_jump = rotation_jump = pose_jump = None
        if prior is not None:
            relative = prior[:3, :3].T @ camera[:3, :3]
            rotation_jump = float(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
            translation_jump = float(np.linalg.norm(prior[:3, :3].T @ (camera[:3, 3] - prior[:3, 3])))
            pose_jump = float(np.hypot(translation_jump, rotation_jump))
        frames.append({"frame_id": frame_id, "camera_pose_encoding_absT_quaR": encoding.tolist(), "camera_to_reference": camera.tolist(), "translation_jump": translation_jump, "rotation_jump_rad": rotation_jump, "pose_jump": pose_jump, "geometric_residual": float(np.median(errors) / max(float(np.median(scales)), 1e-8))})
        prior = camera
    return {"schema_version": "stateguard3r.trajectory.v0", "pose_encoding": "absT_quaR", "matrix_convention": "camera_to_first_input_frame_reference", "reference_frame": {"frame_id": 0, "source_index": 0, "path": str(capability.rgb_paths[0]), "sha256": capability.frames[0].sha256}, "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)", "geometric_residual": "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / max(median_l2(pts3d_in_other_view), 1e-8)", "frames": frames}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Reject a non-production command before resolving or opening a proposed
    # capsule, so CUDA-hidden validation has no runtime data capability.
    if args.capsule != CAPSULE or args.output_dir.parent != ROOT / "outputs" or args.output_dir.exists() or args.detector_config != DETECTOR_CONFIG or args.seed != 0 or args.beta_base != 0.1 or not math.isfinite(args.beta_base) or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RunBoundedPressureV16Error("v16 production command differs from its fixed first release")
    capability = load_dynamic_rgb_capability_v16(args.capsule)
    verify_dynamic_rgb_capability_v16(capability)
    config_payload, config_artifact = _frozen_config(args.detector_config)
    import numpy as np
    import torch
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic
    from stateguard3r.online_detector_v16 import FrozenDetectorV16Config, IncrementalOnlinePrefixDetectorV16
    from stateguard3r.recal3r_bounded_update_pressure_runner_v16 import audit_v16_runner_contract, run_recurrent_bounded_pressure_v16
    from stateguard3r.timestamp_order_v16 import capture_timestamp_records_v16, timestamp_order_sidecar_v16
    from stateguard3r.v16_source_import_audit import audit_v16_production_script
    if not torch.cuda.is_available():
        raise RunBoundedPressureV16Error("v16 CUDA is unavailable")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    sys.path[:0] = [str(RECAL3R_ROOT), str(RECAL3R_ROOT / "src")]
    import add_ckpt_path
    add_ckpt_path.add_path_to_dust3r(str(CHECKPOINT))
    import dust3r.model as dust3r_model
    import dust3r.utils.camera as camera
    import dust3r.utils.device as device_module
    import models.curope.curope2d as curope  # noqa: F401
    if sha256_file_v16(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RunBoundedPressureV16Error("v16 checkpoint hash differs")
    views = prepare_dynamic_rgb_views_v16(capability, torch=torch)
    captures, capture_provenance = capture_timestamp_records_v16(capability.rgb_paths, rgb_txt=capability.listing, dataset_root=capability.dataset_root)
    sidecar = timestamp_order_sidecar_v16(captures, provenance=capture_provenance)
    source_audit = audit_v16_production_script(Path(__file__), (ROOT / "src" / "stateguard3r" / "bounded_update_pressure_v16.py", ROOT / "src" / "stateguard3r" / "recal3r_bounded_update_pressure_runner_v16.py", ROOT / "src" / "stateguard3r" / "online_detector_v16.py", ROOT / "src" / "stateguard3r" / "online_visual_overlap_v16.py", ROOT / "src" / "stateguard3r" / "timestamp_order_v16.py", ROOT / "src" / "stateguard3r" / "dynamic_rgb_capability_v16.py", ROOT / "src" / "stateguard3r" / "health.py"))
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", suffix=".staging", dir=args.output_dir.parent))
    try:
        model, checkpoint_audit = _checkpoint_audit(dust3r_model.ARCroco3DStereo, torch=torch)
        _write_json(staging / "checkpoint-load-audit.json", checkpoint_audit)
        model = model.to("cuda")
        for target in (model, model.config):
            target.model_update_type, target.beta_base, target.entropy_eps, target.entropy_head_reduce, target.uncertainty_clamp_max, target.decay = "recal3r", 0.1, 2e-14, "mean", 1.0, 0.95
        model.eval(); model.enable_u_calibration_trace(oracle_window=1)
        overlaps = _CausalOverlap()
        observer = None if args.state_policy == "always-commit" else _RawObserver(IncrementalOnlinePrefixDetectorV16(FrozenDetectorV16Config.from_mapping(config_payload), captures=captures, capture_provenance=capture_provenance), captures=captures, torch=torch, decode=camera.pose_encoding_to_camera)
        device = torch.device("cuda")
        torch.cuda.synchronize(device); started = time.perf_counter()
        with torch.no_grad():
            result = run_recurrent_bounded_pressure_v16(views, model, device, torch=torch, to_gpu=device_module.to_gpu, to_cpu=device_module.to_cpu, canonicalize_model_update_type=dust3r_model.canonicalize_model_update_type, before_frame=overlaps.before_frame, observer=observer, synchronize=lambda: torch.cuda.synchronize(device))
        torch.cuda.synchronize(device); runtime = time.perf_counter() - started
        if len(result.predictions) != 30 or len(overlaps.values) != 30:
            raise RunBoundedPressureV16Error("v16 output count differs")
        health = observer.records if observer is not None else [replace(record, overlap=overlaps.values[index]) for index, record in enumerate(adapt_recal3r_trace(model.get_u_calibration_trace(), all_frame_ids=list(range(30)), timestamps=[float(row["rgb_capture_timestamp"]) for row in captures], batch_size=1))]
        if len(health) != 30:
            raise RunBoundedPressureV16Error("v16 health count differs")
        write_health_jsonl_atomic(staging / "health.jsonl", health)
        _write_json(staging / "predictions-summary.json", _prediction_summary(result.predictions, torch=torch))
        _write_json(staging / "trajectory.json", _trajectory(result.predictions, capability=capability, decode=camera.pose_encoding_to_camera, np=np))
        _write_json(staging / "timestamp-order-sidecar.json", sidecar)
        _write_json(staging / "state-timeline.json", {"schema_version": SCHEMA, "policy": args.state_policy, "pending_transaction_count": 0, "source_provenance": result.source_provenance, "transactions": result.timeline})
        _write_json(staging / "run.json", {"schema_version": SCHEMA, "status": "succeeded", "state_policy": {"name": args.state_policy}, "runtime_seconds": runtime, "recurrent_policy_runtime_seconds": result.recurrent_policy_runtime_seconds, "capability": dynamic_rgb_capability_provenance_v16(capability), "detector_config": config_artifact, "source_import_audit": source_audit, "runner_contract": audit_v16_runner_contract(ROOT / "src" / "stateguard3r" / "recal3r_bounded_update_pressure_runner_v16.py"), "checkpoint_load_audit": checkpoint_audit, "online_visual_correspondence": overlaps.metadata(), "timestamp_capture_provenance": capture_provenance, "started_at": datetime.now().astimezone().isoformat(), "finished_at": datetime.now().astimezone().isoformat()})
        for child in staging.iterdir():
            child.chmod(0o444)
        staging.chmod(0o555)
        os.rename(staging, args.output_dir)
    except Exception as error:
        try:
            _write_json(staging / "failure.json", {"schema_version": SCHEMA, "status": "failed", "error_class": type(error).__name__, "error": str(error)})
            for child in staging.iterdir():
                if child.is_file() and not child.is_symlink():
                    child.chmod(0o444)
            staging.chmod(0o555)
        except Exception:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
