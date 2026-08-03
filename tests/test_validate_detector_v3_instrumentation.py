from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_detector_v3_instrumentation import (
    BYTE_IDENTICAL_ARTIFACTS,
    InstrumentationValidationError,
    RUN_ARTIFACTS,
    RUN_INVARIANT_FIELDS,
    validate_instrumentation,
)
from stateguard3r.timestamp_order_v3 import timestamp_order_sidecar


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _metadata(*, profile: str, run_dir: Path, pid: int) -> dict[str, object]:
    frame_count = 3
    image_hashes = [hashlib.sha256(f"image-{index}".encode()).hexdigest() for index in range(frame_count)]
    metadata: dict[str, object] = {
        "schema_version": "smoke.v0",
        "status": "succeeded",
        "baseline_root": "/baseline",
        "baseline_commit": "commit",
        "baseline_tracked_worktree_clean": True,
        "module_paths": {"model": "sha"},
        "curope_kernel": {"sha256": "kernel"},
        "runner": {"sha256": "runner"},
        "checkpoint": "/checkpoint",
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_size_bytes": 1,
        "input_mode": "manifest",
        "input_manifest": {"sha256": "manifest-sha"},
        "images": [{"sha256": value} for value in image_hashes],
        "input_frames": [{"sha256": value, "frame_id": index} for index, value in enumerate(image_hashes)],
        "frame_count": frame_count,
        "size": 512,
        "seed": 0,
        "model_update_type": "recal3r",
        "loaded_model_interface": {"head": "dpt"},
        "checkpoint_state_dict": {"strict": False},
        "beta_base": 0.1,
        "recal3r_runtime_config": {"decay": 0.95},
        "recal3r_runtime_config_source": {"sha256": "source"},
        "model_eval": True,
        "calibrated_update_calls": frame_count - 1,
        "expected_calibrated_update_calls": frame_count - 1,
        "trace_frame_steps": list(range(1, frame_count)),
        "health_signal_semantics": {"reliability": "x"},
        "state_args_count": 0,
        "runtime_scope": "inference",
        "peak_memory_scope": "model",
        "device": "cuda",
        "cuda_visible_devices": "2",
        "gpu_name": "L20",
        "torch_version": "2.4",
        "cuda_runtime": "12.1",
        "pid": pid,
        "output_dir": str(run_dir),
        "health_profile": profile,
        "online_visual_correspondence": {
            "schema_version": "stateguard3r.visual-overlap.v1",
            "input": "post_official_loader_post_deferred_transform_normalized_rgb",
            "causal_reference": "immediately_previous_frame_only",
            "model_ready_uint8_rgb_sha256": ["a" * 64 for _ in range(frame_count)],
            "opencv": {"threads": 1, "opencl_enabled": False, "rng_seed": 0},
            "frame_results": [
                {"frame_id": index, "reference_frame_id": index - 1, "score": index / 10}
                for index in range(1, frame_count)
            ],
            "runtime_seconds": 0.2 if profile == "v2" else 0.3,
        },
    }
    assert set(RUN_INVARIANT_FIELDS).issubset(metadata)
    return metadata


def _write_run(path: Path, *, profile: str, pid: int) -> None:
    path.mkdir()
    metadata = _metadata(profile=profile, run_dir=path, pid=pid)
    for name in RUN_ARTIFACTS:
        artifact = path / name
        if name == "health.jsonl":
            artifact.write_text(
                "".join(
                    json.dumps({"frame_id": index, "overlap": None if index == 0 else index / 10}, sort_keys=True)
                    + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
        elif name == "run.json":
            _write_json(artifact, metadata)
        elif name in BYTE_IDENTICAL_ARTIFACTS:
            artifact.write_bytes(f"identical-{name}".encode())
        else:
            artifact.write_bytes(f"other-{profile}-{name}".encode())
    if profile == "v3":
        image_hashes = [item["sha256"] for item in metadata["images"]]  # type: ignore[index]
        captures = [
            {
                "frame_id": index,
                "rgb_capture_timestamp": str(index),
                "rgb_capture_timestamp_text": str(index),
                "rgb_txt_physical_line": index + 1,
                "rgb_path_sha256": image_hash,
            }
            for index, image_hash in enumerate(image_hashes)
        ]
        sidecar = timestamp_order_sidecar(
            captures,
            provenance={
                "parser_version": "fixture",
                "rgb_txt_path": "/raw/rgb.txt",
                "rgb_txt_sha256": "b" * 64,
                "dataset_root": "/raw",
                "input_contract": "ordered_rgb_paths_only_no_source_index_gt_label_or_event_metadata",
            },
        )
        sidecar_path = path / "timestamp-order.json"
        _write_json(sidecar_path, sidecar)
        metadata["capture_timestamp_order"] = {
            "path": str(sidecar_path),
            "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            "schema_version": sidecar["schema_version"],
            "purpose": sidecar["purpose"],
            "violation_positions": [],
        }
        _write_json(path / "run.json", metadata)


def _postflight(path: Path) -> None:
    path.write_text("NVIDIA-SMI\nNo runner process remains\n", encoding="utf-8")


def test_validates_equivalent_v2_and_v3_runs(tmp_path: Path) -> None:
    v2, v3 = tmp_path / "v2", tmp_path / "v3"
    _write_run(v2, profile="v2", pid=123)
    _write_run(v3, profile="v3", pid=456)
    v2_log, v3_log = tmp_path / "v2.log", tmp_path / "v3.log"
    _postflight(v2_log)
    _postflight(v3_log)

    report = validate_instrumentation(
        v2_dir=v2,
        v3_dir=v3,
        v2_postflight_log=v2_log,
        v3_postflight_log=v3_log,
    )

    assert report["status"] == "PASS"
    assert report["frame_count"] == 3
    assert report["timestamp_order"]["violation_positions"] == []
    assert all(report["checks"].values())


def test_rejects_timestamp_sidecar_unbound_from_model_input(tmp_path: Path) -> None:
    v2, v3 = tmp_path / "v2", tmp_path / "v3"
    _write_run(v2, profile="v2", pid=123)
    _write_run(v3, profile="v3", pid=456)
    sidecar_path = v3 / "timestamp-order.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["records"][1]["rgb_path_sha256"] = "c" * 64
    _write_json(sidecar_path, sidecar)
    run = json.loads((v3 / "run.json").read_text(encoding="utf-8"))
    run["capture_timestamp_order"]["sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _write_json(v3 / "run.json", run)
    v2_log, v3_log = tmp_path / "v2.log", tmp_path / "v3.log"
    _postflight(v2_log)
    _postflight(v3_log)

    with pytest.raises(InstrumentationValidationError, match="not bound to model input"):
        validate_instrumentation(
            v2_dir=v2,
            v3_dir=v3,
            v2_postflight_log=v2_log,
            v3_postflight_log=v3_log,
        )
