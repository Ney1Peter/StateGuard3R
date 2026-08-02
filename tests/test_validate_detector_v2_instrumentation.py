from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_detector_v2_instrumentation import (
    BYTE_IDENTICAL_ARTIFACTS,
    InstrumentationValidationError,
    RUN_ARTIFACTS,
    RUN_INVARIANT_FIELDS,
    validate_instrumentation,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _run_metadata(*, profile: str, frame_count: int, pid: int) -> dict[str, object]:
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
        "images": [{"sha256": "image"}],
        "input_frames": [{"sha256": "image", "frame_id": 0}],
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
        "output_dir": f"/{profile}",
    }
    assert set(RUN_INVARIANT_FIELDS).issubset(metadata)
    if profile == "v2":
        metadata["health_profile"] = "v2"
        metadata["online_visual_correspondence"] = {
            "schema_version": "stateguard3r.visual-overlap.v1",
            "input": "post_official_loader_post_deferred_transform_normalized_rgb",
            "causal_reference": "immediately_previous_frame_only",
            "model_ready_uint8_rgb_sha256": ["a" * 64 for _ in range(frame_count)],
            "opencv": {
                "version": "4.11",
                "module_sha256": "b" * 64,
                "binary_sha256": "c" * 64,
                "build_info_sha256": "d" * 64,
                "threads": 1,
                "opencl_enabled": False,
                "rng_seed": 0,
            },
            "frame_results": [
                {"frame_id": index, "reference_frame_id": index - 1, "score": index / 10}
                for index in range(1, frame_count)
            ],
            "runtime_seconds": 0.2,
        }
    return metadata


def _write_run(path: Path, *, profile: str, pid: int) -> None:
    path.mkdir()
    frame_count = 3
    for name in RUN_ARTIFACTS:
        artifact = path / name
        if name == "health.jsonl":
            rows = []
            for index in range(frame_count):
                rows.append(
                    {
                        "frame_id": index,
                        "reliability": 0.5,
                        "overlap": None if profile == "v1" or index == 0 else index / 10,
                    }
                )
            artifact.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
        elif name == "run.json":
            _write_json(artifact, _run_metadata(profile=profile, frame_count=frame_count, pid=pid))
        elif name in BYTE_IDENTICAL_ARTIFACTS:
            artifact.write_bytes(f"identical-{name}".encode())
        else:
            artifact.write_bytes(f"other-{profile}-{name}".encode())


def _postflight(path: Path) -> None:
    path.write_text("NVIDIA-SMI\nNo runner process remains\n", encoding="utf-8")


def test_validates_equivalent_v1_and_v2_runs(tmp_path: Path) -> None:
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    _write_run(v1, profile="v1", pid=123)
    _write_run(v2, profile="v2", pid=456)
    v1_log, v2_log = tmp_path / "v1.log", tmp_path / "v2.log"
    _postflight(v1_log)
    _postflight(v2_log)

    report = validate_instrumentation(
        v1_dir=v1,
        v2_dir=v2,
        v1_postflight_log=v1_log,
        v2_postflight_log=v2_log,
    )

    assert report["status"] == "PASS"
    assert report["frame_count"] == 3
    assert report["recurrent_update_count"] == 2
    assert all(report["checks"].values())
    assert report["visual"]["opencv"]["threads"] == 1


def test_rejects_health_change_beyond_overlap(tmp_path: Path) -> None:
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    _write_run(v1, profile="v1", pid=123)
    _write_run(v2, profile="v2", pid=456)
    health = (v2 / "health.jsonl").read_text(encoding="utf-8").splitlines()
    changed = json.loads(health[1])
    changed["reliability"] = 0.25
    health[1] = json.dumps(changed, sort_keys=True)
    (v2 / "health.jsonl").write_text("\n".join(health) + "\n", encoding="utf-8")
    v1_log, v2_log = tmp_path / "v1.log", tmp_path / "v2.log"
    _postflight(v1_log)
    _postflight(v2_log)

    with pytest.raises(InstrumentationValidationError, match="other than overlap"):
        validate_instrumentation(
            v1_dir=v1,
            v2_dir=v2,
            v1_postflight_log=v1_log,
            v2_postflight_log=v2_log,
        )


def test_rejects_nonfinite_or_missing_v2_overlap(tmp_path: Path) -> None:
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    _write_run(v1, profile="v1", pid=123)
    _write_run(v2, profile="v2", pid=456)
    health = (v2 / "health.jsonl").read_text(encoding="utf-8").splitlines()
    changed = json.loads(health[2])
    changed["overlap"] = None
    health[2] = json.dumps(changed, sort_keys=True)
    (v2 / "health.jsonl").write_text("\n".join(health) + "\n", encoding="utf-8")
    v1_log, v2_log = tmp_path / "v1.log", tmp_path / "v2.log"
    _postflight(v1_log)
    _postflight(v2_log)

    with pytest.raises(InstrumentationValidationError, match="finite"):
        validate_instrumentation(
            v1_dir=v1,
            v2_dir=v2,
            v1_postflight_log=v1_log,
            v2_postflight_log=v2_log,
        )
