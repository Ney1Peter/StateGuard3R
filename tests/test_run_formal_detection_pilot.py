from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from scripts import run_formal_detection_pilot as pilot
from stateguard3r import detection_suite


RUN_LAYOUT = (
    ("development-dynamic", "development", "dynamic_occlusion"),
    ("development-wrong", "development", "wrong_order_segment"),
    ("development-low", "development", "low_overlap_jump"),
    ("holdout-dynamic", "holdout", "dynamic_occlusion"),
    ("holdout-wrong", "holdout", "wrong_order_segment"),
    ("holdout-low", "holdout", "low_overlap_jump"),
)
ENDS = {
    "dynamic_occlusion": 19,
    "wrong_order_segment": 18,
    "low_overlap_jump": 19,
}
RECTANGLE_REFERENCE = "model_input_after_resize_and_center_crop"
EXPECTED_VALIDATOR_CHECKS = {
    "strict_manifest_validation",
    "cpu_pixel_replay",
    "raw_source_snapshot_unchanged",
    "input_tree_snapshot_unchanged",
    "official_tum_associations",
    "production_runtime_provenance",
    "cuda_hidden",
    "six_run_coverage",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _hash(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()


def _artifact(payload: bytes, path: str) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _snapshot(
    path: str,
    digest: str,
    size: int,
    inode: int,
    *,
    count: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path,
        "sha256": digest,
        "size_bytes": size,
        "mode_octal": "0444",
        "link_count": 1,
        "device": 9,
        "inode": inode,
        "mtime_ns": 123456789,
    }
    if count is not None:
        value["data_entry_count"] = count
    return value


def _rectangles() -> list[dict[str, float]]:
    return [
        {
            "x": round(0.25 + offset * 0.04, 8),
            "y": round(0.25 + offset * 0.025, 8),
            "width": 0.5,
            "height": 0.5,
        }
        for offset in range(5)
    ]


def _parameters(corruption_type: str) -> dict[str, object]:
    if corruption_type == "dynamic_occlusion":
        return {
            "coordinate_space": "normalized",
            "coordinate_reference": RECTANGLE_REFERENCE,
            "initial_rectangle": {
                "x": 0.25,
                "y": 0.25,
                "width": 0.5,
                "height": 0.5,
            },
            "velocity": {"dx": 0.04, "dy": 0.025},
            "fill": [255, 0, 0],
            "frame_rectangles": _rectangles(),
            "rectangle_generated_from_seed": False,
            "velocity_generated_from_seed": False,
        }
    if corruption_type == "low_overlap_jump":
        return {
            "source_start": 30,
            "source_end": 34,
            "source_indices": [30, 31, 32, 33, 34],
            "selection": "explicit",
        }
    return {
        "mode": "reverse",
        "permutation": [18, 17, 16, 15],
        "relative_permutation": [3, 2, 1, 0],
    }


def _transform(
    corruption_type: str, position: int, source_index: int
) -> dict[str, object]:
    if corruption_type == "dynamic_occlusion":
        return {
            "type": "rectangle_occlusion",
            "coordinate_space": "normalized",
            "coordinate_reference": RECTANGLE_REFERENCE,
            "rectangle": _rectangles()[position - 15],
            "fill": [255, 0, 0],
        }
    return {
        "type": (
            "source_frame_substitution"
            if corruption_type == "low_overlap_jump"
            else "temporal_reorder"
        ),
        "corruption": corruption_type,
        "original_source_index": position,
        "replacement_source_index": source_index,
    }


def _make_input_root(tmp_path: Path) -> Path:
    root = tmp_path / "inputs"
    records: list[dict[str, object]] = []
    commitments: dict[str, list[dict[str, object]]] = {
        "development": [],
        "holdout": [],
    }
    positions = {"development": 0, "holdout": 0}
    starts = {
        ("development", "dynamic_occlusion"): (248, None),
        ("development", "wrong_order_segment"): (320, None),
        ("development", "low_overlap_jump"): (380, 300),
        ("holdout", "dynamic_occlusion"): (520, None),
        ("holdout", "wrong_order_segment"): (550, None),
        ("holdout", "low_overlap_jump"): (583, 500),
    }
    for run_id, split, corruption_type in RUN_LAYOUT:
        run_root = root / split / run_id
        run_root.mkdir(parents=True)
        base_start, donor_start = starts[(split, corruption_type)]
        source_count = 35 if corruption_type == "low_overlap_jump" else 30
        raw_indices = list(range(base_start, base_start + 30))
        if donor_start is not None:
            raw_indices.extend(range(donor_start, donor_start + 5))
        source_shas = tuple(_hash(run_id, index) for index in range(source_count))
        if corruption_type == "low_overlap_jump":
            source_indices = [*range(15), *range(30, 35), *range(20, 30)]
        elif corruption_type == "wrong_order_segment":
            source_indices = [*range(15), 18, 17, 16, 15, *range(19, 30)]
        else:
            source_indices = list(range(30))
        end = ENDS[corruption_type]
        source_frames: list[dict[str, object]] = []
        for pool_index, raw_index in enumerate(raw_indices):
            timestamp_decimal = Decimal("1000.0") + Decimal(raw_index) * Decimal(
                "0.03"
            )
            depth_timestamp_decimal = timestamp_decimal + Decimal("0.005")
            groundtruth_timestamp_decimal = timestamp_decimal + Decimal("0.007")
            timestamp_text = f"{timestamp_decimal:.6f}"
            depth_timestamp_text = f"{depth_timestamp_decimal:.6f}"
            groundtruth_timestamp_text = f"{groundtruth_timestamp_decimal:.6f}"
            timestamp = float(timestamp_text)
            depth_timestamp = float(depth_timestamp_text)
            groundtruth_timestamp = float(groundtruth_timestamp_text)
            depth_delta = depth_timestamp_decimal - timestamp_decimal
            groundtruth_delta = groundtruth_timestamp_decimal - timestamp_decimal
            source_frames.append(
                {
                    "path": f"../../../../raw/{run_id}/rgb/{raw_index}.png",
                    "source_entry_index": raw_index,
                    "source_line": raw_index + 2,
                    "physical_line_sha256": _hash("rgb-line", raw_index),
                    "physical_line_size_bytes": 80,
                    "timestamp": timestamp,
                    "timestamp_text": timestamp_text,
                    "raw_rgb_source_index": raw_index,
                    "raw_rgb_source_line": raw_index + 2,
                    "rgb_sha256": source_shas[pool_index],
                    "rgb_size_bytes": 1000 + pool_index,
                    "rgb_mode_octal": "0444",
                    "rgb_link_count": 1,
                    "rgb_device": 10,
                    "rgb_inode": 100_000 + raw_index,
                    "rgb_mtime_ns": 1_000_000 + raw_index,
                    "depth": {
                        "source_entry_index": 10_000 + raw_index,
                        "source_line": 10_002 + raw_index,
                        "physical_line_sha256": _hash("depth-line", raw_index),
                        "physical_line_size_bytes": 80,
                        "timestamp": depth_timestamp,
                        "timestamp_text": depth_timestamp_text,
                        "path": f"../../../../raw/{run_id}/depth/{raw_index}.png",
                        "sha256": _hash(f"{run_id}-depth", raw_index),
                        "size_bytes": 2000 + pool_index,
                        "mode_octal": "0444",
                        "link_count": 1,
                        "device": 11,
                        "inode": 200_000 + raw_index,
                        "mtime_ns": 2_000_000 + raw_index,
                        "delta_from_rgb_seconds": float(depth_delta),
                        "absolute_delta_seconds": float(abs(depth_delta)),
                    },
                    "groundtruth": {
                        "source_entry_index": 20_000 + raw_index,
                        "source_line": 20_002 + raw_index,
                        "physical_line_sha256": _hash("gt-line", raw_index),
                        "physical_line_size_bytes": 120,
                        "timestamp": groundtruth_timestamp,
                        "timestamp_text": groundtruth_timestamp_text,
                        "translation_xyz": [0.0, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "translation_xyz_text": ["0", "0", "0"],
                        "quaternion_xyzw_text": ["0", "0", "0", "1"],
                        "delta_from_rgb_seconds": float(groundtruth_delta),
                        "absolute_delta_seconds": float(abs(groundtruth_delta)),
                    },
                    "source_pool_index": pool_index,
                    "source_pool_role": (
                        "base" if pool_index < 30 else "low_overlap_donor"
                    ),
                }
            )
        groundtruth_file_sha = _hash("groundtruth-file", 0)
        corruption = {
            "type": corruption_type,
            "start_frame": 15,
            "end_frame": end,
            "start": 15,
            "end": end,
            "parameters": _parameters(corruption_type),
            "expected_effect": "formal-test-fixture",
        }
        source_manifest = {
            "schema_version": "stateguard3r.tum-formal-pilot-source.v1",
            "sequence": f"{run_id}-source-pool",
            "dataset": "rgbd_dataset_freiburg1_desk",
            "dataset_split": split,
            "run_id": run_id,
            "corruption_type": corruption_type,
            "source_is_read_only": True,
            "file_path_semantics": "relative_to_this_manifest_directory",
            "index_convention": (
                "zero_based_data_entry_index_excluding_comments_and_blank_lines"
            ),
            "physical_line_convention": "one_based_physical_source_line",
            "association_policy": {
                "method": "unique_nearest_absolute_timestamp",
                "tie_policy": "reject",
                "reuse_policy": (
                    "reject_within_source_pool_and_across_consumed_runs"
                ),
                "max_absolute_delta_seconds": 0.02,
            },
            "frame_count": source_count,
            "output_frame_count": 30,
            "ground_truth_path": "../../../../raw/groundtruth.txt",
            "tum_index_files": {
                "rgb": _snapshot(
                    "../../../../raw/rgb.txt",
                    _hash("rgb-index", 0),
                    100,
                    1,
                    count=1000,
                ),
                "depth": _snapshot(
                    "../../../../raw/depth.txt",
                    _hash("depth-index", 0),
                    100,
                    2,
                    count=1000,
                ),
                "groundtruth": _snapshot(
                    "../../../../raw/groundtruth.txt",
                    groundtruth_file_sha,
                    100,
                    3,
                    count=1000,
                ),
            },
            "official_lineage": {
                "dataset_root": (
                    "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/"
                    "rgbd_dataset_freiburg1_desk"
                ),
                "archive": _snapshot(
                    "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/"
                    "rgbd_dataset_freiburg1_desk.tgz",
                    "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
                    344011403,
                    4,
                ),
                "raw_manifest": _snapshot(
                    "/data/wangzheng/Project2/baselines/ReCal3R/logs/"
                    "gate2-fr1-desk-raw-manifest.json",
                    "5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87",
                    231139,
                    5,
                ),
                "raw_manifest_schema_version": "stateguard3r.tum-raw-audit.v1",
            },
            "frames": source_frames,
        }
        source_bytes = _json_bytes(source_manifest)
        input_manifest = {
            "schema_version": "stateguard3r.corruption.v1",
            "frame_count": 30,
            "sequence": run_id,
            "source_sequence": f"{run_id}-source-pool",
            "source_manifest": pilot.SOURCE_MANIFEST_FILENAME,
            "source_frame_count": source_count,
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_ground_truth_path": "../../../../raw/groundtruth.txt",
            "source_ground_truth_resolved_path": (
                "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/"
                "rgbd_dataset_freiburg1_desk/groundtruth.txt"
            ),
            "source_ground_truth_sha256": groundtruth_file_sha,
            "source_is_read_only": True,
            "seed": 0,
            "index_convention": "zero_based_inclusive",
            "materialization": "deferred_transforms_no_image_copy",
            "frames": [
                {
                    "frame_index": position,
                    "source_index": source_index,
                    "path": source_frames[source_index]["path"],
                    "metadata": {
                        key: value
                        for key, value in source_frames[source_index].items()
                        if key != "path"
                    },
                    "transforms": (
                        [_transform(corruption_type, position, source_index)]
                        if 15 <= position <= end
                        else []
                    ),
                }
                for position, source_index in enumerate(source_indices)
            ],
            "corruptions": [corruption],
        }
        input_bytes = _json_bytes(input_manifest)
        source_relative = f"{split}/{run_id}/{pilot.SOURCE_MANIFEST_FILENAME}"
        input_relative = f"{split}/{run_id}/{pilot.INPUT_MANIFEST_FILENAME}"
        (run_root / pilot.SOURCE_MANIFEST_FILENAME).write_bytes(source_bytes)
        (run_root / pilot.INPUT_MANIFEST_FILENAME).write_bytes(input_bytes)
        consumed = tuple(sorted(source_shas[index] for index in source_indices))
        suite_input = detection_suite.DetectionSuiteInput(
            run_id=run_id,
            dataset_split=split,
            corruption_type=corruption_type,
            raw_frame_sha256s=consumed,
            corruption_json=input_bytes,
            input_manifest_json=input_bytes,
            source_manifest_json=source_bytes,
        )
        commitment = detection_suite.make_run_commitment(suite_input)
        commitments[split].append(commitment)
        records.append(
            {
                "run_id": run_id,
                "dataset_split": split,
                "corruption_type": corruption_type,
                "execution_position_within_split": positions[split],
                "raw_frame_sha256s": list(consumed),
                "artifacts": {
                    "source_manifest": _artifact(source_bytes, source_relative),
                    "input_manifest": _artifact(input_bytes, input_relative),
                    "corruption_json": _artifact(input_bytes, input_relative),
                },
                "suite_run_commitment": commitment,
            }
        )
        positions[split] += 1
    split_bytes = detection_suite.make_split_registry(
        commitments["development"], commitments["holdout"]
    )
    holdout_bytes = detection_suite.make_holdout_commitment(commitments["holdout"])
    registry = {
        "schema_version": pilot.INPUT_SCHEMA_VERSION,
        "seed": 0,
        "frame_count": 30,
        "fixed_execution_order": {
            split: [
                run_id
                for run_id, record_split, _ in RUN_LAYOUT
                if record_split == split
            ]
            for split in pilot.SPLITS
        },
        "formal_commitment_artifacts": {
            "split_registry": _artifact(split_bytes, pilot.SPLIT_REGISTRY_FILENAME),
            "holdout_commitment": _artifact(
                holdout_bytes, pilot.HOLDOUT_COMMITMENT_FILENAME
            ),
        },
        "runs": records,
    }
    (root / pilot.SPLIT_REGISTRY_FILENAME).write_bytes(split_bytes)
    (root / pilot.HOLDOUT_COMMITMENT_FILENAME).write_bytes(holdout_bytes)
    (root / pilot.INPUT_REGISTRY_FILENAME).write_bytes(_json_bytes(registry))
    return root


def _fake_runner_provenance() -> dict[str, object]:
    runner = (pilot.REPOSITORY_ROOT / pilot.RUNNER_RELATIVE_PATH).read_bytes()
    return {
        "state_guard_repository": {
            "path": str(pilot.REPOSITORY_ROOT),
            "commit": "a" * 40,
            "tracked_worktree_clean": True,
        },
        "runner_script": _artifact(runner, pilot.RUNNER_RELATIVE_PATH.as_posix()),
        "recal3r_repository": {
            "path": str(pilot.RECAL3R_ROOT.resolve(strict=True)),
            "commit": pilot.EXPECTED_RECAL3R_COMMIT,
            "tracked_worktree_clean": True,
        },
    }


def _write_validation_report(input_root: Path, path: Path) -> Path:
    generator = (
        pilot.REPOSITORY_ROOT / pilot.CPU_VALIDATION_GENERATOR_RELATIVE_PATH
    ).read_bytes()
    loader = (pilot.RECAL3R_ROOT / pilot.OFFICIAL_LOADER_RELATIVE_PATH).read_bytes()
    run_ids = {
        split: [
            run_id for run_id, run_split, _ in RUN_LAYOUT if run_split == split
        ]
        for split in pilot.SPLITS
    }
    source_digest = hashlib.sha256(b"canonical before/after source snapshots").hexdigest()
    input_tree_digest, input_tree_count = pilot._input_tree_snapshot(input_root)
    report = {
        "schema_version": pilot.CPU_VALIDATION_SCHEMA_VERSION,
        "status": "PASS",
        "input_registry": _artifact(
            (input_root / pilot.INPUT_REGISTRY_FILENAME).read_bytes(),
            pilot.INPUT_REGISTRY_FILENAME,
        ),
        "run_ids": run_ids,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "torch_cuda_is_initialized_before": False,
            "torch_cuda_is_initialized_after": False,
        },
        "source_snapshots_before_sha256": source_digest,
        "source_snapshots_after_sha256": source_digest,
        "source_snapshot_artifact_count": 385,
        "input_tree_snapshots_before_sha256": input_tree_digest,
        "input_tree_snapshots_after_sha256": input_tree_digest,
        "input_tree_snapshot_entry_count": input_tree_count,
        "runs": [
            {
                "run_id": run_id,
                "frame_count": 30,
                "manifest_validation_passed": True,
                "pixel_replay_passed": True,
                "transform_digest_sha256": hashlib.sha256(
                    f"{run_id}:transforms".encode()
                ).hexdigest(),
                "pixel_digest_sha256": hashlib.sha256(
                    f"{run_id}:pixels".encode()
                ).hexdigest(),
            }
            for run_id, _, _ in RUN_LAYOUT
        ],
        "generator": {
            "repository_commit": "a" * 40,
            "tracked_worktree_clean": True,
            "script": _artifact(
                generator,
                "scripts/validate_formal_pilot_inputs.py",
            ),
            "git_blob_sha256": hashlib.sha256(generator).hexdigest(),
        },
        "production_runtime": {
            "recal3r_repository": {
                "path": str(pilot.RECAL3R_ROOT.resolve(strict=True)),
                "commit": pilot.EXPECTED_RECAL3R_COMMIT,
                "tracked_worktree_clean": True,
            },
            "python": {
                "executable": str(pilot.RECAL3R_PYTHON),
                "executable_realpath": str(pilot.RECAL3R_PYTHON.resolve(strict=True)),
                "prefix": str((pilot.RECAL3R_ROOT / ".venv").resolve(strict=True)),
            },
            "official_loader": _artifact(
                loader,
                "src/dust3r/utils/image.py",
            ),
        },
        "tum_associations": {
            "method": "unique_nearest_absolute_timestamp",
            "tie_policy": "reject",
            "max_absolute_delta_seconds": 0.02,
            "allocated_rgb_count": 190,
            "globally_unique_depth_count": 190,
            "globally_unique_groundtruth_count": 190,
            "association_digest_sha256": hashlib.sha256(
                b"independent TUM association oracle"
            ).hexdigest(),
        },
        "checks": {name: True for name in EXPECTED_VALIDATOR_CHECKS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(report))
    path.chmod(0o444)
    return path


def _install_fake_cpu_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    output_root = tmp_path / "orchestration-outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pilot, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(
        pilot,
        "_tracked_file_sha256_at_head",
        lambda repository, relative_path: hashlib.sha256(
            (repository / relative_path).read_bytes()
        ).hexdigest(),
    )

    unlock_times = iter(
        (
            "2026-08-02T00:00:00.000000+00:00",
            "2026-08-02T01:00:00.000000+00:00",
        )
    )
    monkeypatch.setattr(pilot, "_utc_now_text", lambda: next(unlock_times))
    monkeypatch.setattr(
        pilot,
        "_utc_now",
        lambda: datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc),
    )

    def invoke(input_root: Path, report_path: Path) -> pilot._Snapshot:
        _write_validation_report(input_root, report_path)
        return pilot._read_snapshot(report_path, name="fake fresh CPU report")

    monkeypatch.setattr(pilot, "_invoke_fixed_cpu_validator", invoke)
    return output_root


def _health_bytes() -> bytes:
    return b"".join(
        _json_bytes(
            {
                "frame_id": index,
                "update_magnitude": float(index + 1),
                "reliability": 0.9,
            }
        )
        + b"\n"
        for index in range(30)
    )


def _write_run_outputs(
    root: Path,
    input_root: Path,
    split: str,
    runner_commit: str = "a" * 40,
) -> None:
    for run_id, run_split, corruption_type in RUN_LAYOUT:
        if run_split != split:
            continue
        run_root = root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        input_run_root = input_root / split / run_id
        source_path = input_run_root / pilot.SOURCE_MANIFEST_FILENAME
        input_path = input_run_root / pilot.INPUT_MANIFEST_FILENAME
        source_manifest = json.loads(source_path.read_bytes())
        input_manifest = json.loads(input_path.read_bytes())
        input_frames = input_manifest["frames"]
        resolved_paths = [
            os.path.realpath(
                os.path.join(source_path.parent, str(frame["path"]))
            )
            for frame in input_frames
        ]
        output_shas = [
            str(frame["metadata"]["rgb_sha256"]) for frame in input_frames
        ]
        transform_name = {
            "dynamic_occlusion": "rectangle_occlusion",
            "low_overlap_jump": "source_frame_substitution",
            "wrong_order_segment": "temporal_reorder",
        }[corruption_type]
        end = ENDS[corruption_type]
        split_position = [
            candidate_id
            for candidate_id, candidate_split, _ in RUN_LAYOUT
            if candidate_split == split
        ].index(run_id)
        hour = 0 if split == "development" else 1
        minute = 10 * (split_position + 1)
        started_at = f"2026-08-02T{hour:02d}:{minute:02d}:00.000000+00:00"
        finished_at = f"2026-08-02T{hour:02d}:{minute:02d}:30.000000+00:00"
        (run_root / "health.jsonl").write_bytes(_health_bytes())
        (run_root / "run.json").write_bytes(
            _json_bytes(
                {
                    "schema_version": "stateguard3r.recal3r-smoke.v0",
                    "status": "succeeded",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "baseline_commit": pilot.EXPECTED_RECAL3R_COMMIT,
                    "checkpoint_sha256": (
                        "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
                    ),
                    "frame_count": 30,
                    "calibrated_update_calls": 29,
                    "expected_calibrated_update_calls": 29,
                    "size": 512,
                    "seed": 0,
                    "beta_base": 0.1,
                    "input_mode": "manifest",
                    "model_update_type": "recal3r",
                    "runner": {
                        "commit": runner_commit,
                        "script_sha256": _fake_runner_provenance()["runner_script"][
                            "sha256"
                        ],
                    },
                    "runtime_seconds": 1.0,
                    "peak_memory_allocated_mib": 100.0,
                    "trace_frame_steps": list(range(1, 30)),
                    "health_signal_semantics": {
                        "reliability": "1 - ReCal3R uncertainty_u",
                        "update_magnitude_source": "global_state_delta",
                        "pose_jump": (
                            "hypot(relative_translation_l2, "
                            "relative_rotation_angle_rad)"
                        ),
                        "geometric_residual": (
                            "relative_median_independent_cross_head_"
                            "pointmap_consistency"
                        ),
                        "geometric_residual_unavailable_reason": None,
                    },
                    "images": [
                        {"path": path, "sha256": digest}
                        for path, digest in zip(resolved_paths, output_shas)
                    ],
                    "input_frames": [
                        {
                            **frame,
                            "path": resolved_paths[position],
                            "sha256": output_shas[position],
                        }
                        for position, frame in enumerate(input_frames)
                    ],
                    "input_manifest": {
                        "path": str(input_path.resolve(strict=True)),
                        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "source_manifest": str(source_path.resolve(strict=True)),
                        "source_manifest_sha256": hashlib.sha256(
                            source_path.read_bytes()
                        ).hexdigest(),
                        "materialization": (
                            "final_manifest_order_with_in_memory_deferred_transforms"
                        ),
                        "rectangle_coordinate_reference": (
                            RECTANGLE_REFERENCE
                            if corruption_type == "dynamic_occlusion"
                            else None
                        ),
                        "transform_counts": {transform_name: end - 14},
                    },
                }
            )
        )
        (run_root / "checkpoint-load-audit.json").write_bytes(_json_bytes({}))
        (run_root / "predictions-summary.json").write_bytes(_json_bytes({}))
        (run_root / "trajectory.json").write_bytes(_json_bytes([]))
        for artifact in run_root.iterdir():
            artifact.chmod(0o444)
        run_root.chmod(0o555)


def _rewrite_frozen_run_json(path: Path, **updates: object) -> None:
    path.chmod(0o644)
    payload = json.loads(path.read_bytes())
    payload.update(updates)
    path.write_bytes(_json_bytes(payload))
    path.chmod(0o444)


def _metrics(split: str) -> dict[str, object]:
    run_rows = [row for row in RUN_LAYOUT if row[1] == split]
    run_ids = [row[0] for row in run_rows]
    labels_by_run = {
        run_id: [int(15 <= index <= ENDS[corruption_type]) for index in range(30)]
        for run_id, _, corruption_type in run_rows
    }
    combined_per_run = {
        run_id: {
            "scores": [float(index) / 30.0 for index in range(30)],
            "used_signals": ["update_magnitude", "reliability"],
            "signal_sources": [
                {"canonical": "update_magnitude", "source": "update_magnitude"},
                {"canonical": "reliability", "source": "reliability"},
            ],
        }
        for run_id in run_ids
    }
    intervals = [
        {
            "run_id": run_id,
            "corruption_type": corruption_type,
            "delay_frames": 0,
        }
        for run_id, _, corruption_type in run_rows
    ]
    return {
        "formal": split == "holdout",
        "runs": [
            {"run_id": run_id, "primary_labels": labels_by_run[run_id]}
            for run_id in run_ids
        ],
        "methods": {
            "random": {"macro": {"auroc": 0.75}},
            "update_magnitude_only": {"macro": {"auroc": 0.87}},
            "reliability_only": {"macro": {"auroc": 0.80}},
            "combined": {
                "macro": {"auroc": 0.85},
                "pooled": {"false_positive_rate": 0.20},
                "diagnostics": {
                    "max_per_run_consecutive_false_positives": 3
                },
                "events": {"intervals": intervals},
                "per_run": combined_per_run,
            },
        },
    }


def _patch_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "window": 5,
        "epsilon": 1e-6,
        "max_z": 10.0,
        "master_seed": 0,
        "thresholds": {method: 0.5 for method in detection_suite.METHOD_NAMES},
    }
    monkeypatch.setattr(
        pilot.detection_suite,
        "search_development_suite",
        lambda runs, candidates, master_seed: {
            "frozen_detection_config": config,
            "candidate_order": candidates,
            "master_seed": master_seed,
        },
    )
    monkeypatch.setattr(
        pilot.detection_suite,
        "evaluate_detection_suite",
        lambda runs, frozen, expected_split: _metrics("development"),
    )
    monkeypatch.setattr(
        pilot.detection_suite,
        "build_formal_v1_config",
        lambda runs, search, **kwargs: {"frozen_detection_config": config},
    )
    monkeypatch.setattr(
        pilot.detection_suite,
        "evaluate_formal_holdout",
        lambda runs, config, **kwargs: _metrics("holdout"),
    )


def _assert_immutable_tree(root: Path) -> None:
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    for path in root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        assert stat.S_IMODE(path.stat().st_mode) == expected


def _prepare_through_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    input_root = _make_input_root(tmp_path)
    protocol = pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    fake_runner = _fake_runner_provenance()
    monkeypatch.setattr(pilot, "_capture_runner_provenance", lambda: fake_runner)
    commit_dir = output_root / "commit"
    pilot.commit_inputs(
        input_root,
        commit_dir,
        protocol=protocol,
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_run_outputs(runs_root, input_root, "development")
    _patch_suite(monkeypatch)
    calibration_dir = output_root / "calibration"
    pilot.calibrate_development(input_root, commit_dir, runs_root, calibration_dir)
    return input_root, commit_dir, calibration_dir, runs_root


def test_commit_binds_exact_inputs_is_immutable_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    protocol = pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    output = output_root / "commit"

    pilot.commit_inputs(
        input_root,
        output,
        protocol=protocol,
    )

    manifest = json.loads((output / pilot.COMMITMENT_MANIFEST_FILENAME).read_bytes())
    assert manifest["input_registry"] == _artifact(
        (input_root / pilot.INPUT_REGISTRY_FILENAME).read_bytes(),
        pilot.INPUT_REGISTRY_FILENAME,
    )
    assert manifest["protocol"]["sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()
    _assert_immutable_tree(output)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="overwrite"):
        pilot.commit_inputs(
            input_root,
            output,
            protocol=protocol,
        )


def test_commit_hash_tamper_and_runner_state_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_capture = pilot._capture_runner_provenance
    input_root = _make_input_root(tmp_path)
    protocol = pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    output = output_root / "commit"
    pilot.commit_inputs(
        input_root,
        output,
        protocol=protocol,
    )
    split = output / pilot.SPLIT_REGISTRY_FILENAME
    split.chmod(0o644)
    split.write_bytes(split.read_bytes() + b" ")
    inputs = pilot._load_input_bundle(input_root)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="exact SHA-256"):
        pilot._load_commit_bundle(output, inputs)

    def dirty_git(repository: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository.resolve())
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[:2] == ("status", "--porcelain"):
            return "?? scripts/untracked_runner.py"
        return pilot.RUNNER_RELATIVE_PATH.as_posix()

    monkeypatch.setattr(pilot, "_git_output", dirty_git)
    monkeypatch.setattr(pilot, "_capture_runner_provenance", real_capture)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="must be clean"):
        pilot._capture_runner_provenance()


def test_commit_rejects_external_or_untracked_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    external = tmp_path / "permissive-protocol.md"
    external.write_text("holdout tuning allowed\n", encoding="utf-8")
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="fixed tracked"):
        pilot.commit_inputs(
            input_root,
            output_root / "external-protocol-commit",
            protocol=external,
        )

    monkeypatch.setattr(
        pilot,
        "_tracked_file_sha256_at_head",
        lambda repository, relative_path: "0" * 64,
    )
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="tracked HEAD blob"):
        pilot.commit_inputs(
            input_root,
            output_root / "tampered-protocol-commit",
            protocol=pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH,
        )


def test_commit_rejects_injected_report_option_and_cleans_failed_validation_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = pilot.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "commit",
                str(tmp_path / "inputs"),
                str(tmp_path / "commit"),
                "--protocol",
                str(pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH),
                "--validation-report",
                str(tmp_path / "handwritten-pass.json"),
            ]
        )

    input_root = _make_input_root(tmp_path)
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    monkeypatch.setattr(
        pilot,
        "_invoke_fixed_cpu_validator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pilot.FormalPilotOrchestrationError("injected validator failure")
        ),
    )
    output = output_root / "failed-commit"
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="injected"):
        pilot.commit_inputs(
            input_root,
            output,
            protocol=pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH,
        )
    assert not output.exists()
    assert list(output_root.glob(".failed-commit.cpu-validation.*")) == []


def test_cpu_report_contract_rejects_checks_tree_and_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    report_path = (
        output_root
        / "contract"
        / pilot.CPU_VALIDATION_DIRECTORY_NAME
        / pilot.CPU_VALIDATION_REPORT_FILENAME
    )
    _write_validation_report(input_root, report_path)
    valid_snapshot = pilot._read_snapshot(report_path, name="valid contract report")
    inputs = pilot._load_input_bundle(input_root)
    runner = _fake_runner_provenance()
    assert pilot._validate_cpu_validation_report(valid_snapshot, inputs, runner)

    mutations = []
    missing_check = json.loads(valid_snapshot.payload)
    del missing_check["checks"]["production_runtime_provenance"]
    mutations.append((missing_check, "check set"))
    bad_tree = json.loads(valid_snapshot.payload)
    bad_tree["input_tree_snapshot_entry_count"] = 23
    mutations.append((bad_tree, "24-entry"))
    bad_runtime = json.loads(valid_snapshot.payload)
    bad_runtime["production_runtime"]["python"]["executable"] = "/usr/bin/python"
    mutations.append((bad_runtime, "Python executable"))

    for payload, message in mutations:
        snapshot = pilot._Snapshot(report_path, _json_bytes(payload))
        with pytest.raises(pilot.FormalPilotOrchestrationError, match=message):
            pilot._validate_cpu_validation_report(snapshot, inputs, runner)


def test_real_subprocess_attestation_is_cpu_hidden_then_consumed_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_invoke = pilot._invoke_fixed_cpu_validator
    original_generator = pilot.CPU_VALIDATION_GENERATOR_RELATIVE_PATH
    input_root = _make_input_root(tmp_path)
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    template = (
        output_root
        / "template"
        / pilot.CPU_VALIDATION_DIRECTORY_NAME
        / pilot.CPU_VALIDATION_REPORT_FILENAME
    )
    _write_validation_report(input_root, template)
    report_payload = template.read_bytes()
    template.unlink()

    fake_generator = tmp_path / "fake-fixed-validator.py"
    fake_generator.write_text(
        "\n".join(
            (
                "import hashlib, json, os, pathlib, sys",
                "assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''",
                "assert os.environ.get('PYTHONNOUSERSITE') == '1'",
                "assert 'PYTHONPATH' not in os.environ",
                f"payload = bytes.fromhex({report_payload.hex()!r})",
                "output = pathlib.Path(sys.argv[2])",
                "output.write_bytes(payload)",
                "output.chmod(0o444)",
                "print(json.dumps({'status':'PASS','output_report':str(output),"
                "'sha256':hashlib.sha256(payload).hexdigest(),"
                "'size_bytes':len(payload)}, sort_keys=True))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pilot, "CPU_VALIDATION_GENERATOR_RELATIVE_PATH", fake_generator
    )
    report_path = (
        output_root
        / "real-subprocess"
        / pilot.CPU_VALIDATION_DIRECTORY_NAME
        / pilot.CPU_VALIDATION_REPORT_FILENAME
    )
    report_path.parent.mkdir(parents=True)

    snapshot = real_invoke(input_root, report_path)

    monkeypatch.setattr(
        pilot, "CPU_VALIDATION_GENERATOR_RELATIVE_PATH", original_generator
    )
    inputs = pilot._load_input_bundle(input_root)
    validated = pilot._validate_cpu_validation_report(
        snapshot, inputs, _fake_runner_provenance()
    )
    assert validated["status"] == "PASS"


def test_input_registry_rejects_path_traversal_aliases_and_symlink_escape(
    tmp_path: Path,
) -> None:
    traversal_root = _make_input_root(tmp_path / "traversal")
    registry_path = traversal_root / pilot.INPUT_REGISTRY_FILENAME
    registry = json.loads(registry_path.read_bytes())
    registry["runs"][0]["run_id"] = "../holdout-dynamic"
    registry["fixed_execution_order"]["development"][0] = "../holdout-dynamic"
    registry_path.write_bytes(_json_bytes(registry))
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="safe path component"):
        pilot._read_input_registry(traversal_root)

    alias_root = _make_input_root(tmp_path / "alias")
    alias_registry_path = alias_root / pilot.INPUT_REGISTRY_FILENAME
    alias_registry = json.loads(alias_registry_path.read_bytes())
    alias_registry["runs"][0]["run_id"] = "development-alias"
    alias_registry["fixed_execution_order"]["development"][0] = "development-alias"
    alias_registry_path.write_bytes(_json_bytes(alias_registry))
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="frozen v1 layout"):
        pilot._read_input_registry(alias_root)

    symlink_root = _make_input_root(tmp_path / "symlink")
    run_directory = symlink_root / "development" / "development-dynamic"
    escaped = tmp_path / "escaped-input-run"
    run_directory.rename(escaped)
    run_directory.symlink_to(escaped, target_is_directory=True)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="non-symlink"):
        pilot._load_input_bundle(symlink_root)


def test_calibrate_reads_development_only_and_rejects_runner_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    protocol = pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    commit_dir = output_root / "commit"
    pilot.commit_inputs(
        input_root,
        commit_dir,
        protocol=protocol,
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_run_outputs(runs_root, input_root, "development")
    _patch_suite(monkeypatch)

    calibration = output_root / "calibration"
    pilot.calibrate_development(input_root, commit_dir, runs_root, calibration)

    assert not any((runs_root / run_id).exists() for run_id in (
        "holdout-dynamic", "holdout-wrong", "holdout-low"
    ))
    _assert_immutable_tree(calibration)
    bad_run = runs_root / "development-dynamic" / "run.json"
    bad_run.chmod(0o644)
    payload = json.loads(bad_run.read_bytes())
    payload["runner"]["commit"] = "b" * 40
    bad_run.write_bytes(_json_bytes(payload))
    bad_run.chmod(0o444)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="runner.commit"):
        pilot.calibrate_development(
            input_root,
            commit_dir,
            runs_root,
            output_root / "bad-calibration",
        )


@pytest.mark.parametrize(
    ("run_id", "updates", "message"),
    [
        (
            "development-dynamic",
            {
                "started_at": "2026-08-01T23:59:00.000000+00:00",
                "finished_at": "2026-08-02T00:00:30.000000+00:00",
            },
            "before its immutable unlock",
        ),
        (
            "development-wrong",
            {
                "started_at": "2026-08-02T00:10:15.000000+00:00",
                "finished_at": "2026-08-02T00:11:00.000000+00:00",
            },
            "serial order",
        ),
        (
            "development-dynamic",
            {
                "started_at": "2026-08-02T00:10:00",
                "finished_at": "2026-08-02T00:10:30",
            },
            "timezone-aware",
        ),
    ],
)
def test_development_timeline_rejects_early_wrong_order_and_naive_iso(
    tmp_path: Path,
    run_id: str,
    updates: dict[str, object],
    message: str,
) -> None:
    input_root = _make_input_root(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_run_outputs(runs_root, input_root, "development")
    _rewrite_frozen_run_json(runs_root / run_id / "run.json", **updates)
    inputs = pilot._load_input_bundle(input_root)
    development = tuple(
        run for run in inputs.runs if run.dataset_split == "development"
    )
    outputs = pilot._snapshot_run_outputs(runs_root, development)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match=message):
        pilot._validated_split_run_timeline(
            development,
            outputs,
            split="development",
            earliest_start=datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc),
            latest_finish=datetime(2026, 8, 2, 3, 0, tzinfo=timezone.utc),
        )


def test_run_output_freeze_and_exact_five_files_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_run_outputs(runs_root, input_root, "holdout")
    inputs = pilot._load_input_bundle(input_root)
    holdout = tuple(run for run in inputs.runs if run.dataset_split == "holdout")
    dynamic_root = runs_root / "holdout-dynamic"
    dynamic_root.chmod(0o755)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="mode 0555"):
        pilot._snapshot_run_outputs(runs_root, holdout)
    dynamic_root.chmod(0o555)

    missing = dynamic_root / "checkpoint-load-audit.json"
    dynamic_root.chmod(0o755)
    missing.unlink()
    dynamic_root.chmod(0o555)
    reads: list[Path] = []

    def forbidden_read(path: Path, *, name: str) -> pilot._Snapshot:
        reads.append(path)
        raise AssertionError("preflight must not read holdout bytes")

    monkeypatch.setattr(pilot, "_read_snapshot", forbidden_read)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="before any holdout"):
        pilot._preflight_holdout_outputs(
            runs_root, [run.run_id for run in holdout]
        )
    assert reads == []


def test_formal_stage_publication_rejects_output_outside_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(pilot, "OUTPUT_ROOT", output_root)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="inside"):
        pilot._publish_immutable_tree(
            tmp_path / "outside-stage", {"artifact.json": b"{}"}
        )


def test_calibration_unlock_failure_rolls_back_new_calibration_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = _make_input_root(tmp_path)
    output_root = _install_fake_cpu_validator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pilot, "_capture_runner_provenance", lambda: _fake_runner_provenance()
    )
    commit_dir = output_root / "commit"
    pilot.commit_inputs(
        input_root,
        commit_dir,
        protocol=pilot.REPOSITORY_ROOT / pilot.PROTOCOL_RELATIVE_PATH,
    )
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _write_run_outputs(runs_root, input_root, "development")
    _patch_suite(monkeypatch)
    calibration = output_root / "calibration"
    monkeypatch.setattr(
        pilot,
        "_publish_holdout_unlock",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pilot.FormalPilotOrchestrationError("injected unlock failure")
        ),
    )
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="injected"):
        pilot.calibrate_development(
            input_root, commit_dir, runs_root, calibration
        )
    assert not calibration.exists()
    assert not pilot._holdout_unlock_directory(calibration).exists()


def test_holdout_timeline_rejects_run_started_before_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root, commit_dir, calibration, runs_root = _prepare_through_calibration(
        tmp_path, monkeypatch
    )
    _write_run_outputs(runs_root, input_root, "holdout")
    _rewrite_frozen_run_json(
        runs_root / "holdout-dynamic" / "run.json",
        started_at="2026-08-02T00:59:00.000000+00:00",
        finished_at="2026-08-02T01:00:30.000000+00:00",
    )
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="immutable unlock"):
        pilot.evaluate_holdout(
            input_root,
            commit_dir,
            calibration,
            runs_root,
            commit_dir.parent / "early-holdout-evaluation",
        )


def test_evaluate_incomplete_holdout_reads_zero_holdout_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root, commit_dir, calibration, runs_root = _prepare_through_calibration(
        tmp_path, monkeypatch
    )
    _write_run_outputs(runs_root, input_root, "holdout")
    missing = runs_root / "holdout-low" / "run.json"
    missing.parent.chmod(0o755)
    missing.unlink()
    original = pilot._read_snapshot
    holdout_reads: list[Path] = []

    def tracking(path: Path, *, name: str):
        if path.parent.name.startswith("holdout-"):
            holdout_reads.append(path)
        return original(path, name=name)

    monkeypatch.setattr(pilot, "_read_snapshot", tracking)
    with pytest.raises(pilot.FormalPilotOrchestrationError, match="before any holdout"):
        pilot.evaluate_holdout(
            input_root,
            commit_dir,
            calibration,
            runs_root,
            commit_dir.parent / "evaluation",
        )
    assert holdout_reads == []


def test_evaluate_publishes_gate_runtime_and_six_compatible_timelines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root, commit_dir, calibration, runs_root = _prepare_through_calibration(
        tmp_path, monkeypatch
    )
    _write_run_outputs(runs_root, input_root, "holdout")
    output = commit_dir.parent / "evaluation"

    result = pilot.evaluate_holdout(
        input_root,
        commit_dir,
        calibration,
        runs_root,
        output,
    )

    assert result["decision"] == "GO"
    gate = json.loads((output / pilot.GO_NO_GO_FILENAME).read_bytes())
    assert gate["passed"] is True
    assert len(gate["checks"]) == 7
    assert len(list((output / pilot.TIMELINE_DIRECTORY).glob("*.svg"))) == 6
    assert json.loads((output / pilot.RUNTIME_SUMMARY_FILENAME).read_bytes())[
        "run_count"
    ] == 6
    _assert_immutable_tree(output)


def test_evaluate_rejects_calibration_artifact_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root, commit_dir, calibration, runs_root = _prepare_through_calibration(
        tmp_path, monkeypatch
    )
    _write_run_outputs(runs_root, input_root, "holdout")
    metrics = calibration / pilot.DEVELOPMENT_METRICS_FILENAME
    metrics.chmod(0o644)
    metrics.write_bytes(metrics.read_bytes() + b" ")

    with pytest.raises(pilot.FormalPilotOrchestrationError, match="exact SHA-256"):
        pilot.evaluate_holdout(
            input_root,
            commit_dir,
            calibration,
            runs_root,
            commit_dir.parent / "evaluation",
        )


def test_evaluate_rejects_self_consistent_formal_config_and_manifest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root, commit_dir, calibration, runs_root = _prepare_through_calibration(
        tmp_path, monkeypatch
    )
    _write_run_outputs(runs_root, input_root, "holdout")
    config_path = calibration / pilot.FORMAL_CONFIG_FILENAME
    config_path.chmod(0o644)
    config = json.loads(config_path.read_bytes())
    config["frozen_detection_config"]["thresholds"]["combined"] = 0.123
    config_bytes = _json_bytes(config)
    config_path.write_bytes(config_bytes)
    manifest_path = calibration / pilot.CALIBRATION_MANIFEST_FILENAME
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["formal_config"] = _artifact(
        config_bytes, pilot.FORMAL_CONFIG_FILENAME
    )
    manifest_bytes = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    unlock_path = (
        pilot._holdout_unlock_directory(calibration)
        / pilot.HOLDOUT_UNLOCK_FILENAME
    )
    unlock_path.chmod(0o644)
    unlock = json.loads(unlock_path.read_bytes())
    unlock["calibration_manifest"] = _artifact(
        manifest_bytes, str(manifest_path)
    )
    unlock_path.write_bytes(_json_bytes(unlock))
    unlock_path.chmod(0o444)

    with pytest.raises(
        pilot.FormalPilotOrchestrationError,
        match="formal_config differs from deterministic replay",
    ):
        pilot.evaluate_holdout(
            input_root,
            commit_dir,
            calibration,
            runs_root,
            commit_dir.parent / "evaluation",
        )


def test_all_seven_gate_conditions_and_registered_boundaries() -> None:
    development = _metrics("development")
    holdout = _metrics("holdout")

    passing = pilot.evaluate_go_no_go(
        development, holdout, provenance_passed=True
    )

    assert passing["decision"] == "GO"
    assert all(check["passed"] for check in passing["checks"])

    failures: list[tuple[dict[str, object], dict[str, object], bool, str]] = []
    changed = copy.deepcopy(holdout)
    changed["methods"]["combined"]["macro"]["auroc"] = 0.75
    failures.append((development, changed, True, "combined_macro_auroc"))
    changed = copy.deepcopy(holdout)
    changed["methods"]["random"]["macro"]["auroc"] = 0.7500001
    failures.append((development, changed, True, "combined_minus_random"))
    changed = copy.deepcopy(holdout)
    changed["methods"]["update_magnitude_only"]["macro"]["auroc"] = 0.88
    failures.append((development, changed, True, "combined_within_0_02"))
    changed_dev = copy.deepcopy(development)
    for interval in changed_dev["methods"]["combined"]["events"]["intervals"][1:]:
        interval["delay_frames"] = None
    failures.append((changed_dev, holdout, True, "same_at_least_two"))
    changed = copy.deepcopy(holdout)
    changed["methods"]["combined"]["pooled"]["false_positive_rate"] = 0.200001
    failures.append((development, changed, True, "false_positive_rate"))
    changed = copy.deepcopy(holdout)
    changed["methods"]["combined"]["diagnostics"][
        "max_per_run_consecutive_false_positives"
    ] = 4
    failures.append((development, changed, True, "false_positive_streak"))
    failures.append((development, holdout, False, "provenance_runtime"))

    for dev_metrics, holdout_metrics, provenance, failed_id in failures:
        result = pilot.evaluate_go_no_go(
            dev_metrics, holdout_metrics, provenance_passed=provenance
        )
        failed = [check["id"] for check in result["checks"] if not check["passed"]]
        assert result["decision"] == "NO-GO"
        assert any(failed_id in check_id for check_id in failed)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_strict_json_rejects_duplicate_and_nonfinite(payload: bytes) -> None:
    with pytest.raises(pilot.FormalPilotOrchestrationError):
        pilot._strict_json_loads(payload, name="tampered")
