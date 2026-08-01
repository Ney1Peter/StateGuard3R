from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from decimal import Decimal
from functools import lru_cache

import numpy as np
import pytest

from stateguard3r.detection import compute_baseline_scores
from stateguard3r.detection_suite import (
    HYPERPARAMETER_SELECTION_RULE,
    METHOD_NAMES,
    THRESHOLD_SELECTION_RULE,
    DetectionSuiteInput,
    DetectionSuiteRun,
    build_evaluation_regions,
    build_formal_v1_config,
    derive_run_seed,
    evaluate_detection_suite,
    evaluate_formal_holdout,
    fixed_candidate_order,
    make_holdout_commitment,
    make_run_commitment,
    make_split_registry,
    search_development_suite,
)


EVENT_ENDS = {
    "low_overlap_jump": 19,
    "dynamic_occlusion": 19,
    "wrong_order_segment": 18,
}
RECTANGLE_REFERENCE = "model_input_after_resize_and_center_crop"


def _dynamic_rectangles() -> list[dict[str, float]]:
    return [
        {
            "x": round(0.25 + offset * 0.04, 8),
            "y": round(0.25 + offset * 0.025, 8),
            "width": 0.5,
            "height": 0.5,
        }
        for offset in range(5)
    ]


def _formal_parameters(corruption_type: str) -> dict[str, object]:
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
            "frame_rectangles": _dynamic_rectangles(),
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


def _formal_transform(
    corruption_type: str, position: int, source_index: int
) -> dict[str, object]:
    if corruption_type == "dynamic_occlusion":
        return {
            "type": "rectangle_occlusion",
            "coordinate_space": "normalized",
            "coordinate_reference": RECTANGLE_REFERENCE,
            "rectangle": _dynamic_rectangles()[position - 15],
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


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _raw_hash(namespace: str, index: int) -> str:
    return hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()


def _make_run(
    run_id: str,
    dataset_split: str,
    corruption_type: str,
    *,
    update_values: list[float] | None = None,
    reliability_values: list[float] | None = None,
    raw_namespace: str | None = None,
    timestamp_origin: str = "1000.0",
) -> DetectionSuiteRun:
    end = EVENT_ENDS[corruption_type]
    if update_values is None:
        update_values = [1.0] * 30
        for position in range(15, end + 1):
            update_values[position] = 10.0
    assert len(update_values) == 30
    if reliability_values is None:
        reliability_values = [
            0.1 if 15 <= position <= end else 0.9 for position in range(30)
        ]
    assert len(reliability_values) == 30
    namespace = raw_namespace or run_id
    starts = {
        ("development", "dynamic_occlusion"): (248, None),
        ("development", "wrong_order_segment"): (320, None),
        ("development", "low_overlap_jump"): (380, 300),
        ("holdout", "dynamic_occlusion"): (520, None),
        ("holdout", "wrong_order_segment"): (550, None),
        ("holdout", "low_overlap_jump"): (583, 500),
    }
    base_start, donor_start = starts[(dataset_split, corruption_type)]
    source_count = 35 if corruption_type == "low_overlap_jump" else 30
    raw_indices = list(range(base_start, base_start + 30))
    if donor_start is not None:
        raw_indices.extend(range(donor_start, donor_start + 5))
    source_shas = tuple(
        _raw_hash(namespace, position) for position in range(source_count)
    )
    if corruption_type == "low_overlap_jump":
        source_indices = [*range(15), *range(30, 35), *range(20, 30)]
    elif corruption_type == "wrong_order_segment":
        source_indices = [*range(15), 18, 17, 16, 15, *range(19, 30)]
    else:
        source_indices = list(range(30))
    output_shas = tuple(source_shas[index] for index in source_indices)
    raw_shas = tuple(sorted(output_shas))
    source_frames = []
    for pool_index, raw_index in enumerate(raw_indices):
        timestamp_decimal = Decimal(timestamp_origin) + Decimal(raw_index) * Decimal(
            "0.03"
        )
        depth_timestamp_decimal = timestamp_decimal + Decimal("0.005")
        gt_timestamp_decimal = timestamp_decimal + Decimal("0.007")
        timestamp_text = f"{timestamp_decimal:.6f}"
        depth_timestamp_text = f"{depth_timestamp_decimal:.6f}"
        gt_timestamp_text = f"{gt_timestamp_decimal:.6f}"
        timestamp = float(timestamp_text)
        depth_timestamp = float(depth_timestamp_text)
        gt_timestamp = float(gt_timestamp_text)
        source_frames.append(
            {
                "path": f"../../../../raw/{namespace}/rgb/{raw_index}.png",
                "source_entry_index": raw_index,
                "source_line": raw_index + 2,
                "physical_line_sha256": _raw_hash("rgb-line", raw_index),
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
                    "physical_line_sha256": _raw_hash("depth-line", raw_index),
                    "physical_line_size_bytes": 80,
                    "timestamp": depth_timestamp,
                    "timestamp_text": depth_timestamp_text,
                    "path": f"../../../../raw/{namespace}/depth/{raw_index}.png",
                    "sha256": _raw_hash(f"{namespace}-depth", raw_index),
                    "size_bytes": 2000 + pool_index,
                    "mode_octal": "0444",
                    "link_count": 1,
                    "device": 11,
                    "inode": 200_000 + raw_index,
                    "mtime_ns": 2_000_000 + raw_index,
                    "delta_from_rgb_seconds": 0.005,
                    "absolute_delta_seconds": 0.005,
                },
                "groundtruth": {
                    "source_entry_index": 20_000 + raw_index,
                    "source_line": 20_002 + raw_index,
                    "physical_line_sha256": _raw_hash("gt-line", raw_index),
                    "physical_line_size_bytes": 120,
                    "timestamp": gt_timestamp,
                    "timestamp_text": gt_timestamp_text,
                    "translation_xyz": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "translation_xyz_text": ["0", "0", "0"],
                    "quaternion_xyzw_text": ["0", "0", "0", "1"],
                    "delta_from_rgb_seconds": 0.007,
                    "absolute_delta_seconds": 0.007,
                },
                "source_pool_index": pool_index,
                "source_pool_role": (
                    "base" if pool_index < 30 else "low_overlap_donor"
                ),
            }
        )

    def snapshot(path: str, digest: str, size: int, inode: int, *, count=None):
        value = {
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

    gt_file_sha = _raw_hash("groundtruth-file", 0)
    source_manifest = {
        "schema_version": "stateguard3r.tum-formal-pilot-source.v1",
        "sequence": f"{run_id}-source-pool",
        "dataset": "rgbd_dataset_freiburg1_desk",
        "dataset_split": dataset_split,
        "run_id": run_id,
        "corruption_type": corruption_type,
        "source_is_read_only": True,
        "file_path_semantics": "relative_to_this_manifest_directory",
        "index_convention": "zero_based_data_entry_index_excluding_comments_and_blank_lines",
        "physical_line_convention": "one_based_physical_source_line",
        "association_policy": {
            "method": "unique_nearest_absolute_timestamp",
            "tie_policy": "reject",
            "reuse_policy": "reject_within_source_pool_and_across_consumed_runs",
            "max_absolute_delta_seconds": 0.02,
        },
        "frame_count": source_count,
        "output_frame_count": 30,
        "ground_truth_path": "../../../../raw/groundtruth.txt",
        "tum_index_files": {
            "rgb": snapshot("../../../../raw/rgb.txt", _raw_hash("rgb-index", 0), 100, 1, count=1000),
            "depth": snapshot("../../../../raw/depth.txt", _raw_hash("depth-index", 0), 100, 2, count=1000),
            "groundtruth": snapshot("../../../../raw/groundtruth.txt", gt_file_sha, 100, 3, count=1000),
        },
        "official_lineage": {
            "dataset_root": "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk",
            "archive": snapshot(
                "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk.tgz",
                "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
                344011403,
                4,
            ),
            "raw_manifest": snapshot(
                "/data/wangzheng/Project2/baselines/ReCal3R/logs/gate2-fr1-desk-raw-manifest.json",
                "5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87",
                231139,
                5,
            ),
            "raw_manifest_schema_version": "stateguard3r.tum-raw-audit.v1",
        },
        "frames": source_frames,
    }
    source_bytes = _json_bytes(source_manifest)
    corruption_interval = {
        "type": corruption_type,
        "start_frame": 15,
        "end_frame": end,
        "start": 15,
        "end": end,
        "parameters": _formal_parameters(corruption_type),
        "expected_effect": "formal-test-fixture",
    }
    input_manifest = {
        "schema_version": "stateguard3r.corruption.v1",
        "frame_count": 30,
        "sequence": run_id,
        "source_sequence": f"{run_id}-source-pool",
        "source_manifest": "source-manifest.json",
        "source_frame_count": source_count,
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_ground_truth_path": "../../../../raw/groundtruth.txt",
        "source_ground_truth_resolved_path": "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt",
        "source_ground_truth_sha256": gt_file_sha,
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
                    [_formal_transform(corruption_type, position, source_index)]
                    if 15 <= position <= end
                    else []
                ),
            }
            for position, source_index in enumerate(source_indices)
        ],
        "corruptions": [corruption_interval],
    }
    input_bytes = _json_bytes(input_manifest)
    run_dir = f"/formal/{dataset_split}/{run_id}"
    resolved_paths = [
        os.path.realpath(os.path.join(run_dir, source_frames[index]["path"]))
        for index in source_indices
    ]
    records = []
    for position, source_index in enumerate(source_indices):
        record = {
            "frame_id": position,
            "timestamp": source_frames[source_index]["timestamp"],
        }
        if position > 0:
            record["global_state_delta"] = float(update_values[position])
            record["uncertainty_u"] = 1.0 - float(reliability_values[position])
        records.append(record)
    run_json = {
        "schema_version": "stateguard3r.recal3r-smoke.v0",
        "status": "succeeded",
        "baseline_commit": "466c7cdf3acd2f589f1d82e5f6391966f19db9ff",
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
        "runtime_seconds": 1.0,
        "peak_memory_allocated_mib": 1024.0,
        "runner": {
            "commit": "a" * 40,
            "script_sha256": (
                "091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e"
            ),
        },
        "trace_frame_steps": list(range(1, 30)),
        "health_signal_semantics": {
            "reliability": "1 - ReCal3R uncertainty_u",
            "update_magnitude_source": "global_state_delta",
            "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)",
            "geometric_residual": "relative_median_independent_cross_head_pointmap_consistency",
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
            for position, frame in enumerate(input_manifest["frames"])
        ],
        "input_manifest": {
            "path": f"{run_dir}/input-manifest.json",
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
            "source_manifest": f"{run_dir}/source-manifest.json",
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "materialization": "final_manifest_order_with_in_memory_deferred_transforms",
            "rectangle_coordinate_reference": (
                RECTANGLE_REFERENCE if corruption_type == "dynamic_occlusion" else None
            ),
            "transform_counts": {
                {
                    "dynamic_occlusion": "rectangle_occlusion",
                    "low_overlap_jump": "source_frame_substitution",
                    "wrong_order_segment": "temporal_reorder",
                }[corruption_type]: end - 14
            },
        },
    }
    health = b"".join(_json_bytes(record) + b"\n" for record in records)
    return DetectionSuiteRun(
        run_id=run_id,
        dataset_split=dataset_split,
        corruption_type=corruption_type,
        raw_frame_sha256s=raw_shas,
        health_jsonl=health,
        corruption_json=input_bytes,
        input_manifest_json=input_bytes,
        source_manifest_json=source_bytes,
        run_json=_json_bytes(run_json),
    )


def _make_split(prefix: str, dataset_split: str) -> list[DetectionSuiteRun]:
    return [
        _make_run(
            f"{dataset_split}-dynamic",
            dataset_split,
            "dynamic_occlusion",
            raw_namespace=f"{prefix}-dynamic",
        ),
        _make_run(
            f"{dataset_split}-wrong",
            dataset_split,
            "wrong_order_segment",
            raw_namespace=f"{prefix}-wrong",
        ),
        _make_run(
            f"{dataset_split}-low",
            dataset_split,
            "low_overlap_jump",
            raw_namespace=f"{prefix}-low",
        ),
    ]


def _replace_source_pool_sha(
    run: DetectionSuiteRun, *, source_pool_position: int, sha256: str
) -> DetectionSuiteRun:
    source_manifest = json.loads(run.source_manifest_json)
    source_manifest["frames"][source_pool_position]["rgb_sha256"] = sha256
    return _replace_source_manifest(run, source_manifest)


def _replace_source_manifest(
    run: DetectionSuiteRun, source_manifest: dict[str, object]
) -> DetectionSuiteRun:
    source_bytes = _json_bytes(source_manifest)
    input_manifest = json.loads(run.input_manifest_json)
    input_manifest["source_manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    input_bytes = _json_bytes(input_manifest)
    return replace(
        run,
        source_manifest_json=source_bytes,
        input_manifest_json=input_bytes,
        corruption_json=input_bytes,
    )


@lru_cache(maxsize=1)
def _formal_fixture():
    development = _make_split("dev", "development")
    holdout = _make_split("holdout", "holdout")
    dev_commitments = [make_run_commitment(run) for run in development]
    holdout_commitments = [make_run_commitment(run) for run in holdout]
    registry = make_split_registry(dev_commitments, holdout_commitments)
    commitment = make_holdout_commitment(holdout_commitments)
    search = search_development_suite(
        development, fixed_candidate_order(), master_seed=0
    )
    config = build_formal_v1_config(
        development,
        search,
        development_run_id="DEV-SUITE-0001",
        split_registry=registry,
        holdout_commitment=commitment,
    )
    return development, holdout, registry, commitment, search, config


@pytest.mark.parametrize(
    ("corruption_type", "end", "expected"),
    [
        ("low_overlap_jump", 19, (5, 5, 20)),
        ("dynamic_occlusion", 19, (5, 5, 20)),
        ("wrong_order_segment", 18, (4, 5, 21)),
    ],
)
def test_fixed_evaluation_region_counts(
    corruption_type: str, end: int, expected: tuple[int, int, int]
) -> None:
    regions = build_evaluation_regions(
        {
            "corruptions": [
                {"type": corruption_type, "start": 15, "end": end}
            ]
        }
    )

    assert (
        regions.positive_frames,
        regions.washout_frames,
        regions.negative_frames,
    ) == expected
    assert np.flatnonzero(regions.recovery_boundary_mask).tolist() == [end + 1]
    assert np.flatnonzero(regions.washout_mask).tolist() == list(
        range(end + 1, end + 6)
    )
    assert np.all(regions.evaluation_mask[:15])


@pytest.mark.parametrize(
    ("corruption_type", "wrong_end"),
    [
        ("dynamic_occlusion", 18),
        ("low_overlap_jump", 18),
        ("wrong_order_segment", 19),
    ],
)
def test_formal_regions_reject_non_registered_interval_length(
    corruption_type: str, wrong_end: int
) -> None:
    with pytest.raises(ValueError, match=r"formal .* end must equal"):
        build_evaluation_regions(
            {
                "corruptions": [
                    {"type": corruption_type, "start": 15, "end": wrong_end}
                ]
            }
        )


@pytest.mark.parametrize(
    ("corruption_type", "field_path", "tampered_value"),
    [
        ("dynamic_occlusion", ("initial_rectangle", "x"), 0.2),
        ("dynamic_occlusion", ("velocity", "dx"), 0.03),
        ("dynamic_occlusion", ("fill", 0), 0),
        (
            "dynamic_occlusion",
            ("coordinate_reference",),
            "source_image_before_model_preprocessing",
        ),
        ("dynamic_occlusion", ("frame_rectangles", 2, "x"), 0.99),
        ("low_overlap_jump", ("source_start",), 29),
        ("low_overlap_jump", ("source_end",), 33),
        ("low_overlap_jump", ("source_indices",), [29, 30, 31, 32, 33]),
        (
            "low_overlap_jump",
            ("selection",),
            "seeded_farthest_disjoint_index_proxy",
        ),
        ("wrong_order_segment", ("mode",), "shuffle"),
        ("wrong_order_segment", ("permutation",), [15, 16, 17, 18]),
    ],
)
def test_pre_forward_commitment_rejects_tampered_formal_parameters(
    corruption_type: str,
    field_path: tuple[str | int, ...],
    tampered_value: object,
) -> None:
    source = _make_run(f"tampered-{corruption_type}", "holdout", corruption_type)
    corruption = json.loads(source.corruption_json)
    parameters = corruption["corruptions"][0]["parameters"]
    target = parameters
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = tampered_value
    input_manifest = json.loads(source.input_manifest_json)
    input_manifest["corruptions"] = copy.deepcopy(corruption["corruptions"])
    pre_forward = DetectionSuiteInput(
        run_id=source.run_id,
        dataset_split=source.dataset_split,
        corruption_type=source.corruption_type,
        raw_frame_sha256s=source.raw_frame_sha256s,
        corruption_json=_json_bytes(corruption),
        input_manifest_json=_json_bytes(input_manifest),
        source_manifest_json=source.source_manifest_json,
    )

    with pytest.raises(ValueError, match="parameters do not match frozen v1 severity"):
        make_run_commitment(pre_forward)


@pytest.mark.parametrize(
    ("corruption_type", "field_path", "tampered_value"),
    [
        ("dynamic_occlusion", ("rectangle", "x"), 0.2),
        ("low_overlap_jump", ("replacement_source_index",), 31),
        ("wrong_order_segment", ("replacement_source_index",), 17),
    ],
)
def test_pre_forward_commitment_rejects_tampered_per_frame_transform(
    corruption_type: str,
    field_path: tuple[str, ...],
    tampered_value: object,
) -> None:
    source = _make_run(
        f"tampered-transform-{corruption_type}", "holdout", corruption_type
    )
    input_manifest = json.loads(source.input_manifest_json)
    transform = input_manifest["frames"][15]["transforms"][0]
    target = transform
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = tampered_value
    pre_forward = DetectionSuiteInput(
        run_id=source.run_id,
        dataset_split=source.dataset_split,
        corruption_type=source.corruption_type,
        raw_frame_sha256s=source.raw_frame_sha256s,
        corruption_json=_json_bytes(input_manifest),
        input_manifest_json=_json_bytes(input_manifest),
        source_manifest_json=source.source_manifest_json,
    )

    with pytest.raises(ValueError, match="frozen formal v1 transform"):
        make_run_commitment(pre_forward)


def test_recovery_boundary_alarm_cannot_rescue_a_missed_event() -> None:
    values = [1.0] * 30
    values[20] = 20.0
    run = _make_run(
        "boundary-only",
        "development",
        "dynamic_occlusion",
        update_values=values,
    )
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 0,
        "thresholds": {method: 5.0 for method in METHOD_NAMES},
    }

    result = evaluate_detection_suite(
        [run], config, expected_split="development"
    )
    update = result["methods"]["update_magnitude_only"]

    assert update["events"]["total_events"] == 1
    assert update["events"]["detected_events"] == 0
    assert update["events"]["missed_events"] == 1
    assert update["events"]["mean_delay_frames"] is None
    assert update["per_run"]["boundary-only"]["diagnostics"][
        "recovery_boundary_alarm"
    ] is True
    assert update["pooled"]["false_positive"] == 0
    assert update["diagnostics"]["washout_alarm_frames"] == 1
    assert update["exact_interval_sensitivity"]["pooled"]["false_positive"] == 1


def test_suite_rejects_nonfinite_score_in_washout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _make_run("washout-nan", "development", "dynamic_occlusion")

    def score_with_washout_nan(*args, **kwargs):
        payload = compute_baseline_scores(*args, **kwargs)
        combined = np.asarray(payload["combined"]["scores"], dtype=np.float64).copy()
        combined[20] = np.nan
        payload["combined"]["scores"] = combined
        return payload

    monkeypatch.setattr(
        "stateguard3r.detection_suite.compute_baseline_scores",
        score_with_washout_nan,
    )
    config = {
        "window": 5,
        "epsilon": 1e-6,
        "max_z": 10.0,
        "master_seed": 0,
        "thresholds": {method: 0.5 for method in METHOD_NAMES},
    }

    with pytest.raises(ValueError, match="all 30 positions"):
        evaluate_detection_suite(
            [run], config, expected_split="development"
        )


def test_primary_metrics_report_longest_consecutive_false_positive_run() -> None:
    reliability = [0.9] * 30
    reliability[5:8] = [0.1, 0.1, 0.1]
    run = _make_run(
        "fp-streak",
        "development",
        "dynamic_occlusion",
        reliability_values=reliability,
    )
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 0,
        "thresholds": {method: -0.5 for method in METHOD_NAMES},
    }

    result = evaluate_detection_suite([run], config)
    reliability_result = result["methods"]["reliability_only"]

    assert reliability_result["per_run"]["fp-streak"]["metrics"][
        "longest_consecutive_false_positives"
    ] == 3
    assert reliability_result["diagnostics"][
        "max_per_run_consecutive_false_positives"
    ] == 3


def test_each_run_builds_its_own_causal_reference() -> None:
    first_values = [0.0] * 29 + [100.0]
    second_values = [1.0, 2.0] + [1.0] * 28
    first = _make_run(
        "a-run",
        "development",
        "dynamic_occlusion",
        update_values=first_values,
    )
    second = _make_run(
        "b-run",
        "development",
        "low_overlap_jump",
        update_values=second_values,
    )
    config = {
        "window": 3,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 9,
        "thresholds": {method: 1_000.0 for method in METHOD_NAMES},
    }

    suite = evaluate_detection_suite(
        [first, second], config, expected_split="development"
    )
    suite_second = np.asarray(
        suite["methods"]["update_magnitude_only"]["per_run"]["b-run"]["scores"]
    )
    records = [json.loads(line) for line in second.health_jsonl.splitlines()]
    independent = compute_baseline_scores(
        records, window=3, epsilon=1.0, max_z=None
    )["update_magnitude_only"]["scores"]
    concatenated_records = [
        json.loads(line)
        for line in first.health_jsonl.splitlines()
        + second.health_jsonl.splitlines()
    ]
    leaked = compute_baseline_scores(
        concatenated_records, window=3, epsilon=1.0, max_z=None
    )["update_magnitude_only"]["scores"][30:]

    np.testing.assert_allclose(suite_second, independent)
    assert suite_second[0] == pytest.approx(0.0)
    assert suite_second[1] == pytest.approx(0.0)
    assert leaked[1] != pytest.approx(suite_second[1])


def test_run_id_random_seeds_are_distinct_and_reproducible() -> None:
    first = _make_run("random-a", "development", "dynamic_occlusion")
    second = _make_run("random-b", "development", "low_overlap_jump")
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": 10.0,
        "master_seed": 123,
        "thresholds": {method: 0.5 for method in METHOD_NAMES},
    }

    one = evaluate_detection_suite([first, second], config)
    two = evaluate_detection_suite([first, second], config)
    first_scores = one["methods"]["random"]["per_run"]["random-a"]["scores"]
    second_scores = one["methods"]["random"]["per_run"]["random-b"]["scores"]

    assert derive_run_seed(123, "development", "random-a") == derive_run_seed(
        123, "development", "random-a"
    )
    assert derive_run_seed(123, "development", "random-a") != derive_run_seed(
        123, "development", "random-b"
    )
    assert derive_run_seed(123, "development", "random-a") != derive_run_seed(
        123, "holdout", "random-a"
    )
    assert first_scores != second_scores
    assert one["methods"]["random"]["per_run"]["random-a"]["scores"] == two[
        "methods"
    ]["random"]["per_run"]["random-a"]["scores"]


def test_development_search_is_deterministic_and_uses_fixed_tie_order() -> None:
    development = _make_split("search-dev", "development")
    candidates = [
        {"window": 5, "epsilon": 1.0, "max_z": 10.0},
        {"window": 10, "epsilon": 1.0, "max_z": 10.0},
    ]

    first = search_development_suite(development, candidates, master_seed=4)
    second = search_development_suite(development, candidates, master_seed=4)

    assert first == second
    assert first["selected_candidate_index"] == 1
    assert set(first["frozen_detection_config"]["thresholds"]) == set(METHOD_NAMES)
    assert first["threshold_selection_rule"] == THRESHOLD_SELECTION_RULE
    assert first["hyperparameter_selection_rule"] == HYPERPARAMETER_SELECTION_RULE


def test_fixed_candidate_order_is_the_exact_fresh_27_item_preregistered_grid() -> None:
    expected = [
        {"window": window, "epsilon": epsilon, "max_z": max_z}
        for window in (5, 10, 15)
        for epsilon in (1e-6, 1e-5, 1e-4)
        for max_z in (10.0, 20.0, None)
    ]

    first = fixed_candidate_order()

    assert first == expected
    assert len(first) == 27
    first[0]["window"] = 1
    assert fixed_candidate_order() == expected


def test_formal_config_builder_rejects_changed_grid_and_nonzero_seed() -> None:
    development, _, registry, commitment, search, _ = _formal_fixture()

    reduced = copy.deepcopy(search)
    selected = reduced["selected_candidate_index"]
    removal = 0 if selected != 0 else 1
    del reduced["candidate_order"][removal]
    del reduced["candidates"][removal]
    if removal < selected:
        reduced["selected_candidate_index"] -= 1

    expanded = copy.deepcopy(search)
    expanded["candidate_order"].append(
        {"window": 1, "epsilon": 1e-6, "max_z": 10.0}
    )
    expanded["candidates"].append(copy.deepcopy(expanded["candidates"][-1]))

    reordered = copy.deepcopy(search)
    movable = [
        index
        for index in range(len(reordered["candidate_order"]))
        if index != reordered["selected_candidate_index"]
    ][:2]
    left, right = movable
    reordered["candidate_order"][left], reordered["candidate_order"][right] = (
        reordered["candidate_order"][right],
        reordered["candidate_order"][left],
    )
    reordered["candidates"][left], reordered["candidates"][right] = (
        reordered["candidates"][right],
        reordered["candidates"][left],
    )

    for changed in (reduced, expanded, reordered):
        with pytest.raises(ValueError, match="fixed 27-candidate formal v1 grid"):
            build_formal_v1_config(
                development,
                changed,
                development_run_id="DEV-SUITE-0001",
                split_registry=registry,
                holdout_commitment=commitment,
            )

    changed_seed = copy.deepcopy(search)
    changed_seed["master_seed"] = 1
    changed_seed["frozen_detection_config"]["master_seed"] = 1
    with pytest.raises(ValueError, match="search_result master_seed must equal 0"):
        build_formal_v1_config(
            development,
            changed_seed,
            development_run_id="DEV-SUITE-0001",
            split_registry=registry,
            holdout_commitment=commitment,
        )


def test_formal_config_reader_rejects_changed_grid_and_nonzero_seed() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()
    changed_grid = copy.deepcopy(config)
    changed_grid["search_provenance"]["candidate_order"].pop()

    with pytest.raises(ValueError, match="fixed 27-candidate formal v1 grid"):
        evaluate_formal_holdout(
            holdout,
            changed_grid,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )

    changed_seed = copy.deepcopy(config)
    changed_seed["search_provenance"]["master_seed"] = 1
    changed_seed["frozen_detection_config"]["master_seed"] = 1
    with pytest.raises(
        ValueError, match="formal_config search_provenance master_seed must equal 0"
    ):
        evaluate_formal_holdout(
            holdout,
            changed_seed,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )


def test_formal_config_reader_rejects_threshold_tamper_without_commitment_update() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()
    changed = copy.deepcopy(config)
    changed["frozen_detection_config"]["thresholds"]["combined"] += 1.0

    with pytest.raises(
        ValueError, match="frozen detection config canonical SHA mismatch"
    ):
        evaluate_formal_holdout(
            holdout,
            changed,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )


def test_formal_suite_binds_provenance_and_reports_pooled_denominators() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()

    result = evaluate_formal_holdout(
        holdout,
        config,
        split_registry=registry,
        holdout_commitment=commitment,
        search_result=search,
        development_runs=development,
        runtime_detection_config=config["frozen_detection_config"],
    )

    assert result["formal"] is True
    assert result["dataset_split"] == "holdout"
    assert [
        (run["positive_frames"], run["washout_frames"], run["negative_frames"])
        for run in result["runs"]
    ] == [(5, 5, 20), (4, 5, 21), (5, 5, 20)]
    for method in METHOD_NAMES:
        assert result["methods"][method]["pooled"]["positive_frames"] == 14
        assert result["methods"][method]["pooled"]["negative_frames"] == 61
        assert result["methods"][method]["pooled"]["evaluated_frames"] == 75
        assert result["methods"][method]["exact_interval_sensitivity"]["pooled"][
            "negative_frames"
        ] == 76
        assert result["methods"][method]["events"]["total_events"] == 3
        assert result["methods"][method]["diagnostics"]["washout_frames"] == 15
        assert "max_per_run_consecutive_false_positives" in result["methods"][
            method
        ]["diagnostics"]
        for run_result in result["methods"][method]["per_run"].values():
            assert run_result["used_signals"]
            assert run_result["signal_sources"]


def test_pre_forward_commitment_requires_no_health_or_run_output() -> None:
    source = _make_run("preforward", "holdout", "dynamic_occlusion")
    pre_forward = DetectionSuiteInput(
        run_id=source.run_id,
        dataset_split="holdout",
        corruption_type=source.corruption_type,
        raw_frame_sha256s=source.raw_frame_sha256s,
        corruption_json=source.corruption_json,
        input_manifest_json=source.input_manifest_json,
        source_manifest_json=source.source_manifest_json,
    )

    commitment = make_run_commitment(pre_forward)

    assert set(commitment["artifacts"]) == {
        "source_manifest",
        "input_manifest",
        "corruption_json",
    }
    source_manifest = json.loads(source.source_manifest_json)
    assert commitment["source_pool_frame_sha256s"] == [
        frame["rgb_sha256"] for frame in source_manifest["frames"]
    ]


def test_pre_forward_commitment_rejects_source_manifest_link_mismatch() -> None:
    source = _make_run("bad-source-link", "holdout", "dynamic_occlusion")
    input_manifest = json.loads(source.input_manifest_json)
    input_manifest["source_manifest_sha256"] = "0" * 64
    input_bytes = _json_bytes(input_manifest)
    pre_forward = DetectionSuiteInput(
        run_id=source.run_id,
        dataset_split="holdout",
        corruption_type=source.corruption_type,
        raw_frame_sha256s=source.raw_frame_sha256s,
        corruption_json=input_bytes,
        input_manifest_json=input_bytes,
        source_manifest_json=source.source_manifest_json,
    )

    with pytest.raises(ValueError, match="input/source manifest SHA mismatch"):
        make_run_commitment(pre_forward)


def test_pre_forward_commitment_rejects_duplicate_consumed_rgb() -> None:
    source = _make_run("duplicate-rgb", "holdout", "dynamic_occlusion")
    source_manifest = json.loads(source.source_manifest_json)
    changed = _replace_source_pool_sha(
        source,
        source_pool_position=1,
        sha256=source_manifest["frames"][0]["rgb_sha256"],
    )

    with pytest.raises(ValueError, match="duplicate RGB SHA identity"):
        make_run_commitment(changed)


def test_pre_forward_commitment_accepts_real_scale_decimal_timestamps() -> None:
    source = _make_run(
        "large-timestamps",
        "holdout",
        "dynamic_occlusion",
        timestamp_origin="1305031102",
    )

    commitment = make_run_commitment(source)

    first = json.loads(source.source_manifest_json)["frames"][0]
    assert first["timestamp"] > 1.3e9
    assert commitment["run_id"] == "large-timestamps"


@pytest.mark.parametrize(
    ("field", "forged_value", "message"),
    [
        (
            "delta_from_rgb_seconds",
            0.005001,
            "timestamp and declared association delta disagree",
        ),
        (
            "absolute_delta_seconds",
            0.005001,
            "timestamp and declared absolute association delta disagree",
        ),
    ],
)
def test_pre_forward_commitment_rejects_forged_decimal_association_delta(
    field: str, forged_value: float, message: str
) -> None:
    source = _make_run(
        "forged-delta",
        "holdout",
        "dynamic_occlusion",
        timestamp_origin="1305031102",
    )
    source_manifest = json.loads(source.source_manifest_json)
    depth = source_manifest["frames"][0]["depth"]
    depth[field] = forged_value
    changed = _replace_source_manifest(source, source_manifest)

    with pytest.raises(ValueError, match=message):
        make_run_commitment(changed)


def test_pre_forward_commitment_rejects_forged_timestamp_text() -> None:
    source = _make_run(
        "forged-timestamp-text",
        "holdout",
        "dynamic_occlusion",
        timestamp_origin="1305031102",
    )
    source_manifest = json.loads(source.source_manifest_json)
    depth = source_manifest["frames"][0]["depth"]
    depth["timestamp_text"] = str(
        Decimal(depth["timestamp_text"]) + Decimal("0.001")
    )
    changed = _replace_source_manifest(source, source_manifest)

    with pytest.raises(ValueError, match="timestamp and timestamp_text disagree"):
        make_run_commitment(changed)


def test_suite_rejects_scoring_interval_that_disagrees_with_run_input() -> None:
    source = _make_run("bad-label-binding", "holdout", "dynamic_occlusion")
    corruption = json.loads(source.corruption_json)
    corruption["corruptions"][0]["parameters"] = {"fixture": False}
    changed = replace(source, corruption_json=_json_bytes(corruption))
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 0,
        "thresholds": {method: 0.0 for method in METHOD_NAMES},
    }

    with pytest.raises(ValueError, match="exact input-manifest bytes"):
        evaluate_detection_suite([changed], config)


def test_suite_health_validation_rejects_missing_usable_signal() -> None:
    source = _make_run("bad-health", "holdout", "dynamic_occlusion")
    records = [json.loads(line) for line in source.health_jsonl.splitlines()]
    del records[1]["global_state_delta"]
    changed = replace(
        source,
        health_jsonl=b"".join(_json_bytes(record) + b"\n" for record in records),
    )
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 0,
        "thresholds": {method: 0.0 for method in METHOD_NAMES},
    }

    with pytest.raises(ValueError, match="require finite global-state delta"):
        evaluate_detection_suite([changed], config)


def test_suite_rejects_fabricated_runner_provenance() -> None:
    source = _make_run("bad-runner", "holdout", "dynamic_occlusion")
    run_json = json.loads(source.run_json)
    run_json["baseline_commit"] = "b" * 40
    changed = replace(source, run_json=_json_bytes(run_json))
    config = {
        "window": 5,
        "epsilon": 1.0,
        "max_z": None,
        "master_seed": 0,
        "thresholds": {method: 0.0 for method in METHOD_NAMES},
    }

    with pytest.raises(ValueError, match="baseline_commit must equal"):
        evaluate_detection_suite([changed], config)


def test_split_registry_rejects_dev_holdout_raw_overlap() -> None:
    development = _make_split("overlap-dev", "development")
    holdout = _make_split("overlap-holdout", "holdout")
    holdout[0] = _make_run(
        "holdout-dynamic",
        "holdout",
        "dynamic_occlusion",
        raw_namespace="overlap-dev-dynamic",
    )

    with pytest.raises(
        ValueError, match="development/holdout source-pool SHA sets overlap"
    ):
        make_split_registry(
            [make_run_commitment(run) for run in development],
            [make_run_commitment(run) for run in holdout],
        )


def test_split_registry_rejects_unused_low_base_overlap_across_splits() -> None:
    development = _make_split("pool-overlap-dev", "development")
    holdout = _make_split("pool-overlap-holdout", "holdout")
    low_index = next(
        index
        for index, run in enumerate(development)
        if run.corruption_type == "low_overlap_jump"
    )
    holdout_dynamic = next(
        run for run in holdout if run.corruption_type == "dynamic_occlusion"
    )
    development[low_index] = _replace_source_pool_sha(
        development[low_index],
        source_pool_position=15,
        sha256=holdout_dynamic.raw_frame_sha256s[0],
    )

    with pytest.raises(
        ValueError, match="development/holdout source-pool SHA sets overlap"
    ):
        make_split_registry(
            [make_run_commitment(run) for run in development],
            [make_run_commitment(run) for run in holdout],
        )


def test_split_registry_rejects_unused_low_base_overlap_within_split() -> None:
    development = _make_split("pool-overlap-within", "development")
    holdout = _make_split("pool-overlap-control", "holdout")
    low_index = next(
        index
        for index, run in enumerate(development)
        if run.corruption_type == "low_overlap_jump"
    )
    development_dynamic = next(
        run for run in development if run.corruption_type == "dynamic_occlusion"
    )
    development[low_index] = _replace_source_pool_sha(
        development[low_index],
        source_pool_position=15,
        sha256=development_dynamic.raw_frame_sha256s[0],
    )

    with pytest.raises(ValueError, match="development source-pool overlap"):
        make_split_registry(
            [make_run_commitment(run) for run in development],
            [make_run_commitment(run) for run in holdout],
        )


def test_formal_suite_rejects_commitment_and_actual_artifact_hash_mismatch() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()

    with pytest.raises(ValueError, match="actual holdout commitment SHA"):
        evaluate_formal_holdout(
            holdout,
            config,
            split_registry=registry,
            holdout_commitment=commitment + b" ",
            search_result=search,
            development_runs=development,
        )

    changed_bytes = holdout[0].input_manifest_json + b"\n"
    changed_run_json = json.loads(holdout[0].run_json)
    changed_run_json["input_manifest"]["sha256"] = hashlib.sha256(
        changed_bytes
    ).hexdigest()
    changed_run = replace(
        holdout[0],
        corruption_json=changed_bytes,
        input_manifest_json=changed_bytes,
        run_json=_json_bytes(changed_run_json),
    )
    with pytest.raises(ValueError, match="actual holdout provenance mismatch"):
        evaluate_formal_holdout(
            [changed_run, *holdout[1:]],
            config,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )


def test_formal_suite_rejects_policy_and_runtime_config_mismatch() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()
    changed_policy = copy.deepcopy(config)
    changed_policy["evaluation_policy"]["washout_length"] = 4

    with pytest.raises(ValueError, match="evaluation policy mismatch"):
        evaluate_formal_holdout(
            holdout,
            changed_policy,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )

    runtime = copy.deepcopy(config["frozen_detection_config"])
    runtime["window"] = 4
    with pytest.raises(ValueError, match="runtime detection config does not match"):
        evaluate_formal_holdout(
            holdout,
            config,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
            runtime_detection_config=runtime,
        )


def test_formal_config_is_strict_about_split_provenance_fields() -> None:
    development, holdout, registry, commitment, search, config = _formal_fixture()
    missing = copy.deepcopy(config)
    del missing["split_registry_sha256"]

    with pytest.raises(ValueError, match="missing: split_registry_sha256"):
        evaluate_formal_holdout(
            holdout,
            missing,
            split_registry=registry,
            holdout_commitment=commitment,
            search_result=search,
            development_runs=development,
        )
