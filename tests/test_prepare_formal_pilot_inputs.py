from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch

import pytest

import scripts.prepare_formal_pilot_inputs as preparer
import stateguard3r.detection_suite as detection_suite
from scripts.prepare_formal_pilot_inputs import (
    DYNAMIC_PARAMETERS,
    EVENT_START,
    FRAME_COUNT,
    INPUT_MANIFEST_FILENAME,
    HOLDOUT_COMMITMENT_FILENAME,
    REGISTRY_FILENAME,
    RUN_SPECS,
    SOURCE_MANIFEST_FILENAME,
    SPLIT_REGISTRY_FILENAME,
    FormalPilotInputError,
    prepare_formal_pilot_inputs,
)
from stateguard3r.input_manifest import InputManifestError, load_input_manifest
from stateguard3r.detection_suite import (
    HOLDOUT_COMMITMENT_SCHEMA_VERSION,
    SPLIT_REGISTRY_SCHEMA_VERSION,
    DetectionSuiteInput,
    make_run_commitment,
)


ENTRY_COUNT = 620
HEADER = "# synthetic TUM fixture\n# comments do not count as entries\n\n"


def _timestamp(index: int, offset: str = "0") -> str:
    value = Decimal("1000") + Decimal(index) / Decimal("10") + Decimal(offset)
    return f"{value:.6f}"


def _rewrite_raw_manifest(root: Path) -> None:
    """Re-sign an intentional synthetic fixture mutation for deeper gate tests."""

    archive_path = root.parent / "rgbd_dataset_freiburg1_desk.tgz"
    raw_files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        status = path.stat()
        raw_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": status.st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mode": f"{stat.S_IMODE(status.st_mode):04o}",
            }
        )
    raw_manifest = {
        "schema_version": preparer.RAW_MANIFEST_SCHEMA_VERSION,
        "status": "PASS",
        "dataset_name": preparer.DATASET_NAME,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archive_size_bytes": archive_path.stat().st_size,
        "regular_file_count": len(raw_files),
        "total_regular_bytes": sum(item["size_bytes"] for item in raw_files),
        "files": raw_files,
    }
    raw_manifest_path = root.parent / "gate2-fr1-desk-raw-manifest.json"
    if raw_manifest_path.exists():
        raw_manifest_path.chmod(0o644)
    raw_manifest_path.write_text(
        json.dumps(raw_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_manifest_path.chmod(0o444)


def _make_tum_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "rgbd_dataset_freiburg1_desk"
    rgb_dir = root / "rgb"
    depth_dir = root / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()

    rgb_lines: list[str] = []
    depth_lines: list[str] = []
    groundtruth_lines: list[str] = []
    for index in range(ENTRY_COUNT):
        rgb_name = f"rgb/{index:06d}.png"
        depth_name = f"depth/{index:06d}.png"
        (root / rgb_name).write_bytes(f"synthetic-rgb-{index}".encode())
        (root / depth_name).write_bytes(f"synthetic-depth-{index}".encode())
        rgb_lines.append(f"{_timestamp(index)} {rgb_name}\n")
        depth_lines.append(f"{_timestamp(index, '0.001')} {depth_name}\n")
        groundtruth_lines.append(
            f"{_timestamp(index, '0.002')} "
            f"{index}.0 {index}.1 {index}.2 0.0 0.0 0.0 1.0\n"
        )
    (root / "rgb.txt").write_text(HEADER + "".join(rgb_lines), encoding="utf-8")
    (root / "depth.txt").write_text(
        HEADER + "".join(depth_lines), encoding="utf-8"
    )
    (root / "groundtruth.txt").write_text(
        HEADER + "".join(groundtruth_lines), encoding="utf-8"
    )
    for path in (candidate for candidate in root.rglob("*") if candidate.is_file()):
        path.chmod(0o444)
    archive_path = tmp_path / "rgbd_dataset_freiburg1_desk.tgz"
    archive_path.write_bytes(b"synthetic-formal-tum-archive")
    archive_path.chmod(0o444)
    _rewrite_raw_manifest(root)
    rgb_dir.chmod(0o555)
    depth_dir.chmod(0o555)
    root.chmod(0o555)
    return root


def _test_source_policy(tum_root: Path) -> preparer._FormalSourcePolicy:
    fixture_root = tum_root.parent
    archive_path = fixture_root / "rgbd_dataset_freiburg1_desk.tgz"
    raw_manifest_path = fixture_root / "gate2-fr1-desk-raw-manifest.json"
    return preparer._FormalSourcePolicy(
        dataset_root=tum_root,
        archive_path=archive_path,
        archive_size_bytes=archive_path.stat().st_size,
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        raw_manifest_path=raw_manifest_path,
        raw_manifest_size_bytes=raw_manifest_path.stat().st_size,
        raw_manifest_sha256=hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest(),
        allowed_output_root=fixture_root,
    )


def _prepare(tum_root: Path, output: Path) -> dict[str, Path]:
    policy = _test_source_policy(tum_root)
    with patch.multiple(
        detection_suite,
        FORMAL_DATASET_ROOT=str(policy.dataset_root),
        FORMAL_ARCHIVE_PATH=str(policy.archive_path),
        FORMAL_ARCHIVE_SIZE_BYTES=policy.archive_size_bytes,
        FORMAL_ARCHIVE_SHA256=policy.archive_sha256,
        FORMAL_RAW_MANIFEST_PATH=str(policy.raw_manifest_path),
        FORMAL_RAW_MANIFEST_SIZE_BYTES=policy.raw_manifest_size_bytes,
        FORMAL_RAW_MANIFEST_SHA256=policy.raw_manifest_sha256,
    ):
        return prepare_formal_pilot_inputs(
            tum_root, output, _source_policy=policy
        )


def _make_test_run_commitment(
    tum_root: Path, suite_input: DetectionSuiteInput
) -> dict:
    policy = _test_source_policy(tum_root)
    with patch.multiple(
        detection_suite,
        FORMAL_DATASET_ROOT=str(policy.dataset_root),
        FORMAL_ARCHIVE_PATH=str(policy.archive_path),
        FORMAL_ARCHIVE_SIZE_BYTES=policy.archive_size_bytes,
        FORMAL_ARCHIVE_SHA256=policy.archive_sha256,
        FORMAL_RAW_MANIFEST_PATH=str(policy.raw_manifest_path),
        FORMAL_RAW_MANIFEST_SIZE_BYTES=policy.raw_manifest_size_bytes,
        FORMAL_RAW_MANIFEST_SHA256=policy.raw_manifest_sha256,
    ):
        return make_run_commitment(suite_input)


def _tree_state(root: Path) -> dict[str, tuple[bytes, int, int, int, int]]:
    result: dict[str, tuple[bytes, int, int, int, int]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        status = path.stat()
        result[path.relative_to(root).as_posix()] = (
            path.read_bytes(),
            status.st_mode,
            status.st_size,
            status.st_mtime_ns,
            status.st_nlink,
        )
    return result


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _run_paths(output: Path, run: dict) -> tuple[Path, Path]:
    run_dir = output / run["dataset_split"] / run["run_id"]
    return run_dir / SOURCE_MANIFEST_FILENAME, run_dir / INPUT_MANIFEST_FILENAME


def test_prepares_six_strict_disjoint_manifest_only_runs(tmp_path: Path) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    source_before = _tree_state(tum_root)
    output = tmp_path / "formal-pilot"

    paths = _prepare(tum_root, output)

    assert paths["registry"] == output / REGISTRY_FILENAME
    assert _tree_state(tum_root) == source_before
    published = sorted(path for path in output.rglob("*") if path.is_file())
    assert len(published) == 15
    assert all(path.suffix == ".json" for path in published)
    assert not any(path.is_symlink() for path in output.rglob("*"))
    assert not list(output.rglob("*.png"))
    assert all(path.stat().st_mode & 0o222 == 0 for path in published)
    assert output.stat().st_mode & 0o222 == 0
    assert all(
        path.stat().st_mode & 0o222 == 0
        for path in output.rglob("*")
        if path.is_dir()
    )

    registry = _json(paths["registry"])
    assert registry["schema_version"] == "stateguard3r.formal-pilot-inputs.v1"
    assert registry["formal_status"] == "pre_forward_inputs_only"
    assert registry["materialization"] == "manifests_only_no_copy_no_symlink"
    assert registry["seed"] == 0
    assert registry["frame_count"] == FRAME_COUNT
    assert registry["event_start"] == EVENT_START
    lineage = registry["official_lineage"]
    assert lineage["dataset_root"] == str(tum_root)
    assert lineage["raw_manifest_schema_version"] == (
        preparer.RAW_MANIFEST_SCHEMA_VERSION
    )
    assert lineage["archive"]["sha256"] == hashlib.sha256(
        (tmp_path / "rgbd_dataset_freiburg1_desk.tgz").read_bytes()
    ).hexdigest()
    assert lineage["raw_manifest"]["sha256"] == hashlib.sha256(
        (tmp_path / "gate2-fr1-desk-raw-manifest.json").read_bytes()
    ).hexdigest()
    assert registry["forbidden_exploratory_rgb_range"] == {"start": 17, "end": 46}
    assert registry["association_policy"] == {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "max_absolute_delta_seconds": 0.02,
    }
    assert registry["fixed_execution_order"] == {
        "development": [
            "development-dynamic",
            "development-wrong",
            "development-low",
        ],
        "holdout": ["holdout-dynamic", "holdout-wrong", "holdout-low"],
    }
    split_registry_path = output / SPLIT_REGISTRY_FILENAME
    holdout_commitment_path = output / HOLDOUT_COMMITMENT_FILENAME
    assert paths["split_registry"] == split_registry_path
    assert paths["holdout_commitment"] == holdout_commitment_path
    split_registry_bytes = split_registry_path.read_bytes()
    holdout_commitment_bytes = holdout_commitment_path.read_bytes()
    assert registry["formal_commitment_artifacts"] == {
        "split_registry": {
            "path": SPLIT_REGISTRY_FILENAME,
            "sha256": hashlib.sha256(split_registry_bytes).hexdigest(),
            "size_bytes": len(split_registry_bytes),
        },
        "holdout_commitment": {
            "path": HOLDOUT_COMMITMENT_FILENAME,
            "sha256": hashlib.sha256(holdout_commitment_bytes).hexdigest(),
            "size_bytes": len(holdout_commitment_bytes),
        },
    }
    split_registry = json.loads(split_registry_bytes)
    holdout_commitment = json.loads(holdout_commitment_bytes)
    assert split_registry["schema_version"] == SPLIT_REGISTRY_SCHEMA_VERSION
    assert holdout_commitment["schema_version"] == (
        HOLDOUT_COMMITMENT_SCHEMA_VERSION
    )
    proof = registry["disjointness_proof"]
    assert proof["development_holdout_disjoint"] is True
    assert proof["unique_consumed_count"] == {
        "rgb": 180,
        "depth": 180,
        "groundtruth": 180,
    }
    assert len(proof["pairwise_intersection_count"]) == 15
    assert all(
        counts == {"rgb": 0, "depth": 0, "groundtruth": 0}
        for counts in proof["pairwise_intersection_count"].values()
    )
    allocation_proof = registry["allocation_disjointness_proof"]
    assert allocation_proof["scope"] == (
        "all_six_complete_base_and_donor_source_pools"
    )
    assert allocation_proof["development_holdout_disjoint"] is True
    assert allocation_proof["unique_allocated_count"] == {
        "rgb": 190,
        "depth": 190,
        "groundtruth": 190,
    }
    assert len(allocation_proof["pairwise_intersection_count"]) == 15
    assert all(
        counts == {"rgb": 0, "depth": 0, "groundtruth": 0}
        for counts in allocation_proof["pairwise_intersection_count"].values()
    )

    expected = {
        "development-dynamic": (248, (), "dynamic_occlusion", 30),
        "development-low": (380, tuple(range(300, 305)), "low_overlap_jump", 35),
        "development-wrong": (320, (), "wrong_order_segment", 30),
        "holdout-dynamic": (520, (), "dynamic_occlusion", 30),
        "holdout-low": (583, tuple(range(500, 505)), "low_overlap_jump", 35),
        "holdout-wrong": (550, (), "wrong_order_segment", 30),
    }
    assert [run["run_id"] for run in registry["runs"]] == [
        "development-dynamic",
        "development-wrong",
        "development-low",
        "holdout-dynamic",
        "holdout-wrong",
        "holdout-low",
    ]
    all_consumed: dict[str, set[int]] = {}
    for run in registry["runs"]:
        base_start, donors, corruption_type, pool_count = expected[run["run_id"]]
        assert run["base_rgb_source_indices"] == list(
            range(base_start, base_start + 30)
        )
        assert run["donor_rgb_source_indices"] == list(donors)
        assert run["source_pool_frame_count"] == pool_count
        assert run["output_frame_count"] == 30
        assert run["corruption_type"] == corruption_type
        assert run["execution_position_within_split"] == registry[
            "fixed_execution_order"
        ][run["dataset_split"]].index(run["run_id"])
        assert len(run["raw_frame_sha256s"]) == 30
        assert len(set(run["raw_frame_sha256s"])) == 30
        source_path, input_path = _run_paths(output, run)
        assert paths[run["run_id"]] == input_path
        source_bytes = source_path.read_bytes()
        input_bytes = input_path.read_bytes()
        assert run["artifacts"]["source_manifest"] == {
            "path": source_path.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size_bytes": len(source_bytes),
        }
        assert run["artifacts"]["input_manifest"] == {
            "path": input_path.relative_to(output).as_posix(),
            "sha256": hashlib.sha256(input_bytes).hexdigest(),
            "size_bytes": len(input_bytes),
        }
        assert run["artifacts"]["corruption_json"] == run["artifacts"][
            "input_manifest"
        ]
        commitment = _make_test_run_commitment(
            tum_root,
            DetectionSuiteInput(
                run_id=run["run_id"],
                dataset_split=run["dataset_split"],
                corruption_type=corruption_type,
                raw_frame_sha256s=tuple(run["raw_frame_sha256s"]),
                corruption_json=input_bytes,
                input_manifest_json=input_bytes,
                source_manifest_json=source_bytes,
            ),
        )
        assert commitment["run_id"] == run["run_id"]
        assert commitment["corruption_type"] == corruption_type
        assert commitment["raw_frame_sha256s"] == run["raw_frame_sha256s"]
        assert run["suite_run_commitment"] == commitment

        source = _json(source_path)
        manifest = _json(input_path)
        loaded = load_input_manifest(input_path)
        assert len(loaded.frames) == 30
        assert source["frame_count"] == pool_count
        assert source["output_frame_count"] == 30
        assert source["official_lineage"] == lineage
        assert source["tum_index_files"]["rgb"]["data_entry_count"] == ENTRY_COUNT
        assert source["tum_index_files"]["groundtruth"]["size_bytes"] == (
            tum_root / "groundtruth.txt"
        ).stat().st_size
        assert source["frames"][0]["source_entry_index"] == run[
            "source_pool_rgb_indices"
        ][0]
        assert source["frames"][0]["source_line"] == (
            run["source_pool_rgb_indices"][0] + 4
        )
        assert len(source["frames"][0]["rgb_sha256"]) == 64
        assert source["frames"][0]["rgb_size_bytes"] > 0
        assert source["frames"][0]["rgb_mode_octal"] == "0444"
        assert source["frames"][0]["rgb_link_count"] == 1
        assert source["frames"][0]["rgb_device"] == (
            tum_root / f"rgb/{run['source_pool_rgb_indices'][0]:06d}.png"
        ).stat().st_dev
        assert source["frames"][0]["rgb_inode"] == (
            tum_root / f"rgb/{run['source_pool_rgb_indices'][0]:06d}.png"
        ).stat().st_ino
        assert source["frames"][0]["rgb_mtime_ns"] == (
            tum_root / f"rgb/{run['source_pool_rgb_indices'][0]:06d}.png"
        ).stat().st_mtime_ns
        assert source["frames"][0]["depth"]["mode_octal"] == "0444"
        assert source["frames"][0]["depth"]["link_count"] == 1
        assert source["tum_index_files"]["rgb"]["mode_octal"] == "0444"
        assert source["tum_index_files"]["rgb"]["link_count"] == 1
        assert source["tum_index_files"]["rgb"]["device"] == (
            tum_root / "rgb.txt"
        ).stat().st_dev
        assert source["tum_index_files"]["rgb"]["inode"] == (
            tum_root / "rgb.txt"
        ).stat().st_ino
        assert source["tum_index_files"]["rgb"]["mtime_ns"] == (
            tum_root / "rgb.txt"
        ).stat().st_mtime_ns
        assert source["frames"][0]["depth"]["source_entry_index"] == run[
            "source_pool_rgb_indices"
        ][0]
        assert source["frames"][0]["groundtruth"]["source_entry_index"] == run[
            "source_pool_rgb_indices"
        ][0]
        assert manifest["source_manifest"] == SOURCE_MANIFEST_FILENAME
        assert manifest["source_manifest_sha256"] == hashlib.sha256(
            source_bytes
        ).hexdigest()
        assert manifest["source_frame_count"] == pool_count
        assert manifest["frame_count"] == 30
        assert manifest["seed"] == 0
        label = manifest["corruptions"]
        assert len(label) == 1
        assert label[0]["type"] == corruption_type
        assert label[0]["start"] == 15

        if corruption_type == "dynamic_occlusion":
            assert label[0]["end"] == 19
            parameters = label[0]["parameters"]
            assert parameters["initial_rectangle"] == DYNAMIC_PARAMETERS["rectangle"]
            assert parameters["velocity"] == DYNAMIC_PARAMETERS["velocity"]
            assert parameters["fill"] == DYNAMIC_PARAMETERS["fill"]
            assert parameters["coordinate_reference"] == (
                "model_input_after_resize_and_center_crop"
            )
            assert len(parameters["frame_rectangles"]) == 5
        elif corruption_type == "low_overlap_jump":
            assert label[0]["end"] == 19
            assert label[0]["parameters"]["source_indices"] == list(range(30, 35))
            assert [manifest["frames"][index]["source_index"] for index in range(15, 20)] == list(
                range(30, 35)
            )
            expected_consumed = [
                *range(base_start, base_start + 15),
                *donors,
                *range(base_start + 20, base_start + 30),
            ]
            assert run["consumed_associations"]["rgb_source_entry_indices"] == expected_consumed
            proxy = run["low_overlap_gt_pose_proxy"]
            assert proxy["status"] == (
                "gt_pose_proxy_only_not_measured_image_overlap"
            )
            assert "no overlap percentage is claimed" in proxy["interpretation"]
            assert [pair["output_position"] for pair in proxy["pairs"]] == list(
                range(15, 20)
            )
            assert [
                pair["base_source_pool_index"] for pair in proxy["pairs"]
            ] == list(range(15, 20))
            assert [
                pair["donor_source_pool_index"] for pair in proxy["pairs"]
            ] == list(range(30, 35))
            separation = base_start + 15 - donors[0]
            expected_distance = float(Decimal(separation) * Decimal(3).sqrt())
            assert [
                pair["translation_l2_meters"] for pair in proxy["pairs"]
            ] == pytest.approx([expected_distance] * 5)
            assert [
                pair["rotation_angle_radians"] for pair in proxy["pairs"]
            ] == pytest.approx([0.0] * 5)
            assert proxy["summary"]["translation_l2_meters"] == pytest.approx(
                {
                    "minimum": expected_distance,
                    "mean": expected_distance,
                    "maximum": expected_distance,
                }
            )
            assert proxy["summary"]["rotation_angle_radians"] == {
                "minimum": 0.0,
                "mean": 0.0,
                "maximum": 0.0,
            }
        else:
            assert label[0]["end"] == 18
            assert label[0]["parameters"]["mode"] == "reverse"
            assert label[0]["parameters"]["permutation"] == [18, 17, 16, 15]

        if corruption_type != "low_overlap_jump":
            assert "low_overlap_gt_pose_proxy" not in run

        consumed_indices = set(
            run["consumed_associations"]["rgb_source_entry_indices"]
        )
        assert len(consumed_indices) == 30
        assert consumed_indices.isdisjoint(range(17, 47))
        all_consumed[run["run_id"]] = consumed_indices

    run_ids = list(all_consumed)
    assert all(
        all_consumed[left].isdisjoint(all_consumed[right])
        for left_index, left in enumerate(run_ids)
        for right in run_ids[left_index + 1 :]
    )
    assert [run["run_id"] for run in split_registry["development"]] == registry[
        "fixed_execution_order"
    ]["development"]
    assert [run["run_id"] for run in split_registry["holdout"]] == registry[
        "fixed_execution_order"
    ]["holdout"]
    assert holdout_commitment["runs"] == split_registry["holdout"]


def test_output_is_deterministic_and_existing_tree_is_never_replaced(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    first = tmp_path / "first" / "pilot"
    second = tmp_path / "second" / "pilot"

    _prepare(tum_root, first)
    before = _tree_state(first)
    with pytest.raises(FormalPilotInputError, match="refusing to replace existing"):
        _prepare(tum_root, first)
    assert _tree_state(first) == before

    _prepare(tum_root, second)
    first_registry = _json(first / REGISTRY_FILENAME)
    second_registry = _json(second / REGISTRY_FILENAME)
    for registry in (first_registry, second_registry):
        for run in registry["runs"]:
            # Relative source paths differ when output nesting differs; semantic
            # frozen selections and all raw-input commitments must not.
            run["artifacts"].pop("source_manifest")
            run["artifacts"].pop("input_manifest")
            run["artifacts"].pop("corruption_json")
    assert first_registry == second_registry


def test_rejects_nearest_tie_reuse_and_cross_run_rgb_content_overlap(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    original_depth = (tum_root / "depth.txt").read_text(encoding="utf-8")
    original_rgb = (tum_root / "rgb.txt").read_text(encoding="utf-8")

    tie_path = tum_root / "depth" / "tie.png"
    tie_path.parent.chmod(0o755)
    tie_path.write_bytes(b"tie-depth")
    tie_path.chmod(0o444)
    tie_path.parent.chmod(0o555)
    (tum_root / "depth.txt").chmod(0o644)
    with (tum_root / "depth.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"{_timestamp(248, '-0.001')} depth/tie.png\n")
    (tum_root / "depth.txt").chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(FormalPilotInputError, match="non-unique nearest depth"):
        _prepare(tum_root, tmp_path / "tie-output")
    assert not (tmp_path / "tie-output").exists()

    (tum_root / "depth.txt").chmod(0o644)
    (tum_root / "depth.txt").write_text(original_depth, encoding="utf-8")
    (tum_root / "depth.txt").chmod(0o444)
    tie_path.parent.chmod(0o755)
    tie_path.unlink()
    tie_path.parent.chmod(0o555)
    rgb_lines = original_rgb.splitlines(keepends=True)
    rgb_lines[3 + 249] = f"{_timestamp(248)} rgb/000249.png\n"
    (tum_root / "rgb.txt").chmod(0o644)
    (tum_root / "rgb.txt").write_text("".join(rgb_lines), encoding="utf-8")
    (tum_root / "rgb.txt").chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(FormalPilotInputError, match="reuses a nearest depth"):
        _prepare(tum_root, tmp_path / "reuse-output")
    assert not (tmp_path / "reuse-output").exists()

    (tum_root / "rgb.txt").chmod(0o644)
    (tum_root / "rgb.txt").write_text(original_rgb, encoding="utf-8")
    (tum_root / "rgb.txt").chmod(0o444)
    (tum_root / "rgb" / "000320.png").chmod(0o644)
    (tum_root / "rgb" / "000320.png").write_bytes(
        (tum_root / "rgb" / "000248.png").read_bytes()
    )
    (tum_root / "rgb" / "000320.png").chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(FormalPilotInputError, match="consumed rgb association overlap"):
        _prepare(tum_root, tmp_path / "overlap-output")
    assert not (tmp_path / "overlap-output").exists()
    assert not list(tmp_path.glob(".*.staging"))


def test_rejects_gt_tie_cross_run_depth_gt_reuse_and_hardlink_alias(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    rgb_index_path = tum_root / "rgb.txt"
    depth_index_path = tum_root / "depth.txt"
    gt_index_path = tum_root / "groundtruth.txt"
    original_rgb = rgb_index_path.read_text(encoding="utf-8")
    original_depth = depth_index_path.read_text(encoding="utf-8")
    original_gt = gt_index_path.read_text(encoding="utf-8")

    gt_index_path.chmod(0o644)
    with gt_index_path.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{_timestamp(248, '-0.002')} 0 0 0 0 0 0 1\n"
        )
    gt_index_path.chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(
        FormalPilotInputError, match="non-unique nearest groundtruth"
    ):
        _prepare(tum_root, tmp_path / "gt-tie-output")

    gt_index_path.chmod(0o644)
    gt_index_path.write_text(original_gt, encoding="utf-8")
    gt_index_path.chmod(0o444)
    rgb_lines = original_rgb.splitlines(keepends=True)
    rgb_lines[3 + 320] = f"{_timestamp(248, '0.005')} rgb/000320.png\n"
    rgb_index_path.chmod(0o644)
    rgb_index_path.write_text("".join(rgb_lines), encoding="utf-8")
    rgb_index_path.chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(
        FormalPilotInputError, match="consumed depth association overlap"
    ):
        _prepare(tum_root, tmp_path / "depth-reuse-output")

    depth_lines = original_depth.splitlines(keepends=True)
    depth_lines[3 + 320] = (
        f"{_timestamp(248, '0.005')} depth/000320.png\n"
    )
    depth_index_path.chmod(0o644)
    depth_index_path.write_text("".join(depth_lines), encoding="utf-8")
    depth_index_path.chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(
        FormalPilotInputError, match="consumed groundtruth association overlap"
    ):
        _prepare(tum_root, tmp_path / "gt-reuse-output")

    rgb_index_path.chmod(0o644)
    rgb_index_path.write_text(original_rgb, encoding="utf-8")
    rgb_index_path.chmod(0o444)
    depth_index_path.chmod(0o644)
    depth_index_path.write_text(original_depth, encoding="utf-8")
    depth_index_path.chmod(0o444)
    alias = tum_root / "rgb" / "000320.png"
    alias.parent.chmod(0o755)
    alias.unlink()
    os.link(tum_root / "rgb" / "000248.png", alias)
    alias.parent.chmod(0o555)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(
        FormalPilotInputError, match="consumed rgb association overlap"
    ):
        _prepare(tum_root, tmp_path / "hardlink-output")

    assert not any(
        (tmp_path / name).exists()
        for name in (
            "gt-tie-output",
            "depth-reuse-output",
            "gt-reuse-output",
            "hardlink-output",
        )
    )


def test_rejects_excess_delta_nonfinite_pose_and_exploratory_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    depth_path = tum_root / "depth.txt"
    original_depth = depth_path.read_text(encoding="utf-8")
    lines = original_depth.splitlines(keepends=True)
    lines[3 + 248] = f"{_timestamp(248, '0.03')} depth/000248.png\n"
    depth_path.chmod(0o644)
    depth_path.write_text("".join(lines), encoding="utf-8")
    depth_path.chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(FormalPilotInputError, match="exceeds 0.02 seconds"):
        _prepare(tum_root, tmp_path / "delta-output")

    depth_path.chmod(0o644)
    depth_path.write_text(original_depth, encoding="utf-8")
    depth_path.chmod(0o444)
    gt_path = tum_root / "groundtruth.txt"
    groundtruth = gt_path.read_text(encoding="utf-8").splitlines(keepends=True)
    groundtruth[3 + 248] = (
        f"{_timestamp(248, '0.002')} 0 0 0 nan 0 0 1\n"
    )
    gt_path.chmod(0o644)
    gt_path.write_text("".join(groundtruth), encoding="utf-8")
    gt_path.chmod(0o444)
    _rewrite_raw_manifest(tum_root)
    with pytest.raises(FormalPilotInputError, match="must be finite"):
        _prepare(tum_root, tmp_path / "nan-output")

    modified = list(RUN_SPECS)
    modified[0] = replace(modified[0], base_start=17)
    monkeypatch.setattr(preparer, "RUN_SPECS", tuple(modified))
    with pytest.raises(FormalPilotInputError, match="exploratory RGB indices"):
        _prepare(tum_root, tmp_path / "exploratory-output")


def test_staging_validation_failure_leaves_no_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "atomic-output"

    def fail_validation(path: Path):
        raise FormalPilotInputError(f"injected validation failure for {path}")

    monkeypatch.setattr(preparer, "load_input_manifest", fail_validation)
    with pytest.raises(FormalPilotInputError, match="injected validation failure"):
        _prepare(tum_root, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-output.*.staging"))


def test_parent_fsync_failure_rolls_back_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "fsync-failure-output"
    real_fsync = preparer._fsync_directory

    def fail_parent_fsync(path: Path) -> None:
        if path == output.parent:
            raise OSError("injected parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(preparer, "_fsync_directory", fail_parent_fsync)
    with pytest.raises(
        FormalPilotInputError, match="publication was atomically rolled back"
    ):
        _prepare(tum_root, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".fsync-failure-output.*.staging"))


def test_post_publication_source_check_failure_rolls_back_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "postcheck-failure-output"
    real_check = preparer._assert_snapshot_unchanged

    def fail_after_publication(snapshot) -> None:
        real_check(snapshot)
        if output.exists():
            raise FormalPilotInputError("injected post-publication source failure")

    monkeypatch.setattr(preparer, "_assert_snapshot_unchanged", fail_after_publication)
    with pytest.raises(
        FormalPilotInputError, match="injected post-publication source failure"
    ):
        _prepare(tum_root, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".postcheck-failure-output.*.staging"))


@pytest.mark.parametrize(
    "relative_path",
    ["rgb.txt", "rgb/000248.png", "depth/000248.png"],
)
def test_rejects_writable_index_or_selected_raw_source_without_mutating_it(
    tmp_path: Path, relative_path: str
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    source = tum_root / relative_path
    source.chmod(0o644)
    before = _tree_state(tum_root)
    output = tmp_path / "writable-output"

    with pytest.raises(FormalPilotInputError, match="must be read-only"):
        _prepare(tum_root, output)

    assert _tree_state(tum_root) == before
    assert not output.exists()


@pytest.mark.parametrize("relative_directory", [".", "rgb", "depth"])
def test_rejects_writable_raw_tree_directory(
    tmp_path: Path, relative_directory: str
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    directory = tum_root if relative_directory == "." else tum_root / relative_directory
    directory.chmod(0o755)

    with pytest.raises(FormalPilotInputError, match="directory must be read-only"):
        _prepare(tum_root, tmp_path / "writable-directory-output")


def test_default_production_policy_rejects_a_synthetic_root(tmp_path: Path) -> None:
    tum_root = _make_tum_fixture(tmp_path)

    with pytest.raises(FormalPilotInputError, match="frozen official raw root"):
        prepare_formal_pilot_inputs(tum_root, tmp_path / "not-production")


def test_injected_policy_still_confines_output_to_its_allowed_root(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    policy = replace(_test_source_policy(tum_root), allowed_output_root=allowed)

    with pytest.raises(FormalPilotInputError, match="must be inside"):
        prepare_formal_pilot_inputs(
            tum_root,
            tmp_path / "outside",
            _source_policy=policy,
        )


@pytest.mark.parametrize("artifact", ["archive", "raw_manifest"])
def test_frozen_archive_and_raw_manifest_exact_identity_is_enforced(
    tmp_path: Path, artifact: str
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    policy = _test_source_policy(tum_root)
    path = policy.archive_path if artifact == "archive" else policy.raw_manifest_path
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"tamper")
    path.chmod(0o444)

    with pytest.raises(FormalPilotInputError, match="exact size/SHA-256 mismatch"):
        prepare_formal_pilot_inputs(
            tum_root,
            tmp_path / f"{artifact}-tamper-output",
            _source_policy=policy,
        )


@pytest.mark.parametrize(
    "relative_path",
    ["rgb/000395.png", "depth/000248.png"],
)
def test_allocation_files_must_match_frozen_raw_manifest(
    tmp_path: Path, relative_path: str
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    source = tum_root / relative_path
    source.chmod(0o644)
    source.write_bytes(source.read_bytes() + b"tamper")
    source.chmod(0o444)

    with pytest.raises(FormalPilotInputError, match="frozen raw manifest"):
        _prepare(tum_root, tmp_path / "raw-file-tamper-output")


def test_formal_loader_rejects_identical_bytes_with_changed_inode(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "inode-drift-pilot"
    paths = _prepare(tum_root, output)
    manifest_path = paths["development-dynamic"]
    source = _json(
        output / "development" / "development-dynamic" / SOURCE_MANIFEST_FILENAME
    )
    frame_path = (
        output / "development" / "development-dynamic" / source["frames"][0]["path"]
    ).resolve()
    original = frame_path.read_bytes()
    frame_path.parent.chmod(0o755)
    frame_path.unlink()
    frame_path.write_bytes(original)
    frame_path.chmod(0o444)
    frame_path.parent.chmod(0o555)

    with pytest.raises(InputManifestError, match="formal declaration"):
        load_input_manifest(manifest_path)


def test_formal_loader_rejects_rgb_symlink_alias_with_same_inode(
    tmp_path: Path,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "symlink-alias-pilot"
    paths = _prepare(tum_root, output)
    manifest_path = paths["development-dynamic"]
    source_path = (
        output / "development" / "development-dynamic" / SOURCE_MANIFEST_FILENAME
    )
    source = _json(source_path)
    declared = source_path.parent / source["frames"][0]["path"]
    frame_path = Path(os.path.abspath(declared))
    relocated = frame_path.with_name("relocated-original.png")
    frame_path.parent.chmod(0o755)
    frame_path.rename(relocated)
    frame_path.symlink_to(relocated.name)
    frame_path.parent.chmod(0o555)

    with pytest.raises(InputManifestError, match="symlink or alias"):
        load_input_manifest(manifest_path)


def test_formal_loader_rejects_groundtruth_or_lineage_drift(tmp_path: Path) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    first_output = tmp_path / "gt-drift-pilot"
    first_paths = _prepare(tum_root, first_output)
    groundtruth = tum_root / "groundtruth.txt"
    groundtruth.chmod(0o644)
    groundtruth.write_bytes(groundtruth.read_bytes() + b"# drift\n")
    groundtruth.chmod(0o444)
    with pytest.raises(InputManifestError, match="formal groundtruth index"):
        load_input_manifest(first_paths["development-dynamic"])

    # Use a fresh fixture because the first raw tree now deliberately differs.
    second_root = _make_tum_fixture(tmp_path / "second")
    second_output = tmp_path / "second" / "lineage-drift-pilot"
    second_paths = _prepare(second_root, second_output)
    raw_manifest_path = second_root.parent / "gate2-fr1-desk-raw-manifest.json"
    payload = raw_manifest_path.read_bytes()
    raw_manifest_path.unlink()
    raw_manifest_path.write_bytes(payload)
    raw_manifest_path.chmod(0o444)
    with pytest.raises(InputManifestError, match="formal declaration"):
        load_input_manifest(second_paths["development-dynamic"])


def test_cli_reports_manifest_only_result_and_rejects_existing_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tum_root = _make_tum_fixture(tmp_path)
    output = tmp_path / "cli-output"
    monkeypatch.setattr(
        preparer, "PRODUCTION_SOURCE_POLICY", _test_source_policy(tum_root)
    )
    policy = _test_source_policy(tum_root)
    with patch.multiple(
        detection_suite,
        FORMAL_DATASET_ROOT=str(policy.dataset_root),
        FORMAL_ARCHIVE_PATH=str(policy.archive_path),
        FORMAL_ARCHIVE_SIZE_BYTES=policy.archive_size_bytes,
        FORMAL_ARCHIVE_SHA256=policy.archive_sha256,
        FORMAL_RAW_MANIFEST_PATH=str(policy.raw_manifest_path),
        FORMAL_RAW_MANIFEST_SIZE_BYTES=policy.raw_manifest_size_bytes,
        FORMAL_RAW_MANIFEST_SHA256=policy.raw_manifest_sha256,
    ):
        assert preparer.main([str(tum_root), str(output)]) == 0
        stdout = capsys.readouterr().out
        assert "six manifest-only formal-pilot runs" in stdout
        assert "no images were copied or linked" in stdout

        with pytest.raises(SystemExit) as raised:
            preparer.main([str(tum_root), str(output)])
        assert raised.value.code == 2
        assert "refusing to replace existing" in capsys.readouterr().err
