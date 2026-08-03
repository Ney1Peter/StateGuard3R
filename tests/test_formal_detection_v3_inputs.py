from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest

import scripts.prepare_formal_detection_v3_inputs as prepare
import scripts.validate_formal_detection_v3_inputs as validator


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _unfreeze_tree(root: Path) -> None:
    root.chmod(0o755)
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        path.chmod(0o755 if path.is_dir() else 0o644)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )


def _raw_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    tum = tmp_path / "tum"
    raw = tum / prepare.DATASET_NAME
    rgb = raw / "rgb"
    depth = raw / "depth"
    rgb.mkdir(parents=True)
    depth.mkdir()
    rgb_rows: list[str] = []
    depth_rows: list[str] = []
    gt_rows: list[str] = []
    for index in range(95):
        timestamp = f"1.{index:09d}"
        rgb_path = rgb / f"{index}.png"
        depth_path = depth / f"{index}.png"
        rgb_path.write_bytes(f"rgb-{index}".encode("ascii"))
        depth_path.write_bytes(f"depth-{index}".encode("ascii"))
        rgb_rows.append(f"{timestamp} rgb/{index}.png")
        depth_rows.append(f"{timestamp} depth/{index}.png")
        gt_rows.append(f"{timestamp} 0 0 0 0 0 0 1")
    (raw / "rgb.txt").write_text("\n".join(rgb_rows) + "\n", encoding="utf-8")
    (raw / "depth.txt").write_text("\n".join(depth_rows) + "\n", encoding="utf-8")
    (raw / "groundtruth.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
    _freeze_tree(raw)

    archive = raw.with_suffix(".tgz")
    archive.write_bytes(b"formal-v3-fixture-archive")
    archive.chmod(0o444)
    acquisition_dir = outputs / "formal-v3-data-0001"
    acquisition_dir.mkdir()
    acquisition = acquisition_dir / "acquisition.json"
    _write_json(
        acquisition,
        {
            "status": "PASS",
            "dataset": prepare.DATASET_NAME,
            "archive": {
                "path": str(archive.resolve()),
                "sha256": _sha256(archive),
                "size_bytes": archive.stat().st_size,
                "mode_octal": "0444",
            },
            "raw_tree": {"manifest": {"dataset": prepare.DATASET_NAME}},
            "postconditions": {"no_model_or_online_overlap_execution": True},
        },
    )
    acquisition.chmod(0o444)

    monkeypatch.setattr(prepare, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(prepare, "RAW_ROOT", raw)
    monkeypatch.setattr(prepare, "ARCHIVE", archive)
    monkeypatch.setattr(prepare, "ACQUISITION", acquisition)
    monkeypatch.setattr(validator, "OUTPUT_ROOT", outputs)
    return outputs, raw


def test_prepare_enables_local_stateguard_source_for_reused_baseline_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = str(prepare.ROOT / "src")
    monkeypatch.setattr(prepare.sys, "path", [entry for entry in sys.path if entry != source])

    assert prepare._enable_stateguard_src() == prepare.ROOT / "src"
    assert prepare.sys.path[0] == source


def _donor_at_90(frames, *, low_base_start: int, base_starts: list[int]):
    assert low_base_start == 60
    assert base_starts == [0, 30, 60]
    assert len(frames) == 95
    return 90, [0.1, 0.2, 0.3, 0.4, 0.5]


def test_prepares_and_validates_frozen_v3_inputs_with_rgb_timestamp_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, _raw = _raw_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = outputs / "formal-v3-inputs-0001"

    assert prepare.prepare(input_root, donor_selector=_donor_at_90) == input_root
    assert stat.S_IMODE(input_root.stat().st_mode) == 0o555
    registry = json.loads((input_root / "formal-v3-manifest.json").read_text(encoding="utf-8"))
    assert [row["run_id"] for row in registry["runs"]] == [
        "blind-dynamic",
        "blind-wrong-order",
        "blind-low-overlap",
    ]
    assert registry["archive"]["sha256"] == _sha256(prepare.ARCHIVE)
    assert registry["acquisition"]["sha256"] == _sha256(prepare.ACQUISITION)

    wrong_source = json.loads(
        (input_root / "blind-wrong-order" / "source-manifest.json").read_text(encoding="utf-8")
    )
    assert wrong_source["rgb_capture_listing"]["sha256"] == _sha256(prepare.RAW_ROOT / "rgb.txt")
    assert [item["source_pool_index"] for item in wrong_source["final_frame_to_raw_rgb_bindings"][14:20]] == [
        14,
        18,
        17,
        16,
        15,
        19,
    ]
    assert wrong_source["frames"][0]["rgb_capture_timestamp_text"] == "1.000000030"
    assert wrong_source["frames"][0]["rgb_txt_physical_line"] == 31
    assert len(wrong_source["frames"][0]["rgb_txt_physical_line_sha256"]) == 64

    report = validator.validate(input_root)
    assert report["status"] == "PASS"
    assert report["runs"][1]["timestamp_order_violation_positions"] == [16, 17, 18]
    assert report["runs"][2]["timestamp_order_violation_positions"] == [20]
    assert all(row["clean_prefix_timestamp_order_violation_positions"] == [] for row in report["runs"])
    assert all(report["checks"].values())


def test_preparation_requires_passed_read_only_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, _raw = _raw_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    acquisition = prepare.ACQUISITION
    acquisition.chmod(0o644)
    payload = json.loads(acquisition.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    _write_json(acquisition, payload)
    acquisition.chmod(0o444)

    with pytest.raises(prepare.FormalV3InputError, match="not PASS"):
        prepare.prepare(outputs / "formal-v3-inputs-0001", donor_selector=_donor_at_90)


def test_validator_rejects_timestamp_text_not_bound_to_raw_rgb_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs, _raw = _raw_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = outputs / "formal-v3-inputs-0001"
    prepare.prepare(input_root, donor_selector=_donor_at_90)

    _unfreeze_tree(input_root)
    run_dir = input_root / "blind-dynamic"
    source_path = run_dir / "source-manifest.json"
    input_path = run_dir / "input-manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["frames"][0]["rgb_capture_timestamp"] = "999.000"
    source["frames"][0]["rgb_capture_timestamp_text"] = "999.000"
    source["final_frame_to_raw_rgb_bindings"][0]["rgb_capture_timestamp"] = "999.000"
    source["final_frame_to_raw_rgb_bindings"][0]["rgb_capture_timestamp_text"] = "999.000"
    _write_json(source_path, source)

    input_payload = json.loads(input_path.read_text(encoding="utf-8"))
    input_payload["source_manifest_sha256"] = _sha256(source_path)
    input_payload["frames"][0]["metadata"]["rgb_capture_timestamp"] = "999.000"
    input_payload["frames"][0]["metadata"]["rgb_capture_timestamp_text"] = "999.000"
    _write_json(input_path, input_payload)

    registry_path = input_root / "formal-v3-manifest.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dynamic = registry["runs"][0]["artifacts"]
    dynamic["source_manifest"]["sha256"] = _sha256(source_path)
    dynamic["source_manifest"]["size_bytes"] = source_path.stat().st_size
    dynamic["input_manifest"]["sha256"] = _sha256(input_path)
    dynamic["input_manifest"]["size_bytes"] = input_path.stat().st_size
    _write_json(registry_path, registry)
    _freeze_tree(input_root)

    with pytest.raises(validator.FormalV3InputValidationError, match="capture timestamp"):
        validator.validate(input_root)


def test_strict_rgb_listing_parser_rejects_nonfinite_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _outputs, raw = _raw_fixture(tmp_path, monkeypatch)
    listing = raw / "rgb.txt"
    listing.chmod(0o644)
    lines = listing.read_text(encoding="utf-8").splitlines()
    lines[1] = "NaN rgb/1.png"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    listing.chmod(0o444)

    with pytest.raises(prepare.FormalV3InputError, match="finite"):
        prepare.parse_tum_listing(listing, value_count=2, label="raw rgb.txt")
