from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateguard3r import dynamic_rgb_capability_v20 as capability


def test_v20_raw_selector_is_fixed_and_has_no_prior_version_or_artifact_route() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "stateguard3r" / "dynamic_rgb_capability_v20.py").read_text(encoding="utf-8")
    for forbidden in ("_v17", "_v18", "_v19", "manifest", "archive", "ground_truth", "outputs/"):
        assert forbidden not in source
    assert capability.FRAME_COUNT == 30
    assert capability.CAPSULE_ID == "native-scalar-hold-v20-dynamic-rgb-0001"


def test_v20_capability_rejects_malformed_or_wrong_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capability, "CAPSULE", tmp_path / "native-scalar-hold-v20-dynamic-rgb-0001.json")
    paths = capability._paths()
    frames = [{"frame_id": index, "rgb_relative_path": path, "sha256": "0" * 64, "size_bytes": 1, "inode": 1, "mtime_ns": 1, "mode_octal": "0444", "transforms": list(capability._transform(index))} for index, path in enumerate(paths)]
    value = {"schema": capability.SCHEMA, "capability_id": capability.CAPSULE_ID, "dataset_root": str(capability.ROOT), "rgb_listing": {"relative_path": "rgb.txt", "sha256": capability.LISTING_SHA256, "mode_octal": "0444"}, "loader": capability.LOADER, "frames": frames}
    target = capability.CAPSULE
    target.write_text(json.dumps(value), encoding="utf-8")
    target.chmod(0o444)
    value["frames"][0]["sha256"] = "Z" * 64
    target.chmod(0o644)
    target.write_text(json.dumps(value), encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV20Error, match="metadata"):
        capability.load_dynamic_rgb_capability_v20(target)
    target.chmod(0o644)
    value["frames"][0]["sha256"] = "0" * 64
    value["frames"][15]["transforms"] = []
    target.write_text(json.dumps(value), encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV20Error, match="schema"):
        capability.load_dynamic_rgb_capability_v20(target)


def test_v20_builder_refuses_an_occupied_one_use_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "native-scalar-hold-v20-dynamic-rgb-0001.json"
    target.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(capability, "CAPSULE", target)
    with pytest.raises(capability.DynamicRGBCapabilityV20Error, match="occupied"):
        capability.build_dynamic_rgb_capability_v20()


def test_v20_frame_dataclass_preserves_only_selected_path_metadata() -> None:
    frame = capability.FrameV20(0, "rgb/example.png", "a" * 64, 4, 5, 6, ())
    assert frame.relative_path == "rgb/example.png"
    assert not hasattr(frame, "timestamp")
