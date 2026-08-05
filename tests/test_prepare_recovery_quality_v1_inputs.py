from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from scripts import prepare_recovery_quality_v1_inputs as prepare
from stateguard3r.input_manifest import load_input_manifest


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _tum_dataset(tmp_path: Path, *, rows: int = 180) -> Path:
    root = tmp_path / "rgbd_dataset_fixture"
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb = ["# rgb\n"]
    depth = ["# depth\n"]
    gt = ["# gt\n"]
    for index in range(rows):
        timestamp = f"{index / 30:.6f}"
        rgb_name = f"rgb/{index:06d}.png"
        depth_name = f"depth/{index:06d}.png"
        (root / rgb_name).write_bytes(f"rgb-{index}".encode())
        (root / depth_name).write_bytes(f"depth-{index}".encode())
        rgb.append(f"{timestamp} {rgb_name}\n")
        depth.append(f"{timestamp} {depth_name}\n")
        gt.append(f"{timestamp} {index / 30:.6f} 0 0 0 0 0 1\n")
    (root / "rgb.txt").write_text("".join(rgb), encoding="ascii")
    (root / "depth.txt").write_text("".join(depth), encoding="ascii")
    (root / "groundtruth.txt").write_text("".join(gt), encoding="ascii")
    archive = root.with_suffix(".tgz")
    archive.write_bytes(b"fixture archive")
    _freeze_tree(root)
    archive.chmod(0o444)
    return root


def test_prepare_builds_four_replayable_conditions_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = prepare.prepare(raw, output_root / "quality-inputs-0001", dataset_id=raw.name)
    registry = json.loads((output / "recovery-quality-inputs.json").read_text())
    assert registry["status"] == "pre_forward_recovery_quality_inputs"
    assert [row["condition"] for row in registry["runs"]] == ["clean", "dynamic", "wrong-order", "low-overlap"]
    for row in registry["runs"]:
        for artifact_name in ("source_manifest", "input_manifest"):
            artifact = row[artifact_name]
            artifact_path = Path(artifact["path"])
            assert artifact_path.exists()
            assert output in artifact_path.parents
    clean = load_input_manifest(output / "clean" / "input-manifest.json")
    wrong = load_input_manifest(output / "wrong-order" / "input-manifest.json")
    assert len(clean.frames) == len(wrong.frames) == 30
    assert [frame.source_index for frame in wrong.frames[15:19]] == [18, 17, 16, 15]
    payload = json.loads((output / "wrong-order" / "input-manifest.json").read_text())
    assert [row["base_source_index"] for row in payload["quality_evaluation_bindings"][15:19]] == [15, 16, 17, 18]
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE((output / "clean" / "input-manifest.json").stat().st_mode) == 0o444


def test_prepare_requires_cuda_disabled_and_readonly_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(prepare.RecoveryQualityInputError, match="CUDA"):
        prepare.prepare(raw, output_root / "quality-inputs-0001", dataset_id=raw.name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    os.chmod(raw / "rgb.txt", 0o644)
    with pytest.raises(prepare.RecoveryQualityInputError, match="0444"):
        prepare.prepare(raw, output_root / "quality-inputs-0002", dataset_id=raw.name)


def test_prepare_excludes_a_frame_with_a_nonunique_nearest_gt_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _tum_dataset(tmp_path)
    gt_path = raw / "groundtruth.txt"
    gt_path.chmod(0o644)
    with gt_path.open("a", encoding="ascii") as stream:
        stream.write("1.666667 1.666667 0 0 0 0 0 1\n")
    gt_path.chmod(0o444)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    frames = prepare.associated_frames(raw)
    assert frames[50] is None
    output = prepare.prepare(raw, output_root / "quality-inputs-0001", dataset_id=raw.name)
    registry = json.loads((output / "recovery-quality-inputs.json").read_text())
    assert registry["base_start_rgb_row"] <= 50 - 30 or registry["base_start_rgb_row"] > 50


def test_prepare_cli_passes_positional_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake(dataset_root: Path, output_dir: Path, *, dataset_id: str) -> Path:
        captured.update(dataset_root=dataset_root, output_dir=output_dir, dataset_id=dataset_id)
        return output_dir

    monkeypatch.setattr(prepare, "prepare", fake)
    assert prepare.main(["--dataset-root", "raw", "--dataset-id", "scene", str(tmp_path / "out")]) == 0
    assert captured["dataset_id"] == "scene"
    assert captured["output_dir"] == tmp_path / "out"
