from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_recovery_quality_alarm_v1 as alarm
from scripts import prepare_recovery_quality_v1_inputs as prepare
from tests.test_prepare_recovery_quality_v1_inputs import _freeze_tree, _tum_dataset


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _baseline(path: Path, manifest: Path, *, forbidden: bool = False) -> None:
    path.mkdir()
    health = []
    for index in range(30):
        row = {
            "frame_id": index,
            "overlap": None if index == 0 else 0.8,
            "update_magnitude": 0.1,
            "reliability": 0.5,
            "pose_jump": None if index == 0 else 0.01,
            "geometric_residual": 0.01,
        }
        if forbidden:
            row["groundtruth_pose"] = 1
        health.append(json.dumps(row, sort_keys=True))
    (path / "health.jsonl").write_text("\n".join(health) + "\n", encoding="utf-8")
    (path / "run.json").write_text(json.dumps({"health_profile": "v2", "input_manifest": _artifact(manifest)}), encoding="utf-8")
    _freeze_tree(path)


def _formal_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "continuous_scoring": {"window": 5, "min_history": 3, "max_z": 5.0, "master_seed": 0},
                "thresholds": {"combined": 2.0},
                "scale_floors": {
                    "update_magnitude": 0.01,
                    "reliability": 0.01,
                    "pose_jump": 0.01,
                    "geometric_residual": 0.01,
                    "overlap": 0.01,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    path.chmod(0o444)


def test_alarm_builder_uses_wrong_order_timestamp_without_gt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(alarm, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, output_root / "inputs-0001", dataset_id=raw.name)
    manifest = inputs / "wrong-order" / "input-manifest.json"
    baseline = tmp_path / "baseline"
    _baseline(baseline, manifest)
    config = tmp_path / "formal-config.json"
    _formal_config(config)
    output = alarm.build(manifest, baseline, config, raw / "rgb.txt", raw, output_root / "alarms-0001")
    payload = json.loads((output / "alarm.json").read_text())
    assert payload["schema_version"] == alarm.SCHEMA_VERSION
    assert payload["hybrid_alarm_positions"] == [16, 17, 18]
    assert payload["policy_alarm_positions"] == [16]
    assert payload["online_input_policy"]["future_frames"] is False


def test_alarm_builder_rejects_gt_key_in_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(alarm, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, output_root / "inputs-0001", dataset_id=raw.name)
    manifest = inputs / "wrong-order" / "input-manifest.json"
    baseline = tmp_path / "baseline"
    _baseline(baseline, manifest, forbidden=True)
    config = tmp_path / "formal-config.json"
    _formal_config(config)
    with pytest.raises(alarm.RecoveryQualityAlarmError, match="forbidden"):
        alarm.build(manifest, baseline, config, raw / "rgb.txt", raw, output_root / "alarms-0001")


def test_alarm_cli_passes_positional_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[Path] = []

    def fake(*args: Path) -> Path:
        captured.extend(args)
        return args[-1]

    monkeypatch.setattr(alarm, "build", fake)
    assert alarm.main(["--input-manifest", "a", "--baseline-dir", "b", "--formal-config", "c", "--rgb-listing", "d", "--dataset-root", "e", str(tmp_path / "out")]) == 0
    assert captured[-1] == tmp_path / "out"
