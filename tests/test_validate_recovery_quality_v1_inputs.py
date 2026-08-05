from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import prepare_recovery_quality_v1_inputs as prepare
from scripts import validate_recovery_quality_v1_inputs as validator
from tests.test_prepare_recovery_quality_v1_inputs import _tum_dataset


def test_validator_replays_all_conditions_and_logical_gt_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(validator, "OUTPUT_ROOT", outputs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, outputs / "inputs-0001", dataset_id=raw.name)
    result = validator.validate(inputs, raw, outputs / "validation-0001")
    report = json.loads((result / "validation.json").read_text())
    assert report["status"] == "PASS"
    wrong = next(row for row in report["runs"] if row["condition"] == "wrong-order")
    assert wrong["timestamp_order_violation_positions"] == [16, 17, 18]


def test_validator_rejects_clean_prefix_timestamp_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(validator, "OUTPUT_ROOT", outputs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, outputs / "inputs-0001", dataset_id=raw.name)
    wrong = inputs / "wrong-order" / "input-manifest.json"
    wrong.chmod(0o644)
    payload = json.loads(wrong.read_text())
    payload["frames"][1], payload["frames"][2] = payload["frames"][2], payload["frames"][1]
    payload["frames"][1]["frame_index"] = 1
    payload["frames"][2]["frame_index"] = 2
    wrong.write_text(json.dumps(payload), encoding="utf-8")
    wrong.chmod(0o444)
    with pytest.raises(validator.RecoveryQualityInputValidationError):
        validator.validate(inputs, raw, outputs / "validation-0001")
