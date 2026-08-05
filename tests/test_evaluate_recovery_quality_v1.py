from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_recovery_quality_v1 as evaluate
from scripts import prepare_recovery_quality_v1_inputs as prepare
from tests.test_prepare_recovery_quality_v1_inputs import _freeze_tree, _tum_dataset


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _trajectory(positions: np.ndarray) -> dict[str, object]:
    frames = []
    for index, position in enumerate(positions):
        matrix = np.eye(4)
        matrix[:3, 3] = position
        frames.append({"frame_id": index, "camera_to_reference": matrix.tolist()})
    return {"frames": frames}


def _run(path: Path, manifest: Path, positions: np.ndarray, *, policy: str | None) -> None:
    path.mkdir()
    run = {"input_manifest": _binding(manifest)}
    if policy is not None:
        run["state_policy"] = {"name": policy}
    (path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (path / "trajectory.json").write_text(json.dumps(_trajectory(positions)), encoding="utf-8")
    for name in ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json"):
        (path / name).write_bytes(b"equivalence fixture\n")
    _freeze_tree(path)


def test_evaluator_requires_equivalence_and_uses_prefix_only_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(evaluate, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, output_root / "inputs-0001", dataset_id=raw.name)
    manifest = inputs / "wrong-order" / "input-manifest.json"
    gt = np.asarray([[index / 30.0, 0.0, 0.0] for index in range(30)])
    baseline_positions = gt.copy()
    baseline_positions[19:, 0] += np.square(np.arange(11)) * 0.01
    policy_positions = gt.copy()
    policy_positions[19:, 0] += np.square(np.arange(11)) * 0.003
    baseline = tmp_path / "baseline"
    always = tmp_path / "always"
    policy = tmp_path / "policy"
    _run(baseline, manifest, baseline_positions, policy=None)
    _run(always, manifest, baseline_positions, policy="always-commit")
    _run(policy, manifest, policy_positions, policy="detector-v3-quality-prior-alarm")
    output = evaluate.evaluate(manifest, baseline, always, policy, output_root / "evaluation-0001")
    payload = json.loads((output / "evaluation.json").read_text())
    assert all(payload["baseline_always_commit_byte_equivalence"].values())
    assert payload["metrics"]["baseline"]["ATE_RMSE_m"] > payload["metrics"]["detector_policy"]["ATE_RMSE_m"]
    assert payload["effects"]["ATE_RMSE_effect"] > 0.0
    assert payload["prefix_only_alignment"]["frames"] == list(range(15))


def test_evaluator_rejects_non_equivalent_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _tum_dataset(tmp_path)
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(prepare, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(evaluate, "OUTPUT_ROOT", output_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    inputs = prepare.prepare(raw, output_root / "inputs-0001", dataset_id=raw.name)
    manifest = inputs / "wrong-order" / "input-manifest.json"
    positions = np.asarray([[index / 30.0, 0.0, 0.0] for index in range(30)])
    baseline = tmp_path / "baseline"
    always = tmp_path / "always"
    policy = tmp_path / "policy"
    _run(baseline, manifest, positions, policy=None)
    _run(always, manifest, positions, policy="always-commit")
    _run(policy, manifest, positions, policy="detector-v3-quality-prior-alarm")
    # Make the control provenance differ while preserving the read-only contract.
    (always / "health.jsonl").chmod(0o644)
    (always / "health.jsonl").write_bytes(b"different\n")
    (always / "health.jsonl").chmod(0o444)
    with pytest.raises(evaluate.RecoveryQualityEvaluationError, match="byte-equivalent"):
        evaluate.evaluate(manifest, baseline, always, policy, output_root / "evaluation-0001")


def test_evaluator_cli_passes_positional_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[Path] = []

    def fake(*args: Path) -> Path:
        captured.extend(args)
        return args[-1]

    monkeypatch.setattr(evaluate, "evaluate", fake)
    assert evaluate.main(["--input-manifest", "a", "--baseline-dir", "b", "--always-commit-dir", "c", "--policy-dir", "d", "--recovery-quality-commitment", "commitment.json", str(tmp_path / "out")]) == 0
    assert captured[-2] == tmp_path / "out"
    assert captured[-1] == Path("commitment.json")
