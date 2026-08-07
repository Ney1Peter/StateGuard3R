from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.evaluate_development_recovery_v1 as evaluation
from scripts.evaluate_development_recovery_v1 import DevelopmentRecoveryEvaluationError, evaluate


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _trajectory(*, candidate: bool = False) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for index in range(30):
        position = float(index)
        if not candidate and index >= 19:
            position -= 0.1 * float(index - 18)
        frames.append(
            {
                "frame_id": index,
                "camera_to_reference": [[1.0, 0.0, 0.0, position], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            }
        )
    return {"frames": frames}


def _run(directory: Path, manifest: Path, *, policy: str, candidate: bool = False, invalid_discard: bool = False) -> None:
    directory.mkdir()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (directory / "run.json").write_text(
        json.dumps({"input_manifest": {"path": str(manifest), "sha256": digest}, "state_policy": {"name": policy}, "runtime_seconds": 2.0}),
        encoding="utf-8",
    )
    for name in ("checkpoint-load-audit.json", "health.jsonl", "predictions-summary.json"):
        (directory / name).write_text('{"same":true}\n', encoding="utf-8")
    (directory / "trajectory.json").write_text(json.dumps(_trajectory(candidate=candidate)), encoding="utf-8")
    timeline = {
        "release_mode": "discard" if policy.endswith("discard") else "replay",
        "transactions": [
            {"frame_id": 0, "action": "hold", "proposed_state_digest_sha256": "unsafe"},
            {"frame_id": 1, "action": "release_discard_then_commit", "discarded_transaction_frame_id": 0, "pre_state_digest_sha256": "unsafe" if invalid_discard else "safe"},
        ],
        "discarded_transaction_frame_ids": [0],
    }
    (directory / "state-timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    _freeze(directory)


def _fixture(tmp_path: Path, *, invalid_discard: bool = False) -> tuple[Path, Path, Path, Path, Path]:
    development = tmp_path / "development"
    run_root = development / "dynamic"
    run_root.mkdir(parents=True)
    source = run_root / "source-manifest.json"
    source.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "source_pool_index": index,
                        "source_pool_role": "base",
                        "groundtruth": {"translation_xyz": [float(index), 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
                    }
                    for index in range(30)
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = run_root / "input-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "stateguard3r.corruption.v1",
                "source_is_read_only": True,
                "source_manifest": "source-manifest.json",
                "frames": [{"frame_index": index, "metadata": {"groundtruth": {"translation_xyz": [-99.0, 0.0, 0.0]}}} for index in range(30)],
            }
        ),
        encoding="utf-8",
    )
    _freeze(run_root)
    baseline = tmp_path / "baseline"
    always = tmp_path / "always"
    replay = tmp_path / "replay"
    discard = tmp_path / "discard"
    _run(baseline, manifest, policy="baseline")
    _run(always, manifest, policy="always-commit")
    _run(replay, manifest, policy="detector-v3-quality-prior-alarm")
    _run(discard, manifest, policy="detector-v3-quality-prior-alarm-discard", candidate=True, invalid_discard=invalid_discard)
    return manifest, baseline, always, replay, discard


def test_development_evaluator_uses_logical_base_gt_and_reports_discard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, baseline, always, replay, discard = _fixture(tmp_path)
    monkeypatch.setattr(evaluation, "DEVELOPMENT_INPUT_ROOT", tmp_path / "development")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "outputs" / "result"
    output.parent.mkdir()
    monkeypatch.setattr(evaluation, "OUTPUT_ROOT", output.parent)

    result = evaluate(manifest, baseline, always, replay, discard, output)

    payload = json.loads((result / "evaluation.json").read_text())
    assert payload["gt_binding"].startswith("source-manifest rows 0--29")
    assert payload["effects"]["discard_vs_baseline"]["ATE_RMSE_effect"] > 0.0
    assert payload["discard_evidence"]["release_count"] == 1
    assert payload["trajectory_byte_different_from_replay"] is True
    assert oct(result.stat().st_mode & 0o777) == "0o555"


def test_development_evaluator_rejects_reinstalled_discard_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, baseline, always, replay, discard = _fixture(tmp_path, invalid_discard=True)
    monkeypatch.setattr(evaluation, "DEVELOPMENT_INPUT_ROOT", tmp_path / "development")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "outputs" / "result"
    output.parent.mkdir()
    monkeypatch.setattr(evaluation, "OUTPUT_ROOT", output.parent)

    with pytest.raises(DevelopmentRecoveryEvaluationError, match="reinstalled"):
        evaluate(manifest, baseline, always, replay, discard, output)
