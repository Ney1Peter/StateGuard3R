from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import aggregate_recovery_quality_v1 as aggregate


def _freeze(root: Path) -> None:
    for path in root.iterdir():
        path.chmod(0o444)
    root.chmod(0o555)


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _evaluation(root: Path, *, scene: str, condition: str, effect: float, replay: bool = True) -> Path:
    root.mkdir()
    base_run = root / "baseline-run.json"
    policy_run = root / "policy-run.json"
    timeline = root / "timeline.json"
    timeline.write_text(json.dumps({"transactions": [{"action": "hold"}, {"action": "release_replay_then_commit"} if replay else {"action": "commit"}], "dropped_transaction_frame_ids": []}), encoding="utf-8")
    base_run.write_text(json.dumps({"runtime_seconds": 10.0}), encoding="utf-8")
    policy_run.write_text(json.dumps({"runtime_seconds": 11.0, "state_policy": {"name": "detector-v3-quality-prior-alarm", "timeline_path": str(timeline)}}), encoding="utf-8")
    for path in (base_run, policy_run, timeline):
        path.chmod(0o444)
    payload = {
        "schema_version": "stateguard3r.recovery-quality-evaluation.v1",
        "baseline_always_commit_byte_equivalence": {"trajectory.json": True},
        "effects": {"ATE_RMSE_effect": effect, "RPE_translation_RMSE_effect": effect},
        "metrics": {},
        "runs": {"baseline": _artifact(base_run), "detector_policy": _artifact(policy_run)},
    }
    (root / "evaluation.json").write_text(json.dumps(payload), encoding="utf-8")
    _freeze(root)
    return root


def test_aggregate_requires_exact_inventory_and_seals_one_go_result(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(aggregate, "OUTPUT_ROOT", outputs)
    values: dict[tuple[str, str], Path] = {}
    for scene in aggregate.SCENES:
        for condition in aggregate.CONDITIONS:
            effect = 0.2 if condition != "clean" else 0.0
            values[(scene, condition)] = _evaluation(tmp_path / f"{scene}-{condition}", scene=scene, condition=condition, effect=effect)
    output = aggregate.aggregate(values, outputs / "result-0001")
    result = json.loads((output / "result.json").read_text())
    assert result["decision"] == "RECOVERY_QUALITY_GO"
    assert result["gates"]["wrong_order_hold_and_replay_each_scene"] is True
    assert (output / "attempt-seal.json").stat().st_mode & 0o777 == 0o444


def test_aggregation_cli_accepts_frozen_commitment(tmp_path: Path, monkeypatch) -> None:
    captured: list[object] = []

    def fake(values, commitment, output):
        captured.extend((values, commitment, output))
        return output

    monkeypatch.setattr(aggregate, "aggregate", fake)
    assert aggregate.main(["--recovery-quality-commitment", "commitment.json", "--evaluation", f"{aggregate.SCENES[0]}=clean=one", str(tmp_path / "out")]) == 0
    assert captured[-1] == tmp_path / "out"
