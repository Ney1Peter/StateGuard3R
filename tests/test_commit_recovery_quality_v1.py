from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import commit_recovery_quality_v1 as commit


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o444)


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size, "mode_octal": "0444"}


def _inputs(outputs: Path, scene: str, name: str) -> tuple[Path, Path]:
    root = outputs / name
    root.mkdir()
    rows = []
    for condition in commit.CONDITIONS:
        directory = root / condition
        directory.mkdir()
        source, manifest = directory / "source-manifest.json", directory / "input-manifest.json"
        _write(source, {"condition": condition})
        _write(manifest, {"condition": condition})
        rows.append({"condition": condition, "source_manifest": _artifact(source), "input_manifest": _artifact(manifest)})
        directory.chmod(0o555)
    registry = root / "recovery-quality-inputs.json"
    _write(registry, {"dataset": scene, "status": "pre_forward_recovery_quality_inputs", "runs": rows})
    root.chmod(0o555)
    validation = outputs / f"{name}-validation"
    validation.mkdir()
    _write(validation / "validation.json", {"status": "PASS", "input_registry": {"sha256": hashlib.sha256(registry.read_bytes()).hexdigest()}})
    validation.chmod(0o555)
    return root, validation


def test_commit_binds_all_eight_inputs_before_forward(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    scene_a, validation_a = _inputs(outputs, commit.SCENES[0], "scene-a")
    scene_b, validation_b = _inputs(outputs, commit.SCENES[1], "scene-b")
    protocol, config, checkpoint = tmp_path / "protocol.md", tmp_path / "config.json", tmp_path / "cut3r_512_dpt_4_64.pth"
    protocol.write_text("protocol", encoding="utf-8")
    _write(config, {"config": True})
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(commit, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(commit, "PROTOCOL", protocol)
    monkeypatch.setattr(commit, "FORMAL_CONFIG", config)
    monkeypatch.setattr(commit, "CHECKPOINT", checkpoint)
    monkeypatch.setattr(commit, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(commit, "_git", lambda *_args: "" if "status" in _args else "deadbeef")
    result = commit.commit(scene_a, validation_a, scene_b, validation_b, outputs / "commit-0001")
    payload = json.loads((result / "commitment.json").read_text())
    assert payload["status"] == "COMMITTED_PRE_FORMAL_FORWARD"
    assert len(payload["run_inventory"]) == 8
    assert (result / "commitment.json").stat().st_mode & 0o777 == 0o444
