from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import recovery_quality_commitment_v1 as binding


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "mode_octal": f"{path.stat().st_mode & 0o777:04o}",
    }


def _freeze(path: Path) -> None:
    path.chmod(0o444)


def _commitment(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol = tmp_path / "protocol.md"
    config = tmp_path / "formal-config.json"
    checkpoint = tmp_path / "cut3r_512_dpt_4_64.pth"
    for path, content in ((protocol, b"protocol\n"), (config, b"{}\n"), (checkpoint, b"checkpoint\n")):
        path.write_bytes(content)
        _freeze(path)
    scenes: dict[str, object] = {}
    inventory: list[dict[str, object]] = []
    manifests: dict[tuple[str, str], Path] = {}
    for scene in binding.SCENES:
        conditions: dict[str, object] = {}
        for condition in binding.CONDITIONS:
            manifest = tmp_path / scene / condition / "input-manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"scene": scene, "condition": condition}) + "\n", encoding="utf-8")
            _freeze(manifest)
            manifests[(scene, condition)] = manifest
            conditions[condition] = {"input_manifest": _artifact(manifest)}
            inventory.append(
                {
                    "run_id": f"{scene}-{condition}",
                    "scene": scene,
                    "condition": condition,
                    "forwards": ["baseline", "always-commit", "detector-policy"],
                }
            )
        scenes[scene] = {"conditions": conditions}
    root = tmp_path / "commitment"
    root.mkdir()
    payload = {
        "schema_version": binding.SCHEMA_VERSION,
        "status": "COMMITTED_PRE_FORMAL_FORWARD",
        "StateGuard3R": {"commit": "state-commit", "clean": True},
        "ReCal3R": {"commit": "recal-commit", "clean": True},
        "protocol": _artifact(protocol),
        "frozen_detector_v3_config": _artifact(config),
        "checkpoint": _artifact(checkpoint),
        "checkpoint_expected_sha256": _sha256(checkpoint),
        "scenes": scenes,
        "run_inventory": inventory,
        "runner_contract": {"device": "cuda", "size": 512, "seed": 0, "beta_base": 0.1, "health_profile": "v2"},
        "policy": {"name": "detector-v3-quality-prior-alarm", "max_hold": 3},
        "no_reconfiguration_after_commitment": True,
    }
    path = root / "commitment.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _freeze(path)
    root.chmod(0o555)
    return path, manifests[(binding.SCENES[0], "wrong-order")], checkpoint


def test_authorize_binds_a_forward_to_exact_committed_input(tmp_path: Path) -> None:
    commitment, manifest, checkpoint = _commitment(tmp_path)

    result = binding.authorize(
        commitment,
        input_manifest=manifest,
        forward="baseline",
        state_guard_commit="state-commit",
        recal3r_commit="recal-commit",
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256(checkpoint),
        runner_contract={"device": "cuda", "size": 512, "seed": 0, "beta_base": 0.1, "health_profile": "v2"},
    )

    assert result["run_id"] == f"{binding.SCENES[0]}-wrong-order"
    assert result["forward"] == "baseline"
    assert result["commitment"]["sha256"] == _sha256(commitment)


def test_authorize_rejects_uncommitted_forward_and_changed_artifact(tmp_path: Path) -> None:
    commitment, manifest, checkpoint = _commitment(tmp_path)
    expected = dict(
        input_manifest=manifest,
        forward="baseline",
        state_guard_commit="state-commit",
        recal3r_commit="recal-commit",
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256(checkpoint),
        runner_contract={"device": "cuda", "size": 512, "seed": 0, "beta_base": 0.1, "health_profile": "v2"},
    )
    with pytest.raises(binding.RecoveryQualityCommitmentError, match="not admitted"):
        binding.authorize(commitment, **{**expected, "forward": "not-a-forward"})

    config = tmp_path / "formal-config.json"
    config.chmod(0o644)
    config.write_text('{"changed": true}\n', encoding="utf-8")
    config.chmod(0o444)
    with pytest.raises(binding.RecoveryQualityCommitmentError, match="artifact hash differs"):
        binding.authorize(commitment, **expected)
