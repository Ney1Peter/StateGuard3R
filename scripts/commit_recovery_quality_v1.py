#!/usr/bin/env python3
"""Freeze the two-scene recovery-quality v1 run inventory before any formal forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
OUTPUT_ROOT = ROOT / "outputs"
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "ReCal3R"
SCENES = ("rgbd_dataset_freiburg3_walking_static", "rgbd_dataset_freiburg3_walking_xyz")
CONDITIONS = ("clean", "dynamic", "wrong-order", "low-overlap")
CHECKPOINT = BASELINE_ROOT / "src" / "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
FORMAL_CONFIG = ROOT / "outputs" / "formal-v3-calibration-0001" / "formal-config.json"
PROTOCOL = ROOT / "docs" / "protocols" / "recovery-quality-v1.md"
SCHEMA_VERSION = "stateguard3r.recovery-quality-v1-commitment.v1"


class RecoveryQualityCommitmentError(ValueError):
    """Raised when a formal input/config/worktree cannot be frozen exactly once."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryQualityCommitmentError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _regular(path: Path, *, label: str, mode: int | None = None) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    if mode is not None:
        _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must be mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str) -> Path:
    details = os.lstat(path)
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode) and stat.S_IMODE(details.st_mode) == 0o555, f"{label} must be a 0555 non-symlink directory")
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _artifact(path: Path, *, mode: int | None = 0o444) -> dict[str, Any]:
    regular = _regular(path, label=str(path), mode=mode)
    return {"path": str(regular), "sha256": _sha256(regular), "size_bytes": regular.stat().st_size, "mode_octal": format(stat.S_IMODE(regular.stat().st_mode), "04o")}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _input_inventory(scene: str, input_root: Path, validation_root: Path) -> dict[str, Any]:
    inputs = _directory(input_root, label=f"{scene} input root")
    validation = _directory(validation_root, label=f"{scene} validation root")
    registry_path = _regular(inputs / "recovery-quality-inputs.json", label=f"{scene} registry", mode=0o444)
    report_path = _regular(validation / "validation.json", label=f"{scene} validation", mode=0o444)
    registry = _json(registry_path, label=f"{scene} registry")
    report = _json(report_path, label=f"{scene} validation")
    _require(registry.get("dataset") == scene and registry.get("status") == "pre_forward_recovery_quality_inputs", f"{scene} input identity differs")
    _require(report.get("status") == "PASS" and report.get("input_registry", {}).get("sha256") == _sha256(registry_path), f"{scene} validation does not bind its registry")
    rows = registry.get("runs")
    _require(isinstance(rows, list) and [row.get("condition") for row in rows if isinstance(row, Mapping)] == list(CONDITIONS), f"{scene} input condition inventory differs")
    artifacts: dict[str, Any] = {}
    for row in rows:
        condition = str(row["condition"])
        manifest = inputs / condition / "input-manifest.json"
        source = inputs / condition / "source-manifest.json"
        _require(row.get("input_manifest", {}).get("sha256") == _sha256(manifest), f"{scene}/{condition} registry input hash differs")
        artifacts[condition] = {"input_manifest": _artifact(manifest), "source_manifest": _artifact(source)}
    return {"input_root": str(inputs), "input_registry": _artifact(registry_path), "validation": _artifact(report_path), "conditions": artifacts}


def commit(scene_a_inputs: Path, scene_a_validation: Path, scene_b_inputs: Path, scene_b_validation: Path, output_dir: Path) -> Path:
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "commitment output must be a new direct child of outputs")
    _require(_git(ROOT, "status", "--porcelain", "--untracked-files=no") == "", "StateGuard3R tracked worktree is not clean")
    _require(_git(BASELINE_ROOT, "status", "--porcelain", "--untracked-files=no") == "", "ReCal3R tracked worktree is not clean")
    _require(_sha256(CHECKPOINT) == CHECKPOINT_SHA256, "pinned checkpoint SHA-256 differs")
    inventory = {
        SCENES[0]: _input_inventory(SCENES[0], scene_a_inputs, scene_a_validation),
        SCENES[1]: _input_inventory(SCENES[1], scene_b_inputs, scene_b_validation),
    }
    output.mkdir()
    run_inventory = [
        {"run_id": f"{scene}-{condition}", "scene": scene, "condition": condition, "forwards": ["baseline", "always-commit", "detector-policy"]}
        for scene in SCENES
        for condition in CONDITIONS
    ]
    payload = {"schema_version": SCHEMA_VERSION, "status": "COMMITTED_PRE_FORMAL_FORWARD", "StateGuard3R": {"commit": _git(ROOT, "rev-parse", "HEAD"), "clean": True}, "ReCal3R": {"commit": _git(BASELINE_ROOT, "rev-parse", "HEAD"), "clean": True}, "protocol": _artifact(PROTOCOL, mode=None), "frozen_detector_v3_config": _artifact(FORMAL_CONFIG), "checkpoint": _artifact(CHECKPOINT, mode=None), "checkpoint_expected_sha256": CHECKPOINT_SHA256, "scenes": inventory, "run_inventory": run_inventory, "policy": {"name": "detector-v3-quality-prior-alarm", "max_hold": 3, "alarm_filter": "causal_hybrid_alarm_rising_edge", "quality_metrics": "prefix_0_14_Sim3__tail_19_29"}, "no_reconfiguration_after_commitment": True}
    _write_json(output / "commitment.json", payload)
    (output / "commitment.json").chmod(0o444)
    output.chmod(0o555)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-a-inputs", required=True, type=Path)
    parser.add_argument("--scene-a-validation", required=True, type=Path)
    parser.add_argument("--scene-b-inputs", required=True, type=Path)
    parser.add_argument("--scene-b-validation", required=True, type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = commit(args.scene_a_inputs, args.scene_a_validation, args.scene_b_inputs, args.scene_b_validation, args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
