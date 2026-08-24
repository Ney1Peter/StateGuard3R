#!/usr/bin/env python3
"""Freeze all 80 Stage 0.7 windows, methods, interactions, and gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAME = "state-triage-v2-stage0p7-inputs-0001"
OUTPUT_PATH = ROOT / "docs" / "protocols" / "state-triage-v2-stage0p7-window-gate-b.json"
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p7 import DECISION_CONTRACT, INTERACTION_CONTRACT  # noqa: E402
from stateguard3r.state_triage_v2 import EvidenceConfig  # noqa: E402


class FreezeStage0p7ProtocolError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeStage0p7ProtocolError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda name: (_ for _ in ()).throw(FreezeStage0p7ProtocolError(f"non-finite JSON constant {name}")), object_pairs_hook=duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, FreezeStage0p7ProtocolError) as error:
        raise FreezeStage0p7ProtocolError(f"cannot load strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeStage0p7ProtocolError("JSON root must be an object")
    return value


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode) if directory else not stat.S_ISREG(metadata.st_mode)) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise FreezeStage0p7ProtocolError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _inventory(input_dir: Path) -> list[dict[str, str]]:
    _read_only(input_dir, directory=True)
    items = _json(input_dir / "protocol.json").get("artifacts")
    if not isinstance(items, list) or len(items) != 80:
        raise FreezeStage0p7ProtocolError("input inventory must contain 80 artifacts")
    result: list[dict[str, str]] = []
    causes: list[str] = []
    for item in items:
        expected = {"capsule_id", "path", "sha256", "event", "source_sequence", "source_group", "recipe_tier"}
        if not isinstance(item, Mapping) or set(item) != expected:
            raise FreezeStage0p7ProtocolError("input artifact schema differs")
        path = (input_dir / str(item["path"])).resolve(strict=True)
        if path.parent != input_dir / "evaluation":
            raise FreezeStage0p7ProtocolError("capsule lies outside evaluation directory")
        _read_only(path)
        if _sha(path) != item["sha256"]:
            raise FreezeStage0p7ProtocolError("capsule digest differs")
        capsule = load_stage0_capsule(path)
        if capsule.capsule_id != item["capsule_id"] or capsule.event.to_dict() != item["event"]:
            raise FreezeStage0p7ProtocolError("capsule differs from inventory")
        causes.append(capsule.event.cause)
        result.append({"capsule_path": str(path.relative_to(ROOT)), "capsule_sha256": str(item["sha256"]), "capsule_id": capsule.capsule_id, "output": f"state-triage-v2-stage0p7-window-{capsule.capsule_id}-observer-0001"})
    if {cause: causes.count(cause) for cause in set(causes)} != {"normal_novelty": 20, "registration_or_order_fault": 20, "bad_observation": 20, "transient_local_content": 20}:
        raise FreezeStage0p7ProtocolError("cause balance differs from 20x4 matrix")
    if len({item["capsule_id"] for item in result}) != 80 or len({item["output"] for item in result}) != 80:
        raise FreezeStage0p7ProtocolError("inventory has duplicate capsule/output names")
    return result


def freeze(*, input_dir: Path, output_path: Path) -> Path:
    input_dir, output_path = input_dir.resolve(strict=True), output_path.resolve(strict=False)
    if input_dir.parent != ROOT / "outputs" or output_path.parent != ROOT / "docs" / "protocols" or output_path.exists():
        raise FreezeStage0p7ProtocolError("input/output location is invalid or already exists")
    inventory = _inventory(input_dir)
    final = ROOT / "docs" / "audits" / "state-triage-v2-stage0p6-final.md"
    runner, observer = ROOT / "scripts" / "run_state_triage_v2_stage0.py", ROOT / "src" / "stateguard3r" / "state_triage_v2.py"
    if not all(path.is_file() for path in (final, runner, observer)):
        raise FreezeStage0p7ProtocolError("required prior audit or source is absent")
    payload = {
        "schema_version": "stateguard3r.state-triage-stage0p7-window-gate-b.v1",
        "stage": "GATE_B_STAGE0P7_FROZEN_PRE_EVALUATION",
        "input_inventory": {"path": str(input_dir.relative_to(ROOT)), "sha256": _sha(input_dir / "protocol.json")},
        "reused_stage0_control_evidence": {"audit_path": str(final.relative_to(ROOT)), "audit_sha256": _sha(final), "scope": "frozen clean control parity reused only if runner and observer source hashes below stay identical"},
        "frozen_sources": {"runner_path": str(runner.relative_to(ROOT)), "runner_sha256": _sha(runner), "observer_path": str(observer.relative_to(ROOT)), "observer_sha256": _sha(observer)},
        "frozen_evidence_config": EvidenceConfig().to_dict(),
        "decision": DECISION_CONTRACT,
        "interaction": INTERACTION_CONTRACT,
        "evaluation_inventory": inventory,
        "gates": {"typed_macro_f1_min": 0.85, "typed_minus_scalar_macro_f1_min": 0.30, "each_cause_recall_min": 0.75, "normal_novelty_false_reject_max": 0.10, "unsafe_confusion_count_max": 2, "valid_event_count": 80, "activated_local_event_min": 8, "activated_typed_recall_min": 0.75, "activated_typed_minus_coverage_recall_min": 0.75},
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload))
        os.chmod(temporary, 0o444)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / INPUT_NAME)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        print(freeze(input_dir=args.input_dir, output_path=args.output_path))
    except (FreezeStage0p7ProtocolError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
