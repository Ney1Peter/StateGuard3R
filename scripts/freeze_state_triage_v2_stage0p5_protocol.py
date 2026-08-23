#!/usr/bin/env python3
"""Pre-register Stage 0.5 evaluation inventory and gates before any forward."""

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
INPUT_NAME = "state-triage-v2-stage0p5-inputs-0001"
OUTPUT_PATH = ROOT / "docs" / "protocols" / "state-triage-v2-stage0p5-gate-b.json"

sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.state_triage_capsule_v2 import load_stage0_capsule  # noqa: E402
from stateguard3r.state_triage_stage0p5 import DECISION_CONTRACT  # noqa: E402
from stateguard3r.state_triage_v2 import EvidenceConfig  # noqa: E402


class FreezeStage0p5ProtocolError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FreezeStage0p5ProtocolError(f"non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FreezeStage0p5ProtocolError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, FreezeStage0p5ProtocolError) as error:
        raise FreezeStage0p5ProtocolError(f"cannot read strict JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeStage0p5ProtocolError(f"{path} must contain an object")
    return value


def _read_only(path: Path, *, directory: bool = False) -> None:
    metadata = os.lstat(path)
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o222:
        raise FreezeStage0p5ProtocolError(f"{path} must be a read-only {'directory' if directory else 'file'}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _inventory(input_dir: Path) -> list[dict[str, str]]:
    _read_only(input_dir, directory=True)
    manifest = _strict_json(input_dir / "protocol.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 12:
        raise FreezeStage0p5ProtocolError("input inventory must contain exactly 12 artifacts")
    result: list[dict[str, str]] = []
    causes: list[str] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"capsule_id", "path", "sha256", "event", "source_sequence", "source_group", "recipe_variant"}:
            raise FreezeStage0p5ProtocolError("input artifact schema differs")
        relative = str(item["path"])
        capsule_path = (input_dir / relative).resolve(strict=True)
        if capsule_path.parent != input_dir / "evaluation":
            raise FreezeStage0p5ProtocolError("capsule is outside evaluation directory")
        _read_only(capsule_path)
        if _sha256(capsule_path) != item["sha256"]:
            raise FreezeStage0p5ProtocolError("input capsule digest differs")
        capsule = load_stage0_capsule(capsule_path)
        if capsule.capsule_id != item["capsule_id"] or capsule.event.to_dict() != item["event"]:
            raise FreezeStage0p5ProtocolError("input capsule event differs")
        causes.append(capsule.event.cause)
        result.append(
            {
                "capsule_path": str(capsule_path.relative_to(ROOT)),
                "capsule_sha256": str(item["sha256"]),
                "capsule_id": capsule.capsule_id,
                "output": f"state-triage-v2-stage0p5-eval-{capsule.capsule_id}-observer-0001",
            }
        )
    if sorted(causes) != sorted(["normal_novelty"] * 3 + ["registration_or_order_fault"] * 3 + ["bad_observation"] * 3 + ["transient_local_content"] * 3):
        raise FreezeStage0p5ProtocolError("input cause balance differs")
    if len({item["capsule_id"] for item in result}) != 12 or len({item["output"] for item in result}) != 12:
        raise FreezeStage0p5ProtocolError("input/output inventory has duplicates")
    return result


def freeze(*, input_dir: Path, output_path: Path) -> Path:
    input_dir = input_dir.resolve(strict=True)
    output_path = output_path.resolve(strict=False)
    if input_dir.parent != ROOT / "outputs" or output_path.parent != ROOT / "docs" / "protocols" or output_path.exists():
        raise FreezeStage0p5ProtocolError("input/output path is not a new approved Stage 0.5 location")
    inventory = _inventory(input_dir)
    stage0_audit = ROOT / "docs" / "audits" / "state-triage-v2-stage0-final.md"
    runner_source = ROOT / "scripts" / "run_state_triage_v2_stage0.py"
    observer_source = ROOT / "src" / "stateguard3r" / "state_triage_v2.py"
    for path in (stage0_audit, runner_source, observer_source):
        if not path.is_file():
            raise FreezeStage0p5ProtocolError(f"missing required provenance source: {path}")
    payload = {
        "schema_version": "stateguard3r.state-triage-stage0p5-gate-b.v1",
        "stage": "GATE_B_STAGE0P5_FROZEN_PRE_EVALUATION",
        "input_inventory": {"path": str(input_dir.relative_to(ROOT)), "sha256": _sha256(input_dir / "protocol.json")},
        "reused_stage0_control_evidence": {
            "audit_path": str(stage0_audit.relative_to(ROOT)),
            "audit_sha256": _sha256(stage0_audit),
            "scope": "four frozen clean source parity controls reused only because runner and observer source digests below are unchanged",
        },
        "frozen_sources": {
            "runner_path": str(runner_source.relative_to(ROOT)),
            "runner_sha256": _sha256(runner_source),
            "observer_path": str(observer_source.relative_to(ROOT)),
            "observer_sha256": _sha256(observer_source),
        },
        "frozen_evidence_config": EvidenceConfig().to_dict(),
        "decision": DECISION_CONTRACT,
        "evaluation_inventory": inventory,
        "gates": {
            "typed_macro_f1_min": 0.75,
            "typed_minus_scalar_macro_f1_min": 0.25,
            "each_cause_recall_min": 2.0 / 3.0,
            "normal_novelty_false_reject_max": 1.0 / 3.0,
            "unsafe_confusion_count_max": 1,
            "valid_event_count": 12,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(payload))
        os.chmod(temporary_name, 0o444)
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / INPUT_NAME)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    try:
        print(freeze(input_dir=args.input_dir, output_path=args.output_path))
    except (FreezeStage0p5ProtocolError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
