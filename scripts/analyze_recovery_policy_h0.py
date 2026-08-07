#!/usr/bin/env python3
"""CPU-only forensic check for the disclosed recovery-quality-v1 replay path.

This development tool reads the sealed formal-v1 outputs but never runs a model.
It checks the exact H0 mechanism: a held candidate restores its pre-state, then a
later replay reinstalls that candidate's proposed post-state before the clear-frame
forward.  Together with byte-identical output files this explains why the replay
policy has no trajectory lever in the sealed v1 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
SCHEMA_VERSION = "stateguard3r.recovery-policy-h0-analysis.v1"
CONDITIONS = ("clean", "dynamic", "wrong-order", "low-overlap")
SCENES = (("a", "rgbd_dataset_freiburg3_walking_static"), ("b", "rgbd_dataset_freiburg3_walking_xyz"))
MODEL_OUTPUT_FILES = ("trajectory.json", "health.jsonl", "predictions-summary.json")


class RecoveryPolicyH0Error(ValueError):
    """Raised when sealed output cannot support the declared H0 finding."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryPolicyH0Error(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str, mode: int = 0o444) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryPolicyH0Error(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISREG(details.st_mode), f"{label} must be a regular non-symlink file")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _directory(path: Path, *, label: str, mode: int = 0o555) -> Path:
    try:
        details = os.lstat(path)
    except OSError as error:
        raise RecoveryPolicyH0Error(f"cannot stat {label}: {error}") from error
    _require(not stat.S_ISLNK(details.st_mode) and stat.S_ISDIR(details.st_mode), f"{label} must be a directory")
    _require(stat.S_IMODE(details.st_mode) == mode, f"{label} must have mode {mode:04o}")
    return path.resolve(strict=True)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RecoveryPolicyH0Error(f"{label} contains non-finite JSON {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_bytes(), parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryPolicyH0Error(f"cannot parse {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _require(not path.is_symlink(), f"refusing to freeze symlink {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _transaction_rows(timeline: Mapping[str, Any], *, label: str) -> list[dict[str, Any]]:
    rows = timeline.get("transactions")
    _require(isinstance(rows, list) and rows, f"{label} has no transactions")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{label} transaction {index} must be an object")
        _require(row.get("frame_id") == index, f"{label} transaction IDs must be contiguous")
        for key in ("action", "pre_state_digest_sha256", "proposed_state_digest_sha256", "committed_state_digest_sha256"):
            _require(isinstance(row.get(key), str) and row[key], f"{label} transaction {index} lacks {key}")
        parsed.append(row)
    dropped = timeline.get("dropped_transaction_frame_ids")
    _require(isinstance(dropped, list) and not dropped, f"{label} has pending dropped transactions")
    return parsed


def analyze_pair(*, label: str, baseline_dir: Path, policy_dir: Path) -> dict[str, Any]:
    """Verify H0 for one sealed baseline/policy pair without writing to either."""

    baseline = _directory(baseline_dir, label=f"{label} baseline")
    policy = _directory(policy_dir, label=f"{label} policy")
    output_equal: dict[str, bool] = {}
    for name in MODEL_OUTPUT_FILES:
        base = _regular(baseline / name, label=f"{label} baseline {name}")
        candidate = _regular(policy / name, label=f"{label} policy {name}")
        output_equal[name] = _sha256(base) == _sha256(candidate)
    _require(all(output_equal.values()), f"{label} policy model outputs are not byte-identical to baseline")

    timeline_path = _regular(policy / "state-timeline.json", label=f"{label} policy state timeline")
    rows = _transaction_rows(_json(timeline_path, label=f"{label} policy state timeline"), label=label)
    releases: list[dict[str, Any]] = []
    holds: list[int] = []
    for row in rows:
        action = row["action"]
        replayed = row.get("replayed_transaction_frame_id")
        if action == "hold":
            _require(replayed is None, f"{label} held transaction may not claim a replay")
            _require(row["committed_state_digest_sha256"] == row["pre_state_digest_sha256"], f"{label} hold did not restore its pre-state")
            holds.append(row["frame_id"])
            continue
        if action == "release_replay_then_commit":
            _require(type(replayed) is int and 0 <= replayed < row["frame_id"], f"{label} release has invalid replay ID")
            held = rows[replayed]
            _require(held["action"] == "hold", f"{label} release does not point to a held transaction")
            _require(row["pre_state_digest_sha256"] == held["proposed_state_digest_sha256"], f"{label} replay did not reinstall the held candidate post-state")
            releases.append({
                "release_frame_id": row["frame_id"],
                "held_frame_id": replayed,
                "reinstalled_post_state_digest_sha256": held["proposed_state_digest_sha256"],
            })
            continue
        _require(action == "commit", f"{label} has unsupported action {action!r}")
        _require(replayed is None, f"{label} ordinary commit may not claim a replay")

    return {
        "label": label,
        "baseline_dir": str(baseline),
        "policy_dir": str(policy),
        "model_output_byte_identical": output_equal,
        "hold_frame_ids": holds,
        "replay_reinstalls": releases,
        "h0_supported": True,
    }


def formal_pairs(outputs_root: Path = OUTPUT_ROOT) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for suffix, scene in SCENES:
        for condition in CONDITIONS:
            stem = f"recovery-quality-formal-scene-{suffix}-{condition}"
            pairs.append((f"{scene}:{condition}", outputs_root / f"{stem}-baseline-0001", outputs_root / f"{stem}-detector-policy-0001"))
    return pairs


def analyze_formal(outputs_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    rows = [analyze_pair(label=label, baseline_dir=baseline, policy_dir=policy) for label, baseline, policy in formal_pairs(outputs_root)]
    _require(len(rows) == 8, "H0 inventory must contain exactly eight formal pairs")
    _require(all(row["h0_supported"] for row in rows), "H0 was not supported for every formal pair")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "read-only disclosed recovery-quality-v1 forensic analysis; not a new model response",
        "pair_count": len(rows),
        "all_model_outputs_byte_identical": all(all(row["model_output_byte_identical"].values()) for row in rows),
        "total_hold_count": sum(len(row["hold_frame_ids"]) for row in rows),
        "total_replay_reinstall_count": sum(len(row["replay_reinstalls"]) for row in rows),
        "pairs": rows,
        "h0_conclusion": "every release reinstalls the held candidate post-state before the clear-frame forward; paired model outputs are byte-identical",
    }


def publish(output_dir: Path, *, outputs_root: Path = OUTPUT_ROOT) -> Path:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "H0 analysis requires CUDA_VISIBLE_DEVICES='' ")
    output = output_dir.resolve(strict=False)
    _require(output.parent == OUTPUT_ROOT and not output.exists(), "output must be a new direct child of StateGuard3R/outputs")
    result = analyze_formal(outputs_root)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent))
    try:
        _write_json(staging / "h0-analysis.json", result)
        _freeze(staging)
        os.replace(staging, output)
    except Exception:
        for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    output = publish(args.output_dir)
    print(json.dumps({"output_dir": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
