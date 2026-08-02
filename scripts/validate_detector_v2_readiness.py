#!/usr/bin/env python3
"""Bind disclosed-development gates to the completed v2 GPU equivalence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


DEVELOPMENT_CHECKS = (
    "full_development_combined_macro_auroc_at_least_0_80",
    "cross_validated_held_out_macro_auroc_at_least_0_75",
    "combined_minus_seeded_random_macro_auroc_at_least_0_15",
    "pooled_primary_fpr_at_most_0_15",
    "max_per_run_fp_streak_at_most_2",
    "all_three_corruptions_detected_with_mean_delay_at_most_1",
    "clean_prefix_fp_at_most_6_without_triplicate_position",
    "reliability_only_pooled_fpr_below_0_50",
    "finite_online_overlap_used_by_combined",
)
INSTRUMENTATION_CHECK = "instrumentation_equivalence_gpu_cleanup"


class ReadinessValidationError(RuntimeError):
    """Raised when an upstream development or instrumentation gate is invalid."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_search_dir", type=Path)
    parser.add_argument("instrumentation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadinessValidationError(f"cannot read {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON payload is not an object: {path}")
    return payload


def validate_readiness(
    *, development_search_dir: Path, instrumentation_dir: Path
) -> dict[str, Any]:
    """Return a PASS report only when all ten Phase 6 checks have evidence."""

    search_path = development_search_dir.resolve() / "search.json"
    instrumentation_path = instrumentation_dir.resolve() / "instrumentation-equivalence.json"
    search = _read_json(search_path)
    instrumentation = _read_json(instrumentation_path)
    readiness = search.get("readiness")
    _require(isinstance(readiness, dict), "development search has no readiness object")
    _require(
        readiness.get("status") == "PREINSTRUMENTATION_PASS",
        "development search is not a pre-instrumentation pass",
    )
    development_checks = readiness.get("checks")
    _require(isinstance(development_checks, dict), "development readiness checks are missing")
    for name in DEVELOPMENT_CHECKS:
        _require(
            development_checks.get(name) is True,
            f"development readiness check did not pass: {name}",
        )
    _require(
        development_checks.get(INSTRUMENTATION_CHECK) is False,
        "development search must preserve its pre-instrumentation state",
    )
    _require(instrumentation.get("status") == "PASS", "instrumentation audit did not pass")
    instrumentation_checks = instrumentation.get("checks")
    _require(isinstance(instrumentation_checks, dict), "instrumentation checks are missing")
    _require(
        all(value is True for value in instrumentation_checks.values()),
        "at least one instrumentation or cleanup check failed",
    )
    for name in (
        "checkpoint_load_audit_byte_identical",
        "prediction_summary_byte_identical",
        "trajectory_byte_identical",
        "run_model_invariants_identical",
        "health_fields_except_overlap_identical",
        "v2_overlap_null_then_finite",
        "v2_causal_visual_provenance_valid",
        "v1_postflight_pid_absent",
        "v2_postflight_pid_absent",
    ):
        _require(instrumentation_checks.get(name) is True, f"missing required audit check: {name}")

    combined_checks = {name: True for name in DEVELOPMENT_CHECKS}
    combined_checks[INSTRUMENTATION_CHECK] = True
    return {
        "schema_version": "stateguard3r.detector-v2-readiness.v1",
        "status": "PASS",
        "decision": "PHASE_7_SINGLE_NEW_SCENE_DOWNLOAD_PERMITTED",
        "scope": (
            "Phase 6 development readiness only; this is not a formal-v2 result and "
            "does not make disclosed legacy data blind evidence."
        ),
        "checks": combined_checks,
        "selected_metrics": readiness.get("selected_metrics"),
        "upstream": {
            "development_search": {
                "path": str(search_path),
                "sha256": _sha256(search_path),
                "size_bytes": search_path.stat().st_size,
                "status": readiness["status"],
            },
            "instrumentation_equivalence": {
                "path": str(instrumentation_path),
                "sha256": _sha256(instrumentation_path),
                "size_bytes": instrumentation_path.stat().st_size,
                "status": instrumentation["status"],
                "frame_count": instrumentation.get("frame_count"),
                "recurrent_update_count": instrumentation.get("recurrent_update_count"),
            },
        },
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _publish(output_dir: Path, report: Mapping[str, Any]) -> None:
    _require(not output_dir.exists(), f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    destination = output_dir / "readiness.json"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_dir, prefix=".readiness.json.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_json_bytes(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    destination.chmod(0o444)
    output_dir.chmod(0o555)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_readiness(
        development_search_dir=args.development_search_dir,
        instrumentation_dir=args.instrumentation_dir,
    )
    script_path = Path(__file__).resolve()
    report["validator"] = {"path": str(script_path), "sha256": _sha256(script_path)}
    _publish(args.output_dir.resolve(strict=False), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
