from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_detector_v2_readiness import (
    DEVELOPMENT_CHECKS,
    INSTRUMENTATION_CHECK,
    ReadinessValidationError,
    validate_readiness,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _search_payload() -> dict[str, object]:
    checks = {name: True for name in DEVELOPMENT_CHECKS}
    checks[INSTRUMENTATION_CHECK] = False
    return {
        "readiness": {
            "status": "PREINSTRUMENTATION_PASS",
            "checks": checks,
            "selected_metrics": {"combined_macro_auroc": 0.9},
        }
    }


def _instrumentation_payload() -> dict[str, object]:
    return {
        "status": "PASS",
        "checks": {
            "checkpoint_load_audit_byte_identical": True,
            "prediction_summary_byte_identical": True,
            "trajectory_byte_identical": True,
            "run_model_invariants_identical": True,
            "health_fields_except_overlap_identical": True,
            "v2_overlap_null_then_finite": True,
            "v2_causal_visual_provenance_valid": True,
            "v1_postflight_pid_absent": True,
            "v2_postflight_pid_absent": True,
        },
        "frame_count": 30,
        "recurrent_update_count": 29,
    }


def test_binds_nine_cpu_gates_and_gpu_equivalence(tmp_path: Path) -> None:
    development, instrumentation = tmp_path / "development", tmp_path / "instrumentation"
    development.mkdir()
    instrumentation.mkdir()
    _write_json(development / "search.json", _search_payload())
    _write_json(instrumentation / "instrumentation-equivalence.json", _instrumentation_payload())

    report = validate_readiness(
        development_search_dir=development, instrumentation_dir=instrumentation
    )

    assert report["status"] == "PASS"
    assert report["decision"] == "PHASE_7_SINGLE_NEW_SCENE_DOWNLOAD_PERMITTED"
    assert all(report["checks"].values())


def test_rejects_failed_instrumentation_check(tmp_path: Path) -> None:
    development, instrumentation = tmp_path / "development", tmp_path / "instrumentation"
    development.mkdir()
    instrumentation.mkdir()
    _write_json(development / "search.json", _search_payload())
    audit = _instrumentation_payload()
    audit["checks"]["v2_postflight_pid_absent"] = False  # type: ignore[index]
    _write_json(instrumentation / "instrumentation-equivalence.json", audit)

    with pytest.raises(ReadinessValidationError, match="instrumentation or cleanup"):
        validate_readiness(development_search_dir=development, instrumentation_dir=instrumentation)
