from __future__ import annotations

from pathlib import Path

import pytest

from stateguard3r.v17_source_import_audit import (
    V17SourceImportAuditError,
    audit_v17_native_runner_contract,
    audit_v17_production_script,
    audit_v17_runtime_import_graph,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "stateguard3r"
RUNNER = PACKAGE / "recal3r_beta_base_floor_runner_v17.py"
OPERATOR = PACKAGE / "beta_base_floor_v17.py"
SCRIPT = ROOT / "scripts" / "run_recal3r_beta_base_one_shot_v17.py"
PRODUCTION_RUNTIME = tuple(
    PACKAGE / name
    for name in (
        "beta_base_floor_v17.py",
        "recal3r_beta_base_floor_runner_v17.py",
        "online_detector_v17.py",
        "dynamic_rgb_capability_v17.py",
        "online_visual_overlap_v17.py",
        "timestamp_order_v17.py",
        "v17_source_import_audit.py",
    )
)


def test_v17_import_walk_is_closed_and_rejects_prior_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = audit_v17_runtime_import_graph((RUNNER, OPERATOR))
    assert report["historical_local_capability"] is False
    assert report["offline_artifact_capability"] is False
    assert report["visited_modules"] == [
        "beta_base_floor_v17",
        "recal3r_beta_base_floor_runner_v17",
    ]
    package = tmp_path / "stateguard3r"
    package.mkdir()
    polluted = package / "beta_base_floor_v17.py"
    polluted.write_text(
        "from . import recal3r_bounded_update_pressure_runner_v16\n"
        + OPERATOR.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr("stateguard3r.v17_source_import_audit.PACKAGE", package)
    with pytest.raises(V17SourceImportAuditError, match="prohibited"):
        audit_v17_runtime_import_graph((polluted,))


def test_v17_native_contract_binds_one_scalar_scope_and_native_reset(tmp_path: Path) -> None:
    report = audit_v17_native_runner_contract(RUNNER)
    assert report["only_candidate_model_mutation"] == "temporary_model_beta_base_scalar_scope"
    assert report["native_reset_method_bound"] == "model._reset_update_pressure_if_needed"
    polluted = tmp_path / "runner.py"
    polluted.write_text(
        RUNNER.read_text(encoding="utf-8").replace(
            "model._reset_update_pressure_if_needed(reset_value)",
            "model.bad_new_native_method(reset_value)",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(V17SourceImportAuditError, match="causal order|unapproved|reset"):
        audit_v17_native_runner_contract(polluted)


def test_v17_production_script_has_its_own_closed_audit(tmp_path: Path) -> None:
    report = audit_v17_production_script(SCRIPT, PRODUCTION_RUNTIME)
    assert report["historical_production_artifact_capability"] is False
    assert "dynamic_rgb_capability_v17" in report["production_script_local_imports"]
    polluted = tmp_path / "run_recal3r_beta_base_one_shot_v17.py"
    polluted.write_text(SCRIPT.read_text(encoding="utf-8") + "\n# development-dynamic\n", encoding="utf-8")
    with pytest.raises(V17SourceImportAuditError, match="outside scripts|historical"):
        audit_v17_production_script(polluted, PRODUCTION_RUNTIME)
