from __future__ import annotations

from pathlib import Path

import pytest

from stateguard3r.v20_source_import_audit import (
    V20SourceImportAuditError,
    audit_v20_native_wrapper_contract,
    audit_v20_runtime_import_graph,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "stateguard3r"
WRAPPER = PACKAGE / "official_native_scalar_hold_v20.py"
DETECTOR = PACKAGE / "native_scalar_detector_v20.py"


def test_v20_import_graph_is_closed_to_its_new_scalar_components() -> None:
    report = audit_v20_runtime_import_graph((WRAPPER, DETECTOR))
    assert report["historical_local_capability"] is False
    assert report["offline_artifact_capability"] is False
    assert report["visited_modules"] == [
        "native_scalar_detector_v20",
        "official_native_scalar_hold_v20",
    ]


def test_v20_import_graph_rejects_a_historical_local_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "stateguard3r"
    package.mkdir()
    polluted = package / "native_scalar_detector_v20.py"
    polluted.write_text(
        "from . import recal3r_beta_base_floor_runner_v17\n" + DETECTOR.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr("stateguard3r.v20_source_import_audit.PACKAGE", package)
    with pytest.raises(V20SourceImportAuditError, match="prohibited"):
        audit_v20_runtime_import_graph((polluted,))


def test_v20_wrapper_audit_binds_native_mask_then_current_scalar_order() -> None:
    report = audit_v20_native_wrapper_contract(WRAPPER)
    assert report["official_mask_called_before_zero_replacement"] is True
    assert report["state_commit_before_scalar_observation"] is True
    assert report["only_candidate_effect"] == "zero_clone_of_native_state_mask"


def test_v20_wrapper_audit_rejects_cpu_state_export(tmp_path: Path) -> None:
    polluted = tmp_path / "official_native_scalar_hold_v20.py"
    polluted.write_text(
        WRAPPER.read_text(encoding="utf-8").replace(
            "state_before.shape != state_after.shape",
            "state_before.cpu().shape != state_after.shape",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(V20SourceImportAuditError, match="serializes"):
        audit_v20_native_wrapper_contract(polluted)
