from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_recal3r_bounded_update_pressure_v16.py"


def _module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]


def test_v16_production_script_uses_only_v16_runtime_and_the_narrow_health_adapter() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    modules = _module_names(SCRIPT)
    forbidden = {"stateguard3r.online_detector_v2", "stateguard3r.online_detector_incremental_v3", "stateguard3r.visual_overlap", "stateguard3r.timestamp_order_v3", "stateguard3r.detection_v2"}
    # ``_v1`` is a prefix of ``_v16``.  Match complete legacy version tokens,
    # not substrings of this version's own module names.
    assert not any(
        module in forbidden
        or any(module.endswith(f"_v{version}") or f"_v{version}." in module for version in range(1, 16))
        for module in modules
    )
    assert "stateguard3r.health" in modules
    assert "stateguard3r.online_detector_v16" in modules
    assert "stateguard3r.online_visual_overlap_v16" in modules
    assert "stateguard3r.timestamp_order_v16" in modules
    assert '"source_index": 0' in source
    assert "capability.frames[0].source_index" not in source


def test_v16_dispatcher_pins_every_runtime_import_source() -> None:
    import scripts.dispatch_recal3r_bounded_update_pressure_v16 as dispatch

    pinned = set(dispatch.COMPONENTS)
    assert set(dispatch.RUNTIME_IMPORT_SOURCES) <= pinned


def test_v16_runner_rejects_cuda_hidden_without_constructing_a_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.run_recal3r_bounded_update_pressure_v16 as run

    capsule = tmp_path / "capsule"
    capsule.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(run.RunBoundedPressureV16Error, match="fixed first release"):
        run.main(["--capsule", str(capsule), "--output-dir", str(tmp_path / "output"), "--state-policy", "always-commit", "--device", "cuda"])
