from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stateguard3r.health import HealthFrame


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_recal3r_beta_base_one_shot_v17.py"


def _load() -> object:
    from scripts import run_recal3r_beta_base_one_shot_v17 as run

    return run


def test_v17_production_script_is_own_version_and_uses_independent_control_serializers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    modules = [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert "stateguard3r.health" in modules
    assert "stateguard3r.dynamic_rgb_capability_v17" in modules
    assert "_output_health_signals_v17" in source
    assert "inference.inference_recurrent_lighter" in source
    assert "get_u_calibration_last_state" not in source
    assert not any(
        module.endswith(f"_v{version}") or f"_v{version}." in module
        for module in modules
        for version in range(1, 17)
    )


def test_v17_control_serializer_has_fixed_canonical_json_contract(tmp_path: Path) -> None:
    run = _load()
    checkpoint = {
        "missing_keys": [],
        "module_class": "ARCroco3DStereo",
        "strict": False,
        "unexpected_keys": [],
    }
    target = tmp_path / "checkpoint-load-audit.json"
    run._write_json(target, checkpoint)
    assert target.read_bytes() == (
        b"{\n"
        b'  "missing_keys": [],\n'
        b'  "module_class": "ARCroco3DStereo",\n'
        b'  "strict": false,\n'
        b'  "unexpected_keys": []\n'
        b"}\n"
    )


class _FakeTensor:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value, dtype=np.float64)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self._value.copy())


def test_v17_trajectory_serializer_uses_formal_pose_and_residual_formula() -> None:
    run = _load()
    camera = np.eye(4)[None]
    prediction = {
        "camera_pose": _FakeTensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        "pts3d_in_self_view": _FakeTensor(np.zeros((1, 2, 3, 3))),
        "pts3d_in_other_view": _FakeTensor(np.ones((1, 2, 3, 3))),
    }
    jumps, residuals, frames = run._output_health_signals_v17(
        [prediction], decode=lambda _value: _FakeTensor(camera), np=np
    )
    assert jumps == [None]
    assert residuals == pytest.approx([1.0])
    assert frames[0]["camera_to_reference"] == camera[0].tolist()


def test_v17_health_adapter_requires_all_native_update_rows() -> None:
    run = _load()
    records: list[HealthFrame] = []
    for index in range(30):
        records.append(
            HealthFrame(
                frame_id=index,
                uncertainty_u=None if index == 0 else 0.2,
                global_state_delta=None if index == 0 else 1.0,
            )
        )
    model = SimpleNamespace(get_u_calibration_trace=lambda: {})
    original = __import__("stateguard3r.health", fromlist=["adapt_recal3r_trace"]).adapt_recal3r_trace
    import stateguard3r.health as health

    health.adapt_recal3r_trace = lambda *_args, **_kwargs: records
    try:
        output = run._health_from_native_trace(
            model,
            captures=[{"rgb_capture_timestamp": str(index)} for index in range(30)],
            overlaps=[None] + [0.5] * 29,
        )
    finally:
        health.adapt_recal3r_trace = original
    assert len(output) == 30
