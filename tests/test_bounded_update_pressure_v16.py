from __future__ import annotations

import ast

import pytest

from stateguard3r import bounded_update_pressure_v16 as pressure


def _torch() -> object:
    return pytest.importorskip("torch")


def test_v16_clear_path_is_exact_identity_and_creates_nothing() -> None:
    torch = _torch()
    native = torch.full(pressure.PRESSURE_SHAPE, 0.4)
    result, evidence = pressure.inject_bounded_pressure_v16(native, alarm=False, device=torch.device("cpu"), dtype=torch.float32, torch=torch)
    assert result is native and evidence is None
    result, evidence = pressure.inject_bounded_pressure_v16(None, alarm=False, device=torch.device("cpu"), dtype=torch.float32, torch=torch)
    assert result is None and evidence is None


def test_v16_alarm_add_cap_bootstrap_and_contract_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    native = torch.full(pressure.PRESSURE_SHAPE, 0.9)
    monkeypatch.setattr(pressure, "_validate_pressure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pressure, "gpu_fingerprint_v16", lambda *_args, **_kwargs: "gpu:test")
    original_full = torch.full
    monkeypatch.setattr(torch, "full", lambda shape, fill, **kwargs: original_full(shape, fill, dtype=kwargs["dtype"]))
    result, evidence = pressure.inject_bounded_pressure_v16(native, alarm=True, device=torch.device("cuda"), dtype=torch.float32, torch=torch)
    assert bool(torch.equal(result, torch.ones_like(native)))
    assert evidence is not None and evidence["increment"] == 0.25 and evidence["cap"] == 1.0
    bootstrap, bootstrap_evidence = pressure.inject_bounded_pressure_v16(None, alarm=True, device=torch.device("cuda"), dtype=torch.float32, torch=torch)
    assert bool(torch.equal(bootstrap, original_full(pressure.PRESSURE_SHAPE, 0.25)))
    assert bootstrap_evidence is not None and bootstrap_evidence["native_pressure_absent_after_reset"] is True
    saturated = torch.zeros(pressure.PRESSURE_SHAPE)
    for _ in range(4):
        saturated, _ = pressure.inject_bounded_pressure_v16(saturated, alarm=True, device=torch.device("cuda"), dtype=torch.float32, torch=torch)
    assert bool(torch.equal(saturated, torch.ones_like(saturated)))
    with pytest.raises(pressure.BoundedUpdatePressureV16Error, match="boolean"):
        pressure.inject_bounded_pressure_v16(native, alarm=1, device=torch.device("cuda"), dtype=torch.float32, torch=torch)
    with pytest.raises(pressure.BoundedUpdatePressureV16Error, match="shape/device"):
        pressure.inject_bounded_pressure_v16(native, alarm=True, device=torch.device("cpu"), dtype=torch.float32, expected_shape=(1, 2, 1), torch=torch)


def test_v16_operator_has_no_cpu_serialization_or_prohibited_tensor_parameter() -> None:
    source = ast.parse(__import__("inspect").getsource(pressure))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "inject_bounded_pressure_v16")
    names = {argument.arg for argument in [*function.args.args, *function.args.kwonlyargs]}
    assert names == {"native_pressure", "alarm", "device", "dtype", "expected_shape", "torch"}
    assert not any(isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"} for node in ast.walk(function))
