from __future__ import annotations

import ast
import inspect

import pytest

from stateguard3r.selective_state_memory_repair_v13 import (
    MEMORY_SHAPE,
    REPAIR_DENOMINATOR,
    STATE_SHAPE,
    SelectiveStateMemoryRepairError,
    capture_state_memory_prestate,
    require_unmutated_prestate,
    selective_state_memory_repair,
)


def _torch() -> object:
    return pytest.importorskip("torch")


def _unique_rows(torch: object, shape: tuple[int, int, int]) -> object:
    rows = torch.arange(1, shape[1] + 1, dtype=torch.float32).reshape(1, shape[1], 1)
    return rows.expand(shape).clone()


def test_v13_repairs_exactly_the_fixed_highest_delta_one_eighth_and_nothing_else() -> None:
    torch = _torch()
    pre_state, pre_mem = torch.zeros(STATE_SHAPE), torch.zeros(MEMORY_SHAPE)
    proposed_state, proposed_mem = _unique_rows(torch, STATE_SHAPE), _unique_rows(torch, MEMORY_SHAPE)

    result = selective_state_memory_repair(
        pre_state, proposed_state, pre_mem, proposed_mem, torch=torch,
    )

    state_count, memory_count = STATE_SHAPE[1] // REPAIR_DENOMINATOR, MEMORY_SHAPE[1] // REPAIR_DENOMINATOR
    state_rows = result.evidence["state_feat"]["selected_rows"]
    memory_rows = result.evidence["mem"]["selected_rows"]
    assert state_rows == list(range(STATE_SHAPE[1] - 1, STATE_SHAPE[1] - state_count - 1, -1))
    assert memory_rows == list(range(MEMORY_SHAPE[1] - 1, MEMORY_SHAPE[1] - memory_count - 1, -1))
    assert result.evidence["state_feat"]["selected_row_count"] == state_count
    assert result.evidence["mem"]["selected_row_count"] == memory_count
    assert result.evidence["raw_camera_pose_numeric_input"] is False
    assert result.evidence["fallback_used"] is False
    assert result.evidence["no_fallback"] is True

    torch.testing.assert_close(result.state_feat[:, : STATE_SHAPE[1] - state_count], proposed_state[:, : STATE_SHAPE[1] - state_count])
    torch.testing.assert_close(result.mem[:, : MEMORY_SHAPE[1] - memory_count], proposed_mem[:, : MEMORY_SHAPE[1] - memory_count])
    torch.testing.assert_close(result.state_feat[:, STATE_SHAPE[1] - state_count :], pre_state[:, STATE_SHAPE[1] - state_count :])
    torch.testing.assert_close(result.mem[:, MEMORY_SHAPE[1] - memory_count :], pre_mem[:, MEMORY_SHAPE[1] - memory_count :])
    for section in ("state_feat", "mem"):
        evidence = result.evidence[section]
        assert evidence["tie_free_boundary"] is True
        assert evidence["selected_rows_equal_pre"] is True
        assert evidence["unselected_rows_equal_proposed"] is True
        assert evidence["selected_nonzero_delta_count"] == evidence["selected_row_count"]
        assert all(evidence[name].startswith("gpu-fingerprint-v1:") for name in ("pre_tensor_digest", "proposed_tensor_digest", "committed_tensor_digest"))


def test_v13_rejects_tie_nonfinite_bad_shape_or_dtype_before_repair() -> None:
    torch = _torch()
    state_pre, mem_pre = torch.zeros(STATE_SHAPE), torch.zeros(MEMORY_SHAPE)
    state_proposed, mem_proposed = _unique_rows(torch, STATE_SHAPE), _unique_rows(torch, MEMORY_SHAPE)
    tie_state = state_proposed.clone()
    boundary = STATE_SHAPE[1] - STATE_SHAPE[1] // REPAIR_DENOMINATOR
    tie_state[:, boundary - 1] = tie_state[:, boundary]
    with pytest.raises(SelectiveStateMemoryRepairError, match="tie"):
        selective_state_memory_repair(state_pre, tie_state, mem_pre, mem_proposed, torch=torch)

    malformed = torch.zeros((1, STATE_SHAPE[1] - 1, STATE_SHAPE[2]))
    with pytest.raises(SelectiveStateMemoryRepairError, match="pinned|divisible|shapes"):
        selective_state_memory_repair(malformed, malformed, mem_pre, mem_proposed, torch=torch)
    with pytest.raises(SelectiveStateMemoryRepairError, match="dtype or device"):
        selective_state_memory_repair(state_pre, state_proposed.double(), mem_pre, mem_proposed, torch=torch)
    broken = state_proposed.clone()
    broken[:, 3, 2] = float("nan")
    with pytest.raises(SelectiveStateMemoryRepairError, match="finite"):
        selective_state_memory_repair(state_pre, broken, mem_pre, mem_proposed, torch=torch)


def test_v13_prestate_is_reference_only_and_rejects_in_place_native_mutation() -> None:
    torch = _torch()
    state, memory = torch.zeros(STATE_SHAPE), torch.zeros(MEMORY_SHAPE)
    snapshot = capture_state_memory_prestate(state, memory, torch=torch)
    require_unmutated_prestate(snapshot)
    state.add_(1.0)
    with pytest.raises(SelectiveStateMemoryRepairError, match="mutated in place"):
        require_unmutated_prestate(snapshot)


def test_v13_operator_has_no_pose_or_model_input_and_no_cpu_state_solver() -> None:
    source = inspect.getsource(selective_state_memory_repair).lower()
    for forbidden in (
        "prediction", "image", "rgb", "pointmap", "decoder",
        "detector", "anchor", "history", "ground_truth", "future", ".cpu(",
        ".numpy(", ".tolist(",
    ):
        assert forbidden not in source
    assert source.count("camera_pose") == 1
    assert '"raw_camera_pose_numeric_input": false' in source
    signature = inspect.signature(selective_state_memory_repair)
    assert tuple(signature.parameters) == (
        "pre_state_feat", "proposed_state_feat", "pre_mem", "proposed_mem", "torch",
    )
    module = ast.parse(inspect.getsourcefile(selective_state_memory_repair) and open(inspect.getsourcefile(selective_state_memory_repair), encoding="utf-8").read())
    operator = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "selective_state_memory_repair")
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(operator)
    )


def test_v13_cpu_evidence_adapter_can_export_only_indices_after_gpu_selection() -> None:
    torch = _torch()
    source = inspect.getsource(selective_state_memory_repair)
    assert "_repair_one" in source
    primitive_source = inspect.getsource(__import__("stateguard3r.selective_state_memory_repair_v13", fromlist=["_repair_one"])._repair_one)
    assert "indices.detach().cpu().reshape(-1).tolist()" in primitive_source
    assert "pre.detach().cpu" not in primitive_source
    assert "proposed.detach().cpu" not in primitive_source
    assert "repaired.detach().cpu" not in primitive_source
    result = selective_state_memory_repair(torch.zeros(STATE_SHAPE), _unique_rows(torch, STATE_SHAPE), torch.zeros(MEMORY_SHAPE), _unique_rows(torch, MEMORY_SHAPE), torch=torch)
    for section in ("state_feat", "mem"):
        evidence = result.evidence[section]
        assert set(evidence) >= {"selected_rows", "row_delta_summary", "pre_tensor_digest", "proposed_tensor_digest", "committed_tensor_digest"}
        assert all("value" not in key for key in evidence)
