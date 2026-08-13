from __future__ import annotations

import ast
import inspect

import pytest

from stateguard3r import beta_base_floor_v17 as floor


def test_v17_arm_is_scalar_only_and_one_shot() -> None:
    assert floor.arm_one_future_update_v17(False, alarm=False) is False
    assert floor.arm_one_future_update_v17(False, alarm=True) is True
    with pytest.raises(floor.BetaBaseFloorV17Error, match="prior"):
        floor.arm_one_future_update_v17(True, alarm=True)
    with pytest.raises(floor.BetaBaseFloorV17Error, match="plain"):
        floor.arm_one_future_update_v17(False, alarm=1)  # type: ignore[arg-type]


def test_v17_selection_consumes_or_reset_cancels_exactly_once() -> None:
    clear = floor.select_one_shot_beta_base_v17(False, reset=False)
    assert clear.beta_base_for_mask == floor.NATIVE_BETA_BASE
    assert not clear.consumed and not clear.reset_cancelled and not clear.armed_after
    consumed = floor.select_one_shot_beta_base_v17(True, reset=False)
    assert consumed.beta_base_for_mask == floor.FLOORED_BETA_BASE
    assert consumed.consumed and not consumed.reset_cancelled and not consumed.armed_after
    cancelled = floor.select_one_shot_beta_base_v17(True, reset=True)
    assert cancelled.beta_base_for_mask == floor.NATIVE_BETA_BASE
    assert not cancelled.consumed and cancelled.reset_cancelled and not cancelled.armed_after


def test_v17_operator_rejects_tensors_for_non_tensor_inputs() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(floor.BetaBaseFloorV17Error, match="plain"):
        floor.select_one_shot_beta_base_v17(torch.tensor(True), reset=False)
    with pytest.raises(floor.BetaBaseFloorV17Error, match="plain"):
        floor.select_one_shot_beta_base_v17(False, reset=torch.tensor(False))


def test_v17_operator_has_only_boolean_inputs_and_no_serialization() -> None:
    source = ast.parse(inspect.getsource(floor))
    functions = {
        node.name: node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
    }
    arm = functions["arm_one_future_update_v17"]
    choose = functions["select_one_shot_beta_base_v17"]
    arm_names = {value.arg for value in [*arm.args.args, *arm.args.kwonlyargs]}
    choose_names = {value.arg for value in [*choose.args.args, *choose.args.kwonlyargs]}
    assert arm_names == {"armed", "alarm"}
    assert choose_names == {"armed", "reset"}
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
        for node in ast.walk(source)
    )
