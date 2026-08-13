from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_beta_base_floor_runner_v17 as runner


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "stateguard3r" / "recal3r_beta_base_floor_runner_v17.py"


def _torch() -> object:
    return pytest.importorskip("torch")


def _view(torch: object, *, reset: bool = False) -> dict[str, object]:
    return {
        "img": torch.zeros((1, 3, 2, 3)),
        "img_mask": torch.tensor([True]),
        "ray_mask": torch.tensor([False]),
        "ray_map": torch.zeros((1, 2, 3, 3)),
        "true_shape": torch.tensor([2, 3]),
        "reset": torch.tensor([reset]) if reset else None,
    }


class _Retriever:
    def __init__(self, torch: object) -> None:
        self.mem = torch.zeros((1, 256, 1536))

    def inquire(self, *_args: object) -> object:
        torch = _torch()
        return torch.zeros((1, 1, 768))

    def update_mem(self, mem: object, *_args: object) -> object:
        return mem + 1


class _Model:
    def __init__(self, torch: object, *, fail_mask: bool = False) -> None:
        self.config = SimpleNamespace(model_update_type="recal3r", beta_base=0.1)
        self.pose_head_flag = True
        self.dec_depth = 1
        self.pose_token = torch.zeros((1, 1, 768))
        self.pose_retriever = _Retriever(torch)
        self.mask_bases: list[float] = []
        self.reset_calls = 0
        self.fail_mask = fail_mask
        self.update_pressure = object()

    def _beta_base(self) -> float:
        return float(getattr(self, "beta_base", self.config.beta_base))

    def _encode_image(self, _img: object, shape: object) -> tuple[list[object], object]:
        torch = _torch()
        return [torch.zeros((1, 5, 1024))], torch.zeros((1, 5, 2), dtype=shape.dtype)

    def _encode_ray_map(self, *_args: object) -> object:
        raise AssertionError("fake input contains no ray map")

    def _init_state(self, _feat: object, pos: object) -> tuple[object, object]:
        torch = _torch()
        return torch.zeros((1, 768, 768)), pos

    def _init_recal3r_reference_state(self, *_args: object) -> None:
        return None

    def _get_img_level_feat(self, feat: object) -> object:
        return feat.mean(dim=1, keepdim=True)

    def _recurrent_rollout(
        self, state: object, *_args: object, **_kwargs: object
    ) -> tuple[object, list[object], object, object, object, object]:
        torch = _torch()
        return (
            state + 1,
            [torch.zeros((1, 5, 1024)), torch.zeros((1, 6, 768))],
            None,
            None,
            None,
            None,
        )

    def _downstream_head(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        torch = _torch()
        return {"camera_pose": torch.full((1, 7), 2.0), "raw_sentinel": "unaltered"}

    def _compute_recal3r_update_mask(self, mask: object, *_args: object, **_kwargs: object) -> object:
        self.mask_bases.append(self._beta_base())
        if self.fail_mask:
            raise RuntimeError("injected native mask failure")
        return mask * self._beta_base()

    def _compute_state_update_mask(self, mask: object, *_args: object) -> tuple[object, object, object]:
        return mask, None, None

    def _advance_recal3r_sequence_age(self, *_args: object) -> None:
        return None

    def _maybe_record_u_calibration_step(self, *_args: object) -> None:
        return None

    def _reset_update_pressure_if_needed(self, *_args: object) -> None:
        self.reset_calls += 1

    def _reset_recal3r_reference_state_if_needed(self, *_args: object) -> None:
        return None


class _Observer:
    def __init__(self, alarms: set[int]) -> None:
        self.alarms = alarms
        self.committed: list[int] = []

    def observe(self, frame_id: int, prediction: object, *_args: object) -> tuple[int, object]:
        return frame_id, prediction

    def alarm(self, observation: tuple[int, object]) -> bool:
        return observation[0] in self.alarms

    def finalize(self, observation: tuple[int, object]) -> dict[str, object]:
        self.committed.append(observation[0])
        return {"detector_history_action": "commit_raw_current_health"}

    def timeline_evidence(self, observation: tuple[int, object]) -> dict[str, object]:
        return {"online_detector": {"frame_id": observation[0]}}


def _run(torch: object, views: list[dict[str, object]], observer: object | None) -> tuple[_Model, object]:
    model = _Model(torch)
    result = runner.run_recurrent_beta_floor_native_v17(
        views,
        model,
        torch.device("cpu"),
        torch=torch,
        to_gpu=lambda value, _device: value,
        to_cpu=lambda value: value,
        canonicalize_model_update_type=lambda value: value,
        observer=observer,
        verify_source=False,
        require_cuda_candidate=False,
    )
    return model, result


def test_v17_control_has_no_arm_scope_or_observer_evidence() -> None:
    torch = _torch()
    model, result = _run(torch, [_view(torch), _view(torch), _view(torch)], None)
    assert model.mask_bases == [0.1, 0.1]
    assert all(row["beta_floor"] is None for row in result.timeline)
    assert all(row["action"] == "commit" for row in result.timeline)
    assert all(value["raw_sentinel"] == "unaltered" for value in result.predictions)


def test_v17_alarm_changes_only_next_native_mask_and_restores_scalar() -> None:
    torch = _torch()
    observer = _Observer({0})
    model, result = _run(torch, [_view(torch), _view(torch), _view(torch)], observer)
    assert observer.committed == [0, 1, 2]
    assert model.mask_bases == [0.0, 0.1]
    assert not hasattr(model, "beta_base")
    consumed = result.timeline[1]["beta_floor"]
    assert consumed is not None
    assert consumed["arm_consumed"] is True
    assert consumed["native_beta_base_during_mask"] == 0.0
    assert consumed["native_beta_base_after_mask"] == 0.1
    assert result.timeline[0]["action"] == "arm_one_future_native_beta_floor"
    assert all(value["raw_sentinel"] == "unaltered" for value in result.predictions)
    assert model.update_pressure is not None


def test_v17_reset_cancels_arm_and_never_carries_it_forward() -> None:
    torch = _torch()
    observer = _Observer({0})
    model, result = _run(torch, [_view(torch), _view(torch, reset=True), _view(torch), _view(torch)], observer)
    assert model.mask_bases == [0.1]
    cancelled = result.timeline[1]["beta_floor"]
    assert cancelled is not None
    assert cancelled["arm_reset_cancelled"] is True
    assert cancelled["arm_consumed"] is False
    assert model.reset_calls == 1


def test_v17_scalar_scope_restores_on_native_exception() -> None:
    torch = _torch()
    model = _Model(torch, fail_mask=True)
    with pytest.raises(RuntimeError, match="injected native mask failure"):
        with runner.temporary_native_beta_floor_v17(model, beta_base=0.0):
            model._compute_recal3r_update_mask(torch.ones((1, 1, 1)))
    assert not hasattr(model, "beta_base")
    assert model._beta_base() == 0.1


def test_v17_pinned_native_source_and_contract_are_bound() -> None:
    provenance = runner.pinned_native_v17_source()
    assert provenance["native_beta_equation_lines"] == "1199-1251"
    assert provenance["lighter_lines"] == "1661-1833"
