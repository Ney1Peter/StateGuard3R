from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_selective_state_memory_repair_runner_v13 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_selective_state_memory_repair_runner_v13.py"


def _torch() -> object:
    return pytest.importorskip("torch")


def _view(torch: object) -> dict[str, object]:
    return {
        "img": torch.zeros((1, 3, 2, 3)), "img_mask": torch.tensor([True]),
        "ray_mask": torch.tensor([False]), "ray_map": torch.zeros((1, 2, 3, 3)),
        "true_shape": torch.tensor([2, 3]), "reset": None,
    }


def _replace_last(source: str, old: str, new: str) -> str:
    before, found, after = source.rpartition(old)
    assert found == old
    return before + new + after


def test_v13_runner_contract_binds_candidate_only_post_update_repair() -> None:
    audit = runner.audit_v13_runner_contract(RUNNER)
    assert audit["candidate_only_selective_state_memory_repair"] is True
    assert audit["repair_relation"] == "after_native_state_memory_update_and_calibration_before_cpu_transfer"
    assert audit["detector_before_repair"] is True
    assert audit["always_control_uses_legacy_raw_transfer"] is True
    assert audit["raw_current_prediction_is_not_replaced_by_runner"] is True
    assert audit["cpu_calibration_state_mirror_is_not_read"] is True
    assert audit["previous_recovery_delegation"] is False


def test_v13_runner_contract_rejects_wrong_order_previous_route_and_contaminated_control(tmp_path: Path) -> None:
    source = RUNNER.read_text(encoding="utf-8")
    moved = tmp_path / "moved.py"
    moved.write_text(
        source.replace(
            "model._maybe_record_u_calibration_step(i, previous_state, state_feat)",
            "selective_state_memory_repair(repair_prestate.state_feat, state_feat, repair_prestate.mem, mem, torch=torch)\n        model._maybe_record_u_calibration_step(i, previous_state, state_feat)",
            1,
        ).replace("repaired = selective_state_memory_repair(", "repaired = None  # ", 1),
        encoding="utf-8",
    )
    with pytest.raises(runner.SelectiveStateMemoryRepairRunnerError, match="ordering|contract"):
        runner.audit_v13_runner_contract(moved)
    contaminated = tmp_path / "control.py"
    entry = source.index("\ndef run_selective_state_memory_repair_recurrent_lighter(")
    contaminated.write_text(
        source[:entry] + source[entry:].replace(
            "predictions.append(to_cpu(res))",
            "selective_state_memory_repair(state_feat, state_feat, mem, mem, torch=torch)\n            predictions.append(to_cpu(res))",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.SelectiveStateMemoryRepairRunnerError, match="contaminated|contract"):
        runner.audit_v13_runner_contract(contaminated)
    delegated = tmp_path / "delegated.py"
    delegated.write_text(source + "\n# CurrentPointmapConsensus\n", encoding="utf-8")
    with pytest.raises(runner.SelectiveStateMemoryRepairRunnerError, match="earlier"):
        runner.audit_v13_runner_contract(delegated)
    cpu_mirror = tmp_path / "cpu-mirror.py"
    cpu_mirror.write_text(source + "\n# _u_calibration_last_state\n", encoding="utf-8")
    with pytest.raises(runner.SelectiveStateMemoryRepairRunnerError, match="CPU calibration"):
        runner.audit_v13_runner_contract(cpu_mirror)


def test_v13_candidate_repairs_persistent_rows_after_raw_detector_before_cpu_transfer() -> None:
    torch = _torch()
    events: list[str] = []

    class _Retriever:
        def __init__(self) -> None:
            self.mem = torch.zeros((1, 256, 1536))
            self.inquired_memory: list[object] = []

        def inquire(self, _global: object, _memory: object) -> object:
            events.append("inquire")
            self.inquired_memory.append(_memory.detach().clone())
            return torch.zeros((1, 1, 768))

        def update_mem(self, memory: object, _global: object, _pose: object) -> object:
            events.append("update_mem")
            row = torch.arange(1, 257, dtype=memory.dtype).reshape(1, 256, 1)
            return memory + row

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_update_type="cut3r")
            self.pose_head_flag, self.dec_depth = True, 1
            self.pose_token = torch.zeros((1, 1, 768))
            self.pose_retriever = _Retriever()
            self._recal3r_sequence_age = 0
            self.seen_state: list[object] = []
            self.seen_memory: list[object] = []

        def _uses_update_pressure_update(self) -> bool:
            return False

        def _encode_image(self, _images: object, shapes: object) -> tuple[list[object], object]:
            events.append("encode")
            return [torch.zeros((1, 5, 1024))], torch.zeros((1, 5, 2), dtype=shapes.dtype)

        def _encode_ray_map(self, _rays: object, _shapes: object) -> tuple[list[object], object]:
            raise AssertionError("ray encoder must not run")

        def _init_state(self, _feature: object, position: object) -> tuple[object, object]:
            return torch.zeros((1, 768, 768)), position

        def _init_recal3r_reference_state(self, _state: object) -> None:
            return None

        def _get_img_level_feat(self, feature: object) -> object:
            return feature.mean(dim=1, keepdim=True)

        def _recurrent_rollout(self, state: object, *_args: object, **_kwargs: object) -> tuple[object, list[object], object, object, object, object]:
            events.append("rollout")
            self.seen_state.append(state.detach().clone())
            self.seen_memory.append(self.pose_retriever.mem.detach().clone())
            row = torch.arange(1, 769, dtype=state.dtype).reshape(1, 768, 1)
            dec0 = torch.zeros((1, 5, 1024))
            return state + row, [dec0, torch.zeros((1, 6, 768))], None, None, None, None

        def _downstream_head(self, _head_input: object, _shapes: object, **_kwargs: object) -> dict[str, object]:
            return {"camera_pose": torch.full((1, 7), 3.0), "sentinel": "raw"}

        def _advance_recal3r_sequence_age(self, _state: object) -> None:
            events.append("advance")
            self._recal3r_sequence_age += 1

        def _maybe_record_u_calibration_step(self, _frame: int, _previous: object, _state: object) -> None:
            events.append("calibration")

        def _reset_update_pressure_if_needed(self, _reset: object) -> None:
            return None

        def _reset_recal3r_reference_state_if_needed(self, _reset: object, _initial: object) -> None:
            return None

    class _Observer:
        def __init__(self) -> None:
            self.observed: list[object] = []
            self.finalized: list[tuple[bool, object, object]] = []

        def observe(self, frame: int, prediction: object, _model: object, _context: object) -> object:
            events.append("observe")
            self.observed.append(prediction)
            return frame

        def alarm(self, observation: object) -> bool:
            events.append("alarm")
            return observation == 1

        def finalize(self, observation: object, *, repaired: bool, prediction: object, repair_evidence: object) -> tuple[object, dict[str, object]]:
            events.append("finalize")
            self.finalized.append((repaired, prediction, repair_evidence))
            return prediction, {"export_action": "export_real_camera_pose", "raw_pose_unchanged": True}

        def timeline_evidence(self, _observation: object) -> dict[str, object]:
            return {"detector_before_repair": True}

    model, observer = _Model(), _Observer()
    transferred: list[object] = []
    result = runner.run_selective_state_memory_repair_recurrent_lighter(
        [_view(torch), _view(torch), _view(torch)], model, torch.device("cpu"), torch=torch,
        to_gpu=lambda value, _device: value,
        to_cpu=lambda value: (transferred.append(value), value)[1],
        canonicalize_model_update_type=lambda value: value,
        before_frame=lambda frame, _view: {"overlap": None if frame == 0 else 0.5},
        observer=observer, verify_source=False, require_cuda_for_repair=False,
    )
    assert len(result.predictions) == len(transferred) == 3
    assert result.timeline[0]["action"] == "commit"
    assert result.timeline[1]["action"] == "selective_state_memory_repair"
    repair = result.timeline[1]["repair"]
    assert repair is not None
    assert repair["state_feat"]["selected_row_count"] == 96
    assert repair["mem"]["selected_row_count"] == 32
    assert result.timeline[1]["export_action"] == "export_real_camera_pose"
    assert observer.finalized[1][0] is True
    assert observer.finalized[1][1]["camera_pose"] is transferred[1]["camera_pose"]
    frame_one = len(events) // 3
    assert events[frame_one:].index("calibration") < events[frame_one:].index("observe") < events[frame_one:].index("alarm") < events[frame_one:].index("finalize")
    # The third recurrent rollout receives frame 1's partially repaired state,
    # not either full pre-state (zero) or full proposed state (two times row).
    state_at_frame_two = model.seen_state[2]
    memory_at_frame_two = model.pose_retriever.inquired_memory[1]
    assert float(state_at_frame_two[0, 767, 0]) == pytest.approx(768.0)
    assert float(state_at_frame_two[0, 0, 0]) == pytest.approx(2.0)
    assert float(memory_at_frame_two[0, 255, 0]) == pytest.approx(256.0)
    assert float(memory_at_frame_two[0, 0, 0]) == pytest.approx(2.0)

    control = _Model()
    control_result = runner.run_selective_state_memory_repair_recurrent_lighter(
        [_view(torch)], control, torch.device("cpu"), torch=torch, to_gpu=lambda value, _device: value,
        to_cpu=lambda value: value, canonicalize_model_update_type=lambda value: value,
        observer=None, verify_source=False,
    )
    assert control_result.timeline[0]["reason"] == "always_commit_control"
    assert control_result.timeline[0]["repair"] is None
