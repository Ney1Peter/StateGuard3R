from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_bounded_update_pressure_runner_v16 as runner


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "stateguard3r" / "recal3r_bounded_update_pressure_runner_v16.py"


def _torch() -> object:
    return pytest.importorskip("torch")


def _view(torch: object) -> dict[str, object]:
    return {"img": torch.zeros((1, 3, 2, 3)), "img_mask": torch.tensor([True]), "ray_mask": torch.tensor([False]), "ray_map": torch.zeros((1, 2, 3, 3)), "true_shape": torch.tensor([2, 3]), "reset": None}


def test_v16_runner_contract_is_own_ordered_and_rejects_legacy_import(tmp_path: Path) -> None:
    audit = runner.audit_v16_runner_contract(RUNNER)
    assert audit["native_update_before_detector"] is True
    assert audit["native_reset_before_detector"] is True
    assert audit["detector_commit_before_pressure"] is True
    assert audit["pressure_before_raw_cpu_transfer"] is True
    polluted = tmp_path / "polluted.py"
    polluted.write_text("from stateguard3r.rgb_listing_capsule_v15 import x\n" + RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(runner.RecurrentBoundedPressureV16Error, match="prohibited"):
        runner.audit_v16_runner_contract(polluted)
    improper = tmp_path / "improper.py"
    improper.write_text(RUNNER.read_text(encoding="utf-8").replace("inject_bounded_pressure_v16(getattr(model, \"update_pressure\", None)", "inject_bounded_pressure_v16(state_feat", 1), encoding="utf-8")
    with pytest.raises(runner.RecurrentBoundedPressureV16Error, match="prohibited value"):
        runner.audit_v16_runner_contract(improper)


def test_v16_fake_recurrence_clears_stale_pressure_and_only_changes_later_native_update(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    rollout_pressure: list[object] = []
    monkeypatch.setattr(runner, "gpu_fingerprint_v16", lambda value, **_kwargs: f"fp:{float(value.reshape(-1)[0])}")
    monkeypatch.setattr(runner, "inject_bounded_pressure_v16", lambda native, **_kwargs: ((torch.full((1, 768, 1), .25) if native is None else native + .25), {"operator": "test", "increment": .25, "cap": 1.0}))

    class Retriever:
        def __init__(self) -> None: self.mem = torch.zeros((1, 256, 1536))
        def inquire(self, *_args: object) -> object: return torch.zeros((1, 1, 768))
        def update_mem(self, mem: object, *_args: object) -> object: return mem + 1
    class Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_update_type="recal3r")
            self.pose_head_flag, self.dec_depth = True, 1
            self.pose_token, self.pose_retriever, self.update_pressure = torch.zeros((1, 1, 768)), Retriever(), torch.full((1, 768, 1), .9)
        def _encode_image(self, _img: object, shape: object) -> tuple[list[object], object]: return [torch.zeros((1, 5, 1024))], torch.zeros((1, 5, 2), dtype=shape.dtype)
        def _encode_ray_map(self, *_args: object) -> object: raise AssertionError("no ray")
        def _init_state(self, _feat: object, pos: object) -> tuple[object, object]: return torch.zeros((1, 768, 768)), pos
        def _init_recal3r_reference_state(self, *_args: object) -> None: pass
        def _get_img_level_feat(self, feat: object) -> object: return feat.mean(dim=1, keepdim=True)
        def _recurrent_rollout(self, state: object, *_args: object, **_kwargs: object) -> tuple[object, list[object], object, object, object, object]:
            rollout_pressure.append(getattr(self, "update_pressure", None)); return state + 1, [torch.zeros((1, 5, 1024)), torch.zeros((1, 6, 768))], None, None, None, None
        def _downstream_head(self, *_args: object, **_kwargs: object) -> dict[str, object]: return {"camera_pose": torch.full((1, 7), 2.0), "sentinel": "raw"}
        def _compute_recal3r_update_mask(self, mask: object, *_args: object, **_kwargs: object) -> object: return mask
        def _advance_recal3r_sequence_age(self, *_args: object) -> None: pass
        def _maybe_record_u_calibration_step(self, *_args: object) -> None: pass
        def _reset_update_pressure_if_needed(self, *_args: object) -> None: pass
        def _reset_recal3r_reference_state_if_needed(self, *_args: object) -> None: pass
    class Observer:
        def observe(self, frame: int, *_args: object) -> int: return frame
        def alarm(self, frame: int) -> bool: return frame == 0
        def finalize(self, _value: object) -> dict[str, object]: return {"detector_history_action": "commit_raw_current_health"}
        def timeline_evidence(self, _value: object) -> dict[str, object]: return {"online_detector": {"hybrid_alarm": True}}

    model = Model()
    output = runner.run_recurrent_bounded_pressure_v16([_view(torch), _view(torch)], model, torch.device("cpu"), torch=torch, to_gpu=lambda value, _device: value, to_cpu=lambda value: value, canonicalize_model_update_type=lambda value: value, observer=Observer(), verify_source=False, require_cuda_candidate=False)
    assert rollout_pressure[0] is None
    assert float(rollout_pressure[1][0, 0, 0]) == pytest.approx(.25)
    assert output.timeline[0]["action"] == "bounded_update_pressure_injection"
    assert output.predictions[0]["sentinel"] == "raw"
