from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_early_spatial_pooled_pose_runner_v9 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_early_spatial_pooled_pose_runner_v9.py"


def _replace_last(source: str, old: str, new: str) -> str:
    before, found, after = source.rpartition(old)
    assert found == old
    return before + new + after


def _torch() -> object:
    return pytest.importorskip("torch")


def _view(torch: object) -> dict[str, object]:
    return {
        "img": torch.zeros((1, 3, 2, 2)),
        "img_mask": torch.tensor([True]),
        "ray_mask": torch.tensor([False]),
        "ray_map": torch.zeros((1, 2, 2, 3)),
        "true_shape": torch.tensor([2, 2]),
        "reset": None,
    }


def test_v9_runner_contract_binds_candidate_pool_before_update_and_raw_control_path() -> None:
    audit = runner.audit_v9_runner_contract(RUNNER)
    assert audit["early_decoder_layer_index"] == 0
    assert audit["preprojection_spatial_token_count"] == 576
    assert audit["preprojection_width"] == 1024
    assert audit["projected_width"] == 768
    assert audit["aggregation"] == "mean"
    assert audit["candidate_only_capture"] is True
    assert audit["always_control_uses_legacy_raw_transfer"] is True
    assert audit["candidate_finalize_before_cpu_transfer"] is True


def test_v9_runner_contract_rejects_special_token_and_after_update_mutations(tmp_path: Path) -> None:
    original = RUNNER.read_text(encoding="utf-8")
    special = tmp_path / "special.py"
    special.write_text(_replace_last(original, "dec[0].mean(dim=1, keepdim=True)", "dec[-1][:, 0:1]"), encoding="utf-8")
    with pytest.raises(runner.EarlySpatialPooledPoseRunnerError, match="source contract"):
        runner.audit_v9_runner_contract(special)

    moved = tmp_path / "moved.py"
    capture = "early_spatial_pooled_pose_token = model.decoder_embed(dec[0].mean(dim=1, keepdim=True)).detach().clone()"
    source = _replace_last(original, capture, "early_spatial_pooled_pose_token = None")
    source = source.replace(
        "        head_input = [dec[0].float()",
        f"        {capture}\n        head_input = [dec[0].float()",
        1,
    )
    moved.write_text(source, encoding="utf-8")
    with pytest.raises(runner.EarlySpatialPooledPoseRunnerError, match="source contract|after rollout"):
        runner.audit_v9_runner_contract(moved)


def test_v9_runner_candidate_pools_before_memory_mutation_and_control_avoids_projection() -> None:
    torch = _torch()

    class _Retriever:
        def __init__(self) -> None:
            self.mem = torch.zeros((1, 1, 768))
            self.model = None

        def inquire(self, _global: object, _mem: object) -> object:
            return torch.zeros((1, 1, 768))

        def update_mem(self, mem: object, _global: object, _final: object) -> object:
            assert self.model is not None
            self.model.dec0.add_(100.0)
            return mem + 1.0

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_update_type="cut3r")
            self.pose_head_flag = True
            self.dec_depth = 1
            self.pose_token = torch.zeros((1, 1, 768))
            self.pose_retriever = _Retriever()
            self.pose_retriever.model = self
            self.decoder_embed = torch.nn.Linear(1024, 768)
            self.dec0 = None
            self.raw_prediction = None
            self.projection_calls = 0
            original_forward = self.decoder_embed.forward

            def counted(value: object) -> object:
                self.projection_calls += 1
                return original_forward(value)

            self.decoder_embed.forward = counted

        def _uses_update_pressure_update(self) -> bool:
            return False

        def _encode_image(self, _images: object, _shapes: object) -> tuple[list[object], object]:
            return [torch.zeros((1, 576, 1024))], torch.zeros((1, 576, 2))

        def _encode_ray_map(self, _rays: object, _shapes: object) -> tuple[list[object], object]:
            raise AssertionError("ray encoder must not run")

        def _init_state(self, feature: object, position: object) -> tuple[object, object]:
            return torch.zeros((1, 1, 768)), position

        def _init_recal3r_reference_state(self, _state: object) -> None:
            return None

        def _get_img_level_feat(self, _feature: object) -> object:
            return torch.zeros((1, 1, 1024))

        def _recurrent_rollout(self, state: object, *_args: object, **_kwargs: object) -> tuple[object, list[object], object, object, object, object]:
            self.dec0 = torch.arange(576 * 1024, dtype=torch.float32).reshape((1, 576, 1024)) / 1024.0
            return state + 1.0, [self.dec0, torch.zeros((1, 577, 768))], None, None, None, None

        def _downstream_head(self, _input: object, _shapes: object, **_kwargs: object) -> object:
            self.raw_prediction = {"camera_pose": torch.zeros((1, 7))}
            return self.raw_prediction

        def _advance_recal3r_sequence_age(self, _state: object) -> None:
            return None

        def _maybe_record_u_calibration_step(self, _frame: int, _previous: object, _state: object) -> None:
            return None

        def _reset_update_pressure_if_needed(self, _reset: object) -> None:
            return None

        def _reset_recal3r_reference_state_if_needed(self, _reset: object, _initial: object) -> None:
            return None

    class _Observer:
        def __init__(self) -> None:
            self.tokens: list[object] = []
            self.finalized = 0

        def observe(self, _frame: int, prediction: object, _model: object, _context: object, token: object) -> object:
            self.tokens.append(token)
            return prediction

        def alarm(self, _observation: object) -> bool:
            return False

        def finalize(self, observation: object, *, quarantined: bool, prediction: object) -> tuple[object, dict[str, object]]:
            assert observation is prediction and quarantined is False
            self.finalized += 1
            return prediction, {"export_action": "export_real_camera_pose"}

        def timeline_evidence(self, _observation: object) -> dict[str, object]:
            return {}

    candidate_model, observer = _Model(), _Observer()
    candidate = runner.run_early_spatial_pooled_pose_recurrent_lighter(
        [_view(torch)], candidate_model, torch.device("cpu"), torch=torch,
        to_gpu=lambda item, _device: item, to_cpu=lambda item: item,
        canonicalize_model_update_type=lambda value: value, observer=observer,
        verify_source=False,
    )
    assert len(candidate.predictions) == 1
    assert observer.finalized == 1
    assert candidate_model.projection_calls == 1
    assert len(observer.tokens) == 1
    expected_input = torch.arange(576 * 1024, dtype=torch.float32).reshape((1, 576, 1024)) / 1024.0
    expected = candidate_model.decoder_embed(expected_input.mean(dim=1, keepdim=True))
    torch.testing.assert_close(observer.tokens[0], expected)
    assert tuple(observer.tokens[0].shape) == (1, 1, 768)
    assert torch.all(candidate_model.dec0 == expected_input + 100.0)

    control_model = _Model()
    control = runner.run_early_spatial_pooled_pose_recurrent_lighter(
        [_view(torch)], control_model, torch.device("cpu"), torch=torch,
        to_gpu=lambda item, _device: item, to_cpu=lambda item: item,
        canonicalize_model_update_type=lambda value: value, observer=None,
        verify_source=False,
    )
    assert control.predictions == [control_model.raw_prediction]
    assert control.timeline[0]["reason"] == "always_commit_control"
    assert control_model.projection_calls == 0
