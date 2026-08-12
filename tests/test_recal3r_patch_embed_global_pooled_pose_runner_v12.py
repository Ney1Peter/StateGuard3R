from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_patch_embed_global_pooled_pose_runner_v12 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_patch_embed_global_pooled_pose_runner_v12.py"


def _replace_last(source: str, old: str, new: str) -> str:
    before, found, after = source.rpartition(old)
    assert found == old
    return before + new + after


def _torch() -> object:
    return pytest.importorskip("torch")


def _view(torch: object) -> dict[str, object]:
    return {
        "img": torch.zeros((1, 3, 2, 3)),
        "img_mask": torch.tensor([True]),
        "ray_mask": torch.tensor([False]),
        "ray_map": torch.zeros((1, 2, 3, 3)),
        "true_shape": torch.tensor([2, 3]),
        "reset": None,
    }


def test_v12_runner_contract_binds_one_candidate_patch_capture_before_encoder_and_raw_control() -> None:
    audit = runner.audit_v12_runner_contract(RUNNER)
    assert audit["patch_feature_shape"] == [1, "N", 1024]
    assert audit["patch_global_shape"] == [1, 1, 1024]
    assert audit["projected_shape"] == [1, 1, 768]
    assert audit["actual_patch_token_count_is_runtime_evidence"] is True
    assert audit["capture_relation"] == "after_feature_selection_before_full_encoder_retrieval_rollout_update_mem"
    assert audit["candidate_only_patch_embed_projection_and_clone"] is True
    assert audit["candidate_metadata"] == "causal_overlap_and_loaded_height_width_only"
    assert audit["always_control_uses_legacy_raw_transfer"] is True
    assert audit["candidate_finalize_after_rollback_before_cpu_transfer"] is True


def test_v12_runner_contract_rejects_decoder_pose_retrieval_and_control_mutations(tmp_path: Path) -> None:
    original = RUNNER.read_text(encoding="utf-8")
    decoder = tmp_path / "decoder.py"
    decoder.write_text(
        _replace_last(original, "patch_tokens_i, _patch_pos_i = model.patch_embed(selected_imgs, true_shape=selected_shapes)", "patch_tokens_i, _patch_pos_i = dec[0], None"),
        encoding="utf-8",
    )
    with pytest.raises(runner.PatchEmbedGlobalPooledPoseRunnerError, match="source contract|forbidden data"):
        runner.audit_v12_runner_contract(decoder)

    pose = tmp_path / "pose.py"
    pose.write_text(
        _replace_last(original, "patch_tokens_i, _patch_pos_i = model.patch_embed(selected_imgs, true_shape=selected_shapes)", "patch_tokens_i, _patch_pos_i = model.pose_token, None"),
        encoding="utf-8",
    )
    with pytest.raises(runner.PatchEmbedGlobalPooledPoseRunnerError, match="source contract|forbidden data"):
        runner.audit_v12_runner_contract(pose)

    moved = tmp_path / "moved.py"
    capture = "patch_embed_global_pose_capture = capture_patch_embed_global_pose_token("
    moved.write_text(
        original.replace(capture, "patch_embed_global_pose_capture = None  # ", 1).replace(
            "        if observer is None:\n",
            "        " + capture + "patch_tokens_i, projection=model.decoder_embed, torch=torch,\n        )\n        if observer is None:\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.PatchEmbedGlobalPooledPoseRunnerError, match="source contract|before encoder|exactly one"):
        runner.audit_v12_runner_contract(moved)

    contaminated = tmp_path / "control.py"
    entry = original.index("\ndef run_patch_embed_global_pooled_pose_recurrent_lighter(")
    contaminated.write_text(
        original[:entry] + original[entry:].replace(
            "predictions.append(to_cpu(res))",
            "model.patch_embed(selected_imgs, true_shape=selected_shapes)\n            predictions.append(to_cpu(res))",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(runner.PatchEmbedGlobalPooledPoseRunnerError, match="exactly one|contaminated"):
        runner.audit_v12_runner_contract(contaminated)


def test_v12_candidate_captures_native_patch_grid_before_encoder_and_rolls_back_before_finalize() -> None:
    torch = _torch()

    class _Retriever:
        def __init__(self) -> None:
            self.mem = torch.zeros((1, 1, 768))

        def inquire(self, _global: object, _mem: object) -> object:
            events.append("inquire")
            return torch.zeros((1, 1, 768))

        def update_mem(self, mem: object, _global: object, _final: object) -> object:
            events.append("update")
            return mem + 1.0

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_update_type="cut3r")
            self.pose_head_flag = True
            self.dec_depth = 1
            self.pose_token = torch.zeros((1, 1, 768))
            self.pose_retriever = _Retriever()
            self.decoder_embed = torch.nn.Linear(1024, 768)
            self.raw_prediction = None
            self.patch_embed_calls = 0
            self.projection_calls = 0
            self._recal3r_sequence_age = 0
            original_projection = self.decoder_embed.forward

            def counted_projection(value: object) -> object:
                self.projection_calls += 1
                events.append("projection")
                return original_projection(value)

            self.decoder_embed.forward = counted_projection

        def _uses_update_pressure_update(self) -> bool:
            return False

        def patch_embed(self, _images: object, *, true_shape: object) -> tuple[object, object]:
            self.patch_embed_calls += 1
            events.append("patch_embed")
            assert tuple(true_shape.shape) == (1, 2)
            feature = torch.arange(5 * 1024, dtype=torch.float32).reshape((1, 5, 1024)) / 1024.0
            return feature, torch.zeros((1, 5, 2))

        def _encode_image(self, images: object, shapes: object) -> tuple[list[object], object]:
            events.append("encode")
            feature, position = self.patch_embed(images, true_shape=shapes)
            return [feature], position

        def _encode_ray_map(self, _rays: object, _shapes: object) -> tuple[list[object], object]:
            raise AssertionError("ray encoder must not run")

        def _init_state(self, _feature: object, position: object) -> tuple[object, object]:
            return torch.zeros((1, 1, 768)), position

        def _init_recal3r_reference_state(self, _state: object) -> None:
            return None

        def _get_img_level_feat(self, feature: object) -> object:
            return feature.mean(dim=1, keepdim=True)

        def _recurrent_rollout(self, state: object, *_args: object, **_kwargs: object) -> tuple[object, list[object], object, object, object, object]:
            events.append("rollout")
            self._recal3r_sequence_age = 99
            dec0 = torch.zeros((1, 5, 1024))
            return state + 1.0, [dec0, torch.zeros((1, 6, 768))], None, None, None, None

        def _downstream_head(self, _input: object, _shapes: object, **_kwargs: object) -> object:
            self.raw_prediction = {"camera_pose": torch.zeros((1, 7))}
            return self.raw_prediction

        def _advance_recal3r_sequence_age(self, _state: object) -> None:
            self._recal3r_sequence_age += 1

        def _maybe_record_u_calibration_step(self, _frame: int, _previous: object, _state: object) -> None:
            return None

        def _reset_update_pressure_if_needed(self, _reset: object) -> None:
            return None

        def _reset_recal3r_reference_state_if_needed(self, _reset: object, _initial: object) -> None:
            return None

    class _Observer:
        def __init__(self, model: object) -> None:
            self.model = model
            self.captures: list[object] = []
            self.contexts: list[object] = []
            self.finalized = 0

        def observe(self, frame: int, prediction: object, _model: object, context: object, capture: object) -> object:
            events.append("observe")
            self.captures.append(capture)
            self.contexts.append(context)
            return frame, prediction

        def alarm(self, observation: object) -> bool:
            return observation[0] == 1

        def finalize(self, observation: object, *, quarantined: bool, prediction: object) -> tuple[object, dict[str, object]]:
            events.append("finalize")
            if quarantined:
                assert self.model._recal3r_sequence_age == 100
            self.finalized += 1
            assert observation[1] is prediction
            return prediction, {"export_action": "export_real_camera_pose"}

        def timeline_evidence(self, _observation: object) -> dict[str, object]:
            return {}

    events: list[str] = []
    candidate_model = _Model()
    observer = _Observer(candidate_model)
    candidate = runner.run_patch_embed_global_pooled_pose_recurrent_lighter(
        [_view(torch), _view(torch)], candidate_model, torch.device("cpu"), torch=torch,
        to_gpu=lambda item, _device: item, to_cpu=lambda item: item,
        canonicalize_model_update_type=lambda value: value, before_frame=lambda frame, _view: {"overlap": None if frame == 0 else 0.5, "img": "must_not_cross_metadata"},
        observer=observer, verify_source=False,
    )
    assert len(candidate.predictions) == 2
    # One observer patch call plus the ordinary encoder's own native patch call per frame.
    assert candidate_model.patch_embed_calls == 4
    assert candidate_model.projection_calls == 2
    assert [capture.patch_feature_shape for capture in observer.captures] == [(1, 5, 1024), (1, 5, 1024)]
    assert observer.contexts == [
        {"overlap": None, "v12_loaded_image_shape": [2, 3]},
        {"overlap": 0.5, "v12_loaded_image_shape": [2, 3]},
    ]
    assert candidate.timeline[1]["action"] == "quarantine_current_rollback"
    assert events.index("patch_embed") < events.index("projection") < events.index("encode") < events.index("rollout") < events.index("update") < events.index("observe")
    assert events.count("finalize") == 2

    control_model = _Model()
    control = runner.run_patch_embed_global_pooled_pose_recurrent_lighter(
        [_view(torch)], control_model, torch.device("cpu"), torch=torch,
        to_gpu=lambda item, _device: item, to_cpu=lambda item: item,
        canonicalize_model_update_type=lambda value: value, observer=None,
        verify_source=False,
    )
    assert control.predictions == [control_model.raw_prediction]
    assert control.timeline[0]["reason"] == "always_commit_control"
    assert control_model.patch_embed_calls == 1
    assert control_model.projection_calls == 0
