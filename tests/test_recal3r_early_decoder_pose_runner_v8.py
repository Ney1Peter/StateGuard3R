from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_early_decoder_pose_runner_v8 as runner
from stateguard3r.early_decoder_pose_v8 import EarlyDecoderPoseError, audit_pinned_early_decoder_pose_sources
from scripts import run_recal3r_early_decoder_pose_export_v8 as script


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECAL3R_SRC = PROJECT_ROOT.parent / "baselines" / "ReCal3R" / "src"
RUNNER = PROJECT_ROOT / "src" / "stateguard3r" / "recal3r_early_decoder_pose_runner_v8.py"


class _Flag:
    def __init__(self, value: bool) -> None:
        self.value = value

    def detach(self) -> "_Flag":
        return self

    def any(self) -> "_Flag":
        return self

    def item(self) -> bool:
        return self.value


def test_reset_predicate_accepts_bool_and_tensor_like_values() -> None:
    assert runner._reset_requested(None) is False
    assert runner._reset_requested(False) is False
    assert runner._reset_requested(_Flag(True)) is True


def test_runner_contract_captures_only_candidate_layer_zero_before_memory_update() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    body = source[source.index("\ndef run_early_decoder_pose_recurrent_lighter(") + 1:]
    audit = runner.audit_v8_runner_contract(RUNNER)
    assert audit["early_decoder_layer_index"] == 0
    assert audit["capture_relation"] == "after_rollout_before_update_mem"
    assert audit["candidate_only_capture"] is True
    assert audit["always_control_uses_legacy_raw_transfer"] is True
    assert audit["candidate_finalize_before_cpu_transfer"] is True
    assert audit["previous_recovery_delegation"] is False

    rollout = body.index("model._recurrent_rollout(")
    capture = body.index("early_decoder_pose_token = dec[0][:, 0:1].detach().clone() if observer is not None else None")
    update = body.index("model.pose_retriever.update_mem(", capture)
    candidate_finalize = body.index("observer.finalize(observation, quarantined=quarantine, prediction=res)")
    candidate_transfer = body.index("predictions.append(to_cpu(exported))", candidate_finalize)
    control_start = body.index("if observer is None:", update)
    candidate_start = body.index("else:\n            observation = observer.observe", control_start)
    control = body[control_start:candidate_start]
    assert rollout < capture < update < candidate_finalize < candidate_transfer
    assert "early_decoder_pose_token =" not in control
    assert "observer." not in control
    assert "predictions.append(to_cpu(res))" in control
    assert ".cpu()" not in body[capture:candidate_transfer]


def test_runner_contract_rejects_layer_and_order_mutations(tmp_path: Path) -> None:
    layer_mutant = tmp_path / "layer_mutant.py"
    original = RUNNER.read_text(encoding="utf-8")
    entry = original.index("\ndef run_early_decoder_pose_recurrent_lighter(")
    layer_mutant.write_text(original[:entry] + original[entry:].replace("dec[0][:, 0:1].detach().clone()", "dec[-1][:, 0:1].detach().clone()", 1), encoding="utf-8")
    with pytest.raises(runner.EarlyDecoderPoseRunnerError, match="source contract"):
        runner.audit_v8_runner_contract(layer_mutant)

    order_mutant = tmp_path / "order_mutant.py"
    source = RUNNER.read_text(encoding="utf-8")
    entry = source.index("\ndef run_early_decoder_pose_recurrent_lighter(")
    prefix, source = source[:entry], source[entry:]
    capture = "early_decoder_pose_token = dec[0][:, 0:1].detach().clone() if observer is not None else None"
    source = source.replace(capture + "\n        new_mem = model.pose_retriever.update_mem", "new_mem = model.pose_retriever.update_mem", 1)
    source = source.replace("        head_input = [dec[0].float()", f"        {capture}\n        head_input = [dec[0].float()", 1)
    order_mutant.write_text(prefix + source, encoding="utf-8")
    with pytest.raises(runner.EarlyDecoderPoseRunnerError, match="source contract|after rollout"):
        runner.audit_v8_runner_contract(order_mutant)


def test_combined_source_audit_rejects_runner_capture_after_update(tmp_path: Path) -> None:
    original = RUNNER.read_text(encoding="utf-8")
    entry = original.index("\ndef run_early_decoder_pose_recurrent_lighter(")
    prefix, source = original[:entry], original[entry:]
    capture = "early_decoder_pose_token = dec[0][:, 0:1].detach().clone() if observer is not None else None"
    source = source.replace(capture + "\n        new_mem = model.pose_retriever.update_mem", "new_mem = model.pose_retriever.update_mem", 1)
    source = source.replace("        head_input = [dec[0].float()", f"        {capture}\n        head_input = [dec[0].float()", 1)
    mutated = tmp_path / "runner.py"
    mutated.write_text(prefix + source, encoding="utf-8")
    dust3r = RECAL3R_SRC / "dust3r"
    with pytest.raises(EarlyDecoderPoseError, match="order"):
        audit_pinned_early_decoder_pose_sources(dust3r / "model.py", dust3r / "heads" / "dpt_head.py", dust3r / "heads" / "postprocess.py", mutated)


def test_v8_script_exposes_only_preregistered_policy_and_components() -> None:
    action = next(item for item in script._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-early-decoder-pose-export")
    assert script._parser().get_default("watchdog") == 8
    assert script.EARLY_DECODER_TOKEN_SOURCE == "decoder_layer_0_after_rollout_before_update_mem"
    source = Path(script.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "SafeAnchorSE3Export",
        "RealSafeGeometricAnchor",
        "CurrentPointmapConsensusExport",
        "PreRolloutPoseQueryExport",
        "register_anchor_orb_3d3d",
    ):
        assert forbidden not in source
    component_paths = {str(item.resolve()) for item in script.V8_COMPONENTS}
    assert str(Path(script.__file__).resolve()) in component_paths
    assert any(path.endswith("recal3r_early_decoder_pose_runner_v8.py") for path in component_paths)


def test_v8_script_fails_closed_without_the_pinned_dpt_pose_head() -> None:
    class _PinnedHead:
        pose_mode = ("exp", -float("inf"), float("inf"))

        def pose_head(self, value: object) -> object:
            return value

    module = SimpleNamespace(DPTPts3dPose=_PinnedHead)
    pose_head, pose_mode = script._pinned_pose_head_interface(SimpleNamespace(downstream_head=_PinnedHead()), module)
    assert callable(pose_head)
    assert pose_mode == _PinnedHead.pose_mode
    with pytest.raises(RuntimeError, match="not pinned"):
        script._pinned_pose_head_interface(SimpleNamespace(downstream_head=object()), module)

    class _NoPose:
        pass

    with pytest.raises(RuntimeError, match="interface"):
        script._pinned_pose_head_interface(SimpleNamespace(downstream_head=_NoPose()), SimpleNamespace(DPTPts3dPose=_NoPose))


def test_v8_provenance_rejects_an_untracked_runtime_component(monkeypatch: pytest.MonkeyPatch) -> None:
    component = Path(script.__file__).resolve()
    monkeypatch.setattr(script, "V8_COMPONENTS", (component,))

    def fake_git_output(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return "commit"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args[:2] == ("ls-files", "--error-unmatch"):
            raise RuntimeError("not tracked")
        raise AssertionError(args)

    monkeypatch.setattr(script.smoke, "_git_output", fake_git_output)
    with pytest.raises(RuntimeError, match="not tracked at HEAD"):
        script._self_provenance()


def test_runner_captures_an_isolated_dec0_token_only_for_candidate() -> None:
    torch = pytest.importorskip("torch")

    class _Retriever:
        def __init__(self) -> None:
            self.mem = torch.zeros((1, 1, 768))
            self.final_token = None
            self.model = None

        def inquire(self, _global: object, _mem: object) -> object:
            return torch.full((1, 1, 768), 17.0)

        def update_mem(self, mem: object, _global: object, final_token: object) -> object:
            self.final_token = final_token.detach().clone()
            assert self.model is not None
            self.model.dec0.add_(100.0)
            return mem + 1.0

    class _Model:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_update_type="cut3r")
            self.pose_head_flag = True
            self.dec_depth = 1
            self.pose_token = torch.full((1, 1, 768), 13.0)
            self.pose_retriever = _Retriever()
            self.pose_retriever.model = self
            self.dec0 = None
            self.raw_prediction = None

        def _uses_update_pressure_update(self) -> bool:
            return False

        def _encode_image(self, _images: object, _shapes: object) -> tuple[list[object], object]:
            return [torch.zeros((1, 1, 768))], torch.zeros((1, 1, 2))

        def _encode_ray_map(self, _rays: object, _shapes: object) -> tuple[list[object], object]:
            raise AssertionError("ray encoder must not run")

        def _init_state(self, feature: object, position: object) -> tuple[object, object]:
            return feature.clone(), position

        def _init_recal3r_reference_state(self, _state: object) -> None:
            return None

        def _get_img_level_feat(self, _feature: object) -> object:
            return torch.zeros((1, 1, 768))

        def _recurrent_rollout(self, state: object, _state_pos: object, _feature: object, _pos: object, _pose_feature: object, _pose_pos: object, _initial: object, **_kwargs: object) -> tuple[object, list[object], object, object, object, object]:
            self.dec0 = torch.full((1, 2, 768), 3.0)
            return state + 1.0, [self.dec0, torch.full((1, 2, 768), 5.0)], None, None, None, None

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

    view = {"img": torch.zeros((1, 3, 2, 2)), "img_mask": torch.tensor([True]), "ray_mask": torch.tensor([False]), "ray_map": torch.zeros((1, 2, 2, 3)), "true_shape": torch.tensor([2, 2]), "reset": None}
    candidate_model, observer = _Model(), _Observer()
    candidate = runner.run_early_decoder_pose_recurrent_lighter([view], candidate_model, torch.device("cpu"), torch=torch, to_gpu=lambda item, _device: item, to_cpu=lambda item: item, canonicalize_model_update_type=lambda value: value, observer=observer, verify_source=False)
    assert len(candidate.predictions) == 1 and observer.finalized == 1
    assert len(observer.tokens) == 1
    token = observer.tokens[0]
    assert token is not candidate_model.dec0
    torch.testing.assert_close(token, torch.full((1, 1, 768), 3.0))
    torch.testing.assert_close(candidate_model.dec0[:, 0:1], torch.full((1, 1, 768), 103.0))
    torch.testing.assert_close(candidate_model.pose_retriever.final_token, torch.full((1, 1, 768), 5.0))

    control_model = _Model()
    control = runner.run_early_decoder_pose_recurrent_lighter([view], control_model, torch.device("cpu"), torch=torch, to_gpu=lambda item, _device: item, to_cpu=lambda item: item, canonicalize_model_update_type=lambda value: value, observer=None, verify_source=False)
    assert control.predictions == [control_model.raw_prediction]
    assert control.timeline[0]["reason"] == "always_commit_control"
