from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateguard3r import recal3r_prerollout_pose_query_runner_v6 as runner
from scripts import run_recal3r_prerollout_pose_query_export_v6 as script


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


def test_runner_captures_query_before_rollout_and_finalizes_before_cpu_transfer() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    capture = source.index("pre_rollout_pose_query = pose_feat_i.detach().clone()")
    rollout = source.index("model._recurrent_rollout(", capture)
    finalize = source.index("observer.finalize(observation, quarantined=quarantine, prediction=res)")
    transfer = source.index("predictions.append(to_cpu(exported))")
    assert capture < rollout < finalize < transfer
    assert ".cpu()" not in source[capture:transfer]
    for forbidden in (
        "recal3r_safe_anchor_runner_v3",
        "recal3r_geometric_registration_runner_v4",
        "recal3r_current_pointmap_runner_v5",
    ):
        assert forbidden not in source


def test_v6_script_exposes_only_preregistered_policy_and_components() -> None:
    action = next(item for item in script._parser()._actions if item.dest == "state_policy")
    assert action.choices == ("always-commit", "detector-v3-incremental-prerollout-pose-query-export")
    assert script._parser().get_default("watchdog") == 8
    source = Path(script.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "SafeAnchorSE3Export",
        "RealSafeGeometricAnchor",
        "CurrentPointmapConsensusExport",
        "current_self_cross_consensus",
        "register_anchor_orb_3d3d",
    ):
        assert forbidden not in source
    component_paths = {str(item.resolve()) for item in script.V6_COMPONENTS}
    assert str(Path(script.__file__).resolve()) in component_paths
    assert any(path.endswith("recal3r_prerollout_pose_query_runner_v6.py") for path in component_paths)


def test_v6_script_fails_closed_without_the_pinned_dpt_pose_head() -> None:
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


def test_v6_provenance_rejects_an_untracked_runtime_component(monkeypatch: pytest.MonkeyPatch) -> None:
    component = Path(script.__file__).resolve()
    monkeypatch.setattr(script, "V6_COMPONENTS", (component,))

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
