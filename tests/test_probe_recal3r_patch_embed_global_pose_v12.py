from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_recal3r_patch_embed_global_pose_v12 as probe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _main_ast() -> ast.FunctionDef:
    module = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    return next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _call_dotted_name(node: ast.Call) -> str | None:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def test_v12_probe_accepts_only_frame_zero_of_the_disclosed_dynamic_manifest() -> None:
    image_path, frame = probe._exact_dynamic_frame_zero(probe.DEVELOPMENT_DYNAMIC_MANIFEST)
    assert image_path.is_file()
    assert frame["frame_index"] == 0
    assert image_path == (
        probe.DEVELOPMENT_DYNAMIC_MANIFEST.parent / frame["path"]
    ).resolve(strict=True)

    with pytest.raises(RuntimeError, match="only the disclosed dynamic manifest"):
        probe._exact_dynamic_frame_zero(
            PROJECT_ROOT / "outputs" / "formal-v1-inputs-0001" / "development" /
            "development-wrong" / "input-manifest.json",
        )


def test_v12_probe_is_single_frame_pre_rope_patch_path_with_no_full_encoder_calls() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    main = _main_ast()
    calls = [
        name for node in ast.walk(main)
        if isinstance(node, ast.Call) and (name := _call_dotted_name(node)) is not None
    ]
    assert calls.count("load_images_for_eval") == 1
    assert calls.count("model.patch_embed") == 1
    assert calls.count("capture_patch_embed_global_pose_token") == 1
    assert calls.count("decode_patch_embed_global_pose_token") == 1
    assert "model._encode_image" not in calls
    assert "model._recurrent_rollout" not in calls
    assert "model.pose_retriever.inquire" not in calls
    assert "model.pose_retriever.update_mem" not in calls
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert ".tolist(" not in source
    assert "PatchEmbedDust3R" in source
    assert "ManyAR_PatchEmbed is forbidden" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "_guard_prohibited_model_paths(model)" in source
    assert "latent_values_serialized\": False" in source


def test_v12_probe_guard_installs_hard_failures_for_encoder_rope_retriever_and_rollout() -> None:
    events: list[str] = []

    class _Module:
        def forward(self, *_args: object, **_kwargs: object) -> object:
            events.append("unblocked")
            return object()

    class _Retriever(_Module):
        def inquire(self, *_args: object) -> object:
            events.append("inquire")
            return object()

        def update_mem(self, *_args: object) -> object:
            events.append("update")
            return object()

    retriever = _Retriever()
    retriever.read_blocks = [_Module()]
    retriever.write_blocks = [_Module()]
    model = SimpleNamespace(
        enc_blocks=[_Module()],
        enc_blocks_ray_map=[_Module()],
        rope=_Module(),
        pose_retriever=retriever,
    )
    probe._guard_prohibited_model_paths(model)
    for action in (
        lambda: model._encode_image(),
        lambda: model._recurrent_rollout(),
        lambda: model.enc_blocks[0].forward(),
        lambda: model.enc_blocks_ray_map[0].forward(),
        lambda: model.rope.forward(),
        lambda: model.pose_retriever.inquire(),
        lambda: model.pose_retriever.update_mem(),
        lambda: model.pose_retriever.forward(),
        lambda: model.pose_retriever.read_blocks[0].forward(),
        lambda: model.pose_retriever.write_blocks[0].forward(),
    ):
        with pytest.raises(AssertionError, match="prohibited path"):
            action()
    assert events == []


def test_v12_probe_requires_explicit_cuda_disable_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=''" ):
        probe._require_cuda_disabled_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    probe._require_cuda_disabled_environment()


def test_v12_probe_run_id_is_fixed_before_any_model_work() -> None:
    parser = probe._parser()
    assert parser.parse_args([
        "--baseline-root", "/tmp/base", "--checkpoint", "/tmp/checkpoint",
        "--checkpoint-sha256", "x", "--input-manifest", "/tmp/manifest",
        "--evidence-json", "/tmp/evidence",
    ]).run_id == probe.RUN_ID
    with pytest.raises(RuntimeError, match="fixed and one-use"):
        probe._require_fixed_run_id("recovery-patch-embed-global-pooled-pose-v12-dynamic-frame0-cpu-probe-0002")
    assert probe._require_fixed_evidence_path(
        probe.ROOT / "logs" / f"{probe.RUN_ID}.json"
    ) == probe.ROOT / "logs" / f"{probe.RUN_ID}.json"
    with pytest.raises(RuntimeError, match="fixed and one-use"):
        probe._require_fixed_evidence_path(probe.ROOT / "logs" / "alternate.json")


def test_v12_probe_requires_exact_pinned_patch_embed_and_pose_head_interfaces() -> None:
    class _PatchEmbed:
        pass

    class _ManyAR:
        pass

    class _DPT:
        pose_mode = ("exp", -float("inf"), float("inf"))

        def pose_head(self, value: object) -> object:
            return value

    patch = _PatchEmbed()
    assert probe._pinned_patch_embed_interface(
        SimpleNamespace(patch_embed=patch), SimpleNamespace(PatchEmbedDust3R=_PatchEmbed),
    ) is patch
    with pytest.raises(RuntimeError, match="ManyAR"):
        probe._pinned_patch_embed_interface(
            SimpleNamespace(patch_embed=_ManyAR()), SimpleNamespace(PatchEmbedDust3R=_PatchEmbed),
        )

    pose_head, pose_mode = probe._pinned_pose_head_interface(
        SimpleNamespace(downstream_head=_DPT()), SimpleNamespace(DPTPts3dPose=_DPT),
    )
    assert callable(pose_head)
    assert pose_mode == _DPT.pose_mode
    with pytest.raises(RuntimeError, match="not pinned"):
        probe._pinned_pose_head_interface(
            SimpleNamespace(downstream_head=object()), SimpleNamespace(DPTPts3dPose=_DPT),
        )


def test_v12_probe_evidence_is_new_direct_immutable_log_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "StateGuard3R"
    logs = sandbox_root / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", sandbox_root)
    target = logs / "probe.json"
    probe._write_immutable_json(target, {"status": "passed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="new direct"):
        probe._write_immutable_json(target, {"status": "again"})
    with pytest.raises(RuntimeError, match="direct logs child"):
        probe._write_immutable_json(logs / "nested" / "probe.json", {"status": "bad"})
    assert not (logs / "nested").exists()
    assert os.path.samefile(target, target)
