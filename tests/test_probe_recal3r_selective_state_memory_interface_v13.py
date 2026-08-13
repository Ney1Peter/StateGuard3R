from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_recal3r_selective_state_memory_interface_v13 as probe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _main_ast() -> ast.FunctionDef:
    module = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")


def _call_name(node: ast.Call) -> str | None:
    parts: list[str] = []
    current: ast.expr = node.func
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def test_v13_interface_probe_binds_only_disclosed_dynamic_frame_zero() -> None:
    image, frame = probe._exact_dynamic_frame_zero(probe.DEVELOPMENT_DYNAMIC_MANIFEST)
    assert image.is_file()
    assert frame["frame_index"] == 0
    with pytest.raises(RuntimeError, match="only the disclosed dynamic manifest"):
        probe._exact_dynamic_frame_zero(
            PROJECT_ROOT / "outputs" / "formal-v1-inputs-0001" / "development" / "development-low" / "input-manifest.json"
        )


def test_v13_interface_probe_never_executes_model_or_repair_paths() -> None:
    source = Path(probe.__file__).read_text(encoding="utf-8")
    calls = [name for node in ast.walk(_main_ast()) if isinstance(node, ast.Call) and (name := _call_name(node))]
    assert "smoke._load_model_with_state_dict_audit" in calls
    for forbidden in (
        "model._encode_image", "model._encode_ray_map", "model._init_state",
        "model._recurrent_rollout", "model._downstream_head", "selective_state_memory_repair",
    ):
        assert forbidden not in calls
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "_guard_all_model_execution(model)" in source
    assert '"model_forward_executed": False' in source
    assert '"state_memory_values_serialized": False' in source


def test_v13_interface_probe_installs_hard_failures_for_every_model_execution_route() -> None:
    events: list[str] = []

    class _Block:
        def forward(self, *_args: object, **_kwargs: object) -> object:
            events.append("unblocked")
            return object()

    class _Retriever(_Block):
        def inquire(self, *_args: object) -> object:
            events.append("inquire")
            return object()

        def update_mem(self, *_args: object) -> object:
            events.append("update")
            return object()

    model = SimpleNamespace(
        enc_blocks=[_Block()], enc_blocks_ray_map=[_Block()], dec_blocks=[_Block()], dec_blocks_state=[_Block()],
        rope=_Block(), pose_retriever=_Retriever(),
    )
    probe._guard_all_model_execution(model)
    for action in (
        lambda: model._encode_image(), lambda: model._encode_ray_map(), lambda: model._init_state(),
        lambda: model._recurrent_rollout(), lambda: model._downstream_head(), lambda: model.forward(),
        lambda: model.forward_recurrent(), lambda: model.forward_recurrent_lighter(),
        lambda: model.enc_blocks[0].forward(), lambda: model.dec_blocks[0].forward(), lambda: model.rope.forward(),
        lambda: model.pose_retriever.inquire(), lambda: model.pose_retriever.update_mem(), lambda: model.pose_retriever.forward(),
    ):
        with pytest.raises(AssertionError, match="prohibited path"):
            action()
    assert events == []


def test_v13_interface_probe_requires_explicit_cuda_disable_and_fixed_one_use_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=''" ):
        probe._require_cuda_disabled_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    probe._require_cuda_disabled_environment()
    parser = probe._parser()
    assert parser.parse_args([
        "--baseline-root", "/tmp/base", "--checkpoint", "/tmp/checkpoint", "--checkpoint-sha256", "x",
        "--input-manifest", "/tmp/manifest", "--evidence-json", "/tmp/evidence",
    ]).run_id == probe.RUN_ID
    with pytest.raises(RuntimeError, match="fixed and one-use"):
        probe._require_fixed_run_id("different")
    assert probe._require_fixed_evidence_path(probe.ROOT / "logs" / f"{probe.RUN_ID}.json")
    with pytest.raises(RuntimeError, match="fixed and one-use"):
        probe._require_fixed_evidence_path(probe.ROOT / "logs" / "other.json")


def test_v13_interface_probe_evidence_is_new_direct_and_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root, logs = tmp_path / "StateGuard3R", tmp_path / "StateGuard3R" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", root)
    path = logs / "probe.json"
    probe._write_immutable_json(path, {"status": "passed"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "passed"}
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="new direct"):
        probe._write_immutable_json(path, {"status": "again"})
    with pytest.raises(RuntimeError, match="new direct"):
        probe._write_immutable_json(logs / "nested" / "bad.json", {"status": "bad"})
    assert not os.path.lexists(logs / "nested" / "bad.json")
