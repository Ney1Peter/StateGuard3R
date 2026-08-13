from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_recal3r_bounded_update_pressure_interface_v16.py"


def test_v16_probe_implementation_is_present() -> None:
    assert PROBE.is_file()


def test_v16_probe_imports_only_its_frame_zero_capability() -> None:
    source = PROBE.read_text(encoding="utf-8")
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    state_modules = [name for name in modules if name.startswith("stateguard3r")]
    assert state_modules == ["stateguard3r.frame_zero_probe_capability_v16"]
    lowered = source.lower()
    for forbidden in ("rgb.txt", "dynamic_rgb", "v15", "v14", "input_manifest", "archive"):
        assert forbidden not in lowered


def test_v16_probe_has_exact_three_resource_ledger_and_cuda_hidden_guards() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "probe_capsule" in source
    assert "frame_zero_rgb" in source
    assert "checkpoint" in source
    assert "assert_exact_resources" in source
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in source
    assert "torch.cuda.is_initialized()" in source
    assert "os.O_EXCL" in source
    assert "_block_model_execution(model)" in source
    assert "load_images_for_eval" in source


def test_v16_probe_blocks_all_forward_encoder_rope_detector_and_operator_routes() -> None:
    from scripts import probe_recal3r_bounded_update_pressure_interface_v16 as probe

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

    retriever = _Retriever()
    retriever.read_blocks, retriever.write_blocks = [_Block()], [_Block()]
    model = SimpleNamespace(
        patch_embed=_Block(), patch_embed_ray_map=_Block(), enc_blocks=[_Block()],
        enc_blocks_ray_map=[_Block()], dec_blocks=[_Block()], dec_blocks_state=[_Block()],
        downstream_head=_Block(), rope=_Block(), pose_retriever=retriever,
    )
    probe._block_model_execution(model)
    for action in (
        lambda: model.forward(), lambda: model._encode_image(),
        lambda: model._recurrent_rollout(), lambda: model._downstream_head(),
        lambda: model._compute_recal3r_update_mask(),
        lambda: model._reset_update_pressure_if_needed(),
        lambda: model.patch_embed.forward(), lambda: model.enc_blocks[0].forward(),
        lambda: model.dec_blocks[0].forward(), lambda: model.rope.forward(),
        lambda: model.downstream_head.forward(), lambda: model.pose_retriever.inquire(),
        lambda: model.pose_retriever.update_mem(), lambda: model.pose_retriever.forward(),
        lambda: model.pose_retriever.read_blocks[0].forward(),
    ):
        with pytest.raises(AssertionError, match="prohibited path"):
            action()
    assert events == []


def test_v16_probe_ledger_rejects_missing_duplicate_or_forbidden_resource(tmp_path: Path) -> None:
    from scripts import probe_recal3r_bounded_update_pressure_interface_v16 as probe

    capsule, rgb, checkpoint = (tmp_path / "capsule"), (tmp_path / "rgb"), (tmp_path / "checkpoint")
    for path in (capsule, rgb, checkpoint):
        path.write_bytes(path.name.encode("ascii"))
        path.chmod(0o444)
    ledger = probe.DataAccessLedgerV16(capsule=capsule, frame_zero_rgb=rgb, checkpoint=checkpoint)
    assert ledger.read(capsule, "probe_capsule") == b"capsule"
    assert ledger.read(rgb, "frame_zero_rgb") == b"rgb"
    assert ledger.read(checkpoint, "checkpoint") == b"checkpoint"
    ledger.assert_exact_resources()
    assert ledger.read(rgb, "frame_zero_rgb") == b"rgb"
    assert ledger.entries[1]["access_count"] == "2"
    other = tmp_path / "other"
    other.write_bytes(b"other")
    other.chmod(0o444)
    with pytest.raises(probe.InterfaceProbeV16Error, match="forbidden"):
        probe.DataAccessLedgerV16(capsule=capsule, frame_zero_rgb=rgb, checkpoint=checkpoint).read(other, "frame_zero_rgb")
    late_bound = probe.DataAccessLedgerV16(capsule=capsule, frame_zero_rgb=None, checkpoint=checkpoint)
    late_bound.read(capsule, "probe_capsule")
    late_bound.bind_frame_zero_rgb(rgb)
    late_bound.read(rgb, "frame_zero_rgb")
    late_bound.read(checkpoint, "checkpoint")
    late_bound.assert_exact_resources()


def test_v16_probe_evidence_is_direct_exclusive_and_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from scripts import probe_recal3r_bounded_update_pressure_interface_v16 as probe

    sandbox, logs = tmp_path / "StateGuard3R", tmp_path / "StateGuard3R" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", sandbox)
    target = logs / "probe.json"
    probe._write_immutable_evidence(target, {"status": "passed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(probe.InterfaceProbeV16Error, match="occupied"):
        probe._write_immutable_evidence(target, {"status": "again"})
    with pytest.raises(probe.InterfaceProbeV16Error, match="direct logs child"):
        probe._write_immutable_evidence(logs / "nested" / "bad", {"status": "bad"})
    assert not os.path.lexists(logs / "nested" / "bad")
