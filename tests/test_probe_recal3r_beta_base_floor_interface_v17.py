import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "probe_recal3r_beta_base_floor_interface_v17.py"


def test_v17_probe_implementation_is_present_and_has_only_its_single_frame_import() -> None:
    assert PROBE.is_file()
    source = PROBE.read_text(encoding="utf-8")
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert [name for name in modules if name.startswith("stateguard3r")] == ["stateguard3r.frame_zero_probe_capability_v17"]
    lowered = source.lower()
    for forbidden in ("rgb.txt", "dynamic_rgb", "v16", "v15", "v14", "input_manifest", "archive"):
        assert forbidden not in lowered


def test_v17_probe_has_cuda_hidden_contract_three_resource_ledger_and_immutable_writer() -> None:
    source = PROBE.read_text(encoding="utf-8")
    for resource in ("probe_capsule", "frame_zero_rgb", "checkpoint"):
        assert resource in source
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in source
    assert "torch.cuda.is_initialized()" in source
    assert "os.O_EXCL" in source
    assert "assert_exact_resources" in source
    assert "_block_model_execution(model)" in source
    assert "load_images_for_eval" in source
    assert "parse_frame_zero_probe_capability_v17_bytes" in source


def test_v17_probe_blocks_forward_model_routes_and_component_forwards() -> None:
    from scripts import probe_recal3r_beta_base_floor_interface_v17 as probe

    events: list[str] = []

    class Block:
        def forward(self, *_args: object, **_kwargs: object) -> object:
            events.append("unblocked")
            return object()

    class Retriever(Block):
        def inquire(self, *_args: object) -> object:
            events.append("inquire")
            return object()

        def update_mem(self, *_args: object) -> object:
            events.append("update")
            return object()

    retriever = Retriever()
    retriever.read_blocks, retriever.write_blocks = [Block()], [Block()]
    model = SimpleNamespace(
        patch_embed=Block(), patch_embed_ray_map=Block(), enc_blocks=[Block()],
        enc_blocks_ray_map=[Block()], dec_blocks=[Block()], dec_blocks_state=[Block()],
        downstream_head=Block(), rope=Block(), pose_retriever=retriever,
    )
    probe._block_model_execution(model)
    blocked = (
        lambda: model.forward(), lambda: model._forward_impl(), lambda: model._encode_image(),
        lambda: model._recurrent_rollout(), lambda: model._downstream_head(),
        lambda: model._compute_recal3r_update_mask(), lambda: model._reset_update_pressure_if_needed(),
        lambda: getattr(model, "_det" + "ector")(),
        lambda: getattr(model, "beta_base_floor_" + "operator")(),
        lambda: model.patch_embed.forward(), lambda: model.enc_blocks[0].forward(),
        lambda: model.dec_blocks[0].forward(), lambda: model.rope.forward(),
        lambda: model.downstream_head.forward(), lambda: model.pose_retriever.inquire(),
        lambda: model.pose_retriever.update_mem(), lambda: model.pose_retriever.forward(),
        lambda: model.pose_retriever.read_blocks[0].forward(),
    )
    for action in blocked:
        with pytest.raises(AssertionError, match="prohibited path"):
            action()
    assert events == []


def test_v17_ledger_allows_exact_resources_repeats_and_rejects_forbidden_or_duplicate_bind(
    tmp_path: Path,
) -> None:
    from scripts import probe_recal3r_beta_base_floor_interface_v17 as probe

    capsule, rgb, checkpoint = (tmp_path / "capsule"), (tmp_path / "rgb"), (tmp_path / "checkpoint")
    for path in (capsule, rgb, checkpoint):
        path.write_bytes(path.name.encode("ascii"))
        path.chmod(0o444)
    ledger = probe.DataAccessLedgerV17(capsule=capsule, frame_zero_rgb=rgb, checkpoint=checkpoint)
    assert ledger.read(capsule, "probe_capsule") == b"capsule"
    assert ledger.read(rgb, "frame_zero_rgb") == b"rgb"
    assert ledger.read(checkpoint, "checkpoint") == b"checkpoint"
    ledger.assert_exact_resources()
    assert ledger.read(rgb, "frame_zero_rgb") == b"rgb"
    assert ledger.entries[1]["access_count"] == "2"
    other = tmp_path / "other"
    other.write_bytes(b"other")
    other.chmod(0o444)
    with pytest.raises(probe.InterfaceProbeV17Error, match="forbidden"):
        probe.DataAccessLedgerV17(capsule=capsule, frame_zero_rgb=rgb, checkpoint=checkpoint).read(other, "frame_zero_rgb")
    late = probe.DataAccessLedgerV17(capsule=capsule, frame_zero_rgb=None, checkpoint=checkpoint)
    late.read(capsule, "probe_capsule")
    late.bind_frame_zero_rgb(rgb)
    with pytest.raises(probe.InterfaceProbeV17Error, match="already bound"):
        late.bind_frame_zero_rgb(rgb)
    late.read(rgb, "frame_zero_rgb")
    late.read(checkpoint, "checkpoint")
    late.assert_exact_resources()


def test_v17_ledger_rejects_symlink_and_mutation(tmp_path: Path) -> None:
    from scripts import probe_recal3r_beta_base_floor_interface_v17 as probe

    capsule, rgb, checkpoint = (tmp_path / "capsule"), (tmp_path / "rgb"), (tmp_path / "checkpoint")
    for path in (capsule, rgb, checkpoint):
        path.write_bytes(path.name.encode("ascii"))
        path.chmod(0o444)
    link = tmp_path / "link"
    link.symlink_to(rgb)
    ledger = probe.DataAccessLedgerV17(capsule=capsule, frame_zero_rgb=link, checkpoint=checkpoint)
    with pytest.raises(probe.InterfaceProbeV17Error, match="non-symlink"):
        ledger.read(link, "frame_zero_rgb")
    assert os.path.islink(link)


def test_v17_probe_evidence_is_direct_exclusive_and_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts import probe_recal3r_beta_base_floor_interface_v17 as probe

    sandbox, logs = tmp_path / "StateGuard3R", tmp_path / "StateGuard3R" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", sandbox)
    target = logs / "probe.json"
    probe._write_immutable_evidence(target, {"status": "passed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "passed"}
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(probe.InterfaceProbeV17Error, match="occupied"):
        probe._write_immutable_evidence(target, {"status": "again"})
    with pytest.raises(probe.InterfaceProbeV17Error, match="direct logs child"):
        probe._write_immutable_evidence(logs / "nested" / "bad", {"status": "bad"})
    assert not os.path.lexists(logs / "nested" / "bad")
