import ast
import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "stateguard3r" / "dynamic_rgb_capability_v17.py"
BUILDER = ROOT / "scripts" / "build_recal3r_dynamic_rgb_capability_v17.py"


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from stateguard3r import dynamic_rgb_capability_v17 as capability

    root = tmp_path / "raw"
    (root / "rgb").mkdir(parents=True)
    paths = [f"rgb/{index:03d}.png" for index in range(30)]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    for relative in paths:
        image = root / relative
        image.write_bytes(b"raw:" + relative.encode("ascii"))
        image.chmod(0o444)
    target = tmp_path / "dynamic.json"
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    monkeypatch.setattr(capability, "RGB_LISTING", listing)
    monkeypatch.setattr(capability, "RGB_LISTING_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(capability, "FIRST_RGB", paths[0])
    monkeypatch.setattr(capability, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(capability, "LISTING_ROW_COUNT", len(paths))
    monkeypatch.setattr(capability, "DYNAMIC_CAPSULE", target)
    return capability, root, listing, paths, target


def test_v17_dynamic_implementation_and_isolated_imports_are_present() -> None:
    assert MODULE.is_file()
    assert BUILDER.is_file()
    modules: list[str] = []
    for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    prohibited = ("v16", "v15", "v14", "recovery", "manifest", "archive", "quarantine", "pressure", "pointmap", "anchor", "registration", "prerollout", "decoder", "spatial", "encoder", "patch", "repair", "rollback")
    assert not any(any(word in module.lower() for word in prohibited) for module in modules)
    source = BUILDER.read_text(encoding="utf-8")
    assert "dynamic_rgb_capability_v17" in source
    assert "frame_zero_probe_capability_v17" not in source


def test_v17_dynamic_builder_reads_only_listing_and_selected_rgb_and_never_rereads_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, root, listing, paths, target = _fixture(monkeypatch, tmp_path)
    reads: list[Path] = []

    def reader(path: Path) -> bytes:
        resolved = path.resolve(strict=True)
        reads.append(resolved)
        if resolved == target.resolve(strict=False):
            raise AssertionError("builder reread its output")
        permitted = {listing.resolve(strict=True), *((root / item).resolve(strict=True) for item in paths)}
        if resolved not in permitted:
            raise AssertionError(f"unexpected builder read: {resolved}")
        return resolved.read_bytes()

    result = capability.build_dynamic_rgb_capability_v17(read_bytes=reader)
    assert result["frame_count"] == 30
    assert target.stat().st_mode & 0o777 == 0o444
    assert reads == [listing.resolve(strict=True), *((root / item).resolve(strict=True) for item in paths)]
    assert result["capsule_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_v17_dynamic_parser_rejects_duplicate_key_transform_path_and_listing_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, _root, _listing, _paths, target = _fixture(monkeypatch, tmp_path)
    capability.build_dynamic_rgb_capability_v17()
    parsed = capability.load_dynamic_rgb_capability_v17(target)
    capability.verify_dynamic_rgb_capability_v17(parsed)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    duplicate.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="duplicate"):
        capability.load_dynamic_rgb_capability_v17(duplicate)
    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["frames"][15]["transforms"][0]["rectangle"]["x"] = 0.3
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(changed), encoding="utf-8")
    altered.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="transform"):
        capability.load_dynamic_rgb_capability_v17(altered)
    escaped = json.loads(target.read_text(encoding="utf-8"))
    escaped["frames"][0]["rgb_relative_path"] = "rgb/../outside.png"
    bad_path = tmp_path / "escape.json"
    bad_path.write_text(json.dumps(escaped), encoding="utf-8")
    bad_path.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="path"):
        capability.load_dynamic_rgb_capability_v17(bad_path)
    changed_order = json.loads(target.read_text(encoding="utf-8"))
    changed_order["frames"][1]["rgb_relative_path"] = changed_order["frames"][0]["rgb_relative_path"]
    bad_order = tmp_path / "order.json"
    bad_order.write_text(json.dumps(changed_order), encoding="utf-8")
    bad_order.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="order"):
        capability.load_dynamic_rgb_capability_v17(bad_order)


def test_v17_dynamic_verifier_rejects_symlink_and_mutable_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability, root, _listing, paths, target = _fixture(monkeypatch, tmp_path)
    capability.build_dynamic_rgb_capability_v17()
    parsed = capability.load_dynamic_rgb_capability_v17(target)
    image = root / paths[0]
    image.chmod(0o644)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="0444"):
        capability.verify_dynamic_rgb_capability_v17(parsed)
    image.chmod(0o444)
    replacement = root / "replacement.png"
    replacement.write_bytes(b"replacement")
    replacement.chmod(0o444)
    image.unlink()
    image.symlink_to(replacement)
    with pytest.raises(capability.DynamicRGBCapabilityV17Error, match="non-symlink"):
        capability.verify_dynamic_rgb_capability_v17(parsed)
    assert os.path.islink(image)
