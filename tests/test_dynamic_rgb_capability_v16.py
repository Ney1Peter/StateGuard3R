from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "stateguard3r" / "dynamic_rgb_capability_v16.py"
BUILDER = ROOT / "scripts" / "build_recal3r_dynamic_rgb_capability_v16.py"


def test_v16_dynamic_capability_implementation_is_present() -> None:
    assert MODULE.is_file()
    assert BUILDER.is_file()


def test_v16_dynamic_capability_has_no_prior_component_import() -> None:
    source = MODULE.read_text(encoding="utf-8")
    modules = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    prohibited = (
        "v15", "v14", "recovery", "manifest", "archive", "quarantine",
        "pointmap", "anchor", "registration", "prerollout", "decoder",
        "spatial", "encoder", "patch",
    )
    assert not any(any(word in name for word in prohibited) for name in modules)


def test_v16_dynamic_builder_is_own_raw_listing_only_script() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert "dynamic_rgb_capability_v16" in source
    assert "frame_zero_probe_capability_v16" not in source
    assert "outputs/recovery" not in source
    assert "v15" not in source.lower()


def test_v16_dynamic_builder_never_rereads_or_stats_its_created_output() -> None:
    source = MODULE.read_text(encoding="utf-8")
    build = source[source.index("def build_dynamic_rgb_capability_v16("): source.index("def load_dynamic_rgb_capability_v16(")]
    writer = source[source.index("def _write_fresh_frozen("): source.index("def build_dynamic_rgb_capability_v16(")]
    assert "sha256_file_v16(DYNAMIC_CAPSULE)" not in build
    assert "DYNAMIC_CAPSULE.read" not in build
    assert "_read_only_regular(path" not in writer
    assert "os.lstat(path)" not in writer


def test_v16_dynamic_builder_postcreate_reader_cannot_access_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateguard3r import dynamic_rgb_capability_v16 as capability

    root = tmp_path / "raw"
    (root / "rgb").mkdir(parents=True)
    paths = [f"rgb/{index:03d}.png" for index in range(30)]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    for path in paths:
        item = root / path
        item.write_bytes(path.encode("ascii"))
        item.chmod(0o444)
    target = tmp_path / "dynamic.json"
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    monkeypatch.setattr(capability, "RGB_LISTING", listing)
    monkeypatch.setattr(capability, "RGB_LISTING_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(capability, "FIRST_RGB", paths[0])
    monkeypatch.setattr(capability, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(capability, "LISTING_ROW_COUNT", 30)
    monkeypatch.setattr(capability, "DYNAMIC_CAPSULE", target)
    original = Path.read_bytes

    def guard(path: Path) -> bytes:
        if path.resolve(strict=False) == target.resolve(strict=False):
            raise AssertionError("builder reread its own output")
        return original(path)

    result = capability.build_dynamic_rgb_capability_v16(read_bytes=guard)
    assert result["capsule_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_v16_dynamic_builder_reads_only_listing_and_thirty_rgb_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateguard3r import dynamic_rgb_capability_v16 as capability

    root = tmp_path / "raw"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    paths = [f"rgb/{index:03d}.png" for index in range(30)]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    for path in paths:
        target = root / path
        target.write_bytes(b"rgb:" + path.encode("ascii"))
        target.chmod(0o444)
    target = tmp_path / "dynamic.json"
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    monkeypatch.setattr(capability, "RGB_LISTING", listing)
    monkeypatch.setattr(capability, "RGB_LISTING_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(capability, "FIRST_RGB", paths[0])
    monkeypatch.setattr(capability, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(capability, "LISTING_ROW_COUNT", 30)
    monkeypatch.setattr(capability, "DYNAMIC_CAPSULE", target)
    reads: list[Path] = []

    def poison_reader(path: Path) -> bytes:
        resolved = path.resolve(strict=True)
        reads.append(resolved)
        permitted = {listing.resolve(strict=True), *(root / item for item in paths)}
        if resolved not in permitted:
            raise AssertionError(f"unexpected builder read: {resolved}")
        return resolved.read_bytes()

    result = capability.build_dynamic_rgb_capability_v16(read_bytes=poison_reader)
    assert result["frame_count"] == 30
    assert target.stat().st_mode & 0o777 == 0o444
    assert reads == [listing.resolve(strict=True), *((root / item).resolve(strict=True) for item in paths)]
    assert not any("outputs" in str(path) for path in reads)


def test_v16_dynamic_parser_fails_closed_on_duplicate_key_transform_and_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateguard3r import dynamic_rgb_capability_v16 as capability

    root = tmp_path / "raw"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    paths = [f"rgb/{index:03d}.png" for index in range(30)]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    for path in paths:
        data = root / path
        data.write_bytes(path.encode("ascii"))
        data.chmod(0o444)
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    monkeypatch.setattr(capability, "RGB_LISTING", listing)
    monkeypatch.setattr(capability, "RGB_LISTING_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(capability, "FIRST_RGB", paths[0])
    monkeypatch.setattr(capability, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(capability, "LISTING_ROW_COUNT", 30)
    target = tmp_path / "dynamic.json"
    monkeypatch.setattr(capability, "DYNAMIC_CAPSULE", target)
    capability.build_dynamic_rgb_capability_v16()
    parsed = capability.load_dynamic_rgb_capability_v16(target)
    capability.verify_dynamic_rgb_capability_v16(parsed)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    duplicate.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV16Error, match="duplicate"):
        capability.load_dynamic_rgb_capability_v16(duplicate)
    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["frames"][15]["transforms"][0]["rectangle"]["x"] = 0.3
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(changed), encoding="utf-8")
    altered.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV16Error, match="transform"):
        capability.load_dynamic_rgb_capability_v16(altered)
    escaped = json.loads(target.read_text(encoding="utf-8"))
    escaped["frames"][0]["rgb_relative_path"] = "rgb/../outside.png"
    bad_path = tmp_path / "escape.json"
    bad_path.write_text(json.dumps(escaped), encoding="utf-8")
    bad_path.chmod(0o444)
    with pytest.raises(capability.DynamicRGBCapabilityV16Error, match="path"):
        capability.load_dynamic_rgb_capability_v16(bad_path)
