from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "stateguard3r" / "frame_zero_probe_capability_v16.py"
BUILDER = ROOT / "scripts" / "build_recal3r_frame_zero_probe_capability_v16.py"


def test_v16_probe_capability_implementation_is_present() -> None:
    assert MODULE.is_file()
    assert BUILDER.is_file()


def test_v16_probe_capability_parser_cannot_reference_dynamic_or_listing() -> None:
    source = MODULE.read_text(encoding="utf-8").lower()
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any("dynamic_rgb" in module or "v15" in module for module in modules)
    assert "raw_selector" not in source
    assert "first_rgb" not in source
    assert "build_frame_zero" not in source
    assert "transforms" not in source


def test_v16_probe_builder_reads_only_listing_and_frame_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_recal3r_frame_zero_probe_capability_v16 as builder
    from stateguard3r import frame_zero_probe_capability_v16 as capability

    root = tmp_path / "raw"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    first = "rgb/first.png"
    paths = [first, *(f"rgb/{index:03d}.png" for index in range(1, 30))]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    (root / first).write_bytes(b"first")
    (root / first).chmod(0o444)
    target = tmp_path / "probe.json"
    monkeypatch.setattr(builder, "DATASET_ROOT", root)
    monkeypatch.setattr(builder, "RAW_SELECTOR", listing)
    monkeypatch.setattr(builder, "RAW_SELECTOR_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "FIRST_RGB", first)
    monkeypatch.setattr(builder, "FRAME_COUNT", 30)
    monkeypatch.setattr(builder, "LISTING_ROW_COUNT", 30)
    monkeypatch.setattr(builder, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(builder, "PROBE_CAPSULE", target)
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    reads: list[Path] = []

    def poison_reader(path: Path) -> bytes:
        resolved = path.resolve(strict=True)
        reads.append(resolved)
        if resolved not in {listing.resolve(strict=True), (root / first).resolve(strict=True)}:
            raise AssertionError(f"unexpected probe-builder read: {resolved}")
        return resolved.read_bytes()

    result = builder.build_frame_zero_probe_capability_v16(read_bytes=poison_reader)
    assert result["frame_id"] == 0
    assert reads == [listing.resolve(strict=True), (root / first).resolve(strict=True)]
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "capability_id", "dataset_root", "rgb_root", "loader", "frame"}
    assert set(payload["frame"]) == {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal"}


def test_v16_probe_builder_never_rereads_or_stats_its_created_output() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    build = source[source.index("def build_frame_zero_probe_capability_v16("): source.index("def main(")]
    writer = source[source.index("def _write_fresh_frozen("): source.index("def build_frame_zero_probe_capability_v16(")]
    assert "PROBE_CAPSULE.read" not in build
    assert "os.lstat(path)" not in writer


def test_v16_probe_builder_postcreate_reader_cannot_access_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_recal3r_frame_zero_probe_capability_v16 as builder

    root = tmp_path / "raw"
    (root / "rgb").mkdir(parents=True)
    first = "rgb/first.png"
    paths = [first, *(f"rgb/{index:03d}.png" for index in range(1, 30))]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    image = root / first
    image.write_bytes(b"first")
    image.chmod(0o444)
    target = tmp_path / "probe.json"
    monkeypatch.setattr(builder, "DATASET_ROOT", root)
    monkeypatch.setattr(builder, "RAW_SELECTOR", listing)
    monkeypatch.setattr(builder, "RAW_SELECTOR_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "FIRST_RGB", first)
    monkeypatch.setattr(builder, "FRAME_COUNT", 30)
    monkeypatch.setattr(builder, "LISTING_ROW_COUNT", 30)
    monkeypatch.setattr(builder, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(builder, "PROBE_CAPSULE", target)
    original = Path.read_bytes

    def guard(path: Path) -> bytes:
        if path.resolve(strict=False) == target.resolve(strict=False):
            raise AssertionError("builder reread its own output")
        return original(path)

    result = builder.build_frame_zero_probe_capability_v16(read_bytes=guard)
    assert result["capsule_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_v16_probe_parser_reads_capsule_then_only_frame_zero_never_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateguard3r import frame_zero_probe_capability_v16 as capability

    root = tmp_path / "raw"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    image = rgb / "first.png"
    image.write_bytes(b"frame zero")
    image.chmod(0o444)
    stat = image.stat()
    payload = {
        "schema": capability.PROBE_SCHEMA,
        "capability_id": capability.PROBE_CAPSULE_ID,
        "dataset_root": str(root),
        "rgb_root": str(rgb),
        "loader": capability.MODEL_LOADER,
        "frame": {
            "frame_id": 0,
            "rgb_relative_path": "rgb/first.png",
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "size_bytes": stat.st_size,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "mode_octal": "0444",
        },
    }
    capsule = tmp_path / "probe.json"
    capsule.write_text(json.dumps(payload), encoding="utf-8")
    capsule.chmod(0o444)
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    reads: list[Path] = []

    def reader(path: Path) -> bytes:
        reads.append(path.resolve(strict=True))
        return path.read_bytes()

    parsed = capability.load_frame_zero_probe_capability_v16(capsule, read_bytes=reader)
    assert reads == [capsule.resolve(strict=True)]
    parsed_from_bytes = capability.parse_frame_zero_probe_capability_v16_bytes(capsule.read_bytes(), capsule_path=capsule)
    assert parsed_from_bytes == parsed
    capability.verify_frame_zero_probe_capability_v16(parsed, read_bytes=reader)
    assert reads == [capsule.resolve(strict=True), image.resolve(strict=True)]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    duplicate.chmod(0o444)
    with pytest.raises(capability.FrameZeroProbeCapabilityV16Error, match="duplicate"):
        capability.load_frame_zero_probe_capability_v16(duplicate)
    poisoned = dict(payload)
    poisoned_frame = dict(payload["frame"])
    poisoned_frame["source_index"] = 0
    poisoned["frame"] = poisoned_frame
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(poisoned), encoding="utf-8")
    extra.chmod(0o444)
    with pytest.raises(capability.FrameZeroProbeCapabilityV16Error, match="schema"):
        capability.load_frame_zero_probe_capability_v16(extra)
