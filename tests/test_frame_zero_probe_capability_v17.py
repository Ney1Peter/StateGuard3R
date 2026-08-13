import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "stateguard3r" / "frame_zero_probe_capability_v17.py"
BUILDER = ROOT / "scripts" / "build_recal3r_frame_zero_probe_capability_v17.py"


def _fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from scripts import build_recal3r_frame_zero_probe_capability_v17 as builder

    root = tmp_path / "raw"
    (root / "rgb").mkdir(parents=True)
    first = "rgb/first.png"
    paths = [first, *(f"rgb/{index:03d}.png" for index in range(1, 30))]
    listing = root / "rgb.txt"
    listing.write_text("\n".join(f"1.{index:06d} {path}" for index, path in enumerate(paths)) + "\n", encoding="utf-8")
    listing.chmod(0o444)
    image = root / first
    image.write_bytes(b"frame zero")
    image.chmod(0o444)
    target = tmp_path / "probe.json"
    monkeypatch.setattr(builder, "DATASET_ROOT", root)
    monkeypatch.setattr(builder, "RAW_SELECTOR", listing)
    monkeypatch.setattr(builder, "RAW_SELECTOR_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(builder, "FIRST_RGB", first)
    monkeypatch.setattr(builder, "LISTING_ROW_COUNT", len(paths))
    monkeypatch.setattr(builder, "PATH_LIST_SHA256", hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode("utf-8")).hexdigest())
    monkeypatch.setattr(builder, "PROBE_CAPSULE", target)
    return builder, root, listing, first, paths, target


def test_v17_frame_zero_parser_isolated_from_selector_and_dynamic_capability() -> None:
    assert MODULE.is_file()
    assert BUILDER.is_file()
    source = MODULE.read_text(encoding="utf-8").lower()
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any("dynamic_rgb" in module or any(f"v{prior}" in module for prior in range(1, 17)) for module in modules)
    for forbidden in ("raw_selector", "first_rgb", "build_frame_zero", "transforms", "rgb.txt"):
        assert forbidden not in source


def test_v17_frame_zero_builder_reads_listing_and_first_rgb_only_and_does_not_reread_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder, root, listing, first, _paths, target = _fixture(monkeypatch, tmp_path)
    reads: list[Path] = []

    def reader(path: Path) -> bytes:
        resolved = path.resolve(strict=True)
        reads.append(resolved)
        if resolved == target.resolve(strict=False):
            raise AssertionError("builder reread its output")
        if resolved not in {listing.resolve(strict=True), (root / first).resolve(strict=True)}:
            raise AssertionError(f"unexpected builder read: {resolved}")
        return resolved.read_bytes()

    result = builder.build_frame_zero_probe_capability_v17(read_bytes=reader)
    assert result["frame_id"] == 0
    assert reads == [listing.resolve(strict=True), (root / first).resolve(strict=True)]
    assert target.stat().st_mode & 0o777 == 0o444
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "capability_id", "dataset_root", "rgb_root", "loader", "frame"}
    assert set(payload["frame"]) == {"frame_id", "rgb_relative_path", "sha256", "size_bytes", "inode", "mtime_ns", "mode_octal"}


def test_v17_frame_zero_parser_reads_capsule_then_verifier_reads_one_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stateguard3r import frame_zero_probe_capability_v17 as capability

    root = tmp_path / "raw"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    image = rgb / "first.png"
    image.write_bytes(b"frame zero")
    image.chmod(0o444)
    current = image.stat()
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
            "size_bytes": current.st_size,
            "inode": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "mode_octal": "0444",
        },
    }
    capsule = tmp_path / "capsule.json"
    capsule.write_text(json.dumps(payload), encoding="utf-8")
    capsule.chmod(0o444)
    monkeypatch.setattr(capability, "DATASET_ROOT", root)
    reads: list[Path] = []

    def reader(path: Path) -> bytes:
        reads.append(path.resolve(strict=True))
        return path.read_bytes()

    parsed = capability.load_frame_zero_probe_capability_v17(capsule, read_bytes=reader)
    assert reads == [capsule.resolve(strict=True)]
    assert capability.parse_frame_zero_probe_capability_v17_bytes(capsule.read_bytes(), capsule_path=capsule) == parsed
    capability.verify_frame_zero_probe_capability_v17(parsed, read_bytes=reader)
    assert reads == [capsule.resolve(strict=True), image.resolve(strict=True)]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    duplicate.chmod(0o444)
    with pytest.raises(capability.FrameZeroProbeCapabilityV17Error, match="duplicate"):
        capability.load_frame_zero_probe_capability_v17(duplicate)
    escaped = dict(payload)
    escaped["frame"] = dict(payload["frame"], rgb_relative_path="rgb/../outside.png")
    bad_path = tmp_path / "escape.json"
    bad_path.write_text(json.dumps(escaped), encoding="utf-8")
    bad_path.chmod(0o444)
    with pytest.raises(capability.FrameZeroProbeCapabilityV17Error, match="path"):
        capability.load_frame_zero_probe_capability_v17(bad_path)
