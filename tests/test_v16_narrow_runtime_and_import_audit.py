from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stateguard3r import online_detector_v16 as detector
from stateguard3r import online_visual_overlap_v16 as overlap
from stateguard3r import timestamp_order_v16 as timestamp
from stateguard3r.v16_source_import_audit import V16SourceImportAuditError, audit_v16_production_script, audit_v16_runtime_import_graph


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "stateguard3r"
RUNTIME = (
    PACKAGE / "bounded_update_pressure_v16.py",
    PACKAGE / "recal3r_bounded_update_pressure_runner_v16.py",
    PACKAGE / "online_detector_v16.py",
    PACKAGE / "online_visual_overlap_v16.py",
    PACKAGE / "timestamp_order_v16.py",
    PACKAGE / "dynamic_rgb_capability_v16.py",
    PACKAGE / "health.py",
)


def test_v16_static_import_walk_is_closed_and_rejects_exact_prior_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = audit_v16_runtime_import_graph(RUNTIME)
    assert report["legacy_recovery_or_input_import"] is False
    assert report["offline_annotation_or_artifact_path"] is False
    assert "bounded_update_pressure_v16" in report["visited_modules"]
    # Restore the real package after the isolated polluted-graph assertion.
    monkeypatch.setattr("stateguard3r.v16_source_import_audit.PACKAGE", PACKAGE)
    script = audit_v16_production_script(ROOT / "scripts" / "run_recal3r_bounded_update_pressure_v16.py", RUNTIME)
    assert "dynamic_rgb_capability_v16" in script["production_script_local_imports"]
    source = (PACKAGE / "bounded_update_pressure_v16.py").read_text(encoding="utf-8")
    package = tmp_path / "stateguard3r"
    package.mkdir()
    polluted = package / "bounded_update_pressure_v16.py"
    polluted.write_text("from . import rgb_listing_capsule_v15\n" + source, encoding="utf-8")
    monkeypatch.setattr("stateguard3r.v16_source_import_audit.PACKAGE", package)
    with pytest.raises(V16SourceImportAuditError, match="prohibited"):
        audit_v16_runtime_import_graph((polluted,))


def test_v16_detector_is_bounded_prefix_only_and_rejects_extra_capability() -> None:
    config = detector.FrozenDetectorV16Config(
        detector.V16ScoringConfig(window=3, min_history=2, max_z=5.0, master_seed=0),
        {name: 1e-3 for name in detector.CANONICAL_SIGNALS},
        2.0,
    )
    captures = [{"rgb_capture_timestamp": str(index + 1)} for index in range(5)]
    value = detector.IncrementalOnlinePrefixDetectorV16(config, captures=captures, capture_provenance={"raw": "rgb"})
    for frame in range(5):
        record = {"frame_id": frame, "overlap": None if frame == 0 else 0.9, "reliability": 0.9}
        decision = value.observe(record)
        assert decision.frame_id == frame
        value.commit(record)
    assert value.retained_row_count == 3
    with pytest.raises(detector.OnlineDetectorV16Error, match="non-v16"):
        value.observe({"frame_id": 5, "label": 1})


def test_v16_timestamp_parser_binds_only_selected_raw_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    rgb = root / "rgb"
    rgb.mkdir(parents=True)
    paths = []
    for index in range(2):
        path = rgb / f"{index}.png"
        path.write_bytes(f"rgb-{index}".encode("ascii"))
        path.chmod(0o444)
        paths.append(path)
    listing = root / "rgb.txt"
    listing.write_text("1.0 rgb/0.png\n2.0 rgb/1.png\n", encoding="utf-8")
    listing.chmod(0o444)
    records, provenance = timestamp.capture_timestamp_records_v16(paths, rgb_txt=listing, dataset_root=root)
    assert [row["rgb_capture_timestamp"] for row in records] == ["1.0", "2.0"]
    sidecar = timestamp.timestamp_order_sidecar_v16(records, provenance=provenance)
    assert [row["timestamp_order_violation"] for row in sidecar["records"]] == [None, False]


def test_v16_online_overlap_requires_only_normalized_previous_current_rgb() -> None:
    first = np.zeros((1, 3, 3, 4), dtype=np.float32)
    second = np.ones((1, 3, 3, 4), dtype=np.float32)
    converted = overlap.normalized_tensor_to_uint8_rgb_v16(first)
    assert converted.shape == (3, 4, 3) and converted.dtype == np.uint8
    assert converted.flags["C_CONTIGUOUS"] and not np.shares_memory(converted, first)
    with pytest.raises(overlap.OnlineVisualOverlapV16Error, match="normalized"):
        overlap.normalized_tensor_to_uint8_rgb_v16(second * 2.0)
