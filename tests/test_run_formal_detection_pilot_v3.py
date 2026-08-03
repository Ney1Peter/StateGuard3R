from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_formal_detection_pilot_v3 as pilot
from stateguard3r.timestamp_order_v3 import (
    capture_timestamp_records,
    timestamp_order_sidecar,
)


def _spec(tmp_path: Path, run_id: str, *, split: str = "holdout") -> pilot.RunSpec:
    listing = tmp_path / f"{run_id}-rgb.txt"
    listing.write_text("0.0 image.png\n", encoding="utf-8")
    return pilot.RunSpec(
        run_id,
        split,
        "wrong_order_segment",
        tmp_path,
        pilot.Snapshot(tmp_path / "source-manifest.json", b"{}\n"),
        pilot.Snapshot(tmp_path / "input-manifest.json", b"{}\n"),
        pilot.Snapshot(listing, listing.read_bytes()),
        tmp_path,
    )


def _frozen_run(root: Path, run_id: str) -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    for filename in pilot.RUN_FILES.values():
        (directory / filename).write_bytes(b"{}\n")
        (directory / filename).chmod(0o444)
    directory.chmod(0o555)


def test_blind_preflight_never_reads_response_or_timestamp_sidecar_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = [_spec(tmp_path, name) for name in ("blind-dynamic", "blind-wrong-order", "blind-low-overlap")]
    for spec in specs[:2]:
        _frozen_run(tmp_path, spec.run_id)
    with pytest.raises(pilot.FormalV3Error, match="all three blind outputs"):
        pilot._preflight_holdout_outputs(tmp_path, specs)
    _frozen_run(tmp_path, specs[2].run_id)

    def fail_if_read(self: Path, *args, **kwargs):
        raise AssertionError(f"blind preflight read bytes from {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_if_read)
    pilot._preflight_holdout_outputs(tmp_path, specs)


def test_blind_preflight_requires_timestamp_sidecar_in_exact_layout(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "blind-dynamic")
    _frozen_run(tmp_path, spec.run_id)
    sidecar = tmp_path / spec.run_id / "timestamp-order.json"
    sidecar.chmod(0o644)

    with pytest.raises(pilot.FormalV3Error, match="timestamp-order.json is not frozen"):
        pilot._preflight_holdout_outputs(tmp_path, [spec])


def _timestamp_fixture(tmp_path: Path) -> tuple[pilot.RunSpec, pilot.Snapshot, dict[str, object]]:
    root = tmp_path / "raw"
    root.mkdir()
    images = []
    for index in range(30):
        image = root / f"frame-{index}.png"
        image.write_bytes(f"frame-{index}".encode())
        images.append(image)
    listing = root / "rgb.txt"
    listing.write_text(
        "".join(f"{index / 10:.1f} frame-{index}.png" + chr(10) for index in range(30)),
        encoding="utf-8",
    )
    captures, provenance = capture_timestamp_records(images, rgb_txt=listing, dataset_root=root)
    sidecar = timestamp_order_sidecar(captures, provenance=provenance)
    sidecar_path = tmp_path / "timestamp-order.json"
    payload = (json.dumps(sidecar, sort_keys=True, indent=2) + "\n").encode()
    sidecar_path.write_bytes(payload)
    snapshot = pilot.Snapshot(sidecar_path, payload)
    spec = pilot.RunSpec(
        "blind-dynamic",
        "holdout",
        "dynamic_occlusion",
        tmp_path,
        pilot.Snapshot(tmp_path / "source.json", b"{}\n"),
        pilot.Snapshot(tmp_path / "input.json", b"{}\n"),
        pilot.Snapshot(listing, listing.read_bytes()),
        root,
    )
    metadata: dict[str, object] = {
        "capture_timestamp_order": {
            "path": str(sidecar_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "schema_version": sidecar["schema_version"],
            "purpose": sidecar["purpose"],
            "violation_positions": [],
        },
        "images": [{"sha256": row["rgb_path_sha256"]} for row in sidecar["records"]],
    }
    return spec, snapshot, metadata


def test_timestamp_sidecar_validation_binds_raw_listing_and_final_model_images(
    tmp_path: Path,
) -> None:
    spec, snapshot, metadata = _timestamp_fixture(tmp_path)

    assert pilot._validate_timestamp_sidecar(snapshot, metadata, spec) == [None] + [False] * 29

    metadata["images"][1]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(pilot.FormalV3Error, match="RGB binding"):
        pilot._validate_timestamp_sidecar(snapshot, metadata, spec)


def test_fixed_v2_config_is_bound_to_the_declared_immutable_hash() -> None:
    config, calibration, value = pilot._v2_fixed_config()

    assert hashlib.sha256(config.payload).hexdigest() == pilot.V2_FORMAL_CONFIG_SHA256
    assert calibration.path.name == "calibration-manifest.json"
    assert value["thresholds"]["combined"] == pytest.approx(2.0686323694270046)


def _go_metrics(*, wrong_delay: int | None = 0) -> dict[str, object]:
    combined = {
        "macro_auroc": 0.9,
        "per_run": {
            "blind-wrong-order": {"delay": {"intervals": [{"delay_frames": wrong_delay}]}},
        },
        "events": {"detected_corruption_types": ["dynamic_occlusion", "wrong_order_segment", "low_overlap_jump"]},
        "pooled": {"false_positive_rate": 0.0},
        "clean_prefix": {"false_positive_rate": 0.0},
        "diagnostics": {"max_per_run_consecutive_false_positives": 0, "startup_max_streak": 0},
    }
    return {
        "methods": {
            "combined": combined,
            "random": {"macro_auroc": 0.5},
            "update_magnitude_only": {"macro_auroc": 0.8},
            "reliability_only": {"macro_auroc": 0.7},
        }
    }


def test_v3_gate_requires_wrong_order_detection_within_one_frame() -> None:
    development = _go_metrics()
    assert pilot._go_no_go(development, _go_metrics(), provenance=True)["decision"] == "DETECTOR_V3_GO"
    assert pilot._go_no_go(development, _go_metrics(wrong_delay=2), provenance=True)["decision"] == "DETECTOR_V3_NO_GO"


def test_runner_command_is_explicit_v3_timestamp_execution(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "blind-wrong-order")
    command = pilot._runner_command(spec, tmp_path / "run")

    assert command[command.index("--health-profile") + 1] == "v3"
    assert command[command.index("--rgb-timestamp-listing") + 1] == str(spec.timestamp_listing.path)
    assert command[command.index("--timestamp-dataset-root") + 1] == str(spec.timestamp_root)


def test_evaluate_does_not_seal_or_read_when_structural_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(pilot, "OUTPUTS", tmp_path)
    monkeypatch.setattr(pilot, "V3_INPUTS", inputs)
    specs = [_spec(tmp_path, name) for name in ("blind-dynamic", "blind-wrong-order", "blind-low-overlap")]
    monkeypatch.setattr(pilot, "_run_specs", lambda: tuple(specs))

    def fail_preflight(*_args, **_kwargs):
        raise pilot.FormalV3Error("all three blind outputs must finish")

    monkeypatch.setattr(pilot, "_preflight_holdout_outputs", fail_preflight)
    output = tmp_path / "formal-v3-evaluation-0001"
    with pytest.raises(pilot.FormalV3Error, match="all three blind outputs"):
        pilot.evaluate_holdout(inputs, tmp_path / "commit", tmp_path / "calibration", runs, output)
    assert not output.exists()
    assert not output.with_name(output.name + "-attempt").exists()


def test_publish_is_atomic_immutable_and_no_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pilot, "OUTPUTS", tmp_path)
    output = tmp_path / "formal-v3-output"
    pilot._publish(output, {"nested/evidence.json": b"{}\n"})

    assert output.stat().st_mode & 0o777 == 0o555
    assert (output / "nested" / "evidence.json").stat().st_mode & 0o777 == 0o444
    with pytest.raises(pilot.FormalV3Error, match="already exists"):
        pilot._publish(output, {"again.json": b"{}\n"})
