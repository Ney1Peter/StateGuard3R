from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_synthetic_smoke_inputs import (
    CORRUPTION_INTERVALS,
    FRAME_COUNT,
    SYNTHETIC_MARKER,
    SyntheticSmokeInputError,
    generate_synthetic_smoke_inputs,
    main as generate_main,
)
from stateguard3r.corruption import main as corruption_main
from stateguard3r.detection import main as detection_main
from stateguard3r.health import read_health_jsonl


def _make_frames(tmp_path: Path) -> tuple[Path, Path]:
    frame_a = tmp_path / "frame_a.png"
    frame_b = tmp_path / "frame_b.png"
    frame_a.write_bytes(b"synthetic-frame-a")
    frame_b.write_bytes(b"synthetic-frame-b")
    return frame_a, frame_b


def _file_state(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_generator_writes_marked_deterministic_inputs_without_copying_sources(
    tmp_path: Path,
) -> None:
    frame_a, frame_b = _make_frames(tmp_path)
    source_state = _file_state((frame_a, frame_b))

    first = generate_synthetic_smoke_inputs(
        tmp_path / "first", frame_a, frame_b
    )
    second = generate_synthetic_smoke_inputs(
        tmp_path / "second", frame_a, frame_b
    )

    assert _file_state((frame_a, frame_b)) == source_state
    assert {path.name for path in first.values()} == {
        "source_manifest.json",
        "corruption_specs.json",
        "health.jsonl",
    }
    assert sorted(tmp_path.rglob("*.png")) == [frame_a, frame_b]
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()

    source = json.loads(first["source_manifest"].read_text(encoding="utf-8"))
    assert source["synthetic"] is True
    assert source["not_real_experiment"] is True
    assert source["warning"] == SYNTHETIC_MARKER
    assert source["frame_count"] == FRAME_COUNT
    assert len(source["frames"]) == FRAME_COUNT
    assert {frame["path"] for frame in source["frames"]} == {
        str(frame_a.resolve()),
        str(frame_b.resolve()),
    }
    assert all(frame["synthetic"] is True for frame in source["frames"])

    config = json.loads(first["corruption_specs"].read_text(encoding="utf-8"))
    assert config["synthetic"] is True
    assert config["not_real_experiment"] is True
    assert config["warning"] == SYNTHETIC_MARKER
    assert [spec["type"] for spec in config["corruptions"]] == [
        item[0] for item in CORRUPTION_INTERVALS
    ]
    target_sets = [
        set(range(spec["start"], spec["end"] + 1))
        for spec in config["corruptions"]
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(target_sets)
        for right in target_sets[index + 1 :]
    )


def test_health_jsonl_is_strict_stable_and_anomalous_on_all_labeled_fields(
    tmp_path: Path,
) -> None:
    frame_a, frame_b = _make_frames(tmp_path)
    outputs = generate_synthetic_smoke_inputs(
        tmp_path / "inputs", frame_a, frame_b
    )

    records = read_health_jsonl(outputs["health_jsonl"])
    assert len(records) == FRAME_COUNT
    assert [record.frame_id for record in records] == list(range(FRAME_COUNT))
    assert all(SYNTHETIC_MARKER in record.decision for record in records)

    labeled = {
        frame_id
        for _, start, end in CORRUPTION_INTERVALS
        for frame_id in range(start, end + 1)
    }
    stable = [record for record in records if record.frame_id not in labeled]
    assert {
        (
            record.geometric_residual,
            record.pose_jump,
            record.update_magnitude,
            record.overlap,
            record.reliability,
        )
        for record in stable
    } == {(0.01, 0.005, 0.02, 0.95, 0.95)}

    for _, start, end in CORRUPTION_INTERVALS:
        interval = records[start : end + 1]
        assert all(record.geometric_residual > 0.01 for record in interval)
        assert all(record.pose_jump > 0.005 for record in interval)
        assert all(record.update_magnitude > 0.02 for record in interval)
        assert all(record.overlap < 0.95 for record in interval)
        assert all(record.reliability < 0.95 for record in interval)

    raw_records = [
        json.loads(line)
        for line in outputs["health_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert all(raw["decision"].startswith(SYNTHETIC_MARKER) for raw in raw_records)
    assert all(
        raw["uncertainty_u"] == pytest.approx(1 - raw["reliability"])
        for raw in raw_records
    )


def test_outputs_run_through_corruption_and_detection_clis(
    tmp_path: Path,
) -> None:
    frame_a, frame_b = _make_frames(tmp_path)
    inputs = generate_synthetic_smoke_inputs(
        tmp_path / "inputs", frame_a, frame_b
    )
    corruption_path = tmp_path / "run" / "corruption_manifest.json"
    metrics_path = tmp_path / "run" / "metrics.json"

    assert (
        corruption_main(
            [
                str(inputs["source_manifest"]),
                str(corruption_path),
                "--config",
                str(inputs["corruption_specs"]),
                "--seed",
                "0",
                "--check-paths",
            ]
        )
        == 0
    )
    assert (
        detection_main(
            [
                str(inputs["health_jsonl"]),
                str(corruption_path),
                "--output",
                str(metrics_path),
                "--window",
                "15",
                "--threshold",
                "3.0",
                "--method-threshold",
                "random=0.5",
                "--method-threshold",
                "update_magnitude_only=3.0",
                "--method-threshold",
                "reliability_only=-0.5",
                "--method-threshold",
                "combined=3.0",
                "--seed",
                "0",
            ]
        )
        == 0
    )

    corruption = json.loads(corruption_path.read_text(encoding="utf-8"))
    assert corruption["frame_count"] == FRAME_COUNT
    assert [item["type"] for item in corruption["corruptions"]] == [
        item[0] for item in CORRUPTION_INTERVALS
    ]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["frame_count"] == FRAME_COUNT
    assert metrics["positive_frames"] == 15
    assert set(metrics["by_corruption_type"]) == {
        item[0] for item in CORRUPTION_INTERVALS
    }
    for method in ("update_magnitude_only", "reliability_only", "combined"):
        assert metrics["methods"][method]["metrics"]["auroc"] == pytest.approx(1.0)
        assert metrics["methods"][method]["metrics"]["f1"] == pytest.approx(1.0)


def test_cli_prints_explicit_synthetic_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame_a, frame_b = _make_frames(tmp_path)

    assert (
        generate_main(
            [
                "--output-dir",
                str(tmp_path / "inputs"),
                "--frame-a",
                str(frame_a),
                "--frame-b",
                str(frame_b),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert SYNTHETIC_MARKER in output
    assert "not real experiment results" in output


def test_generator_rejects_missing_aliased_or_existing_outputs(tmp_path: Path) -> None:
    frame_a, frame_b = _make_frames(tmp_path)

    with pytest.raises(SyntheticSmokeInputError, match="existing file"):
        generate_synthetic_smoke_inputs(
            tmp_path / "missing", frame_a, tmp_path / "missing.png"
        )

    with pytest.raises(SyntheticSmokeInputError, match="distinct files"):
        generate_synthetic_smoke_inputs(tmp_path / "aliased", frame_a, frame_a)

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    existing = output_dir / "health.jsonl"
    existing.write_text("keep-me\n", encoding="utf-8")
    with pytest.raises(SyntheticSmokeInputError, match="refusing to replace"):
        generate_synthetic_smoke_inputs(output_dir, frame_a, frame_b)
    assert existing.read_text(encoding="utf-8") == "keep-me\n"
