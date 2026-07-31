from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import stateguard3r.timeline as timeline_module
from stateguard3r.timeline import (
    TimelineError,
    _write_svg_atomic,
    generate_timeline,
    main,
    render_timeline_svg,
)


def _inputs() -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "frame_id": 10,
            "reliability": 0.9,
            "global_state_delta": 0.1,
            "pose_jump": None,
            "geometric_residual": 0.2,
        },
        {
            "frame_id": 20,
            "reliability": 0.8,
            "global_state_delta": 0.4,
            "pose_jump": None,
            "geometric_residual": 0.3,
        },
        {
            "frame_id": 30,
            "reliability": None,
            "global_state_delta": None,
            "pose_jump": None,
            "geometric_residual": None,
        },
        {
            "frame_id": 40,
            "reliability": 0.4,
            "global_state_delta": 1.2,
            "pose_jump": None,
            "geometric_residual": 0.9,
        },
        {
            "frame_id": 50,
            "reliability": 0.7,
            "global_state_delta": 0.2,
            "pose_jump": None,
            "geometric_residual": 0.25,
        },
    ]
    metrics: dict[str, object] = {
        "schema_version": "detection-only-v0",
        "frame_count": 5,
        "frame_ids": [10, 20, 30, 40, 50],
        "labels": [0, 1, 1, 0, 1],
        "methods": {
            "combined": {"scores": [0.0, 4.0, None, 6.0, 2.0]},
        },
        "by_corruption_type": {
            "dynamic_occlusion": {"labels": [0, 0, 0, 0, 1]},
            "low_overlap_jump": {"labels": [0, 1, 1, 0, 0]},
        },
    }
    return records, metrics


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    records, metrics = _inputs()
    health_path = tmp_path / "health.jsonl"
    metrics_path = tmp_path / "metrics.json"
    health_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return health_path, metrics_path


def test_svg_has_type_bands_all_panels_and_null_breaks() -> None:
    records, metrics = _inputs()

    svg = render_timeline_svg(records, metrics, title="tiny & strict")
    root = ET.fromstring(svg)
    elements = list(root.iter())

    assert root.tag.endswith("svg")
    assert "tiny &amp; strict" in svg
    corruption_types = {
        element.attrib["data-corruption-type"]
        for element in elements
        if "data-corruption-type" in element.attrib
    }
    assert corruption_types == {"dynamic_occlusion", "low_overlap_jump"}
    panels = {
        element.attrib["data-signal"]: element
        for element in elements
        if element.attrib.get("class") == "signal-panel"
    }
    assert set(panels) == {
        "combined_risk",
        "reliability",
        "update_magnitude",
        "pose_jump",
        "geometric_residual",
    }
    assert panels["update_magnitude"].attrib["data-source"] == "global_state_delta"
    combined_segments = [
        element
        for element in elements
        if element.tag.endswith("polyline")
        and element.attrib.get("data-signal") == "combined_risk"
    ]
    assert len(combined_segments) == 2
    assert [
        (element.attrib["data-start-position"], element.attrib["data-end-position"])
        for element in combined_segments
    ] == [("0", "1"), ("3", "4")]
    assert any(
        element.attrib.get("data-missing") == "true"
        for element in panels["pose_jump"].iter()
    )
    metadata = next(element for element in elements if element.tag.endswith("metadata"))
    payload = json.loads(metadata.text or "{}")
    assert payload["null_policy"] == "line_break_no_interpolation"
    assert payload["signal_sources"]["pose_jump"] == "pose_jump"


def test_generate_timeline_cli_writes_parseable_svg_atomically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    health_path, metrics_path = _write_inputs(tmp_path)
    output_path = tmp_path / "nested" / "timeline.svg"

    exit_code = main(
        [
            str(health_path),
            str(metrics_path),
            "--output",
            str(output_path),
            "--width",
            "800",
            "--panel-height",
            "90",
        ]
    )

    assert exit_code == 0
    assert "for 5 frames" in capsys.readouterr().out
    ET.parse(output_path)
    assert list(output_path.parent.glob(".timeline.svg.*.tmp")) == []


def test_frame_id_mismatch_is_rejected() -> None:
    records, metrics = _inputs()
    metrics["frame_ids"] = [10, 20, 30, 40, 999]

    with pytest.raises(TimelineError, match="exactly match"):
        render_timeline_svg(records, metrics)


def test_direct_renderer_rejects_empty_health_records() -> None:
    with pytest.raises(TimelineError, match="must not be empty"):
        render_timeline_svg([], {"frame_count": 0})


def test_output_cannot_alias_either_input(tmp_path: Path) -> None:
    health_path, metrics_path = _write_inputs(tmp_path)
    original = health_path.read_bytes()

    with pytest.raises(TimelineError, match="must not overwrite"):
        generate_timeline(
            health_path,
            metrics_path,
            health_path,
            overwrite=True,
        )

    assert health_path.read_bytes() == original


def test_atomic_overwrite_failure_preserves_previous_svg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "timeline.svg"
    output_path.write_text("known-good\n", encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(timeline_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        _write_svg_atomic(output_path, "<svg/>\n", overwrite=True)

    assert output_path.read_text(encoding="utf-8") == "known-good\n"
    assert list(tmp_path.glob(".timeline.svg.*.tmp")) == []
