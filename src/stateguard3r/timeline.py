"""Dependency-light SVG timelines for detection-only experiments.

The renderer consumes the exact health JSONL and detection metrics JSON used by
the evaluator. Missing measurements remain gaps: it never interpolates a line
through ``null`` values and never substitutes one signal for another.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

SCHEMA_VERSION = "stateguard3r.timeline.v0"
DEFAULT_WIDTH = 1200
DEFAULT_PANEL_HEIGHT = 120

PANEL_SPECS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("combined_risk", "Combined risk", (), "#d62728"),
    (
        "reliability",
        "Reliability",
        ("reliability", "reliability_mean", "state_reliability_mean"),
        "#2ca02c",
    ),
    (
        "update_magnitude",
        "Update magnitude",
        (
            "update_magnitude",
            "global_state_delta",
            "global_state_delta_mean",
            "state_delta_mean",
            "delta_norm_mean",
        ),
        "#9467bd",
    ),
    (
        "pose_jump",
        "Pose jump",
        ("pose_jump", "pose_jump_translation", "pose_translation_jump"),
        "#ff7f0e",
    ),
    (
        "geometric_residual",
        "Geometric residual",
        (
            "geometric_residual",
            "geometric_or_pointmap_residual",
            "pointmap_residual",
            "relative_residual",
            "residual_score",
        ),
        "#1f77b4",
    ),
)

CORRUPTION_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
)


class TimelineError(ValueError):
    """Raised when timeline inputs are malformed or disagree."""


def _read_health_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise TimelineError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise TimelineError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            records.append(value)
    if not records:
        raise TimelineError(f"{path}: health ledger contains no records")
    return records


def _read_metrics_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimelineError(f"cannot read detection metrics JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TimelineError("detection metrics must be a JSON object")
    return value


def _numeric_series(values: Sequence[Any], *, name: str, length: int) -> FloatArray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TimelineError(f"{name} must be a JSON array")
    if len(values) != length:
        raise TimelineError(f"{name} has {len(values)} entries; expected {length}")
    result = np.full(length, np.nan, dtype=np.float64)
    for index, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TimelineError(f"{name}[{index}] must be finite or null")
        number = float(value)
        if not math.isfinite(number):
            raise TimelineError(f"{name}[{index}] must be finite or null")
        result[index] = number
    return result


def _binary_labels(values: Any, *, name: str, length: int) -> NDArray[np.int64]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TimelineError(f"{name} must be a JSON array")
    if len(values) != length:
        raise TimelineError(f"{name} has {len(values)} entries; expected {length}")
    labels = np.zeros(length, dtype=np.int64)
    for index, value in enumerate(values):
        if isinstance(value, bool):
            labels[index] = int(value)
        elif type(value) is int and value in (0, 1):
            labels[index] = value
        else:
            raise TimelineError(f"{name}[{index}] must be 0 or 1")
    return labels


def _select_health_series(
    records: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> tuple[FloatArray, str | None]:
    length = len(records)
    first_present: str | None = None
    for alias in aliases:
        if any(alias in record for record in records) and first_present is None:
            first_present = alias
        values = [record.get(alias) for record in records]
        series = _numeric_series(values, name=f"health.{alias}", length=length)
        if np.isfinite(series).any():
            return series, alias
    return np.full(length, np.nan, dtype=np.float64), first_present


def _validated_inputs(
    records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> tuple[
    list[Any],
    dict[str, tuple[FloatArray, str | None]],
    list[tuple[str, NDArray[np.int64]]],
]:
    length = len(records)
    if length == 0:
        raise TimelineError("health records must not be empty")
    frame_count = metrics.get("frame_count")
    if type(frame_count) is not int or frame_count != length:
        raise TimelineError(
            f"metrics frame_count {frame_count!r} does not match {length} health records"
        )

    health_frame_ids = [record.get("frame_id", index) for index, record in enumerate(records)]
    metric_frame_ids = metrics.get("frame_ids")
    if not isinstance(metric_frame_ids, list) or metric_frame_ids != health_frame_ids:
        raise TimelineError("metrics frame_ids must exactly match health JSONL frame_ids")

    overall_labels = _binary_labels(
        metrics.get("labels"), name="metrics.labels", length=length
    )
    methods = metrics.get("methods")
    if not isinstance(methods, Mapping):
        raise TimelineError("metrics.methods must be a JSON object")
    combined = methods.get("combined")
    if not isinstance(combined, Mapping):
        raise TimelineError("metrics.methods.combined must be a JSON object")
    combined_scores = _numeric_series(
        combined.get("scores"), name="metrics.methods.combined.scores", length=length
    )

    signals: dict[str, tuple[FloatArray, str | None]] = {
        "combined_risk": (combined_scores, "metrics.methods.combined.scores")
    }
    for canonical, _, aliases, _ in PANEL_SPECS[1:]:
        signals[canonical] = _select_health_series(records, aliases)

    bands: list[tuple[str, NDArray[np.int64]]] = []
    grouped = metrics.get("by_corruption_type", {})
    if not isinstance(grouped, Mapping):
        raise TimelineError("metrics.by_corruption_type must be a JSON object")
    for corruption_type in sorted(grouped):
        payload = grouped[corruption_type]
        if not isinstance(corruption_type, str) or not corruption_type:
            raise TimelineError("corruption type names must be non-empty strings")
        if not isinstance(payload, Mapping):
            raise TimelineError(
                f"metrics.by_corruption_type[{corruption_type!r}] must be an object"
            )
        labels = _binary_labels(
            payload.get("labels"),
            name=f"metrics.by_corruption_type[{corruption_type!r}].labels",
            length=length,
        )
        bands.append((corruption_type, labels))
    if bands:
        grouped_union = np.maximum.reduce([labels for _, labels in bands])
        if not np.array_equal(grouped_union, overall_labels):
            raise TimelineError(
                "the union of corruption-type labels must equal metrics.labels"
            )
    if not bands and np.any(overall_labels):
        bands.append(("corruption", overall_labels))
    return health_frame_ids, signals, bands


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == mask.size - 1):
            end = index if active and index == mask.size - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def _scale(series: FloatArray) -> tuple[float, float]:
    finite = series[np.isfinite(series)]
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1e-6)
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.05
    return minimum - padding, maximum + padding


def _frame_x(index: int, *, left: float, plot_width: float, count: int) -> float:
    return left + (index + 0.5) * plot_width / count


def render_timeline_svg(
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    *,
    width: int = DEFAULT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    title: str = "StateGuard3R detection timeline",
) -> str:
    """Render validated health and detection data as a standalone SVG."""

    if type(width) is not int or width < 640:
        raise TimelineError("width must be an integer of at least 640")
    if type(panel_height) is not int or panel_height < 80:
        raise TimelineError("panel_height must be an integer of at least 80")
    if not isinstance(title, str) or not title.strip():
        raise TimelineError("title must be a non-empty string")

    frame_ids, signals, bands = _validated_inputs(records, metrics)
    count = len(frame_ids)
    left = 180.0
    right = 30.0
    plot_width = width - left - right
    band_height = 18
    band_stride = 25
    band_rows = max(1, len(bands))
    panels_top = 58 + band_rows * band_stride + 22
    panel_stride = panel_height + 26
    height = panels_top + len(PANEL_SPECS) * panel_stride + 38

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "frame_count": count,
        "frame_ids": frame_ids,
        "null_policy": "line_break_no_interpolation",
        "signal_sources": {name: source for name, (_, source) in signals.items()},
        "corruption_types": [name for name, _ in bands],
    }
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f"<metadata>{escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}</metadata>",
        "<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}"
        ".axis{stroke:#999;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
        ".panel{fill:#fafafa;stroke:#ccc;stroke-width:1}</style>",
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="20" y="30" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]

    if not bands:
        parts.append(
            '<text x="20" y="65" font-size="12" data-no-corruption="true">'
            "No positive corruption labels</text>"
        )
    for band_index, (corruption_type, labels) in enumerate(bands):
        y = 50 + band_index * band_stride
        color = CORRUPTION_COLORS[band_index % len(CORRUPTION_COLORS)]
        parts.append(
            f'<text x="20" y="{y + 14}" font-size="12">{escape(corruption_type)}</text>'
        )
        parts.append(
            f'<rect x="{left:.2f}" y="{y}" width="{plot_width:.2f}" '
            f'height="{band_height}" fill="#f4f4f4" stroke="#ddd"/>'
        )
        for start, end in _runs(labels.astype(bool)):
            cell_width = plot_width / count
            x = left + start * cell_width
            run_width = (end - start + 1) * cell_width
            parts.append(
                f'<rect class="corruption-band" data-corruption-type="{escape(corruption_type, quote=True)}" '
                f'data-start-position="{start}" data-end-position="{end}" '
                f'x="{x:.2f}" y="{y}" width="{run_width:.2f}" '
                f'height="{band_height}" fill="{color}" fill-opacity="0.42"/>'
            )

    tick_indices = sorted(
        {int(round(value)) for value in np.linspace(0, count - 1, min(5, count))}
    )
    for panel_index, (canonical, label, _, color) in enumerate(PANEL_SPECS):
        series, source = signals[canonical]
        panel_y = panels_top + panel_index * panel_stride
        inner_top = panel_y + 23
        inner_height = panel_height - 42
        parts.append(
            f'<g class="signal-panel" data-signal="{canonical}" '
            f'data-source="{escape(source or "", quote=True)}">'
        )
        parts.append(
            f'<rect class="panel" x="{left:.2f}" y="{panel_y}" '
            f'width="{plot_width:.2f}" height="{panel_height}"/>'
        )
        source_suffix = f" ({source})" if source else ""
        parts.append(
            f'<text x="20" y="{panel_y + 19}" font-size="13" font-weight="bold">'
            f"{escape(label + source_suffix)}</text>"
        )
        for tick_index in tick_indices:
            x = _frame_x(tick_index, left=left, plot_width=plot_width, count=count)
            parts.append(
                f'<line class="grid" x1="{x:.2f}" y1="{inner_top}" '
                f'x2="{x:.2f}" y2="{inner_top + inner_height}"/>'
            )

        finite = np.isfinite(series)
        if not finite.any():
            parts.append(
                f'<text data-missing="true" x="{left + 12:.2f}" '
                f'y="{panel_y + panel_height / 2:.2f}" font-size="12" fill="#777">'
                "No finite measured data; no proxy or interpolation used</text>"
            )
        else:
            minimum, maximum = _scale(series)
            parts.append(
                f'<text x="{left - 8:.2f}" y="{inner_top + 5:.2f}" '
                f'text-anchor="end" font-size="10">{maximum:.6g}</text>'
            )
            parts.append(
                f'<text x="{left - 8:.2f}" y="{inner_top + inner_height:.2f}" '
                f'text-anchor="end" font-size="10">{minimum:.6g}</text>'
            )
            for segment_index, (start, end) in enumerate(_runs(finite)):
                points: list[str] = []
                for index in range(start, end + 1):
                    x = _frame_x(index, left=left, plot_width=plot_width, count=count)
                    y = inner_top + (maximum - series[index]) * inner_height / (
                        maximum - minimum
                    )
                    points.append(f"{x:.2f},{y:.2f}")
                attributes = (
                    f'data-signal="{canonical}" data-segment="{segment_index}" '
                    f'data-start-position="{start}" data-end-position="{end}"'
                )
                if len(points) == 1:
                    x, y = points[0].split(",")
                    parts.append(
                        f'<circle {attributes} cx="{x}" cy="{y}" r="2.5" fill="{color}"/>'
                    )
                else:
                    parts.append(
                        f'<polyline {attributes} points="{" ".join(points)}" '
                        f'fill="none" stroke="{color}" stroke-width="2"/>'
                    )
        parts.append("</g>")

    axis_y = panels_top + len(PANEL_SPECS) * panel_stride - 17
    parts.append(
        f'<line class="axis" x1="{left:.2f}" y1="{axis_y}" '
        f'x2="{left + plot_width:.2f}" y2="{axis_y}"/>'
    )
    for tick_index in tick_indices:
        x = _frame_x(tick_index, left=left, plot_width=plot_width, count=count)
        parts.append(
            f'<text x="{x:.2f}" y="{axis_y + 17}" text-anchor="middle" '
            f'font-size="10">{escape(str(frame_ids[tick_index]))}</text>'
        )
    parts.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{axis_y + 33}" '
        'text-anchor="middle" font-size="11">frame_id</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _paths_alias(left: Path, right: Path) -> bool:
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        if os.path.abspath(left) == os.path.abspath(right):
            return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _write_svg_atomic(path: Path, svg: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise TimelineError(
            f"timeline output already exists: {path}; use overwrite=True or --overwrite"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(svg)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise TimelineError(f"timeline output already exists: {path}") from error
            temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def generate_timeline(
    health_jsonl: str | os.PathLike[str],
    metrics_json: str | os.PathLike[str],
    output_svg: str | os.PathLike[str],
    *,
    width: int = DEFAULT_WIDTH,
    panel_height: int = DEFAULT_PANEL_HEIGHT,
    title: str = "StateGuard3R detection timeline",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate inputs, render an SVG, and publish it atomically."""

    health_path = Path(health_jsonl)
    metrics_path = Path(metrics_json)
    output_path = Path(output_svg)
    if _paths_alias(output_path, health_path) or _paths_alias(output_path, metrics_path):
        raise TimelineError("timeline output must not overwrite either input file")
    records = _read_health_jsonl(health_path)
    metrics = _read_metrics_json(metrics_path)
    svg = render_timeline_svg(
        records,
        metrics,
        width=width,
        panel_height=panel_height,
        title=title,
    )
    _write_svg_atomic(output_path, svg, overwrite=overwrite)
    return {
        "schema_version": SCHEMA_VERSION,
        "frame_count": len(records),
        "output": str(output_path),
        "bytes": len(svg.encode("utf-8")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render detection metrics and measured health signals as SVG."
    )
    parser.add_argument("health_jsonl", type=Path)
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--panel-height", type=int, default=DEFAULT_PANEL_HEIGHT)
    parser.add_argument("--title", default="StateGuard3R detection timeline")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = generate_timeline(
            args.health_jsonl,
            args.metrics_json,
            args.output,
            width=args.width,
            panel_height=args.panel_height,
            title=args.title,
            overwrite=args.overwrite,
        )
    except (OSError, TimelineError) as error:
        parser.error(str(error))
    print(
        f"wrote {summary['output']} for {summary['frame_count']} frames "
        f"({summary['bytes']} bytes)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
