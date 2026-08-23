"""Read-only, causal evidence extraction for StateTriage3R Stage 0.

This module deliberately has no ReCal3R, Torch, image-loader, GT, manifest, or
recovery-policy imports.  A caller gives it *one already-produced frame* and
the current model-ready RGB.  The observer keeps only bounded, copied 3D
anchors and scalar histories from previous calls.  It therefore cannot alter a
model, nor can a future prediction affect an already emitted ledger row.

The geometric quantities are output-space proxies.  In particular, reference
anchor support is not a claim that a ReCal3R token has a spatial identity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "stateguard3r.state-triage-v2-evidence.v1"

FloatArray = NDArray[np.float64]


class StateTriageEvidenceError(ValueError):
    """Raised when a claimed current-frame observation is malformed."""


@dataclass(frozen=True)
class EvidenceConfig:
    """Fixed resource bounds and geometry conventions for one observer.

    All thresholds here define evidence extraction, not a learned classifier.
    The Stage-0 typed decision thresholds are frozen separately in Gate B.
    """

    grid_rows: int = 12
    grid_columns: int = 16
    image_quality_rows: int = 32
    image_quality_columns: int = 32
    max_anchor_history: int = 1_024
    max_conflict_anchor_history: int = 512
    scalar_history_window: int = 15
    support_radius_relative: float = 0.05
    support_radius_floor: float = 1e-3
    residual_mad_multiplier: float = 3.0
    scalar_scale_floor: float = 1e-6

    def __post_init__(self) -> None:
        positive_ints = (
            "grid_rows",
            "grid_columns",
            "image_quality_rows",
            "image_quality_columns",
            "max_anchor_history",
            "max_conflict_anchor_history",
            "scalar_history_window",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise StateTriageEvidenceError(f"{name} must be a positive integer")
        for name in (
            "support_radius_relative",
            "support_radius_floor",
            "residual_mad_multiplier",
            "scalar_scale_floor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise StateTriageEvidenceError(f"{name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceFrame:
    """One JSON-safe causal evidence row.

    ``*_z`` values use only scalar values from strictly earlier calls.  A
    missing native health value is represented by ``None`` rather than by an
    invented value.  The three provenance-like identifiers intentionally do
    not appear here: callers may put them in their outer ledger but must not
    use them as typed-decision features.
    """

    frame_id: int
    reference_anchor_count_before: int
    conflict_anchor_count_before: int
    sampled_anchor_count: int
    support_radius: float
    coverage_ratio: float
    novel_ratio: float
    temporal_conflict_support: float
    normalized_residual_median: float
    normalized_residual_mad: float
    residual_high_fraction: float
    residual_globality: float
    residual_locality: float
    pose_jump: float | None
    pose_jump_z: float | None
    native_geometric_residual: float | None
    native_geometric_residual_z: float | None
    uncertainty_u: float | None
    confidence_mean: float
    confidence_std: float
    confidence_disagreement_mean: float
    sharpness: float
    sharpness_z: float | None
    low_light_fraction: float
    clipped_fraction: float
    saturation_mean: float

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise StateTriageEvidenceError("frame_id must be a non-negative integer")
        for name in ("reference_anchor_count_before", "conflict_anchor_count_before", "sampled_anchor_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StateTriageEvidenceError(f"{name} must be a non-negative integer")
        if self.sampled_anchor_count < 1:
            raise StateTriageEvidenceError("sampled_anchor_count must be positive")
        finite_fields = (
            "support_radius",
            "coverage_ratio",
            "novel_ratio",
            "temporal_conflict_support",
            "normalized_residual_median",
            "normalized_residual_mad",
            "residual_high_fraction",
            "residual_globality",
            "residual_locality",
            "confidence_mean",
            "confidence_std",
            "confidence_disagreement_mean",
            "sharpness",
            "low_light_fraction",
            "clipped_fraction",
            "saturation_mean",
        )
        unit_fields = (
            "coverage_ratio",
            "novel_ratio",
            "temporal_conflict_support",
            "residual_high_fraction",
            "residual_globality",
            "residual_locality",
            "low_light_fraction",
            "clipped_fraction",
            "saturation_mean",
        )
        for name in finite_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise StateTriageEvidenceError(f"{name} must be finite and non-negative")
            if name in unit_fields and value > 1.0:
                raise StateTriageEvidenceError(f"{name} must be in [0, 1]")
        if not math.isclose(self.coverage_ratio + self.novel_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise StateTriageEvidenceError("coverage_ratio and novel_ratio must sum to one")
        if not math.isclose(self.residual_globality + self.residual_locality, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise StateTriageEvidenceError("residual_globality and residual_locality must sum to one")
        for name in ("pose_jump", "pose_jump_z", "native_geometric_residual", "native_geometric_residual_z", "uncertainty_u", "sharpness_z"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                raise StateTriageEvidenceError(f"{name} must be finite or null")
        if self.uncertainty_u is not None and not 0.0 <= float(self.uncertainty_u) <= 1.0:
            raise StateTriageEvidenceError("uncertainty_u must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_numpy(value: Any, *, name: str) -> FloatArray:
    """Copy tensor-like values into a finite float64 NumPy array.

    ``detach``/``cpu`` are duck-typed to support the CPU predictions returned
    by ReCal3R without importing Torch.  The returned array is always an
    independent copy so no input/model storage can be modified by this module.
    """

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    try:
        array = np.array(current, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise StateTriageEvidenceError(f"{name} cannot be converted to float64") from error
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StateTriageEvidenceError(f"{name} must be finite and non-empty")
    return array


def _optional_scalar(values: Mapping[str, Any] | None, name: str) -> float | None:
    if values is None or name not in values or values[name] is None:
        return None
    raw = values[name]
    if isinstance(raw, bool):
        raise StateTriageEvidenceError(f"health.{name} must be a finite scalar or null")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise StateTriageEvidenceError(f"health.{name} must be a finite scalar or null") from error
    if not math.isfinite(value):
        raise StateTriageEvidenceError(f"health.{name} must be a finite scalar or null")
    return value


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise StateTriageEvidenceError(f"{name} does not expose a shape")
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError) as error:
        raise StateTriageEvidenceError(f"{name} shape is invalid") from error


def _sample_pointmap(value: Any, name: str, *, y: NDArray[np.int64], x: NDArray[np.int64]) -> FloatArray:
    """Copy only fixed grid anchors, never a full pointmap into the observer."""

    shape = _shape(value, name=f"prediction.{name}")
    y_indices, x_indices = y.tolist(), x.tolist()
    if len(shape) == 4:
        if shape[0] != 1 or shape[-1] != 3:
            raise StateTriageEvidenceError(f"prediction.{name} must have shape (1, H, W, 3)")
        sampled = value[0, y_indices, x_indices]
    elif len(shape) == 3:
        if shape[-1] != 3:
            raise StateTriageEvidenceError(f"prediction.{name} must have shape (H, W, 3)")
        sampled = value[y_indices, x_indices]
    else:
        raise StateTriageEvidenceError(f"prediction.{name} must have a pointmap shape")
    values = _to_numpy(sampled, name=f"prediction.{name} sampled anchors")
    if values.shape != (len(y), 3):
        raise StateTriageEvidenceError(f"prediction.{name} sampled anchors have invalid shape")
    return values


def _sample_map(value: Any, name: str, *, y: NDArray[np.int64], x: NDArray[np.int64], shape: tuple[int, int]) -> FloatArray:
    raw_shape = _shape(value, name=f"prediction.{name}")
    y_indices, x_indices = y.tolist(), x.tolist()
    if raw_shape == (1, *shape):
        sampled = value[0, y_indices, x_indices]
    elif raw_shape == shape:
        sampled = value[y_indices, x_indices]
    else:
        raise StateTriageEvidenceError(f"prediction.{name} must have shape {shape} or (1, *shape)")
    values = _to_numpy(sampled, name=f"prediction.{name} sampled anchors")
    if values.shape != (len(y),):
        raise StateTriageEvidenceError(f"prediction.{name} sampled anchors have invalid shape")
    return values


def _pointmap_hw(value: Any, name: str) -> tuple[int, int]:
    shape = _shape(value, name=f"prediction.{name}")
    if len(shape) == 4 and shape[0] == 1 and shape[-1] == 3:
        return shape[1], shape[2]
    if len(shape) == 3 and shape[-1] == 3:
        return shape[0], shape[1]
    raise StateTriageEvidenceError(f"prediction.{name} must have a pointmap shape")


def _require_prediction(prediction: Mapping[str, Any], name: str) -> Any:
    if name not in prediction:
        raise StateTriageEvidenceError(f"prediction is missing {name}")
    return prediction[name]


def _grid_axis_indices(size: int, count: int) -> NDArray[np.int64]:
    return np.rint(np.linspace(0, size - 1, count)).astype(np.int64)


def _sample_model_rgb(value: Any, *, rows: int, columns: int) -> FloatArray:
    """Copy a fixed quality grid instead of an entire model-ready RGB image."""

    shape = _shape(value, name="model_ready_rgb")
    if len(shape) == 4:
        if shape[0] != 1 or shape[1] != 3:
            raise StateTriageEvidenceError("model_ready_rgb must have shape (1, 3, H, W)")
        height, width = shape[2], shape[3]
    elif len(shape) == 3:
        if shape[0] != 3:
            raise StateTriageEvidenceError("model_ready_rgb must have shape (3, H, W)")
        height, width = shape[1], shape[2]
    else:
        raise StateTriageEvidenceError("model_ready_rgb must have a channel-first image shape")
    if min(height, width) < 2:
        raise StateTriageEvidenceError("model_ready_rgb must have at least two pixels per dimension")
    y, x = _grid_axis_indices(height, rows), _grid_axis_indices(width, columns)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")
    y_indices, x_indices = y_grid.reshape(-1).tolist(), x_grid.reshape(-1).tolist()
    sampled = value[0, :, y_indices, x_indices] if len(shape) == 4 else value[:, y_indices, x_indices]
    channels = _to_numpy(sampled, name="model_ready_rgb sampled quality grid")
    if channels.shape == (rows * columns, 3):
        channels = channels.T
    if channels.shape != (3, rows * columns):
        raise StateTriageEvidenceError("model_ready_rgb sampled quality grid has invalid shape")
    rgb = np.moveaxis(channels.reshape(3, rows, columns), 0, -1)
    if float(rgb.min()) < -1.0 - 1e-6 or float(rgb.max()) > 1.0 + 1e-6:
        raise StateTriageEvidenceError("model_ready_rgb must be normalized to [-1, 1]")
    return rgb


def _camera_matrix(value: Any) -> FloatArray:
    matrix = _to_numpy(value, name="camera_to_reference")
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if matrix.shape != (4, 4):
        raise StateTriageEvidenceError("camera_to_reference must have shape (4, 4)")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-5):
        raise StateTriageEvidenceError("camera_to_reference must be homogeneous")
    return matrix


def _grid_indices(height: int, width: int, config: EvidenceConfig) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rows = _grid_axis_indices(height, config.grid_rows)
    columns = _grid_axis_indices(width, config.grid_columns)
    y, x = np.meshgrid(rows, columns, indexing="ij")
    return y.reshape(-1), x.reshape(-1)


def _nearest_distance(points: FloatArray, anchors: FloatArray) -> FloatArray:
    if anchors.size == 0:
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    # At most 192x2048 distances under the frozen default, deliberately small
    # enough to be a bounded observer rather than a retained pointmap.
    distances = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=2)
    return distances.min(axis=1)


def _scalar_z(history: Sequence[float], value: float | None, config: EvidenceConfig) -> float | None:
    if value is None:
        return None
    finite_history = np.asarray([item for item in history if math.isfinite(item)], dtype=np.float64)
    if finite_history.size == 0:
        return 0.0
    median = float(np.median(finite_history))
    mad = float(np.median(np.abs(finite_history - median)))
    return float(abs(value - median) / max(mad, config.scalar_scale_floor))


class EvidenceObserver:
    """Sequential bounded-history observer for one already-produced stream.

    The caller must invoke :meth:`observe` in consecutive frame order.  No API
    accepts a future frame or an entire prediction sequence, which makes
    prefix-invariance directly testable.
    """

    def __init__(self, config: EvidenceConfig = EvidenceConfig()) -> None:
        self.config = config
        self._next_frame_id = 0
        self._anchors: deque[FloatArray] = deque()
        self._conflict_anchors: deque[FloatArray] = deque()
        self._anchor_count = 0
        self._conflict_anchor_count = 0
        self._scalar_history: dict[str, deque[float]] = {
            "pose_jump": deque(maxlen=config.scalar_history_window),
            "native_geometric_residual": deque(maxlen=config.scalar_history_window),
            "sharpness": deque(maxlen=config.scalar_history_window),
        }

    @property
    def anchor_count(self) -> int:
        """Number of retained previous reference anchors (bounded by config)."""

        return self._anchor_count

    @property
    def conflict_anchor_count(self) -> int:
        """Number of retained previous high-residual anchors (bounded by config)."""

        return self._conflict_anchor_count

    def _stack(self, queue: deque[FloatArray]) -> FloatArray:
        if not queue:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(tuple(queue), axis=0)

    def _append_bounded(self, queue: deque[FloatArray], values: FloatArray, *, limit: int, conflict: bool) -> None:
        copied = np.array(values, dtype=np.float64, copy=True)
        if copied.size == 0:
            return
        queue.append(copied)
        count_name = "_conflict_anchor_count" if conflict else "_anchor_count"
        count = getattr(self, count_name) + copied.shape[0]
        while queue and count > limit:
            oldest = queue[0]
            excess = count - limit
            if oldest.shape[0] <= excess:
                queue.popleft()
                count -= oldest.shape[0]
            else:
                queue[0] = oldest[excess:].copy()
                count -= excess
        setattr(self, count_name, count)

    def observe(
        self,
        *,
        frame_id: int,
        prediction: Mapping[str, Any],
        camera_to_reference: Any,
        model_ready_rgb: Any,
        native_health: Mapping[str, Any] | None = None,
    ) -> EvidenceFrame:
        """Extract one row using this frame and strictly preceding history only."""

        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id != self._next_frame_id:
            raise StateTriageEvidenceError(
                f"frame_id must be the next consecutive ID {self._next_frame_id}, got {frame_id!r}"
            )
        if not isinstance(prediction, Mapping):
            raise StateTriageEvidenceError("prediction must be a mapping")

        raw_self_points = _require_prediction(prediction, "pts3d_in_self_view")
        raw_other_points = _require_prediction(prediction, "pts3d_in_other_view")
        height, width = _pointmap_hw(raw_self_points, "pts3d_in_self_view")
        if _pointmap_hw(raw_other_points, "pts3d_in_other_view") != (height, width):
            raise StateTriageEvidenceError("independent pointmaps must have the same shape")
        camera = _camera_matrix(camera_to_reference)
        rgb = _sample_model_rgb(
            model_ready_rgb,
            rows=self.config.image_quality_rows,
            columns=self.config.image_quality_columns,
        )

        y, x = _grid_indices(height, width, self.config)
        self_samples = _sample_pointmap(raw_self_points, "pts3d_in_self_view", y=y, x=x)
        other_samples = _sample_pointmap(raw_other_points, "pts3d_in_other_view", y=y, x=x)
        conf_samples = _sample_map(_require_prediction(prediction, "conf"), "conf", y=y, x=x, shape=(height, width))
        conf_self_samples = _sample_map(_require_prediction(prediction, "conf_self"), "conf_self", y=y, x=x, shape=(height, width))
        transformed_self = self_samples @ camera[:3, :3].T + camera[:3, 3]
        errors = np.linalg.norm(transformed_self - other_samples, axis=1)
        scales = np.linalg.norm(other_samples, axis=1)
        normalized_errors = errors / np.maximum(scales, self.config.scalar_scale_floor)
        median_residual = float(np.median(normalized_errors))
        mad_residual = float(np.median(np.abs(normalized_errors - median_residual)))
        high_threshold = max(
            median_residual + self.config.residual_mad_multiplier * max(mad_residual, self.config.scalar_scale_floor),
            median_residual * 1.25 + self.config.scalar_scale_floor,
        )
        high_mask = normalized_errors > high_threshold
        residual_high_fraction = float(high_mask.mean())
        coefficient_of_variation = float(normalized_errors.std() / max(median_residual, self.config.scalar_scale_floor))
        residual_globality = float(1.0 / (1.0 + coefficient_of_variation))
        residual_locality = float(1.0 - residual_globality)

        previous_anchors = self._stack(self._anchors)
        previous_conflicts = self._stack(self._conflict_anchors)
        support_radius = max(
            self.config.support_radius_floor,
            self.config.support_radius_relative * float(np.median(scales)),
        )
        support_distance = _nearest_distance(other_samples, previous_anchors)
        coverage_ratio = float((support_distance <= support_radius).mean())
        novel_ratio = float(1.0 - coverage_ratio)
        high_points = other_samples[high_mask]
        if high_points.size == 0:
            temporal_conflict_support = 0.0
        else:
            temporal_conflict_support = float((_nearest_distance(high_points, previous_conflicts) <= support_radius).mean())

        luma = ((rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) + 1.0) * 0.5
        laplacian = -4.0 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
        sharpness = float(np.var(laplacian))
        normalized_rgb = (rgb + 1.0) * 0.5
        low_light_fraction = float((luma <= 0.05).mean())
        clipped_fraction = float(((normalized_rgb <= 0.01) | (normalized_rgb >= 0.99)).any(axis=2).mean())
        saturation_mean = float((normalized_rgb.max(axis=2) - normalized_rgb.min(axis=2)).mean())

        pose_jump = _optional_scalar(native_health, "pose_jump")
        native_geometric_residual = _optional_scalar(native_health, "geometric_residual")
        uncertainty_u = _optional_scalar(native_health, "uncertainty_u")
        if uncertainty_u is not None and not 0.0 <= uncertainty_u <= 1.0:
            raise StateTriageEvidenceError("health.uncertainty_u must be in [0, 1]")
        pose_jump_z = _scalar_z(self._scalar_history["pose_jump"], pose_jump, self.config)
        geometric_z = _scalar_z(
            self._scalar_history["native_geometric_residual"], native_geometric_residual, self.config
        )
        sharpness_z = _scalar_z(self._scalar_history["sharpness"], sharpness, self.config)

        evidence = EvidenceFrame(
            frame_id=frame_id,
            reference_anchor_count_before=self._anchor_count,
            conflict_anchor_count_before=self._conflict_anchor_count,
            sampled_anchor_count=int(other_samples.shape[0]),
            support_radius=float(support_radius),
            coverage_ratio=coverage_ratio,
            novel_ratio=novel_ratio,
            temporal_conflict_support=temporal_conflict_support,
            normalized_residual_median=median_residual,
            normalized_residual_mad=mad_residual,
            residual_high_fraction=residual_high_fraction,
            residual_globality=residual_globality,
            residual_locality=residual_locality,
            pose_jump=pose_jump,
            pose_jump_z=pose_jump_z,
            native_geometric_residual=native_geometric_residual,
            native_geometric_residual_z=geometric_z,
            uncertainty_u=uncertainty_u,
            confidence_mean=float(conf_samples.mean()),
            confidence_std=float(conf_samples.std()),
            confidence_disagreement_mean=float(np.abs(conf_samples - conf_self_samples).mean()),
            sharpness=sharpness,
            sharpness_z=sharpness_z,
            low_light_fraction=low_light_fraction,
            clipped_fraction=clipped_fraction,
            saturation_mean=saturation_mean,
        )

        self._append_bounded(
            self._anchors,
            other_samples,
            limit=self.config.max_anchor_history,
            conflict=False,
        )
        self._append_bounded(
            self._conflict_anchors,
            high_points,
            limit=self.config.max_conflict_anchor_history,
            conflict=True,
        )
        for name, value in (
            ("pose_jump", pose_jump),
            ("native_geometric_residual", native_geometric_residual),
            ("sharpness", sharpness),
        ):
            if value is not None:
                self._scalar_history[name].append(float(value))
        self._next_frame_id += 1
        return evidence


__all__ = [
    "SCHEMA_VERSION",
    "EvidenceConfig",
    "EvidenceFrame",
    "EvidenceObserver",
    "StateTriageEvidenceError",
]
