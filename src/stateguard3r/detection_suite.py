"""Leak-resistant multi-run calibration and formal detection evaluation.

This module deliberately sits above :mod:`stateguard3r.detection`.  The
single-run module remains responsible for causal score construction; this
module never concatenates ledgers before calling ``compute_baseline_scores``.
It adds the pieces that are inherently split-level concerns:

* a fixed 30-position evaluation policy with an event at position 15 and a
  five-position post-event washout;
* pooled, macro, per-run, event-delay, boundary, and washout metrics;
* deterministic development-grid and per-method threshold selection;
* independent deterministic random streams derived from a master seed and
  stable run ID; and
* a strict v1 calibration/holdout binding based on exact-byte SHA-256
  commitments and raw-RGB SHA-set disjointness.

The API accepts immutable byte snapshots instead of paths.  Parsing and
fingerprinting therefore concern the same bytes, while callers retain control
over how files are opened and frozen.  Ignored washout positions are removed
only from metric denominators: all 30 records are scored first, so recurrent
state and causal rolling references are never rewritten by the evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .detection import (
    METHOD_NAMES,
    auroc,
    compute_baseline_scores,
    detection_delay,
    frame_labels_from_corruption,
    threshold_metrics,
)
from .health import HealthFrame


FORMAL_CONFIG_SCHEMA_VERSION = "stateguard3r.detection-suite-calibration.v1"
SEARCH_SCHEMA_VERSION = "stateguard3r.detection-suite-search.v1"
RESULT_SCHEMA_VERSION = "stateguard3r.detection-suite-result.v1"
SPLIT_REGISTRY_SCHEMA_VERSION = "stateguard3r.detection-split-registry.v1"
HOLDOUT_COMMITMENT_SCHEMA_VERSION = (
    "stateguard3r.detection-holdout-commitment.v1"
)
EVALUATION_POLICY_SCHEMA_VERSION = (
    "stateguard3r.detection-evaluation-policy.v1"
)

FRAME_COUNT = 30
EVENT_START = 15
WASHOUT_LENGTH = 5
CORRUPTION_TYPES = frozenset(
    ("low_overlap_jump", "dynamic_occlusion", "wrong_order_segment")
)
FORMAL_EVENT_ENDS = {
    "low_overlap_jump": 19,
    "dynamic_occlusion": 19,
    "wrong_order_segment": 18,
}
FORMAL_RECTANGLE_COORDINATE_REFERENCE = (
    "model_input_after_resize_and_center_crop"
)
PRE_FORWARD_ARTIFACTS = frozenset(
    ("source_manifest", "input_manifest", "corruption_json")
)
FULL_ARTIFACTS = frozenset(
    (
        "source_manifest",
        "input_manifest",
        "corruption_json",
        "health_jsonl",
        "run_json",
    )
)
THRESHOLD_CANDIDATE_RULE = (
    "sorted unique finite masked development scores plus nextafter(max,+inf)"
)
THRESHOLD_SELECTION_RULE = (
    "maximize macro-F1; then minimize pooled FPR; then maximize detected "
    "events; then minimize detected-event mean delay (None=inf); then choose "
    "the higher threshold"
)
HYPERPARAMETER_SELECTION_RULE = (
    "use combined method's selected macro-F1/FPR/detected/delay quality; then "
    "maximize combined macro-AUROC; then choose the earlier candidate"
)
RUN_SEED_DERIVATION = (
    "uint64_be(sha256(str(master_seed)+NUL+dataset_split+NUL+run_id)[:8])"
)
RECAL3R_RUN_SCHEMA_VERSION = "stateguard3r.recal3r-smoke.v0"
EXPECTED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
EXPECTED_CHECKPOINT_SHA256 = (
    "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"
)
EXPECTED_RUNNER_SHA256 = (
    "091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e"
)
FORMAL_SOURCE_SCHEMA_VERSION = "stateguard3r.tum-formal-pilot-source.v1"
FORMAL_INPUT_SCHEMA_VERSION = "stateguard3r.corruption.v1"
FORMAL_RAW_MANIFEST_SCHEMA_VERSION = "stateguard3r.tum-raw-audit.v1"
FORMAL_DATASET = "rgbd_dataset_freiburg1_desk"
FORMAL_DATASET_ROOT = (
    "/data/wangzheng/Project2/baselines/ReCal3R/data/tum/"
    "rgbd_dataset_freiburg1_desk"
)
FORMAL_ARCHIVE_PATH = FORMAL_DATASET_ROOT + ".tgz"
FORMAL_ARCHIVE_SIZE_BYTES = 344_011_403
FORMAL_ARCHIVE_SHA256 = (
    "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c"
)
FORMAL_RAW_MANIFEST_PATH = (
    "/data/wangzheng/Project2/baselines/ReCal3R/logs/"
    "gate2-fr1-desk-raw-manifest.json"
)
FORMAL_RAW_MANIFEST_SIZE_BYTES = 231_139
FORMAL_RAW_MANIFEST_SHA256 = (
    "5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87"
)
FORMAL_INDEX_CONVENTION = (
    "zero_based_data_entry_index_excluding_comments_and_blank_lines"
)
FORMAL_PHYSICAL_LINE_CONVENTION = "one_based_physical_source_line"
MAX_ASSOCIATION_DELTA_SECONDS = 0.02
FORMAL_RUN_SPECS: tuple[dict[str, Any], ...] = (
    {
        "run_id": "development-dynamic",
        "dataset_split": "development",
        "corruption_type": "dynamic_occlusion",
        "base_indices": tuple(range(248, 278)),
        "donor_indices": (),
    },
    {
        "run_id": "development-wrong",
        "dataset_split": "development",
        "corruption_type": "wrong_order_segment",
        "base_indices": tuple(range(320, 350)),
        "donor_indices": (),
    },
    {
        "run_id": "development-low",
        "dataset_split": "development",
        "corruption_type": "low_overlap_jump",
        "base_indices": tuple(range(380, 410)),
        "donor_indices": tuple(range(300, 305)),
    },
    {
        "run_id": "holdout-dynamic",
        "dataset_split": "holdout",
        "corruption_type": "dynamic_occlusion",
        "base_indices": tuple(range(520, 550)),
        "donor_indices": (),
    },
    {
        "run_id": "holdout-wrong",
        "dataset_split": "holdout",
        "corruption_type": "wrong_order_segment",
        "base_indices": tuple(range(550, 580)),
        "donor_indices": (),
    },
    {
        "run_id": "holdout-low",
        "dataset_split": "holdout",
        "corruption_type": "low_overlap_jump",
        "base_indices": tuple(range(583, 613)),
        "donor_indices": tuple(range(500, 505)),
    },
)
FORMAL_RUN_SPEC_BY_ID = {item["run_id"]: item for item in FORMAL_RUN_SPECS}
FORMAL_RUN_IDS_BY_SPLIT = {
    split: tuple(
        item["run_id"] for item in FORMAL_RUN_SPECS if item["dataset_split"] == split
    )
    for split in ("development", "holdout")
}
EXPECTED_HEALTH_SIGNAL_SEMANTICS = {
    "reliability": "1 - ReCal3R uncertainty_u",
    "update_magnitude_source": "global_state_delta",
    "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)",
    "geometric_residual": (
        "relative_median_independent_cross_head_pointmap_consistency"
    ),
    "geometric_residual_unavailable_reason": None,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def fixed_evaluation_policy() -> dict[str, Any]:
    """Return a fresh copy of the only policy accepted by the formal v1 gate."""

    return {
        "schema_version": EVALUATION_POLICY_SCHEMA_VERSION,
        "frame_count": FRAME_COUNT,
        "event_start": EVENT_START,
        "washout_length": WASHOUT_LENGTH,
        "primary_positive": "manifest_start_end_inclusive_stream_positions",
        "washout": "event_end_plus_1_through_event_end_plus_5_inclusive",
        "startup": "positions_before_event_are_evaluated_negatives",
        "mask_application": "metrics_only_after_full_run_score_computation",
        "fpr_denominator": "finite_evaluated_primary_negative_frames",
        "delay": "first_crossing_inside_primary_interval_from_event_start",
    }


def fixed_candidate_order() -> list[dict[str, Any]]:
    """Return the pre-registered formal-v1 grid in its exact search order."""

    return [
        {"window": window, "epsilon": epsilon, "max_z": max_z}
        for window in (5, 10, 15)
        for epsilon in (1e-6, 1e-5, 1e-4)
        for max_z in (10.0, 20.0, None)
    ]


@dataclass(frozen=True)
class DetectionSuiteRun:
    """One immutable single-corruption run supplied to the suite.

    ``raw_frame_sha256s`` is the unique set of all RGB files actually consumed,
    including low-overlap donor frames.  The input manifest and run JSON are
    independently checked to contain the same set.
    """

    run_id: str
    dataset_split: str
    corruption_type: str
    raw_frame_sha256s: tuple[str, ...]
    health_jsonl: bytes
    corruption_json: bytes
    input_manifest_json: bytes
    source_manifest_json: bytes
    run_json: bytes


@dataclass(frozen=True)
class DetectionSuiteInput:
    """Pre-forward snapshots sufficient to freeze a run commitment.

    Health and run outputs intentionally do not exist at this stage.  Their
    provenance is validated later from :class:`DetectionSuiteRun` snapshots.
    """

    run_id: str
    dataset_split: str
    corruption_type: str
    raw_frame_sha256s: tuple[str, ...]
    corruption_json: bytes
    input_manifest_json: bytes
    source_manifest_json: bytes


@dataclass(frozen=True)
class EvaluationRegions:
    """Primary labels and mutually explicit evaluation regions for one run."""

    primary_labels: NDArray[np.int64]
    evaluation_mask: NDArray[np.bool_]
    recovery_boundary_mask: NDArray[np.bool_]
    washout_mask: NDArray[np.bool_]
    event_start: int
    event_end: int

    @property
    def positive_frames(self) -> int:
        return int(self.primary_labels.sum())

    @property
    def washout_frames(self) -> int:
        return int(self.washout_mask.sum())

    @property
    def negative_frames(self) -> int:
        return int(np.sum(self.evaluation_mask & (self.primary_labels == 0)))


@dataclass(frozen=True)
class _PreparedRun:
    source: DetectionSuiteRun
    records: tuple[dict[str, Any], ...]
    corruption: Mapping[str, Any]
    regions: EvaluationRegions
    artifacts: dict[str, dict[str, Any]]
    raw_frame_sha256s: tuple[str, ...]
    source_pool_frame_sha256s: tuple[str, ...]
    source_pool_identities: tuple[dict[str, Any], ...]
    consumed_frames: tuple[dict[str, Any], ...]
    source_index_files: dict[str, dict[str, Any]]
    official_lineage: dict[str, Any] | None
    execution_provenance: dict[str, Any]


@dataclass(frozen=True)
class _PreparedInput:
    corruption: Mapping[str, Any]
    regions: EvaluationRegions
    artifacts: dict[str, dict[str, Any]]
    raw_frame_sha256s: tuple[str, ...]
    source_pool_frame_sha256s: tuple[str, ...]
    source_pool_identities: tuple[dict[str, Any], ...]
    consumed_frames: tuple[dict[str, Any], ...]
    source_index_files: dict[str, dict[str, Any]]
    official_lineage: dict[str, Any] | None


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], *, name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        problems: list[str] = []
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if unknown:
            problems.append("unknown: " + ", ".join(unknown))
        raise ValueError(f"{name} has invalid keys; " + "; ".join(problems))


def _strict_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_number(
    value: Any, *, name: str, positive: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_json_loads(raw: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} is not allowed")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict JSON: {error}") from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _artifact(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise ValueError("artifact snapshots must be bytes")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _validated_artifact(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(value, {"sha256", "size_bytes"}, name=name)
    return {
        "sha256": _validated_sha256(value["sha256"], name=f"{name}.sha256"),
        "size_bytes": _strict_integer(
            value["size_bytes"], name=f"{name}.size_bytes", minimum=1
        ),
    }


def _validated_raw_sha_set(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty JSON array")
    items = tuple(
        _validated_sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not items:
        raise ValueError(f"{name} must be non-empty")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must contain unique SHA-256 values")
    if len(items) != FRAME_COUNT:
        raise ValueError(f"{name} must contain exactly {FRAME_COUNT} consumed RGB hashes")
    return tuple(sorted(items))


def _validated_source_pool_sha_sequence(
    value: Any, *, name: str, corruption_type: str
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    items = tuple(
        _validated_sha256(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    expected_count = 35 if corruption_type == "low_overlap_jump" else FRAME_COUNT
    if len(items) != expected_count:
        raise ValueError(
            f"{name} must contain exactly {expected_count} ordered source-pool hashes"
        )
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must contain unique SHA-256 values")
    return items


def _validated_path(value: Any, *, name: str, absolute: bool = False) -> str:
    path = _nonempty_string(value, name=name)
    if "\0" in path:
        raise ValueError(f"{name} may not contain a NUL byte")
    if absolute and not os.path.isabs(path):
        raise ValueError(f"{name} must be an absolute path")
    return path


def _formal_spec_for_split_type(
    dataset_split: str, corruption_type: str
) -> Mapping[str, Any]:
    matches = [
        spec
        for spec in FORMAL_RUN_SPECS
        if spec["dataset_split"] == dataset_split
        and spec["corruption_type"] == corruption_type
    ]
    if len(matches) != 1:  # guarded by constants and earlier enum checks
        raise ValueError("formal split/corruption mapping is not uniquely defined")
    return matches[0]


def _require_exact_formal_run_identity(
    run_id: str, dataset_split: str, corruption_type: str
) -> Mapping[str, Any]:
    spec = FORMAL_RUN_SPEC_BY_ID.get(run_id)
    if spec is None:
        expected = ", ".join(FORMAL_RUN_IDS_BY_SPLIT.get(dataset_split, ()))
        raise ValueError(
            f"formal {dataset_split} run_id {run_id!r} is not frozen; expected one of: "
            f"{expected}"
        )
    if (
        spec["dataset_split"] != dataset_split
        or spec["corruption_type"] != corruption_type
    ):
        raise ValueError(
            f"formal run_id {run_id!r} split/corruption assignment is not frozen v1"
        )
    return spec


def _validated_snapshot_artifact(
    value: Any, *, name: str, include_entry_count: bool
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    keys = {
        "path",
        "sha256",
        "size_bytes",
        "mode_octal",
        "link_count",
        "device",
        "inode",
        "mtime_ns",
    }
    if include_entry_count:
        keys.add("data_entry_count")
    _expect_exact_keys(value, keys, name=name)
    mode = value["mode_octal"]
    if not isinstance(mode, str) or re.fullmatch(r"[0-7]{4}", mode) is None:
        raise ValueError(f"{name}.mode_octal must contain four octal digits")
    result = {
        "path": _validated_path(value["path"], name=f"{name}.path"),
        "sha256": _validated_sha256(value["sha256"], name=f"{name}.sha256"),
        "size_bytes": _strict_integer(
            value["size_bytes"], name=f"{name}.size_bytes", minimum=1
        ),
        "mode_octal": mode,
        "link_count": _strict_integer(
            value["link_count"], name=f"{name}.link_count", minimum=1
        ),
        "device": _strict_integer(
            value["device"], name=f"{name}.device", minimum=0
        ),
        "inode": _strict_integer(value["inode"], name=f"{name}.inode", minimum=1),
        "mtime_ns": _strict_integer(
            value["mtime_ns"], name=f"{name}.mtime_ns", minimum=0
        ),
    }
    if include_entry_count:
        result["data_entry_count"] = _strict_integer(
            value["data_entry_count"],
            name=f"{name}.data_entry_count",
            minimum=1,
        )
    return result


def _validated_official_lineage(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(
        value,
        {"dataset_root", "archive", "raw_manifest", "raw_manifest_schema_version"},
        name=name,
    )
    if value["raw_manifest_schema_version"] != FORMAL_RAW_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{name}.raw_manifest_schema_version mismatch")
    dataset_root = _validated_path(
        value["dataset_root"], name=f"{name}.dataset_root", absolute=True
    )
    if dataset_root != FORMAL_DATASET_ROOT:
        raise ValueError(f"{name}.dataset_root is not the frozen TUM root")
    archive = _validated_snapshot_artifact(
            value["archive"], name=f"{name}.archive", include_entry_count=False
        )
    raw_manifest = _validated_snapshot_artifact(
            value["raw_manifest"],
            name=f"{name}.raw_manifest",
            include_entry_count=False,
        )
    expected = (
        (archive, FORMAL_ARCHIVE_PATH, FORMAL_ARCHIVE_SHA256, FORMAL_ARCHIVE_SIZE_BYTES),
        (
            raw_manifest,
            FORMAL_RAW_MANIFEST_PATH,
            FORMAL_RAW_MANIFEST_SHA256,
            FORMAL_RAW_MANIFEST_SIZE_BYTES,
        ),
    )
    for artifact, expected_path, expected_sha, expected_size in expected:
        if (
            artifact["path"] != expected_path
            or artifact["sha256"] != expected_sha
            or artifact["size_bytes"] != expected_size
        ):
            raise ValueError(f"{name} official artifact identity mismatch")
    return {
        "dataset_root": dataset_root,
        "archive": archive,
        "raw_manifest": raw_manifest,
        "raw_manifest_schema_version": FORMAL_RAW_MANIFEST_SCHEMA_VERSION,
    }


def _validated_line_provenance(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    timestamp = _finite_number(value["timestamp"], name=f"{name}.timestamp")
    timestamp_text = _nonempty_string(
        value["timestamp_text"], name=f"{name}.timestamp_text"
    )
    if re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        timestamp_text,
    ) is None:
        raise ValueError(f"{name}.timestamp_text must be a decimal number")
    try:
        timestamp_decimal = Decimal(timestamp_text)
    except InvalidOperation as error:
        raise ValueError(f"{name}.timestamp_text must be a decimal number") from error
    if not timestamp_decimal.is_finite():
        raise ValueError(f"{name}.timestamp_text must be finite")
    try:
        timestamp_from_text = float(timestamp_decimal)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name}.timestamp_text is outside the finite JSON number range"
        ) from error
    if not math.isfinite(timestamp_from_text):
        raise ValueError(
            f"{name}.timestamp_text is outside the finite JSON number range"
        )
    if timestamp != timestamp_from_text:
        raise ValueError(f"{name}.timestamp and timestamp_text disagree")
    return {
        "source_entry_index": _strict_integer(
            value["source_entry_index"],
            name=f"{name}.source_entry_index",
            minimum=0,
        ),
        "source_line": _strict_integer(
            value["source_line"], name=f"{name}.source_line", minimum=1
        ),
        "physical_line_sha256": _validated_sha256(
            value["physical_line_sha256"], name=f"{name}.physical_line_sha256"
        ),
        "physical_line_size_bytes": _strict_integer(
            value["physical_line_size_bytes"],
            name=f"{name}.physical_line_size_bytes",
            minimum=1,
        ),
        "timestamp": timestamp,
        "timestamp_text": timestamp_text,
    }


def _validated_association_delta(
    value: Mapping[str, Any],
    *,
    name: str,
    rgb_timestamp_text: str,
    associated_timestamp_text: str,
) -> tuple[float, float]:
    signed = _finite_number(
        value["delta_from_rgb_seconds"], name=f"{name}.delta_from_rgb_seconds"
    )
    absolute = _finite_number(
        value["absolute_delta_seconds"], name=f"{name}.absolute_delta_seconds"
    )
    signed_decimal = Decimal(str(signed))
    absolute_decimal = Decimal(str(absolute))
    expected_decimal = Decimal(associated_timestamp_text) - Decimal(
        rgb_timestamp_text
    )
    maximum_decimal = Decimal(str(MAX_ASSOCIATION_DELTA_SECONDS))
    if absolute_decimal < 0 or absolute_decimal > maximum_decimal:
        raise ValueError(
            f"{name}.absolute_delta_seconds must be in [0, "
            f"{MAX_ASSOCIATION_DELTA_SECONDS}]"
        )
    if signed_decimal != expected_decimal:
        raise ValueError(f"{name} timestamp and declared association delta disagree")
    if absolute_decimal != abs(expected_decimal):
        raise ValueError(
            f"{name} timestamp and declared absolute association delta disagree"
        )
    return signed, absolute


def _validated_identity_components_unique(
    identities: Sequence[Mapping[str, Any]], *, name: str
) -> None:
    fields = {
        "RGB raw index": [item["rgb"]["raw_source_index"] for item in identities],
        "RGB path": [item["rgb"]["path"] for item in identities],
        "RGB device/inode": [
            (item["rgb"]["device"], item["rgb"]["inode"]) for item in identities
        ],
        "RGB SHA": [item["rgb"]["sha256"] for item in identities],
        "depth index": [item["depth"]["source_entry_index"] for item in identities],
        "depth path": [item["depth"]["path"] for item in identities],
        "depth device/inode": [
            (item["depth"]["device"], item["depth"]["inode"])
            for item in identities
        ],
        "depth SHA": [item["depth"]["sha256"] for item in identities],
        "ground-truth index": [
            item["groundtruth"]["source_entry_index"] for item in identities
        ],
        "ground-truth line": [item["groundtruth"]["source_line"] for item in identities],
        "ground-truth line SHA": [
            item["groundtruth"]["physical_line_sha256"] for item in identities
        ],
    }
    for label, values in fields.items():
        if len(set(values)) != len(values):
            raise ValueError(f"{name} contains duplicate {label} identity")


def _validated_source_pool_identities(
    value: Any,
    *,
    name: str,
    dataset_split: str,
    corruption_type: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    spec = _formal_spec_for_split_type(dataset_split, corruption_type)
    expected_indices = tuple(spec["base_indices"]) + tuple(spec["donor_indices"])
    if len(value) != len(expected_indices):
        raise ValueError(
            f"{name} must contain exactly {len(expected_indices)} source identities"
        )
    result: list[dict[str, Any]] = []
    for position, raw in enumerate(value):
        item_name = f"{name}[{position}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{item_name} must be a JSON object")
        _expect_exact_keys(
            raw,
            {"source_pool_index", "source_pool_role", "rgb", "depth", "groundtruth"},
            name=item_name,
        )
        if raw["source_pool_index"] != position:
            raise ValueError(f"{item_name}.source_pool_index is out of order")
        expected_role = "base" if position < FRAME_COUNT else "low_overlap_donor"
        if raw["source_pool_role"] != expected_role:
            raise ValueError(f"{item_name}.source_pool_role mismatch")

        rgb = raw["rgb"]
        if not isinstance(rgb, Mapping):
            raise ValueError(f"{item_name}.rgb must be a JSON object")
        _expect_exact_keys(
            rgb,
            {
                "raw_source_index",
                "source_line",
                "physical_line_sha256",
                "timestamp",
                "path",
                "device",
                "inode",
                "sha256",
                "size_bytes",
            },
            name=f"{item_name}.rgb",
        )
        rgb_projection = {
            "raw_source_index": _strict_integer(
                rgb["raw_source_index"],
                name=f"{item_name}.rgb.raw_source_index",
                minimum=0,
            ),
            "source_line": _strict_integer(
                rgb["source_line"], name=f"{item_name}.rgb.source_line", minimum=1
            ),
            "physical_line_sha256": _validated_sha256(
                rgb["physical_line_sha256"],
                name=f"{item_name}.rgb.physical_line_sha256",
            ),
            "timestamp": _finite_number(
                rgb["timestamp"], name=f"{item_name}.rgb.timestamp"
            ),
            "path": _validated_path(rgb["path"], name=f"{item_name}.rgb.path"),
            "device": _strict_integer(
                rgb["device"], name=f"{item_name}.rgb.device", minimum=0
            ),
            "inode": _strict_integer(
                rgb["inode"], name=f"{item_name}.rgb.inode", minimum=1
            ),
            "sha256": _validated_sha256(
                rgb["sha256"], name=f"{item_name}.rgb.sha256"
            ),
            "size_bytes": _strict_integer(
                rgb["size_bytes"], name=f"{item_name}.rgb.size_bytes", minimum=1
            ),
        }
        if rgb_projection["raw_source_index"] != expected_indices[position]:
            raise ValueError(
                f"{item_name}.rgb.raw_source_index does not match frozen allocation"
            )

        depth = raw["depth"]
        if not isinstance(depth, Mapping):
            raise ValueError(f"{item_name}.depth must be a JSON object")
        _expect_exact_keys(
            depth,
            {
                "source_entry_index",
                "source_line",
                "physical_line_sha256",
                "timestamp",
                "path",
                "device",
                "inode",
                "sha256",
                "size_bytes",
                "absolute_delta_seconds",
            },
            name=f"{item_name}.depth",
        )
        depth_projection = {
            "source_entry_index": _strict_integer(
                depth["source_entry_index"],
                name=f"{item_name}.depth.source_entry_index",
                minimum=0,
            ),
            "source_line": _strict_integer(
                depth["source_line"], name=f"{item_name}.depth.source_line", minimum=1
            ),
            "physical_line_sha256": _validated_sha256(
                depth["physical_line_sha256"],
                name=f"{item_name}.depth.physical_line_sha256",
            ),
            "timestamp": _finite_number(
                depth["timestamp"], name=f"{item_name}.depth.timestamp"
            ),
            "path": _validated_path(depth["path"], name=f"{item_name}.depth.path"),
            "device": _strict_integer(
                depth["device"], name=f"{item_name}.depth.device", minimum=0
            ),
            "inode": _strict_integer(
                depth["inode"], name=f"{item_name}.depth.inode", minimum=1
            ),
            "sha256": _validated_sha256(
                depth["sha256"], name=f"{item_name}.depth.sha256"
            ),
            "size_bytes": _strict_integer(
                depth["size_bytes"],
                name=f"{item_name}.depth.size_bytes",
                minimum=1,
            ),
            "absolute_delta_seconds": _finite_number(
                depth["absolute_delta_seconds"],
                name=f"{item_name}.depth.absolute_delta_seconds",
            ),
        }
        if not 0 <= depth_projection["absolute_delta_seconds"] <= MAX_ASSOCIATION_DELTA_SECONDS:
            raise ValueError(f"{item_name}.depth association exceeds 20 ms")

        groundtruth = raw["groundtruth"]
        if not isinstance(groundtruth, Mapping):
            raise ValueError(f"{item_name}.groundtruth must be a JSON object")
        _expect_exact_keys(
            groundtruth,
            {
                "source_entry_index",
                "source_line",
                "physical_line_sha256",
                "timestamp",
                "absolute_delta_seconds",
            },
            name=f"{item_name}.groundtruth",
        )
        groundtruth_projection = {
            "source_entry_index": _strict_integer(
                groundtruth["source_entry_index"],
                name=f"{item_name}.groundtruth.source_entry_index",
                minimum=0,
            ),
            "source_line": _strict_integer(
                groundtruth["source_line"],
                name=f"{item_name}.groundtruth.source_line",
                minimum=1,
            ),
            "physical_line_sha256": _validated_sha256(
                groundtruth["physical_line_sha256"],
                name=f"{item_name}.groundtruth.physical_line_sha256",
            ),
            "timestamp": _finite_number(
                groundtruth["timestamp"], name=f"{item_name}.groundtruth.timestamp"
            ),
            "absolute_delta_seconds": _finite_number(
                groundtruth["absolute_delta_seconds"],
                name=f"{item_name}.groundtruth.absolute_delta_seconds",
            ),
        }
        if not 0 <= groundtruth_projection["absolute_delta_seconds"] <= MAX_ASSOCIATION_DELTA_SECONDS:
            raise ValueError(f"{item_name}.groundtruth association exceeds 20 ms")
        result.append(
            {
                "source_pool_index": position,
                "source_pool_role": expected_role,
                "rgb": rgb_projection,
                "depth": depth_projection,
                "groundtruth": groundtruth_projection,
            }
        )
    _validated_identity_components_unique(result, name=name)
    return tuple(result)


def _validated_consumed_frames(
    value: Any,
    *,
    name: str,
    source_pool_identities: Sequence[Mapping[str, Any]],
    corruption_type: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array")
    if len(value) != FRAME_COUNT:
        raise ValueError(f"{name} must contain exactly {FRAME_COUNT} frames")
    if corruption_type == "low_overlap_jump":
        expected_indices = [*range(15), *range(30, 35), *range(20, 30)]
    elif corruption_type == "wrong_order_segment":
        expected_indices = [*range(15), 18, 17, 16, 15, *range(19, 30)]
    else:
        expected_indices = list(range(FRAME_COUNT))
    result: list[dict[str, Any]] = []
    for position, raw in enumerate(value):
        item_name = f"{name}[{position}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{item_name} must be a JSON object")
        _expect_exact_keys(
            raw,
            {
                "output_position",
                "source_pool_index",
                "path",
                "rgb_sha256",
                "timestamp",
                "source_metadata_canonical_sha256",
            },
            name=item_name,
        )
        source_index = _strict_integer(
            raw["source_pool_index"], name=f"{item_name}.source_pool_index", minimum=0
        )
        if raw["output_position"] != position or source_index != expected_indices[position]:
            raise ValueError(f"{item_name} position/source mapping is not frozen v1")
        identity = source_pool_identities[source_index]["rgb"]
        row = {
            "output_position": position,
            "source_pool_index": source_index,
            "path": _validated_path(raw["path"], name=f"{item_name}.path"),
            "rgb_sha256": _validated_sha256(
                raw["rgb_sha256"], name=f"{item_name}.rgb_sha256"
            ),
            "timestamp": _finite_number(
                raw["timestamp"], name=f"{item_name}.timestamp"
            ),
            "source_metadata_canonical_sha256": _validated_sha256(
                raw["source_metadata_canonical_sha256"],
                name=f"{item_name}.source_metadata_canonical_sha256",
            ),
        }
        if (
            row["path"] != identity["path"]
            or row["rgb_sha256"] != identity["sha256"]
            or row["timestamp"] != identity["timestamp"]
        ):
            raise ValueError(f"{item_name} disagrees with its source-pool identity")
        result.append(row)
    if len({row["rgb_sha256"] for row in result}) != FRAME_COUNT:
        raise ValueError(f"{name} must consume 30 unique RGB identities")
    return tuple(result)


def _validated_formal_source_manifest(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    dataset_split: str,
    corruption_type: str,
) -> dict[str, Any]:
    base_keys = {
        "schema_version",
        "sequence",
        "dataset",
        "dataset_split",
        "run_id",
        "corruption_type",
        "source_is_read_only",
        "file_path_semantics",
        "index_convention",
        "physical_line_convention",
        "association_policy",
        "frame_count",
        "output_frame_count",
        "ground_truth_path",
        "tum_index_files",
        "frames",
    }
    _expect_exact_keys(
        manifest, base_keys | {"official_lineage"}, name="source_manifest_json"
    )
    fixed = {
        "schema_version": FORMAL_SOURCE_SCHEMA_VERSION,
        "sequence": f"{run_id}-source-pool",
        "dataset": FORMAL_DATASET,
        "dataset_split": dataset_split,
        "run_id": run_id,
        "corruption_type": corruption_type,
        "source_is_read_only": True,
        "file_path_semantics": "relative_to_this_manifest_directory",
        "index_convention": FORMAL_INDEX_CONVENTION,
        "physical_line_convention": FORMAL_PHYSICAL_LINE_CONVENTION,
        "output_frame_count": FRAME_COUNT,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise ValueError(f"run {run_id}: source manifest {field} mismatch")
    spec = _formal_spec_for_split_type(dataset_split, corruption_type)
    expected_count = len(spec["base_indices"]) + len(spec["donor_indices"])
    if manifest.get("frame_count") != expected_count:
        raise ValueError(
            f"run {run_id}: source manifest frame_count must equal {expected_count}"
        )
    policy = manifest.get("association_policy")
    expected_policy = {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "reuse_policy": "reject_within_source_pool_and_across_consumed_runs",
        "max_absolute_delta_seconds": MAX_ASSOCIATION_DELTA_SECONDS,
    }
    if policy != expected_policy:
        raise ValueError(f"run {run_id}: source association policy mismatch")
    _validated_path(
        manifest.get("ground_truth_path"), name="source_manifest.ground_truth_path"
    )

    raw_index_files = manifest.get("tum_index_files")
    if not isinstance(raw_index_files, Mapping):
        raise ValueError("source_manifest.tum_index_files must be a JSON object")
    _expect_exact_keys(
        raw_index_files, {"rgb", "depth", "groundtruth"}, name="tum_index_files"
    )
    source_index_files = {
        kind: _validated_snapshot_artifact(
            raw_index_files[kind],
            name=f"tum_index_files.{kind}",
            include_entry_count=True,
        )
        for kind in ("rgb", "depth", "groundtruth")
    }
    official_lineage = _validated_official_lineage(
        manifest.get("official_lineage"), name="source_manifest.official_lineage"
    )

    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != expected_count:
        raise ValueError(f"run {run_id}: source manifest frames do not match frame_count")
    common_line_keys = {
        "source_entry_index",
        "source_line",
        "physical_line_sha256",
        "physical_line_size_bytes",
        "timestamp",
        "timestamp_text",
    }
    source_projection: list[dict[str, Any]] = []
    for position, frame in enumerate(raw_frames):
        name = f"source manifest frame {position}"
        if not isinstance(frame, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        _expect_exact_keys(
            frame,
            common_line_keys
            | {
                "path",
                "raw_rgb_source_index",
                "raw_rgb_source_line",
                "rgb_sha256",
                "rgb_size_bytes",
                "rgb_mode_octal",
                "rgb_link_count",
                "rgb_device",
                "rgb_inode",
                "rgb_mtime_ns",
                "depth",
                "groundtruth",
                "source_pool_index",
                "source_pool_role",
            },
            name=name,
        )
        line = _validated_line_provenance(frame, name=f"{name}.rgb_line")
        raw_index = _strict_integer(
            frame["raw_rgb_source_index"],
            name=f"{name}.raw_rgb_source_index",
            minimum=0,
        )
        raw_line = _strict_integer(
            frame["raw_rgb_source_line"],
            name=f"{name}.raw_rgb_source_line",
            minimum=1,
        )
        if raw_index != line["source_entry_index"] or raw_line != line["source_line"]:
            raise ValueError(f"{name} duplicate RGB raw-index/line fields disagree")
        mode = frame["rgb_mode_octal"]
        if not isinstance(mode, str) or re.fullmatch(r"[0-7]{4}", mode) is None:
            raise ValueError(f"{name}.rgb_mode_octal must contain four octal digits")

        depth = frame["depth"]
        if not isinstance(depth, Mapping):
            raise ValueError(f"{name}.depth must be a JSON object")
        depth_keys = common_line_keys | {
            "path",
            "sha256",
            "size_bytes",
            "mode_octal",
            "link_count",
            "device",
            "inode",
            "mtime_ns",
            "delta_from_rgb_seconds",
            "absolute_delta_seconds",
        }
        _expect_exact_keys(depth, depth_keys, name=f"{name}.depth")
        depth_line = _validated_line_provenance(depth, name=f"{name}.depth")
        _validated_association_delta(
            depth,
            name=f"{name}.depth",
            rgb_timestamp_text=line["timestamp_text"],
            associated_timestamp_text=depth_line["timestamp_text"],
        )
        depth_snapshot = _validated_snapshot_artifact(
            {
                key: depth[key]
                for key in (
                    "path",
                    "sha256",
                    "size_bytes",
                    "mode_octal",
                    "link_count",
                    "device",
                    "inode",
                    "mtime_ns",
                )
            },
            name=f"{name}.depth_snapshot",
            include_entry_count=False,
        )

        groundtruth = frame["groundtruth"]
        if not isinstance(groundtruth, Mapping):
            raise ValueError(f"{name}.groundtruth must be a JSON object")
        groundtruth_keys = common_line_keys | {
            "translation_xyz",
            "quaternion_xyzw",
            "translation_xyz_text",
            "quaternion_xyzw_text",
            "delta_from_rgb_seconds",
            "absolute_delta_seconds",
        }
        _expect_exact_keys(groundtruth, groundtruth_keys, name=f"{name}.groundtruth")
        gt_line = _validated_line_provenance(
            groundtruth, name=f"{name}.groundtruth"
        )
        _validated_association_delta(
            groundtruth,
            name=f"{name}.groundtruth",
            rgb_timestamp_text=line["timestamp_text"],
            associated_timestamp_text=gt_line["timestamp_text"],
        )
        for field, count in (("translation_xyz", 3), ("quaternion_xyzw", 4)):
            values = groundtruth[field]
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(f"{name}.groundtruth.{field} has invalid shape")
            for index, value in enumerate(values):
                _finite_number(
                    value, name=f"{name}.groundtruth.{field}[{index}]"
                )
        for field, count in (
            ("translation_xyz_text", 3),
            ("quaternion_xyzw_text", 4),
        ):
            values = groundtruth[field]
            if not isinstance(values, list) or len(values) != count or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name}.groundtruth.{field} has invalid shape")

        source_projection.append(
            {
                "source_pool_index": frame["source_pool_index"],
                "source_pool_role": frame["source_pool_role"],
                "rgb": {
                    "raw_source_index": raw_index,
                    "source_line": line["source_line"],
                    "physical_line_sha256": line["physical_line_sha256"],
                    "timestamp": line["timestamp"],
                    "path": frame["path"],
                    "device": frame["rgb_device"],
                    "inode": frame["rgb_inode"],
                    "sha256": frame["rgb_sha256"],
                    "size_bytes": frame["rgb_size_bytes"],
                },
                "depth": {
                    "source_entry_index": depth_line["source_entry_index"],
                    "source_line": depth_line["source_line"],
                    "physical_line_sha256": depth_line["physical_line_sha256"],
                    "timestamp": depth_line["timestamp"],
                    "path": depth_snapshot["path"],
                    "device": depth_snapshot["device"],
                    "inode": depth_snapshot["inode"],
                    "sha256": depth_snapshot["sha256"],
                    "size_bytes": depth_snapshot["size_bytes"],
                    "absolute_delta_seconds": depth["absolute_delta_seconds"],
                },
                "groundtruth": {
                    "source_entry_index": gt_line["source_entry_index"],
                    "source_line": gt_line["source_line"],
                    "physical_line_sha256": gt_line["physical_line_sha256"],
                    "timestamp": gt_line["timestamp"],
                    "absolute_delta_seconds": groundtruth[
                        "absolute_delta_seconds"
                    ],
                },
            }
        )
    identities = _validated_source_pool_identities(
        source_projection,
        name=f"run {run_id} source_pool_identities",
        dataset_split=dataset_split,
        corruption_type=corruption_type,
    )
    return {
        "frames": raw_frames,
        "source_pool_identities": identities,
        "source_pool_frame_sha256s": tuple(
            item["rgb"]["sha256"] for item in identities
        ),
        "source_index_files": source_index_files,
        "official_lineage": official_lineage,
    }


def derive_run_seed(master_seed: int, dataset_split: str, run_id: str) -> int:
    """Derive a stable independent seed from master seed, split, and run ID."""

    seed = _strict_integer(master_seed, name="master_seed", minimum=0)
    if dataset_split not in ("development", "holdout"):
        raise ValueError("dataset_split must be development or holdout")
    stable_id = _nonempty_string(run_id, name="run_id")
    digest = hashlib.sha256(
        f"{seed}\0{dataset_split}\0{stable_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _enabled_primary_interval(corruption: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = corruption.get("corruptions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("corruption JSON must contain a corruptions array")
    enabled = [item for item in raw if isinstance(item, Mapping) and item.get("enabled", True) is not False]
    if len(enabled) != 1:
        raise ValueError("formal suite requires exactly one enabled corruption interval")
    if any(not isinstance(item, Mapping) for item in raw):
        raise ValueError("every corruption interval must be a JSON object")
    return enabled[0]


def build_evaluation_regions(
    corruption: Mapping[str, Any],
    *,
    frame_count: int = FRAME_COUNT,
) -> EvaluationRegions:
    """Build the fixed primary/washout mask without changing score history."""

    if frame_count != FRAME_COUNT:
        raise ValueError(f"formal evaluation requires frame_count={FRAME_COUNT}")
    interval = _enabled_primary_interval(corruption)
    corruption_type = _nonempty_string(
        interval.get("type"), name="corruption interval type"
    )
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"unsupported formal corruption type {corruption_type!r}")
    if "start" not in interval or "end" not in interval:
        raise ValueError("formal corruption interval must use start/end positions")
    start = _strict_integer(interval["start"], name="corruption start")
    end = _strict_integer(interval["end"], name="corruption end")
    if start != EVENT_START:
        raise ValueError(f"formal corruption start must equal {EVENT_START}")
    expected_end = FORMAL_EVENT_ENDS[corruption_type]
    if end != expected_end:
        raise ValueError(
            f"formal {corruption_type} end must equal {expected_end}"
        )
    if end + WASHOUT_LENGTH >= frame_count:
        raise ValueError(
            "formal 30-frame run must retain all five post-event washout positions"
        )

    frame_ids = list(range(frame_count))
    labels = frame_labels_from_corruption(frame_ids, corruption)
    expected = np.zeros(frame_count, dtype=np.int64)
    expected[start : end + 1] = 1
    if not np.array_equal(labels, expected):
        raise ValueError("primary labels must equal the sole manifest start/end interval")

    washout = np.zeros(frame_count, dtype=np.bool_)
    washout[end + 1 : end + WASHOUT_LENGTH + 1] = True
    boundary = np.zeros(frame_count, dtype=np.bool_)
    boundary[end + 1] = True
    evaluation = ~washout
    if np.any((labels == 1) & ~evaluation):
        raise ValueError("evaluation policy may not mask a primary-positive frame")
    return EvaluationRegions(
        primary_labels=labels,
        evaluation_mask=evaluation,
        recovery_boundary_mask=boundary,
        washout_mask=washout,
        event_start=start,
        event_end=end,
    )


def _parse_health_jsonl(raw: bytes) -> tuple[dict[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"health_jsonl is not UTF-8: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(
            line.encode("utf-8"), name=f"health_jsonl:{line_number}"
        )
        if not isinstance(value, dict):
            raise ValueError(f"health_jsonl:{line_number} must be a JSON object")
        try:
            records.append(HealthFrame.from_dict(value).to_dict())
        except ValueError as error:
            raise ValueError(f"health_jsonl:{line_number}: {error}") from error
    if len(records) != FRAME_COUNT:
        raise ValueError(f"formal health ledger must contain exactly {FRAME_COUNT} records")
    frame_ids = [record.get("frame_id") for record in records]
    if frame_ids != list(range(FRAME_COUNT)):
        raise ValueError("formal health frame_ids must equal output positions 0..29")
    for position, record in enumerate(records):
        if record["timestamp"] is None:
            raise ValueError(
                f"formal health position {position} requires a finite input timestamp"
            )
        if record["update_magnitude"] is not None:
            raise ValueError(
                "formal health update_magnitude must remain missing because the frozen "
                "runner records global_state_delta"
            )
        if position == 0:
            if (
                record["global_state_delta"] is not None
                or record["uncertainty_u"] is not None
                or record["reliability"] is not None
            ):
                raise ValueError(
                    "formal health position 0 must precede the first calibrated update"
                )
            continue
        if (
            record["global_state_delta"] is None
            or record["uncertainty_u"] is None
            or record["reliability"] is None
        ):
            raise ValueError(
                "formal health positions 1..29 require finite global-state delta and "
                f"uncertainty/reliability signals; missing at position {position}"
            )
    return tuple(records)


def _input_manifest_rgb_shas(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    if manifest.get("frame_count") != FRAME_COUNT:
        raise ValueError(f"input manifest frame_count must equal {FRAME_COUNT}")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise ValueError(f"input manifest must contain exactly {FRAME_COUNT} frames")
    hashes: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"input manifest frame {index} must be an object")
        metadata = frame.get("metadata")
        nested = metadata.get("rgb_sha256") if isinstance(metadata, Mapping) else None
        raw_sha = frame.get("sha256", nested)
        hashes.append(
            _validated_sha256(raw_sha, name=f"input manifest frame {index} RGB SHA")
        )
    if len(set(hashes)) != FRAME_COUNT:
        raise ValueError("input manifest must consume 30 unique RGB SHA-256 values")
    return tuple(sorted(hashes))


def _source_manifest_rgb_shas(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    frame_count = _strict_integer(
        manifest.get("frame_count"), name="source manifest frame_count", minimum=1
    )
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError("source manifest frames do not match frame_count")
    hashes: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"source manifest frame {index} must be an object")
        metadata = frame.get("metadata")
        nested = metadata.get("rgb_sha256") if isinstance(metadata, Mapping) else None
        raw_sha = frame.get("rgb_sha256", frame.get("sha256", nested))
        hashes.append(
            _validated_sha256(raw_sha, name=f"source manifest frame {index} RGB SHA")
        )
    if len(set(hashes)) != len(hashes):
        raise ValueError("source manifest RGB SHA-256 values must be unique")
    return tuple(hashes)


def _interval_projection(interval: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    required = ("type", "start", "end", "parameters")
    missing = [field for field in required if field not in interval]
    if missing:
        raise ValueError(f"{name} missing required field(s): {', '.join(missing)}")
    corruption_type = _nonempty_string(interval["type"], name=f"{name}.type")
    start = _strict_integer(interval["start"], name=f"{name}.start")
    end = _strict_integer(interval["end"], name=f"{name}.end")
    parameters = interval["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{name}.parameters must be a JSON object")
    if "start_frame" in interval and interval["start_frame"] != start:
        raise ValueError(f"{name}.start_frame disagrees with start")
    if "end_frame" in interval and interval["end_frame"] != end:
        raise ValueError(f"{name}.end_frame disagrees with end")
    return {
        "type": corruption_type,
        "start": start,
        "end": end,
        "parameters": dict(parameters),
    }


def _formal_dynamic_rectangles() -> list[dict[str, float]]:
    return [
        {
            "x": round(0.25 + offset * 0.04, 8),
            "y": round(0.25 + offset * 0.025, 8),
            "width": 0.5,
            "height": 0.5,
        }
        for offset in range(5)
    ]


def _expected_formal_parameters(corruption_type: str) -> dict[str, Any]:
    if corruption_type == "dynamic_occlusion":
        return {
            "coordinate_space": "normalized",
            "coordinate_reference": FORMAL_RECTANGLE_COORDINATE_REFERENCE,
            "initial_rectangle": {
                "x": 0.25,
                "y": 0.25,
                "width": 0.5,
                "height": 0.5,
            },
            "velocity": {"dx": 0.04, "dy": 0.025},
            "fill": [255, 0, 0],
            "frame_rectangles": _formal_dynamic_rectangles(),
            "rectangle_generated_from_seed": False,
            "velocity_generated_from_seed": False,
        }
    if corruption_type == "low_overlap_jump":
        return {
            "source_start": 30,
            "source_end": 34,
            "source_indices": [30, 31, 32, 33, 34],
            "selection": "explicit",
        }
    if corruption_type == "wrong_order_segment":
        return {
            "mode": "reverse",
            "permutation": [18, 17, 16, 15],
            "relative_permutation": [3, 2, 1, 0],
        }
    raise ValueError(f"unsupported formal corruption type {corruption_type!r}")


def _validate_formal_parameters(interval: Mapping[str, Any]) -> None:
    corruption_type = str(interval["type"])
    expected = _expected_formal_parameters(corruption_type)
    if _canonical_json_bytes(interval["parameters"]) != _canonical_json_bytes(expected):
        raise ValueError(
            f"formal {corruption_type} parameters do not match frozen v1 severity"
        )


def _expected_formal_transform(
    corruption_type: str, position: int
) -> dict[str, Any]:
    if corruption_type == "dynamic_occlusion":
        return {
            "type": "rectangle_occlusion",
            "coordinate_space": "normalized",
            "coordinate_reference": FORMAL_RECTANGLE_COORDINATE_REFERENCE,
            "rectangle": _formal_dynamic_rectangles()[position - EVENT_START],
            "fill": [255, 0, 0],
        }
    replacement = (
        position + 15
        if corruption_type == "low_overlap_jump"
        else FORMAL_EVENT_ENDS[corruption_type] - (position - EVENT_START)
    )
    return {
        "type": (
            "source_frame_substitution"
            if corruption_type == "low_overlap_jump"
            else "temporal_reorder"
        ),
        "corruption": corruption_type,
        "original_source_index": position,
        "replacement_source_index": replacement,
    }


def _validate_manifest_transform_binding(
    input_manifest: Mapping[str, Any], interval: Mapping[str, Any]
) -> None:
    frames = input_manifest["frames"]
    start = int(interval["start"])
    end = int(interval["end"])
    expected_transform = {
        "low_overlap_jump": "source_frame_substitution",
        "dynamic_occlusion": "rectangle_occlusion",
        "wrong_order_segment": "temporal_reorder",
    }[str(interval["type"])]
    for position, frame in enumerate(frames):
        if not isinstance(frame, Mapping):  # guarded by RGB extraction
            raise ValueError(f"input manifest frame {position} must be an object")
        if "frame_index" in frame and frame["frame_index"] != position:
            raise ValueError("input manifest frame_index must equal output position")
        transforms = frame.get("transforms")
        if not isinstance(transforms, list):
            raise ValueError(f"input manifest frame {position} needs transforms array")
        if start <= position <= end:
            if len(transforms) != 1 or not isinstance(transforms[0], Mapping):
                raise ValueError(
                    "every primary-positive frame must carry exactly one transform"
                )
            if transforms[0].get("type") != expected_transform:
                raise ValueError(
                    f"input transform at position {position} does not match "
                    f"corruption type {interval['type']!r}"
                )
            expected = _expected_formal_transform(str(interval["type"]), position)
            if _canonical_json_bytes(transforms[0]) != _canonical_json_bytes(expected):
                raise ValueError(
                    f"input transform at position {position} does not match "
                    "the frozen formal v1 transform"
                )
        elif transforms:
            raise ValueError(
                f"input transform outside primary interval at position {position}"
            )


def _prepare_input_snapshots(
    run: DetectionSuiteInput | DetectionSuiteRun,
) -> _PreparedInput:
    run_id = _nonempty_string(run.run_id, name="run_id")
    if run.dataset_split not in ("development", "holdout"):
        raise ValueError(f"run {run_id}: dataset_split must be development or holdout")
    if run.corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"run {run_id}: unsupported corruption type")
    raw_frame_sha256s = _validated_raw_sha_set(
        run.raw_frame_sha256s, name=f"run {run_id} raw_frame_sha256s"
    )
    artifacts = {
        "corruption_json": _artifact(run.corruption_json),
        "input_manifest": _artifact(run.input_manifest_json),
        "source_manifest": _artifact(run.source_manifest_json),
    }
    input_manifest = _strict_json_loads(
        run.input_manifest_json, name="input_manifest_json"
    )
    source_manifest = _strict_json_loads(
        run.source_manifest_json, name="source_manifest_json"
    )
    corruption = _strict_json_loads(run.corruption_json, name="corruption_json")
    for value, name in (
        (input_manifest, "input_manifest_json"),
        (source_manifest, "source_manifest_json"),
        (corruption, "corruption_json"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must contain a JSON object")
    if run.corruption_json != run.input_manifest_json:
        raise ValueError(
            f"run {run_id}: formal corruption JSON must be the exact input-manifest bytes"
        )
    source = _validated_formal_source_manifest(
        source_manifest,
        run_id=run_id,
        dataset_split=run.dataset_split,
        corruption_type=run.corruption_type,
    )
    source_rgb_sequence = source["source_pool_frame_sha256s"]
    source_identities = source["source_pool_identities"]

    input_keys = {
        "schema_version",
        "sequence",
        "source_sequence",
        "source_manifest",
        "source_manifest_sha256",
        "source_ground_truth_path",
        "source_ground_truth_resolved_path",
        "source_ground_truth_sha256",
        "source_is_read_only",
        "source_frame_count",
        "frame_count",
        "seed",
        "index_convention",
        "materialization",
        "frames",
        "corruptions",
    }
    _expect_exact_keys(input_manifest, input_keys, name="input_manifest_json")
    input_fixed = {
        "schema_version": FORMAL_INPUT_SCHEMA_VERSION,
        "sequence": run_id,
        "source_sequence": source_manifest["sequence"],
        "source_manifest": "source-manifest.json",
        "source_is_read_only": True,
        "source_frame_count": len(source_rgb_sequence),
        "frame_count": FRAME_COUNT,
        "seed": 0,
        "index_convention": "zero_based_inclusive",
        "materialization": "deferred_transforms_no_image_copy",
    }
    for field, expected in input_fixed.items():
        if input_manifest.get(field) != expected:
            raise ValueError(f"run {run_id}: input manifest {field} mismatch")
    source_sha = _validated_sha256(
        input_manifest.get("source_manifest_sha256"),
        name="input manifest source_manifest_sha256",
    )
    if source_sha != artifacts["source_manifest"]["sha256"]:
        raise ValueError(f"run {run_id}: input/source manifest SHA mismatch")
    if input_manifest.get("source_ground_truth_path") != source_manifest.get(
        "ground_truth_path"
    ):
        raise ValueError(f"run {run_id}: input/source ground-truth path mismatch")
    expected_gt_sha = source["source_index_files"]["groundtruth"]["sha256"]
    if input_manifest.get("source_ground_truth_sha256") != expected_gt_sha:
        raise ValueError(f"run {run_id}: input/source ground-truth SHA mismatch")
    gt_resolved = _validated_path(
        input_manifest.get("source_ground_truth_resolved_path"),
        name="input manifest source_ground_truth_resolved_path",
        absolute=True,
    )
    if gt_resolved != os.path.join(FORMAL_DATASET_ROOT, "groundtruth.txt"):
        raise ValueError(f"run {run_id}: resolved ground-truth path is not frozen")

    input_frames = input_manifest.get("frames")
    if not isinstance(input_frames, list) or len(input_frames) != FRAME_COUNT:
        raise ValueError(f"run {run_id}: input manifest must contain 30 frames")
    consumed_projection: list[dict[str, Any]] = []
    for position, frame in enumerate(input_frames):
        if not isinstance(frame, Mapping):
            raise ValueError(f"input manifest frame {position} must be an object")
        _expect_exact_keys(
            frame,
            {"frame_index", "source_index", "path", "metadata", "transforms"},
            name=f"input manifest frame {position}",
        )
        if frame["frame_index"] != position:
            raise ValueError("input manifest frame_index must equal output position")
        source_index = _strict_integer(
            frame.get("source_index"),
            name=f"input manifest frame {position} source_index",
            minimum=0,
        )
        if source_index >= len(source_rgb_sequence):
            raise ValueError(f"run {run_id}: source_index is outside source pool")
        metadata = frame.get("metadata")
        source_frame = source["frames"][source_index]
        expected_metadata = {
            field: value for field, value in source_frame.items() if field != "path"
        }
        if not isinstance(metadata, Mapping) or dict(metadata) != expected_metadata:
            raise ValueError(
                f"run {run_id}: output frame {position} metadata disagrees with source_index"
            )
        frame_path = _validated_path(
            frame.get("path"), name=f"input manifest frame {position} path"
        )
        if frame_path != source_frame["path"]:
            raise ValueError(
                f"run {run_id}: output frame {position} path disagrees with source_index"
            )
        consumed_projection.append(
            {
                "output_position": position,
                "source_pool_index": source_index,
                "path": frame_path,
                "rgb_sha256": source_identities[source_index]["rgb"]["sha256"],
                "timestamp": source_identities[source_index]["rgb"]["timestamp"],
                "source_metadata_canonical_sha256": hashlib.sha256(
                    _canonical_json_bytes(expected_metadata)
                ).hexdigest(),
            }
        )
    consumed_frames = _validated_consumed_frames(
        consumed_projection,
        name=f"run {run_id} consumed_frames",
        source_pool_identities=source_identities,
        corruption_type=run.corruption_type,
    )
    input_raw = tuple(sorted(frame["rgb_sha256"] for frame in consumed_frames))
    if input_raw != raw_frame_sha256s:
        raise ValueError(
            f"run {run_id}: input/declared consumed raw RGB SHA sets differ"
        )

    scoring_interval = _enabled_primary_interval(corruption)
    input_interval = _enabled_primary_interval(input_manifest)
    scoring_projection = _interval_projection(
        scoring_interval, name="corruption JSON interval"
    )
    input_projection = _interval_projection(
        input_interval, name="input manifest interval"
    )
    if scoring_projection != input_projection:
        raise ValueError(
            f"run {run_id}: scoring corruption interval disagrees with input manifest"
        )
    if scoring_projection["type"] != run.corruption_type:
        raise ValueError(f"run {run_id}: declared and manifest corruption types differ")
    regions = build_evaluation_regions(corruption)
    _validate_formal_parameters(scoring_projection)
    _validate_manifest_transform_binding(input_manifest, scoring_projection)
    return _PreparedInput(
        corruption=corruption,
        regions=regions,
        artifacts=artifacts,
        raw_frame_sha256s=raw_frame_sha256s,
        source_pool_frame_sha256s=source_rgb_sequence,
        source_pool_identities=source_identities,
        consumed_frames=consumed_frames,
        source_index_files=source["source_index_files"],
        official_lineage=source["official_lineage"],
    )


def _expected_transform_summary(corruption_type: str) -> tuple[dict[str, int], str | None]:
    if corruption_type == "dynamic_occlusion":
        return {"rectangle_occlusion": 5}, FORMAL_RECTANGLE_COORDINATE_REFERENCE
    if corruption_type == "low_overlap_jump":
        return {"source_frame_substitution": 5}, None
    return {"temporal_reorder": 4}, None


def _resolved_manifest_frame_path(source_manifest_path: str, raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return os.path.realpath(raw_path)
    return os.path.realpath(os.path.join(os.path.dirname(source_manifest_path), raw_path))


def _validated_run_input_binding(
    run_json: Mapping[str, Any],
    *,
    run_id: str,
    prepared_input: _PreparedInput,
) -> tuple[str, ...]:
    if run_json.get("frame_count") != FRAME_COUNT:
        raise ValueError(f"run JSON frame_count must equal {FRAME_COUNT}")
    linked = run_json.get("input_manifest")
    if not isinstance(linked, Mapping):
        raise ValueError(f"run {run_id}: run JSON lacks input_manifest provenance")
    _expect_exact_keys(
        linked,
        {
            "path",
            "sha256",
            "source_manifest",
            "source_manifest_sha256",
            "materialization",
            "rectangle_coordinate_reference",
            "transform_counts",
        },
        name=f"run {run_id} input_manifest",
    )
    input_path = _validated_path(
        linked["path"], name=f"run {run_id} input_manifest.path", absolute=True
    )
    source_path = _validated_path(
        linked["source_manifest"],
        name=f"run {run_id} input_manifest.source_manifest",
        absolute=True,
    )
    if os.path.basename(input_path) != "input-manifest.json":
        raise ValueError(f"run {run_id}: unexpected input-manifest filename")
    if os.path.basename(source_path) != "source-manifest.json":
        raise ValueError(f"run {run_id}: unexpected source-manifest filename")
    if os.path.dirname(input_path) != os.path.dirname(source_path):
        raise ValueError(f"run {run_id}: input/source manifests must share one run directory")
    if linked["materialization"] != (
        "final_manifest_order_with_in_memory_deferred_transforms"
    ):
        raise ValueError(f"run {run_id}: input materialization provenance mismatch")
    transform_counts, rectangle_reference = _expected_transform_summary(
        prepared_input.corruption["corruptions"][0]["type"]
    )
    if linked["transform_counts"] != transform_counts:
        raise ValueError(f"run {run_id}: transform-count provenance mismatch")
    if linked["rectangle_coordinate_reference"] != rectangle_reference:
        raise ValueError(f"run {run_id}: rectangle-reference provenance mismatch")

    images = run_json.get("images")
    input_frames = run_json.get("input_frames")
    if not isinstance(images, list) or len(images) != FRAME_COUNT:
        raise ValueError(f"run JSON must contain exactly {FRAME_COUNT} images")
    if not isinstance(input_frames, list) or len(input_frames) != FRAME_COUNT:
        raise ValueError(f"run JSON must contain exactly {FRAME_COUNT} input_frames")

    ordered_shas: list[str] = []
    # The exact input-frame metadata/transforms are recovered from the committed
    # source identities and corruption construction; paths are resolved relative
    # to the run JSON's bound source-manifest path just as the strict loader does.
    interval = _enabled_primary_interval(prepared_input.corruption)
    for position, (image, run_frame, consumed) in enumerate(
        zip(images, input_frames, prepared_input.consumed_frames)
    ):
        if not isinstance(image, Mapping):
            raise ValueError(f"run JSON image {position} must be an object")
        if not isinstance(run_frame, Mapping):
            raise ValueError(f"run JSON input_frame {position} must be an object")
        _expect_exact_keys(image, {"path", "sha256"}, name=f"run image {position}")
        _expect_exact_keys(
            run_frame,
            {"frame_index", "source_index", "path", "sha256", "metadata", "transforms"},
            name=f"run input_frame {position}",
        )
        expected_path = _resolved_manifest_frame_path(source_path, consumed["path"])
        expected_sha = consumed["rgb_sha256"]
        actual_path = _validated_path(
            run_frame["path"], name=f"run input_frame {position}.path", absolute=True
        )
        actual_sha = _validated_sha256(
            run_frame["sha256"], name=f"run input_frame {position}.sha256"
        )
        if (
            run_frame["frame_index"] != position
            or run_frame["source_index"] != consumed["source_pool_index"]
            or actual_path != expected_path
            or actual_sha != expected_sha
        ):
            raise ValueError(
                f"run {run_id}: input_frame {position} does not match committed order"
            )
        metadata = run_frame["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"run input_frame {position}.metadata must be an object")
        if hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest() != consumed[
            "source_metadata_canonical_sha256"
        ]:
            raise ValueError(
                f"run {run_id}: input_frame {position} metadata identity mismatch"
            )
        transforms = run_frame["transforms"]
        if not isinstance(transforms, list):
            raise ValueError(f"run input_frame {position}.transforms must be an array")
        if interval["start"] <= position <= interval["end"]:
            expected_transforms = [
                _expected_formal_transform(str(interval["type"]), position)
            ]
        else:
            expected_transforms = []
        if transforms != expected_transforms:
            raise ValueError(
                f"run {run_id}: input_frame {position} transform provenance mismatch"
            )
        image_path = _validated_path(
            image["path"], name=f"run image {position}.path", absolute=True
        )
        image_sha = _validated_sha256(
            image["sha256"], name=f"run image {position}.sha256"
        )
        if image_path != actual_path or image_sha != actual_sha:
            raise ValueError(
                f"run {run_id}: images/input_frames differ at position {position}"
            )
        ordered_shas.append(actual_sha)
    if len(set(ordered_shas)) != FRAME_COUNT:
        raise ValueError("run JSON must contain 30 unique ordered RGB identities")
    return tuple(ordered_shas)


def _validated_execution_provenance(
    run_json: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    fixed = {
        "schema_version": RECAL3R_RUN_SCHEMA_VERSION,
        "status": "succeeded",
        "baseline_commit": EXPECTED_RECAL3R_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "frame_count": FRAME_COUNT,
        "calibrated_update_calls": FRAME_COUNT - 1,
        "expected_calibrated_update_calls": FRAME_COUNT - 1,
        "size": 512,
        "seed": 0,
        "beta_base": 0.1,
        "input_mode": "manifest",
        "model_update_type": "recal3r",
    }
    for field, expected in fixed.items():
        if run_json.get(field) != expected:
            raise ValueError(
                f"run {run_id}: run JSON {field} must equal {expected!r}"
            )
    runner = run_json.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError(f"run {run_id}: run JSON lacks runner provenance")
    runner_sha = _validated_sha256(
        runner.get("script_sha256"), name=f"run {run_id} runner.script_sha256"
    )
    if runner_sha != EXPECTED_RUNNER_SHA256:
        raise ValueError(f"run {run_id}: unexpected runner script SHA")
    runner_commit = runner.get("commit")
    if (
        not isinstance(runner_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", runner_commit) is None
    ):
        raise ValueError(f"run {run_id}: runner.commit must be a full Git SHA")
    runtime = _finite_number(
        run_json.get("runtime_seconds"),
        name=f"run {run_id}.runtime_seconds",
        positive=True,
    )
    peak_memory = _finite_number(
        run_json.get("peak_memory_allocated_mib"),
        name=f"run {run_id}.peak_memory_allocated_mib",
        positive=True,
    )
    if run_json.get("trace_frame_steps") != list(range(1, FRAME_COUNT)):
        raise ValueError(f"run {run_id}: trace_frame_steps must equal positions 1..29")
    if run_json.get("health_signal_semantics") != EXPECTED_HEALTH_SIGNAL_SEMANTICS:
        raise ValueError(f"run {run_id}: health_signal_semantics mismatch")
    return {
        **fixed,
        "runner_script_sha256": runner_sha,
        "runner_commit": runner_commit,
        "runtime_seconds": runtime,
        "peak_memory_allocated_mib": peak_memory,
        "trace_frame_steps": list(range(1, FRAME_COUNT)),
        "health_signal_semantics": dict(EXPECTED_HEALTH_SIGNAL_SEMANTICS),
    }


def _prepare_run(run: DetectionSuiteRun) -> _PreparedRun:
    run_id = _nonempty_string(run.run_id, name="run_id")
    if run.dataset_split not in ("development", "holdout"):
        raise ValueError(f"run {run_id}: dataset_split must be development or holdout")
    prepared_input = _prepare_input_snapshots(run)
    artifacts = {
        **prepared_input.artifacts,
        "health_jsonl": _artifact(run.health_jsonl),
        "run_json": _artifact(run.run_json),
    }
    records = _parse_health_jsonl(run.health_jsonl)
    run_json = _strict_json_loads(run.run_json, name="run_json")
    if not isinstance(run_json, Mapping):
        raise ValueError("run_json must contain a JSON object")
    execution_provenance = _validated_execution_provenance(run_json, run_id=run_id)
    run_ordered = _validated_run_input_binding(
        run_json, run_id=run_id, prepared_input=prepared_input
    )
    expected_ordered = tuple(
        frame["rgb_sha256"] for frame in prepared_input.consumed_frames
    )
    if run_ordered != expected_ordered:
        raise ValueError(
            f"run {run_id}: ordered RGB identities differ across input/run provenance"
        )
    linked_manifest = run_json.get("input_manifest")
    if not isinstance(linked_manifest, Mapping):
        raise ValueError(f"run {run_id}: run JSON lacks input_manifest provenance")
    linked_sha = _validated_sha256(
        linked_manifest.get("sha256"), name=f"run {run_id} input_manifest.sha256"
    )
    if linked_sha != artifacts["input_manifest"]["sha256"]:
        raise ValueError(f"run {run_id}: run JSON input-manifest SHA mismatch")
    linked_source_sha = _validated_sha256(
        linked_manifest.get("source_manifest_sha256"),
        name=f"run {run_id} input_manifest.source_manifest_sha256",
    )
    if linked_source_sha != artifacts["source_manifest"]["sha256"]:
        raise ValueError(f"run {run_id}: run JSON source-manifest SHA mismatch")
    for position, (record, consumed) in enumerate(
        zip(records, prepared_input.consumed_frames)
    ):
        if record["timestamp"] != consumed["timestamp"]:
            raise ValueError(
                f"run {run_id}: health timestamp/input identity mismatch at "
                f"position {position}"
            )

    return _PreparedRun(
        source=run,
        records=records,
        corruption=prepared_input.corruption,
        regions=prepared_input.regions,
        artifacts=artifacts,
        raw_frame_sha256s=prepared_input.raw_frame_sha256s,
        source_pool_frame_sha256s=prepared_input.source_pool_frame_sha256s,
        source_pool_identities=prepared_input.source_pool_identities,
        consumed_frames=prepared_input.consumed_frames,
        source_index_files=prepared_input.source_index_files,
        official_lineage=prepared_input.official_lineage,
        execution_provenance=execution_provenance,
    )


def _run_commitment_from_prepared(run: _PreparedRun) -> dict[str, Any]:
    return {
        "run_id": run.source.run_id,
        "dataset_split": run.source.dataset_split,
        "corruption_type": run.source.corruption_type,
        "raw_frame_sha256s": list(run.raw_frame_sha256s),
        "source_pool_frame_sha256s": list(run.source_pool_frame_sha256s),
        "source_pool_identities": list(run.source_pool_identities),
        "consumed_frames": list(run.consumed_frames),
        "source_index_files": run.source_index_files,
        "official_lineage": run.official_lineage,
        "artifacts": {
            name: run.artifacts[name] for name in sorted(PRE_FORWARD_ARTIFACTS)
        },
    }


def _full_run_provenance(run: _PreparedRun) -> dict[str, Any]:
    return {
        "run_id": run.source.run_id,
        "dataset_split": run.source.dataset_split,
        "corruption_type": run.source.corruption_type,
        "raw_frame_sha256s": list(run.raw_frame_sha256s),
        "source_pool_frame_sha256s": list(run.source_pool_frame_sha256s),
        "source_pool_identities": list(run.source_pool_identities),
        "consumed_frames": list(run.consumed_frames),
        "source_index_files": run.source_index_files,
        "official_lineage": run.official_lineage,
        "artifacts": {name: run.artifacts[name] for name in sorted(FULL_ARTIFACTS)},
        "execution_provenance": run.execution_provenance,
    }


def make_run_commitment(
    run: DetectionSuiteInput | DetectionSuiteRun,
) -> dict[str, Any]:
    """Freeze input/corruption provenance without requiring forward outputs."""

    prepared = _prepare_input_snapshots(run)
    return {
        "run_id": run.run_id,
        "dataset_split": run.dataset_split,
        "corruption_type": run.corruption_type,
        "raw_frame_sha256s": list(prepared.raw_frame_sha256s),
        "source_pool_frame_sha256s": list(prepared.source_pool_frame_sha256s),
        "source_pool_identities": list(prepared.source_pool_identities),
        "consumed_frames": list(prepared.consumed_frames),
        "source_index_files": prepared.source_index_files,
        "official_lineage": prepared.official_lineage,
        "artifacts": prepared.artifacts,
    }


def _validated_run_commitment(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(
        value,
        {
            "run_id",
            "dataset_split",
            "corruption_type",
            "raw_frame_sha256s",
            "source_pool_frame_sha256s",
            "source_pool_identities",
            "consumed_frames",
            "source_index_files",
            "official_lineage",
            "artifacts",
        },
        name=name,
    )
    run_id = _nonempty_string(value["run_id"], name=f"{name}.run_id")
    dataset_split = value["dataset_split"]
    if dataset_split not in ("development", "holdout"):
        raise ValueError(f"{name}.dataset_split is invalid")
    corruption_type = value["corruption_type"]
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"{name}.corruption_type is unsupported")
    _require_exact_formal_run_identity(run_id, dataset_split, corruption_type)
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{name}.artifacts must be a JSON object")
    _expect_exact_keys(artifacts, PRE_FORWARD_ARTIFACTS, name=f"{name}.artifacts")
    raw_frame_sha256s = _validated_raw_sha_set(
        value["raw_frame_sha256s"], name=f"{name}.raw_frame_sha256s"
    )
    source_pool_frame_sha256s = _validated_source_pool_sha_sequence(
        value["source_pool_frame_sha256s"],
        name=f"{name}.source_pool_frame_sha256s",
        corruption_type=corruption_type,
    )
    if not set(raw_frame_sha256s).issubset(source_pool_frame_sha256s):
        raise ValueError(f"{name} consumed RGB hashes are outside its source pool")
    source_pool_identities = _validated_source_pool_identities(
        value["source_pool_identities"],
        name=f"{name}.source_pool_identities",
        dataset_split=dataset_split,
        corruption_type=corruption_type,
    )
    identity_shas = tuple(item["rgb"]["sha256"] for item in source_pool_identities)
    if identity_shas != source_pool_frame_sha256s:
        raise ValueError(f"{name} source-pool hashes/identity projection disagree")
    consumed_frames = _validated_consumed_frames(
        value["consumed_frames"],
        name=f"{name}.consumed_frames",
        source_pool_identities=source_pool_identities,
        corruption_type=corruption_type,
    )
    if tuple(sorted(item["rgb_sha256"] for item in consumed_frames)) != raw_frame_sha256s:
        raise ValueError(f"{name} consumed-frame projection/raw SHA set disagree")
    raw_index_files = value["source_index_files"]
    if not isinstance(raw_index_files, Mapping):
        raise ValueError(f"{name}.source_index_files must be a JSON object")
    _expect_exact_keys(
        raw_index_files,
        {"rgb", "depth", "groundtruth"},
        name=f"{name}.source_index_files",
    )
    source_index_files = {
        kind: _validated_snapshot_artifact(
            raw_index_files[kind],
            name=f"{name}.source_index_files.{kind}",
            include_entry_count=True,
        )
        for kind in ("rgb", "depth", "groundtruth")
    }
    official_lineage = _validated_official_lineage(
        value["official_lineage"], name=f"{name}.official_lineage"
    )
    return {
        "run_id": run_id,
        "dataset_split": dataset_split,
        "corruption_type": corruption_type,
        "raw_frame_sha256s": list(raw_frame_sha256s),
        "source_pool_frame_sha256s": list(source_pool_frame_sha256s),
        "source_pool_identities": list(source_pool_identities),
        "consumed_frames": list(consumed_frames),
        "source_index_files": source_index_files,
        "official_lineage": official_lineage,
        "artifacts": {
            artifact_name: _validated_artifact(
                artifacts[artifact_name], name=f"{name}.artifacts.{artifact_name}"
            )
            for artifact_name in sorted(PRE_FORWARD_ARTIFACTS)
        },
    }


def _validated_commitment_list(
    value: Any, *, name: str, expected_split: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    result = [
        _validated_run_commitment(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(CORRUPTION_TYPES):
        raise ValueError(f"{name} must contain exactly three runs")
    run_ids = [item["run_id"] for item in result]
    types = [item["corruption_type"] for item in result]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"{name} contains duplicate run IDs")
    if any(item["dataset_split"] != expected_split for item in result):
        raise ValueError(f"{name} contains a run assigned to the wrong split")
    if tuple(run_ids) != FORMAL_RUN_IDS_BY_SPLIT[expected_split]:
        raise ValueError(f"{name} run IDs/order do not match frozen formal v1")
    if set(types) != CORRUPTION_TYPES or len(set(types)) != len(types):
        raise ValueError(f"{name} must contain each corruption type exactly once")
    _assert_pairwise_disjoint_source_pools(result, name=name)
    return result


def _assert_pairwise_disjoint_source_pools(
    runs: Sequence[Mapping[str, Any]], *, name: str
) -> None:
    owner: dict[str, str] = {}
    for run in runs:
        run_id = str(run["run_id"])
        for digest in run["source_pool_frame_sha256s"]:
            previous = owner.get(digest)
            if previous is not None:
                raise ValueError(
                    f"{name} source-pool overlap between {previous!r} and {run_id!r}: "
                    f"{digest}"
                )
            owner[digest] = run_id
    _validated_identity_components_unique(
        [identity for run in runs for identity in run["source_pool_identities"]],
        name=f"{name} complete source pools",
    )


def _assert_split_disjoint(
    development: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
) -> None:
    duplicate_run_ids = {run["run_id"] for run in development} & {
        run["run_id"] for run in holdout
    }
    if duplicate_run_ids:
        raise ValueError("development/holdout run IDs overlap")
    development_source_pool = {
        digest
        for run in development
        for digest in run["source_pool_frame_sha256s"]
    }
    holdout_source_pool = {
        digest for run in holdout for digest in run["source_pool_frame_sha256s"]
    }
    overlap = sorted(development_source_pool & holdout_source_pool)
    if overlap:
        raise ValueError(
            "development/holdout source-pool SHA sets overlap: "
            + ", ".join(overlap)
        )
    _validated_identity_components_unique(
        [
            identity
            for run in [*development, *holdout]
            for identity in run["source_pool_identities"]
        ],
        name="development/holdout complete source pools",
    )


def make_split_registry(
    development: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize a strict, deterministic, pre-forward split registry."""

    dev = _validated_commitment_list(
        list(development), name="development", expected_split="development"
    )
    frozen = _validated_commitment_list(
        list(holdout), name="holdout", expected_split="holdout"
    )
    _assert_split_disjoint(dev, frozen)
    return _canonical_json_bytes(
        {
            "schema_version": SPLIT_REGISTRY_SCHEMA_VERSION,
            "development": dev,
            "holdout": frozen,
        }
    )


def make_holdout_commitment(holdout: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize the holdout-only projection independently of the registry."""

    frozen = _validated_commitment_list(
        list(holdout), name="holdout", expected_split="holdout"
    )
    return _canonical_json_bytes(
        {
            "schema_version": HOLDOUT_COMMITMENT_SCHEMA_VERSION,
            "runs": frozen,
        }
    )


def _parse_split_registry(raw: bytes) -> dict[str, Any]:
    value = _strict_json_loads(raw, name="split_registry")
    if not isinstance(value, Mapping):
        raise ValueError("split_registry must be a JSON object")
    _expect_exact_keys(
        value,
        {"schema_version", "development", "holdout"},
        name="split_registry",
    )
    if value["schema_version"] != SPLIT_REGISTRY_SCHEMA_VERSION:
        raise ValueError("split_registry schema_version mismatch")
    development = _validated_commitment_list(
        value["development"],
        name="split_registry.development",
        expected_split="development",
    )
    holdout = _validated_commitment_list(
        value["holdout"], name="split_registry.holdout", expected_split="holdout"
    )
    _assert_split_disjoint(development, holdout)
    return {
        "schema_version": SPLIT_REGISTRY_SCHEMA_VERSION,
        "development": development,
        "holdout": holdout,
    }


def _parse_holdout_commitment(raw: bytes) -> dict[str, Any]:
    value = _strict_json_loads(raw, name="holdout_commitment")
    if not isinstance(value, Mapping):
        raise ValueError("holdout_commitment must be a JSON object")
    _expect_exact_keys(value, {"schema_version", "runs"}, name="holdout_commitment")
    if value["schema_version"] != HOLDOUT_COMMITMENT_SCHEMA_VERSION:
        raise ValueError("holdout_commitment schema_version mismatch")
    return {
        "schema_version": HOLDOUT_COMMITMENT_SCHEMA_VERSION,
        "runs": _validated_commitment_list(
            value["runs"],
            name="holdout_commitment.runs",
            expected_split="holdout",
        ),
    }


def _validated_candidate(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(value, {"window", "epsilon", "max_z"}, name=name)
    window = _strict_integer(value["window"], name=f"{name}.window", minimum=1)
    if window > EVENT_START:
        raise ValueError(f"{name}.window may not exceed event_start={EVENT_START}")
    epsilon = _finite_number(value["epsilon"], name=f"{name}.epsilon", positive=True)
    raw_max_z = value["max_z"]
    max_z = (
        None
        if raw_max_z is None
        else _finite_number(raw_max_z, name=f"{name}.max_z", positive=True)
    )
    return {"window": window, "epsilon": epsilon, "max_z": max_z}


def _validated_candidate_order(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidate_order must be a non-empty JSON array")
    candidates = [
        _validated_candidate(item, name=f"candidate_order[{index}]")
        for index, item in enumerate(value)
    ]
    fingerprints = [_canonical_json_bytes(candidate) for candidate in candidates]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate_order may not contain duplicate candidates")
    return candidates


def _require_formal_search_protocol(
    candidate_order: Sequence[Mapping[str, Any]],
    master_seed: int,
    *,
    name: str,
) -> None:
    if list(candidate_order) != fixed_candidate_order():
        raise ValueError(
            f"{name} candidate_order must equal the fixed 27-candidate formal v1 grid"
        )
    if master_seed != 0:
        raise ValueError(f"{name} master_seed must equal 0")


def _validated_thresholds(value: Any, *, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(value, set(METHOD_NAMES), name=name)
    return {
        method: _finite_number(value[method], name=f"{name}.{method}")
        for method in METHOD_NAMES
    }


def _validated_detection_config(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(
        value,
        {"window", "epsilon", "max_z", "master_seed", "thresholds"},
        name=name,
    )
    # Validate the candidate fields without accepting the two config-only keys.
    window = _strict_integer(value["window"], name=f"{name}.window", minimum=1)
    if window > EVENT_START:
        raise ValueError(f"{name}.window may not exceed event_start={EVENT_START}")
    epsilon = _finite_number(value["epsilon"], name=f"{name}.epsilon", positive=True)
    raw_max_z = value["max_z"]
    max_z = (
        None
        if raw_max_z is None
        else _finite_number(raw_max_z, name=f"{name}.max_z", positive=True)
    )
    return {
        "window": window,
        "epsilon": epsilon,
        "max_z": max_z,
        "master_seed": _strict_integer(
            value["master_seed"], name=f"{name}.master_seed", minimum=0
        ),
        "thresholds": _validated_thresholds(
            value["thresholds"], name=f"{name}.thresholds"
        ),
    }


def _prepare_suite(
    runs: Sequence[DetectionSuiteRun],
    *,
    expected_split: str | None,
    require_three: bool,
) -> list[_PreparedRun]:
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise ValueError("suite runs must be a non-empty sequence")
    prepared = [_prepare_run(run) for run in runs]
    if not require_three:
        prepared.sort(key=lambda item: item.source.run_id)
    run_ids = [run.source.run_id for run in prepared]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("suite run IDs must be unique")
    if expected_split is not None and any(
        run.source.dataset_split != expected_split for run in prepared
    ):
        raise ValueError(f"all suite runs must belong to {expected_split}")
    if require_three:
        types = [run.source.corruption_type for run in prepared]
        if len(prepared) != 3 or set(types) != CORRUPTION_TYPES or len(set(types)) != 3:
            raise ValueError("suite must contain exactly one run of each corruption type")
        if expected_split is None or tuple(run_ids) != FORMAL_RUN_IDS_BY_SPLIT[
            expected_split
        ]:
            raise ValueError("formal suite run IDs/order do not match frozen v1")
    _assert_pairwise_disjoint_source_pools(
        [_run_commitment_from_prepared(run) for run in prepared], name="suite"
    )
    stable_fields = (
        "schema_version",
        "status",
        "baseline_commit",
        "checkpoint_sha256",
        "frame_count",
        "calibrated_update_calls",
        "expected_calibrated_update_calls",
        "size",
        "seed",
        "beta_base",
        "input_mode",
        "model_update_type",
        "runner_script_sha256",
        "runner_commit",
        "trace_frame_steps",
        "health_signal_semantics",
    )
    first = prepared[0].execution_provenance
    for run in prepared[1:]:
        if any(run.execution_provenance[field] != first[field] for field in stable_fields):
            raise ValueError("suite runs do not share fixed execution provenance")
    return prepared


def _score_prepared_runs(
    runs: Sequence[_PreparedRun],
    candidate: Mapping[str, Any],
    master_seed: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runs:
        # This per-run call is the central no-cross-run-reference guarantee.
        baselines = compute_baseline_scores(
            run.records,
            window=int(candidate["window"]),
            epsilon=float(candidate["epsilon"]),
            max_z=candidate["max_z"],
            seed=derive_run_seed(
                master_seed, run.source.dataset_split, run.source.run_id
            ),
        )
        run_payloads: dict[str, dict[str, Any]] = {}
        expected_signals = {
            "random": ("seeded_random",),
            "update_magnitude_only": ("update_magnitude",),
            "reliability_only": ("reliability",),
            "combined": (
                "geometric_residual",
                "pose_jump",
                "update_magnitude",
                "overlap",
                "reliability",
            ),
        }
        for method in METHOD_NAMES:
            payload = baselines[method]
            used_signals = list(payload["used_signals"])
            run_payloads[method] = {
                "scores": np.asarray(payload["scores"], dtype=np.float64),
                "used_signals": used_signals,
                "missing_signals": [
                    signal
                    for signal in expected_signals[method]
                    if signal not in used_signals
                ],
                "signal_sources": [dict(item) for item in payload["signal_sources"]],
            }
        required = {
            "update_magnitude_only": {"update_magnitude"},
            "reliability_only": {"reliability"},
            "combined": {"update_magnitude", "reliability"},
        }
        for method, required_signals in required.items():
            used = set(run_payloads[method]["used_signals"])
            if not required_signals.issubset(used):
                raise ValueError(
                    f"run {run.source.run_id} method {method} lacks required "
                    f"usable signals: {sorted(required_signals - used)}"
                )
        scores[run.source.run_id] = run_payloads
    return scores


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _longest_true_run(values: NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _evaluate_method(
    runs: Sequence[_PreparedRun],
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]],
    method: str,
    threshold: float,
    *,
    include_scores: bool,
) -> dict[str, Any]:
    threshold = _finite_number(threshold, name=f"thresholds.{method}")
    per_run: dict[str, Any] = {}
    pooled_labels: list[NDArray[np.int64]] = []
    pooled_scores: list[NDArray[np.float64]] = []
    exact_labels: list[NDArray[np.int64]] = []
    exact_scores: list[NDArray[np.float64]] = []
    exact_per_run: dict[str, Any] = {}
    event_rows: list[dict[str, Any]] = []
    detected_delays: list[int] = []
    boundary_alarms = 0
    washout_alarm_frames = 0
    washout_frames = 0
    max_consecutive_false_positives = 0

    for run in runs:
        run_id = run.source.run_id
        score_payload = scores[run_id][method]
        risk = np.asarray(score_payload["scores"], dtype=np.float64)
        if risk.shape != (FRAME_COUNT,):
            raise ValueError(f"run {run_id} method {method} emitted wrong score shape")
        mask = run.regions.evaluation_mask
        if np.any(~np.isfinite(risk)):
            raise ValueError(
                f"run {run_id} method {method} has non-finite score(s) among "
                f"all {FRAME_COUNT} positions"
            )
        labels = run.regions.primary_labels
        evaluated_labels = labels[mask]
        evaluated_scores = risk[mask]
        metrics = threshold_metrics(evaluated_labels, evaluated_scores, threshold)
        metrics["auroc"] = auroc(evaluated_labels, evaluated_scores)
        predicted_false_positive = (
            mask & (labels == 0) & (risk >= threshold)
        )
        longest_false_positives = _longest_true_run(predicted_false_positive)
        metrics["longest_consecutive_false_positives"] = longest_false_positives
        delay = detection_delay(
            labels,
            risk,
            threshold,
            frame_ids=list(range(FRAME_COUNT)),
        )
        metrics["detection_delay"] = delay

        boundary_position = run.regions.event_end + 1
        boundary_alarm = bool(risk[boundary_position] >= threshold)
        washout_positions = np.flatnonzero(run.regions.washout_mask)
        washout_alarm_positions = [
            int(position)
            for position in washout_positions
            if risk[position] >= threshold
        ]
        diagnostic = {
            "recovery_boundary_position": boundary_position,
            "recovery_boundary_score": float(risk[boundary_position]),
            "recovery_boundary_alarm": boundary_alarm,
            "washout_positions": [int(value) for value in washout_positions],
            "washout_alarm_positions": washout_alarm_positions,
            "washout_alarm_frames": len(washout_alarm_positions),
        }
        row: dict[str, Any] = {
            "corruption_type": run.source.corruption_type,
            "used_signals": list(score_payload["used_signals"]),
            "missing_signals": list(score_payload["missing_signals"]),
            "signal_sources": [
                dict(item) for item in score_payload["signal_sources"]
            ],
            "metrics": metrics,
            "diagnostics": diagnostic,
        }
        if include_scores:
            row["scores"] = [float(value) for value in risk]
        per_run[run_id] = row

        pooled_labels.append(evaluated_labels)
        pooled_scores.append(evaluated_scores)
        exact_metric = threshold_metrics(labels, risk, threshold)
        exact_metric["auroc"] = auroc(labels, risk)
        exact_per_run[run_id] = exact_metric
        exact_labels.append(labels)
        exact_scores.append(risk)
        interval = dict(delay["intervals"][0])
        interval.update(
            {"run_id": run_id, "corruption_type": run.source.corruption_type}
        )
        event_rows.append(interval)
        if interval["delay_frames"] is not None:
            detected_delays.append(int(interval["delay_frames"]))
        boundary_alarms += int(boundary_alarm)
        washout_alarm_frames += len(washout_alarm_positions)
        washout_frames += int(run.regions.washout_mask.sum())
        max_consecutive_false_positives = max(
            max_consecutive_false_positives, longest_false_positives
        )

    pooled_truth = np.concatenate(pooled_labels)
    pooled_risk = np.concatenate(pooled_scores)
    pooled = threshold_metrics(pooled_truth, pooled_risk, threshold)
    pooled["auroc"] = auroc(pooled_truth, pooled_risk)
    pooled["max_per_run_consecutive_false_positives"] = (
        max_consecutive_false_positives
    )
    exact_truth = np.concatenate(exact_labels)
    exact_risk = np.concatenate(exact_scores)
    exact_pooled = threshold_metrics(exact_truth, exact_risk, threshold)
    exact_pooled["auroc"] = auroc(exact_truth, exact_risk)
    total_events = len(event_rows)
    detected_events = len(detected_delays)
    return {
        "threshold": threshold,
        "per_run": per_run,
        "pooled": pooled,
        "macro": {
            "f1": _mean_defined(
                [row["metrics"]["f1"] for row in per_run.values()]
            ),
            "false_positive_rate": _mean_defined(
                [
                    row["metrics"]["false_positive_rate"]
                    for row in per_run.values()
                ]
            ),
            "auroc": _mean_defined(
                [row["metrics"]["auroc"] for row in per_run.values()]
            ),
        },
        "exact_interval_sensitivity": {
            "definition": "all primary-label-external frames are negatives",
            "per_run": exact_per_run,
            "pooled": exact_pooled,
            "macro": {
                "f1": _mean_defined(
                    [row["f1"] for row in exact_per_run.values()]
                ),
                "false_positive_rate": _mean_defined(
                    [
                        row["false_positive_rate"]
                        for row in exact_per_run.values()
                    ]
                ),
                "auroc": _mean_defined(
                    [row["auroc"] for row in exact_per_run.values()]
                ),
            },
        },
        "events": {
            "total_events": total_events,
            "detected_events": detected_events,
            "missed_events": total_events - detected_events,
            "detection_rate": float(detected_events / total_events),
            "mean_delay_frames": (
                float(np.mean(detected_delays)) if detected_delays else None
            ),
            "median_delay_frames": (
                float(np.median(detected_delays)) if detected_delays else None
            ),
            "intervals": event_rows,
        },
        "diagnostics": {
            "recovery_boundary_events": total_events,
            "recovery_boundary_alarms": boundary_alarms,
            "washout_frames": washout_frames,
            "washout_alarm_frames": washout_alarm_frames,
            "max_per_run_consecutive_false_positives": (
                max_consecutive_false_positives
            ),
        },
    }


def _regions_payload(run: _PreparedRun) -> dict[str, Any]:
    return {
        "run_id": run.source.run_id,
        "corruption_type": run.source.corruption_type,
        "event_start": run.regions.event_start,
        "event_end": run.regions.event_end,
        "positive_frames": run.regions.positive_frames,
        "washout_frames": run.regions.washout_frames,
        "negative_frames": run.regions.negative_frames,
        "primary_labels": run.regions.primary_labels.tolist(),
        "evaluation_mask": run.regions.evaluation_mask.tolist(),
        "recovery_boundary_mask": run.regions.recovery_boundary_mask.tolist(),
        "washout_mask": run.regions.washout_mask.tolist(),
    }


def _evaluate_prepared_suite(
    runs: Sequence[_PreparedRun],
    config: Mapping[str, Any],
    *,
    include_scores: bool,
) -> dict[str, Any]:
    candidate = {
        "window": config["window"],
        "epsilon": config["epsilon"],
        "max_z": config["max_z"],
    }
    scores = _score_prepared_runs(runs, candidate, int(config["master_seed"]))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation_policy": fixed_evaluation_policy(),
        "detection_config": dict(config),
        "run_count": len(runs),
        "runs": [
            {
                **_regions_payload(run),
                "dataset_split": run.source.dataset_split,
                "derived_random_seed": derive_run_seed(
                    int(config["master_seed"]),
                    run.source.dataset_split,
                    run.source.run_id,
                ),
                "artifact_provenance": run.artifacts,
            }
            for run in runs
        ],
        "methods": {
            method: _evaluate_method(
                runs,
                scores,
                method,
                config["thresholds"][method],
                include_scores=include_scores,
            )
            for method in METHOD_NAMES
        },
    }


def evaluate_detection_suite(
    runs: Sequence[DetectionSuiteRun],
    frozen_detection_config: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> dict[str, Any]:
    """Evaluate already-frozen thresholds without any formal split claim."""

    config = _validated_detection_config(
        frozen_detection_config, name="frozen_detection_config"
    )
    prepared = _prepare_suite(
        runs, expected_split=expected_split, require_three=False
    )
    return _evaluate_prepared_suite(prepared, config, include_scores=True)


def _threshold_candidates(
    runs: Sequence[_PreparedRun],
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]],
    method: str,
) -> list[float]:
    values = np.concatenate(
        [
            np.asarray(
                scores[run.source.run_id][method]["scores"], dtype=np.float64
            )[run.regions.evaluation_mask]
            for run in runs
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"method {method} has non-finite development scores")
    unique = np.unique(values)
    sentinel = float(np.nextafter(unique[-1], math.inf))
    if not math.isfinite(sentinel):
        raise ValueError(f"method {method} score range cannot produce a finite sentinel")
    return [float(value) for value in unique] + [sentinel]


def _delay_for_selection(result: Mapping[str, Any]) -> float:
    delay = result["events"]["mean_delay_frames"]
    return math.inf if delay is None else float(delay)


def _threshold_quality(result: Mapping[str, Any]) -> tuple[float, ...]:
    macro_f1 = result["macro"]["f1"]
    pooled_fpr = result["pooled"]["false_positive_rate"]
    if macro_f1 is None or pooled_fpr is None:
        raise ValueError("development threshold metrics require both classes")
    return (
        float(macro_f1),
        -float(pooled_fpr),
        float(result["events"]["detected_events"]),
        -_delay_for_selection(result),
        float(result["threshold"]),
    )


def _selection_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "threshold": result["threshold"],
        "per_run": result["per_run"],
        "pooled": result["pooled"],
        "macro": result["macro"],
        "exact_interval_sensitivity": result["exact_interval_sensitivity"],
        "events": result["events"],
        "diagnostics": result["diagnostics"],
    }


def _development_suite_sha256(runs: Sequence[_PreparedRun]) -> str:
    provenance = [_full_run_provenance(run) for run in runs]
    return hashlib.sha256(_canonical_json_bytes(provenance)).hexdigest()


def search_development_suite(
    runs: Sequence[DetectionSuiteRun],
    candidate_order: Sequence[Mapping[str, Any]],
    *,
    master_seed: int,
) -> dict[str, Any]:
    """Deterministically select one shared score config and four thresholds.

    Every candidate scores every development ledger independently.  Thresholds
    are optimized per method over the fixed masked development runs.  Only the
    selected ``combined`` result controls the shared score hyperparameters.
    """

    prepared = _prepare_suite(
        runs, expected_split="development", require_three=True
    )
    candidates = _validated_candidate_order(list(candidate_order))
    seed = _strict_integer(master_seed, name="master_seed", minimum=0)
    candidate_results: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        scores = _score_prepared_runs(prepared, candidate, seed)
        methods: dict[str, Any] = {}
        for method in METHOD_NAMES:
            thresholds = _threshold_candidates(prepared, scores, method)
            evaluated = [
                _evaluate_method(
                    prepared,
                    scores,
                    method,
                    threshold,
                    include_scores=False,
                )
                for threshold in thresholds
            ]
            selected = max(evaluated, key=_threshold_quality)
            methods[method] = {
                "threshold_candidate_count": len(thresholds),
                "threshold_candidates_sha256": hashlib.sha256(
                    _canonical_json_bytes(thresholds)
                ).hexdigest(),
                "selected": _selection_summary(selected),
            }
        candidate_results.append(
            {
                "candidate_index": candidate_index,
                "candidate": candidate,
                "methods": methods,
            }
        )

    def hyperparameter_quality(row: Mapping[str, Any]) -> tuple[float, ...]:
        combined = row["methods"]["combined"]["selected"]
        macro_f1 = combined["macro"]["f1"]
        pooled_fpr = combined["pooled"]["false_positive_rate"]
        macro_auroc = combined["macro"]["auroc"]
        if macro_f1 is None or pooled_fpr is None or macro_auroc is None:
            raise ValueError("combined development selection metrics are undefined")
        return (
            float(macro_f1),
            -float(pooled_fpr),
            float(combined["events"]["detected_events"]),
            -_delay_for_selection(combined),
            float(macro_auroc),
            -float(row["candidate_index"]),
        )

    selected_row = max(candidate_results, key=hyperparameter_quality)
    selected_candidate = selected_row["candidate"]
    frozen_config = {
        **selected_candidate,
        "master_seed": seed,
        "thresholds": {
            method: selected_row["methods"][method]["selected"]["threshold"]
            for method in METHOD_NAMES
        },
    }
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "evaluation_policy": fixed_evaluation_policy(),
        "development_run_ids": [run.source.run_id for run in prepared],
        "development_suite_sha256": _development_suite_sha256(prepared),
        "master_seed": seed,
        "run_seed_derivation": RUN_SEED_DERIVATION,
        "candidate_order": candidates,
        "threshold_candidate_rule": THRESHOLD_CANDIDATE_RULE,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "hyperparameter_selection_rule": HYPERPARAMETER_SELECTION_RULE,
        "selected_candidate_index": selected_row["candidate_index"],
        "frozen_detection_config": frozen_config,
        "candidates": candidate_results,
    }


def _validated_full_run_provenance(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    _expect_exact_keys(
        value,
        {
            "run_id",
            "dataset_split",
            "corruption_type",
            "raw_frame_sha256s",
            "source_pool_frame_sha256s",
            "source_pool_identities",
            "consumed_frames",
            "source_index_files",
            "official_lineage",
            "artifacts",
            "execution_provenance",
        },
        name=name,
    )
    base = _validated_run_commitment(
        {
            "run_id": value["run_id"],
            "dataset_split": value["dataset_split"],
            "corruption_type": value["corruption_type"],
            "raw_frame_sha256s": value["raw_frame_sha256s"],
            "source_pool_frame_sha256s": value[
                "source_pool_frame_sha256s"
            ],
            "source_pool_identities": value["source_pool_identities"],
            "consumed_frames": value["consumed_frames"],
            "source_index_files": value["source_index_files"],
            "official_lineage": value["official_lineage"],
            "artifacts": {
                key: value["artifacts"][key]
                for key in PRE_FORWARD_ARTIFACTS
                if isinstance(value.get("artifacts"), Mapping)
                and key in value["artifacts"]
            },
        },
        name=name,
    )
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{name}.artifacts must be a JSON object")
    _expect_exact_keys(artifacts, FULL_ARTIFACTS, name=f"{name}.artifacts")
    base["artifacts"] = {
        artifact_name: _validated_artifact(
            artifacts[artifact_name], name=f"{name}.artifacts.{artifact_name}"
        )
        for artifact_name in sorted(FULL_ARTIFACTS)
    }
    execution = value["execution_provenance"]
    if not isinstance(execution, Mapping):
        raise ValueError(f"{name}.execution_provenance must be a JSON object")
    expected_execution_keys = {
        "schema_version",
        "status",
        "baseline_commit",
        "checkpoint_sha256",
        "frame_count",
        "calibrated_update_calls",
        "expected_calibrated_update_calls",
        "size",
        "seed",
        "beta_base",
        "input_mode",
        "model_update_type",
        "runner_script_sha256",
        "runner_commit",
        "runtime_seconds",
        "peak_memory_allocated_mib",
        "trace_frame_steps",
        "health_signal_semantics",
    }
    _expect_exact_keys(
        execution,
        expected_execution_keys,
        name=f"{name}.execution_provenance",
    )
    # Reuse the same fixed-value validation shape as a run JSON.
    synthetic_run_json = {
        **dict(execution),
        "runner": {
            "script_sha256": execution["runner_script_sha256"],
            "commit": execution["runner_commit"],
        },
    }
    synthetic_run_json.pop("runner_script_sha256")
    synthetic_run_json.pop("runner_commit")
    base["execution_provenance"] = _validated_execution_provenance(
        synthetic_run_json, run_id=base["run_id"]
    )
    return base


def _validate_search_result(
    value: Mapping[str, Any], prepared: Sequence[_PreparedRun]
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "evaluation_policy",
        "development_run_ids",
        "development_suite_sha256",
        "master_seed",
        "run_seed_derivation",
        "candidate_order",
        "threshold_candidate_rule",
        "threshold_selection_rule",
        "hyperparameter_selection_rule",
        "selected_candidate_index",
        "frozen_detection_config",
        "candidates",
    }
    _expect_exact_keys(value, expected_keys, name="search_result")
    if value["schema_version"] != SEARCH_SCHEMA_VERSION:
        raise ValueError("search_result schema_version mismatch")
    if value["evaluation_policy"] != fixed_evaluation_policy():
        raise ValueError("search_result evaluation policy mismatch")
    if value["run_seed_derivation"] != RUN_SEED_DERIVATION:
        raise ValueError("search_result seed derivation mismatch")
    for field, expected in (
        ("threshold_candidate_rule", THRESHOLD_CANDIDATE_RULE),
        ("threshold_selection_rule", THRESHOLD_SELECTION_RULE),
        ("hyperparameter_selection_rule", HYPERPARAMETER_SELECTION_RULE),
    ):
        if value[field] != expected:
            raise ValueError(f"search_result {field} mismatch")
    candidate_order = _validated_candidate_order(value["candidate_order"])
    selected_index = _strict_integer(
        value["selected_candidate_index"], name="selected_candidate_index", minimum=0
    )
    if selected_index >= len(candidate_order):
        raise ValueError("selected_candidate_index is outside candidate_order")
    config = _validated_detection_config(
        value["frozen_detection_config"], name="frozen_detection_config"
    )
    expected_candidate = candidate_order[selected_index]
    if any(config[key] != expected_candidate[key] for key in expected_candidate):
        raise ValueError("frozen detection config does not match selected candidate")
    seed = _strict_integer(value["master_seed"], name="master_seed", minimum=0)
    if config["master_seed"] != seed:
        raise ValueError("frozen detection config master_seed mismatch")
    expected_run_ids = [run.source.run_id for run in prepared]
    if value["development_run_ids"] != expected_run_ids:
        raise ValueError("search_result development run IDs mismatch")
    expected_suite_sha = _development_suite_sha256(prepared)
    if value["development_suite_sha256"] != expected_suite_sha:
        raise ValueError("search_result development suite provenance mismatch")
    if not isinstance(value["candidates"], list) or len(value["candidates"]) != len(
        candidate_order
    ):
        raise ValueError("search_result candidates do not match candidate_order")
    return {
        "candidate_order": candidate_order,
        "selected_candidate_index": selected_index,
        "master_seed": seed,
        "frozen_detection_config": config,
    }


def build_formal_v1_config(
    development_runs: Sequence[DetectionSuiteRun],
    search_result: Mapping[str, Any],
    *,
    development_run_id: str,
    split_registry: bytes,
    holdout_commitment: bytes,
) -> dict[str, Any]:
    """Bind a completed dev search to pre-forward split/holdout commitments."""

    prepared = _prepare_suite(
        development_runs, expected_split="development", require_three=True
    )
    registry = _parse_split_registry(split_registry)
    commitment = _parse_holdout_commitment(holdout_commitment)
    if registry["holdout"] != commitment["runs"]:
        raise ValueError("holdout commitment does not match split registry")
    actual_dev = [_run_commitment_from_prepared(run) for run in prepared]
    registry_dev_by_id = {run["run_id"]: run for run in registry["development"]}
    if {run["run_id"] for run in actual_dev} != set(registry_dev_by_id):
        raise ValueError("development runs do not match split registry run IDs")
    for run in actual_dev:
        if run != registry_dev_by_id[run["run_id"]]:
            raise ValueError(
                f"development run {run['run_id']!r} does not match split commitment"
            )
    validated_search = _validate_search_result(search_result, prepared)
    _require_formal_search_protocol(
        validated_search["candidate_order"],
        validated_search["master_seed"],
        name="search_result",
    )
    recomputed_search = search_development_suite(
        [run.source for run in prepared],
        validated_search["candidate_order"],
        master_seed=validated_search["master_seed"],
    )
    if _canonical_json_bytes(search_result) != _canonical_json_bytes(
        recomputed_search
    ):
        raise ValueError("search_result does not match deterministic recomputation")
    recomputed_frozen_config = _validated_detection_config(
        recomputed_search["frozen_detection_config"],
        name="recomputed_search.frozen_detection_config",
    )
    search_sha = hashlib.sha256(_canonical_json_bytes(search_result)).hexdigest()
    frozen_config_sha = hashlib.sha256(
        _canonical_json_bytes(recomputed_frozen_config)
    ).hexdigest()
    return {
        "schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
        "dataset_split": "development",
        "development_run_id": _nonempty_string(
            development_run_id, name="development_run_id"
        ),
        "split_registry_sha256": hashlib.sha256(split_registry).hexdigest(),
        "holdout_commitment_sha256": hashlib.sha256(
            holdout_commitment
        ).hexdigest(),
        "evaluation_policy": fixed_evaluation_policy(),
        "search_provenance": {
            "search_result_sha256": search_sha,
            "frozen_detection_config_canonical_sha256": frozen_config_sha,
            "candidate_order": validated_search["candidate_order"],
            "threshold_candidate_rule": THRESHOLD_CANDIDATE_RULE,
            "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
            "hyperparameter_selection_rule": HYPERPARAMETER_SELECTION_RULE,
            "run_seed_derivation": RUN_SEED_DERIVATION,
            "master_seed": validated_search["master_seed"],
        },
        "development_runs": [_full_run_provenance(run) for run in prepared],
        "frozen_detection_config": recomputed_frozen_config,
    }


def _validate_formal_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("formal_config must be a JSON object")
    expected_keys = {
        "schema_version",
        "dataset_split",
        "development_run_id",
        "split_registry_sha256",
        "holdout_commitment_sha256",
        "evaluation_policy",
        "search_provenance",
        "development_runs",
        "frozen_detection_config",
    }
    _expect_exact_keys(value, expected_keys, name="formal_config")
    if value["schema_version"] != FORMAL_CONFIG_SCHEMA_VERSION:
        raise ValueError("formal_config schema_version mismatch")
    if value["dataset_split"] != "development":
        raise ValueError("formal_config dataset_split must be development")
    _nonempty_string(value["development_run_id"], name="development_run_id")
    split_sha = _validated_sha256(
        value["split_registry_sha256"], name="split_registry_sha256"
    )
    commitment_sha = _validated_sha256(
        value["holdout_commitment_sha256"], name="holdout_commitment_sha256"
    )
    if value["evaluation_policy"] != fixed_evaluation_policy():
        raise ValueError("formal_config evaluation policy mismatch")

    search = value["search_provenance"]
    if not isinstance(search, Mapping):
        raise ValueError("search_provenance must be a JSON object")
    search_keys = {
        "search_result_sha256",
        "frozen_detection_config_canonical_sha256",
        "candidate_order",
        "threshold_candidate_rule",
        "threshold_selection_rule",
        "hyperparameter_selection_rule",
        "run_seed_derivation",
        "master_seed",
    }
    _expect_exact_keys(search, search_keys, name="search_provenance")
    _validated_sha256(
        search["search_result_sha256"], name="search_result_sha256"
    )
    frozen_config_sha = _validated_sha256(
        search["frozen_detection_config_canonical_sha256"],
        name="frozen_detection_config_canonical_sha256",
    )
    candidates = _validated_candidate_order(search["candidate_order"])
    for field, expected in (
        ("threshold_candidate_rule", THRESHOLD_CANDIDATE_RULE),
        ("threshold_selection_rule", THRESHOLD_SELECTION_RULE),
        ("hyperparameter_selection_rule", HYPERPARAMETER_SELECTION_RULE),
        ("run_seed_derivation", RUN_SEED_DERIVATION),
    ):
        if search[field] != expected:
            raise ValueError(f"formal_config {field} mismatch")
    master_seed = _strict_integer(
        search["master_seed"], name="search_provenance.master_seed", minimum=0
    )
    _require_formal_search_protocol(
        candidates, master_seed, name="formal_config search_provenance"
    )
    frozen = _validated_detection_config(
        value["frozen_detection_config"], name="frozen_detection_config"
    )
    if hashlib.sha256(_canonical_json_bytes(frozen)).hexdigest() != frozen_config_sha:
        raise ValueError(
            "formal_config frozen detection config canonical SHA mismatch"
        )
    if frozen["master_seed"] != master_seed:
        raise ValueError("formal_config master seed mismatch")
    if not any(
        all(frozen[key] == candidate[key] for key in candidate)
        for candidate in candidates
    ):
        raise ValueError("formal_config frozen hyperparameters are outside candidate_order")

    development_runs = value["development_runs"]
    if not isinstance(development_runs, list) or len(development_runs) != 3:
        raise ValueError("formal_config must contain three development runs")
    validated_dev = [
        _validated_full_run_provenance(run, name=f"development_runs[{index}]")
        for index, run in enumerate(development_runs)
    ]
    types = [run["corruption_type"] for run in validated_dev]
    if set(types) != CORRUPTION_TYPES or len(set(types)) != 3:
        raise ValueError("formal_config development runs must cover three types")
    if any(run["dataset_split"] != "development" for run in validated_dev):
        raise ValueError("formal_config development run has wrong split")
    if len(
        {
            run["execution_provenance"]["runner_commit"]
            for run in validated_dev
        }
    ) != 1:
        raise ValueError("formal_config development runner commits differ")
    _assert_pairwise_disjoint_source_pools(
        validated_dev, name="formal_config.development"
    )
    return {
        "development_run_id": value["development_run_id"],
        "split_registry_sha256": split_sha,
        "holdout_commitment_sha256": commitment_sha,
        "development_runs": validated_dev,
        "frozen_detection_config": frozen,
        "candidate_order": candidates,
    }


def evaluate_formal_holdout(
    holdout_runs: Sequence[DetectionSuiteRun],
    formal_config: Mapping[str, Any],
    *,
    split_registry: bytes,
    holdout_commitment: bytes,
    search_result: Mapping[str, Any],
    development_runs: Sequence[DetectionSuiteRun],
    runtime_detection_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every frozen binding, then evaluate the untouched holdout.

    All configuration, exact-byte hashes, manifest/run links, raw-frame sets,
    and run shapes are checked before the first call to the causal scorer.
    ``runtime_detection_config`` models CLI overrides and, when supplied, must
    equal the frozen values exactly after validation.
    """

    validated = _validate_formal_config(formal_config)
    if hashlib.sha256(split_registry).hexdigest() != validated["split_registry_sha256"]:
        raise ValueError("actual split registry SHA does not match formal config")
    if (
        hashlib.sha256(holdout_commitment).hexdigest()
        != validated["holdout_commitment_sha256"]
    ):
        raise ValueError("actual holdout commitment SHA does not match formal config")
    rebuilt_config = build_formal_v1_config(
        development_runs,
        search_result,
        development_run_id=validated["development_run_id"],
        split_registry=split_registry,
        holdout_commitment=holdout_commitment,
    )
    if _canonical_json_bytes(formal_config) != _canonical_json_bytes(rebuilt_config):
        raise ValueError(
            "formal_config does not match exact deterministic development recalibration"
        )
    registry = _parse_split_registry(split_registry)
    commitment = _parse_holdout_commitment(holdout_commitment)
    if registry["holdout"] != commitment["runs"]:
        raise ValueError("holdout commitment does not match split registry")

    config_dev_by_id = {
        run["run_id"]: run for run in validated["development_runs"]
    }
    registry_dev_by_id = {run["run_id"]: run for run in registry["development"]}
    if set(config_dev_by_id) != set(registry_dev_by_id):
        raise ValueError("formal development provenance run IDs mismatch registry")
    for run_id, full in config_dev_by_id.items():
        projected = {
            "run_id": full["run_id"],
            "dataset_split": full["dataset_split"],
            "corruption_type": full["corruption_type"],
            "raw_frame_sha256s": full["raw_frame_sha256s"],
            "source_pool_frame_sha256s": full[
                "source_pool_frame_sha256s"
            ],
            "source_pool_identities": full["source_pool_identities"],
            "consumed_frames": full["consumed_frames"],
            "source_index_files": full["source_index_files"],
            "official_lineage": full["official_lineage"],
            "artifacts": {
                name: full["artifacts"][name]
                for name in sorted(PRE_FORWARD_ARTIFACTS)
            },
        }
        if projected != registry_dev_by_id[run_id]:
            raise ValueError(f"formal development provenance mismatch for {run_id!r}")
    _assert_split_disjoint(validated["development_runs"], registry["holdout"])

    # Prepare every actual run, including all four artifact snapshots, before
    # score construction so provenance failures are fail-fast.
    prepared = _prepare_suite(
        holdout_runs, expected_split="holdout", require_three=True
    )
    development_runner_commit = validated["development_runs"][0][
        "execution_provenance"
    ]["runner_commit"]
    if any(
        run.execution_provenance["runner_commit"] != development_runner_commit
        for run in prepared
    ):
        raise ValueError("holdout runner commit differs from development")
    actual_by_id = {
        run.source.run_id: _run_commitment_from_prepared(run) for run in prepared
    }
    expected_by_id = {run["run_id"]: run for run in commitment["runs"]}
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("actual holdout run IDs do not match commitment")
    for run_id, actual in actual_by_id.items():
        if actual != expected_by_id[run_id]:
            raise ValueError(f"actual holdout provenance mismatch for {run_id!r}")

    frozen = validated["frozen_detection_config"]
    if runtime_detection_config is not None:
        runtime = _validated_detection_config(
            runtime_detection_config, name="runtime_detection_config"
        )
        if runtime != frozen:
            raise ValueError("runtime detection config does not match frozen config")

    result = _evaluate_prepared_suite(prepared, frozen, include_scores=True)
    result.update(
        {
            "formal": True,
            "dataset_split": "holdout",
            "formal_provenance": {
                "formal_config_canonical_sha256": hashlib.sha256(
                    _canonical_json_bytes(formal_config)
                ).hexdigest(),
                "split_registry_sha256": validated["split_registry_sha256"],
                "holdout_commitment_sha256": validated[
                    "holdout_commitment_sha256"
                ],
                "actual_holdout_runs": [
                    _full_run_provenance(run) for run in prepared
                ],
            },
        }
    )
    return result


__all__ = [
    "CORRUPTION_TYPES",
    "EVENT_START",
    "FORMAL_CONFIG_SCHEMA_VERSION",
    "FRAME_COUNT",
    "HOLDOUT_COMMITMENT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SEARCH_SCHEMA_VERSION",
    "SPLIT_REGISTRY_SCHEMA_VERSION",
    "WASHOUT_LENGTH",
    "DetectionSuiteInput",
    "DetectionSuiteRun",
    "EvaluationRegions",
    "build_evaluation_regions",
    "build_formal_v1_config",
    "derive_run_seed",
    "evaluate_detection_suite",
    "evaluate_formal_holdout",
    "fixed_candidate_order",
    "fixed_evaluation_policy",
    "make_holdout_commitment",
    "make_run_commitment",
    "make_split_registry",
    "search_development_suite",
]
