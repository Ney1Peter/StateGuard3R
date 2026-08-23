from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.state_triage_capsule_v2 import (
    COORDINATE_REFERENCE,
    SCHEMA_VERSION,
    Stage0CapsuleError,
    apply_stage0_transforms,
    load_stage0_capsule,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)


def _capsule(tmp_path: Path, *, cause: str = "transient_local_content") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    image_a.chmod(0o444)
    image_b.chmod(0o444)
    coverage = "novel" if cause == "normal_novelty" else "covered"
    transform = []
    if cause == "transient_local_content":
        transform = [{"type": "rectangle_occlusion", "coordinate_reference": COORDINATE_REFERENCE, "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5}, "fill": [1.0, -1.0, -1.0]}]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "capsule_id": "unit-capsule",
        "source_sequence": "unit-sequence",
        "source_group": "unit-group",
        "recipe_id": "unit-recipe",
        "event": {"event_id": "unit-event", "cause": cause, "start_frame": 0, "end_frame": 1, "coverage_expectation": coverage, "label_provenance": "pre_forward_deterministic_recipe"},
        "online_evidence_contract": "current_frame_and_strict_prefix_only",
        "frames": [
            {"frame_id": 0, "path": str(image_a), "sha256": hashlib.sha256(b"a").hexdigest(), "timestamp": 1.0, "transforms": transform},
            {"frame_id": 1, "path": str(image_b), "sha256": hashlib.sha256(b"b").hexdigest(), "timestamp": 2.0, "transforms": transform},
        ],
    }
    path = tmp_path / "capsule.json"
    _write(path, payload)
    return path


def test_loads_and_rehashes_frozen_stage0_capsule(tmp_path: Path) -> None:
    path = _capsule(tmp_path)
    capsule = load_stage0_capsule(path)

    assert capsule.event.cause == "transient_local_content"
    assert len(capsule.frames) == 2
    assert capsule.frames[0].transforms[0]["type"] == "rectangle_occlusion"


def test_rejects_label_coverage_mismatch_and_changed_rgb(tmp_path: Path) -> None:
    path = _capsule(tmp_path, cause="normal_novelty")
    payload = json.loads(path.read_text())
    path.chmod(0o644)
    payload["event"]["coverage_expectation"] = "covered"
    _write(path, payload)
    with pytest.raises(Stage0CapsuleError, match="requires coverage_expectation"):
        load_stage0_capsule(path)

    path = _capsule(tmp_path / "changed", cause="bad_observation")
    image = tmp_path / "changed" / "a.png"
    image.chmod(0o644)
    image.write_bytes(b"changed")
    image.chmod(0o444)
    with pytest.raises(Stage0CapsuleError, match="changed from capsule"):
        load_stage0_capsule(path)


def test_materializer_clones_and_only_changes_declared_rectangle(tmp_path: Path) -> None:
    capsule = load_stage0_capsule(_capsule(tmp_path))
    first = np.zeros((1, 3, 8, 8), dtype=np.float64)
    second = np.zeros((1, 3, 8, 8), dtype=np.float64)
    views = [{"img": first}, {"img": second}]

    transformed = apply_stage0_transforms(views, capsule)

    assert transformed[0]["img"] is not first
    np.testing.assert_array_equal(first, np.zeros_like(first))
    np.testing.assert_allclose(transformed[0]["img"][:, 0, 2:6, 2:6], 1.0)
    np.testing.assert_allclose(transformed[0]["img"][:, 1:, 2:6, 2:6], -1.0)


def test_rejects_writable_capsule(tmp_path: Path) -> None:
    path = _capsule(tmp_path)
    path.chmod(0o644)
    with pytest.raises(Stage0CapsuleError, match="read-only"):
        load_stage0_capsule(path)
