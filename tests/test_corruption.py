from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import stateguard3r.corruption as corruption_module
from stateguard3r.corruption import (
    CorruptionManifestError,
    build_corruption_manifest,
    generate_corruption_manifest,
    main,
)


def _make_source_manifest(
    tmp_path: Path, frame_count: int = 16
) -> tuple[Path, dict[str, object], list[Path]]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths = []
    frames = []
    for index in range(frame_count):
        frame_path = frame_dir / f"frame_{index:03d}.png"
        frame_path.write_bytes(f"unchanged-frame-{index}".encode())
        frame_paths.append(frame_path)
        frames.append(
            {
                "path": str(frame_path.relative_to(tmp_path)),
                "timestamp": round(index / 10.0, 3),
            }
        )
    ground_truth_path = tmp_path / "groundtruth.txt"
    ground_truth_path.write_text("# immutable synthetic GT\n0.0 0 0 0 0 0 0 1\n")
    manifest: dict[str, object] = {
        "sequence": "clean_sequence",
        "frames": frames,
        "camera": "test-camera",
        "ground_truth_path": ground_truth_path.name,
    }
    manifest_path = tmp_path / "source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, manifest, frame_paths


def _file_state(paths: list[Path]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_all_three_corruptions_have_labels_and_deferred_transforms(
    tmp_path: Path,
) -> None:
    _, source, frame_paths = _make_source_manifest(tmp_path)
    source_before = copy.deepcopy(source)
    files_before = _file_state(frame_paths)
    specs = [
        {
            "type": "low_overlap_jump",
            "start": 1,
            "end": 2,
            "parameters": {"source_start": 12},
        },
        {
            "type": "dynamic_occlusion",
            "start": 4,
            "end": 6,
            "parameters": {
                "rectangle": {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.3},
                "velocity": {"dx": 0.05, "dy": -0.02},
                "fill": [7, 8, 9],
            },
        },
        {
            "type": "wrong_order_segment",
            "start": 8,
            "end": 11,
            "parameters": {"mode": "shuffle"},
        },
    ]

    result = build_corruption_manifest(
        source,
        specs,
        seed=1234,
        check_paths=True,
        source_base_dir=tmp_path,
    )

    assert source == source_before
    assert _file_state(frame_paths) == files_before
    assert result["source_is_read_only"] is True
    assert result["materialization"] == "deferred_transforms_no_image_copy"
    assert result["frame_count"] == len(source["frames"])
    assert [label["type"] for label in result["corruptions"]] == [
        "low_overlap_jump",
        "dynamic_occlusion",
        "wrong_order_segment",
    ]
    for label in result["corruptions"]:
        assert set(label) == {
            "type",
            "start_frame",
            "end_frame",
            "start",
            "end",
            "parameters",
            "expected_effect",
        }
        assert label["start_frame"] == label["start"]
        assert label["end_frame"] == label["end"]
        assert label["expected_effect"]

    assert [result["frames"][index]["source_index"] for index in (1, 2)] == [
        12,
        13,
    ]
    assert result["frames"][1]["transforms"] == [
        {
            "type": "source_frame_substitution",
            "corruption": "low_overlap_jump",
            "original_source_index": 1,
            "replacement_source_index": 12,
        }
    ]

    rectangle_transforms = [
        result["frames"][index]["transforms"][0] for index in range(4, 7)
    ]
    assert [transform["rectangle"] for transform in rectangle_transforms] == [
        {"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.3},
        {"x": 0.15, "y": 0.18, "width": 0.2, "height": 0.3},
        {"x": 0.2, "y": 0.16, "width": 0.2, "height": 0.3},
    ]
    assert all(
        transform["type"] == "rectangle_occlusion"
        and transform["coordinate_space"] == "normalized"
        and transform["fill"] == [7, 8, 9]
        for transform in rectangle_transforms
    )

    wrong_order = [result["frames"][index]["source_index"] for index in range(8, 12)]
    assert set(wrong_order) == set(range(8, 12))
    assert wrong_order != list(range(8, 12))
    assert result["corruptions"][2]["parameters"]["permutation"] == wrong_order
    assert [result["frames"][index]["source_index"] for index in range(4, 7)] == [
        4,
        5,
        6,
    ]
    affected = {1, 2, 4, 5, 6, 8, 9, 10, 11}
    for index, frame in enumerate(result["frames"]):
        if index not in affected:
            assert frame["source_index"] == index
            assert frame["transforms"] == []


def test_fixed_seed_makes_generated_parameters_and_shuffle_reproducible() -> None:
    source = {
        "sequence": "seed-test",
        "frame_paths": [f"frame-{index}.png" for index in range(12)],
    }
    specs = [
        {"type": "dynamic_occlusion", "start_frame": 1, "end_frame": 3},
        {"type": "wrong_order_segment", "start": 6, "end": 10},
    ]

    first = build_corruption_manifest(source, specs, seed=99)
    second = build_corruption_manifest(source, specs, seed=99)
    different_seed = build_corruption_manifest(source, specs, seed=100)

    assert first == second
    assert first != different_seed
    assert first["seed"] == 99
    dynamic_parameters = first["corruptions"][0]["parameters"]
    assert dynamic_parameters["rectangle_generated_from_seed"] is True
    assert dynamic_parameters["velocity_generated_from_seed"] is True
    assert len(dynamic_parameters["frame_rectangles"]) == 3


def test_generate_and_cli_leave_source_manifest_and_frames_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_path, _, frame_paths = _make_source_manifest(tmp_path, frame_count=8)
    source_before = (source_path.read_bytes(), source_path.stat().st_mtime_ns)
    frames_before = _file_state(frame_paths)
    ground_truth_path = tmp_path / "groundtruth.txt"
    ground_truth_before = (
        ground_truth_path.read_bytes(),
        ground_truth_path.stat().st_mtime_ns,
    )
    output_path = tmp_path / "generated" / "corrupted.json"
    spec = {
        "type": "wrong_order_segment",
        "start": 2,
        "end": 5,
        "parameters": {"mode": "reverse"},
    }

    exit_code = main(
        [
            str(source_path),
            str(output_path),
            "--spec",
            json.dumps(spec),
            "--seed",
            "17",
            "--check-paths",
        ]
    )

    assert exit_code == 0
    assert "1 corruption(s)" in capsys.readouterr().out
    assert (source_path.read_bytes(), source_path.stat().st_mtime_ns) == source_before
    assert _file_state(frame_paths) == frames_before
    assert (
        ground_truth_path.read_bytes(),
        ground_truth_path.stat().st_mtime_ns,
    ) == ground_truth_before
    generated = json.loads(output_path.read_text(encoding="utf-8"))
    assert generated["source_is_read_only"] is True
    assert generated["source_manifest"] == str(source_path.resolve())
    assert generated["source_manifest_sha256"] == hashlib.sha256(
        source_before[0]
    ).hexdigest()
    assert generated["source_ground_truth_path"] == ground_truth_path.name
    assert generated["source_ground_truth_resolved_path"] == str(
        ground_truth_path.resolve()
    )
    assert generated["source_ground_truth_sha256"] == hashlib.sha256(
        ground_truth_before[0]
    ).hexdigest()
    assert [generated["frames"][index]["source_index"] for index in range(2, 6)] == [
        5,
        4,
        3,
        2,
    ]
    assert sorted(tmp_path.rglob("*.png")) == sorted(frame_paths)


def test_generate_rejects_overwriting_source_manifest(tmp_path: Path) -> None:
    source_path, _, _ = _make_source_manifest(tmp_path, frame_count=4)
    source_before = source_path.read_bytes()

    with pytest.raises(CorruptionManifestError, match="must not overwrite"):
        generate_corruption_manifest(
            source_path,
            source_path,
            [{"type": "wrong_order_segment", "start": 1, "end": 2}],
        )

    assert source_path.read_bytes() == source_before


def test_generate_rejects_output_that_aliases_a_source_frame(tmp_path: Path) -> None:
    source_path, _, frame_paths = _make_source_manifest(tmp_path, frame_count=4)
    frames_before = _file_state(frame_paths)

    with pytest.raises(CorruptionManifestError, match="read-only source frame"):
        generate_corruption_manifest(
            source_path,
            frame_paths[0],
            [{"type": "wrong_order_segment", "start": 1, "end": 2}],
            overwrite=True,
        )

    assert _file_state(frame_paths) == frames_before


@pytest.mark.parametrize(
    "ground_truth_kind",
    ["relative", "absolute", "symlink", "hardlink"],
)
def test_generate_rejects_overwriting_any_source_ground_truth_alias(
    tmp_path: Path, ground_truth_kind: str
) -> None:
    source_path, source, frame_paths = _make_source_manifest(tmp_path, frame_count=4)
    ground_truth_path = tmp_path / "groundtruth.txt"

    if ground_truth_kind == "relative":
        source["ground_truth_path"] = ground_truth_path.name
    elif ground_truth_kind == "absolute":
        source["ground_truth_path"] = str(ground_truth_path.resolve())
    elif ground_truth_kind == "symlink":
        alias = tmp_path / "groundtruth-symlink.txt"
        alias.symlink_to(ground_truth_path.name)
        source["ground_truth_path"] = alias.name
    else:
        alias = tmp_path / "groundtruth-hardlink.txt"
        alias.hardlink_to(ground_truth_path)
        source["ground_truth_path"] = alias.name

    source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    source_before = (source_path.read_bytes(), source_path.stat().st_mtime_ns)
    frames_before = _file_state(frame_paths)
    ground_truth_before = (
        ground_truth_path.read_bytes(),
        ground_truth_path.stat().st_mtime_ns,
    )

    with pytest.raises(CorruptionManifestError, match="source ground truth"):
        generate_corruption_manifest(
            source_path,
            ground_truth_path,
            [{"type": "wrong_order_segment", "start": 1, "end": 2}],
            overwrite=True,
        )

    assert (source_path.read_bytes(), source_path.stat().st_mtime_ns) == source_before
    assert _file_state(frame_paths) == frames_before
    assert (
        ground_truth_path.read_bytes(),
        ground_truth_path.stat().st_mtime_ns,
    ) == ground_truth_before


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    source_path, _, _ = _make_source_manifest(tmp_path, frame_count=4)
    output_path = tmp_path / "existing.json"
    output_path.write_text("keep-existing-output\n", encoding="utf-8")

    with pytest.raises(CorruptionManifestError, match="already exists"):
        generate_corruption_manifest(
            source_path,
            output_path,
            [{"type": "wrong_order_segment", "start": 1, "end": 2}],
        )

    assert output_path.read_text(encoding="utf-8") == "keep-existing-output\n"


def test_atomic_overwrite_failure_preserves_existing_output_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, _, _ = _make_source_manifest(tmp_path, frame_count=4)
    output_path = tmp_path / "existing.json"
    output_path.write_text("known-good-output\n", encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(corruption_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        generate_corruption_manifest(
            source_path,
            output_path,
            [{"type": "wrong_order_segment", "start": 1, "end": 2}],
            overwrite=True,
        )

    assert output_path.read_text(encoding="utf-8") == "known-good-output\n"
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {"type": "dynamic_occlusion", "start": -1, "end": 1},
            "invalid inclusive range",
        ),
        (
            {"type": "dynamic_occlusion", "start": 2, "end": 1},
            "invalid inclusive range",
        ),
        (
            {"type": "dynamic_occlusion", "start": 0, "end": 5},
            "invalid inclusive range",
        ),
        (
            {"type": "wrong_order_segment", "start": 2, "end": 2},
            "at least two frames",
        ),
        (
            {
                "type": "low_overlap_jump",
                "start": 1,
                "end": 2,
                "parameters": {"source_start": 2},
            },
            "must not overlap",
        ),
    ],
)
def test_invalid_ranges_and_segments_are_rejected(
    spec: dict[str, object], message: str
) -> None:
    source = {"sequence": "short", "frame_paths": [f"{index}.png" for index in range(5)]}

    with pytest.raises(CorruptionManifestError, match=message):
        build_corruption_manifest(source, [spec])


def test_path_existence_validation_is_optional(tmp_path: Path) -> None:
    source = {"sequence": "missing", "frame_paths": ["does-not-exist.png"]}
    spec = {"type": "dynamic_occlusion", "start": 0, "end": 0}

    result = build_corruption_manifest(
        source, [spec], source_base_dir=tmp_path, check_paths=False
    )
    assert result["frames"][0]["path"] == "does-not-exist.png"

    with pytest.raises(CorruptionManifestError, match="does not exist"):
        build_corruption_manifest(
            source, [spec], source_base_dir=tmp_path, check_paths=True
        )


@pytest.mark.parametrize(
    "fill",
    (
        [0, 0],
        [0, 0, 256],
        [0, False, 0],
        [0, float("nan"), 0],
        "black",
    ),
)
def test_dynamic_occlusion_rejects_non_rgb_fill(fill: object) -> None:
    source = {
        "sequence": "fill-validation",
        "frame_paths": ["0.png", "1.png"],
    }

    with pytest.raises(CorruptionManifestError, match="fill"):
        build_corruption_manifest(
            source,
            [
                {
                    "type": "dynamic_occlusion",
                    "start": 0,
                    "end": 1,
                    "parameters": {"fill": fill},
                }
            ],
        )


def test_generate_rejects_non_object_source_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    source_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(CorruptionManifestError, match="JSON object"):
        generate_corruption_manifest(
            source_path,
            tmp_path / "output.json",
            [{"type": "dynamic_occlusion", "start": 0, "end": 0}],
        )


def test_check_paths_also_validates_declared_ground_truth(tmp_path: Path) -> None:
    source = {
        "sequence": "missing-gt",
        "frame_paths": ["frame.png"],
        "ground_truth_path": "missing-groundtruth.txt",
    }
    (tmp_path / "frame.png").write_bytes(b"frame")

    with pytest.raises(CorruptionManifestError, match="ground_truth_path"):
        build_corruption_manifest(
            source,
            [{"type": "dynamic_occlusion", "start": 0, "end": 0}],
            source_base_dir=tmp_path,
            check_paths=True,
        )
