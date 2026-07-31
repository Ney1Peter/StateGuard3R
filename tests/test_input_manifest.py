from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import stateguard3r.input_manifest as input_manifest_module
from stateguard3r.corruption import generate_corruption_manifest
from stateguard3r.input_manifest import (
    InputManifestError,
    apply_deferred_transforms,
    load_input_manifest,
    materialize_manifest_views,
)


def _generated_manifest(tmp_path: Path) -> tuple[Path, list[Path]]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_paths: list[Path] = []
    frames: list[dict[str, object]] = []
    for index in range(6):
        path = frame_dir / f"frame_{index}.png"
        path.write_bytes(f"immutable-{index}".encode())
        frame_paths.append(path)
        frames.append({"path": f"frames/{path.name}", "timestamp": index / 10})
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps({"sequence": "tiny", "frames": frames}), encoding="utf-8"
    )
    output_path = tmp_path / "corruption.json"
    generate_corruption_manifest(
        source_path,
        output_path,
        [
            {
                "type": "low_overlap_jump",
                "start": 0,
                "end": 0,
                "parameters": {"source_start": 5},
            },
            {
                "type": "dynamic_occlusion",
                "start": 2,
                "end": 2,
                "parameters": {
                    "rectangle": {"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5},
                    "velocity": {"dx": 0.0, "dy": 0.0},
                    "fill": [255, 127.5, 0],
                },
            },
            {
                "type": "wrong_order_segment",
                "start": 3,
                "end": 4,
                "parameters": {"mode": "reverse"},
            },
        ],
        check_paths=True,
    )
    return output_path, frame_paths


def _file_state(paths: list[Path]) -> dict[Path, tuple[bytes, int]]:
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_materializer_preserves_final_order_clones_and_applies_rgb_fill(
    tmp_path: Path,
) -> None:
    manifest_path, frame_paths = _generated_manifest(tmp_path)
    before = _file_state(frame_paths)
    seen_paths: list[str] = []
    loaded: list[dict[str, object]] = []

    def loader(paths: list[str]):
        seen_paths.extend(paths)
        for path in paths:
            source_index = int(Path(path).stem.split("_")[-1])
            loaded.append(
                {"img": np.full((1, 3, 4, 4), source_index / 10, dtype=np.float32)}
            )
        return loaded

    output = materialize_manifest_views(manifest_path, loader)

    assert [Path(path).name for path in seen_paths] == [
        "frame_5.png",
        "frame_1.png",
        "frame_2.png",
        "frame_4.png",
        "frame_3.png",
        "frame_5.png",
    ]
    assert len(output) == 6
    assert all(result["img"] is not source["img"] for result, source in zip(output, loaded))
    np.testing.assert_allclose(output[0]["img"], 0.5)
    np.testing.assert_allclose(output[3]["img"], 0.4)
    expected = np.full((1, 3, 4, 4), 0.2, dtype=np.float32)
    expected[:, 0, 1:3, 1:3] = 1.0
    expected[:, 1, 1:3, 1:3] = 0.0
    expected[:, 2, 1:3, 1:3] = -1.0
    np.testing.assert_allclose(output[2]["img"], expected)
    np.testing.assert_allclose(loaded[2]["img"], 0.2)
    assert _file_state(frame_paths) == before
    assert sorted(tmp_path.rglob("*.png")) == frame_paths


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.clone_calls = 0

    @property
    def shape(self):
        return self.array.shape

    def clone(self):
        self.clone_calls += 1
        return _FakeTensor(self.array.copy())

    def __setitem__(self, key, value):
        self.array[key] = value


class _IntegerTensorLike(_FakeTensor):
    def is_floating_point(self):
        return False


def test_torch_like_view_uses_clone_without_importing_torch(tmp_path: Path) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)
    manifest = load_input_manifest(manifest_path)
    tensors = [_FakeTensor(np.zeros((1, 3, 4, 4), dtype=np.float32)) for _ in manifest.frames]

    output = apply_deferred_transforms(
        [{"img": tensor} for tensor in tensors], manifest
    )

    assert all(tensor.clone_calls == 1 for tensor in tensors)
    assert all(result["img"] is not source for result, source in zip(output, tensors))
    np.testing.assert_allclose(output[2]["img"].array[:, 0, 1:3, 1:3], 1.0)
    tree = ast.parse(Path(input_manifest_module.__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported_roots


def test_integer_tensor_like_view_is_rejected(tmp_path: Path) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)
    manifest = load_input_manifest(manifest_path)
    views = [
        {"img": _FakeTensor(np.zeros((1, 3, 4, 4), dtype=np.float32))}
        for _ in manifest.frames
    ]
    views[0] = {
        "img": _IntegerTensorLike(np.zeros((1, 3, 4, 4), dtype=np.int32))
    }

    with pytest.raises(InputManifestError, match="floating dtype"):
        apply_deferred_transforms(views, manifest)


def _mutate_manifest(path: Path, mutation) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "schema_version",
    [None, "stateguard3r.corruption.v0", "stateguard3r.corruption.v2"],
)
def test_loader_accepts_only_current_v1_schema(
    tmp_path: Path, schema_version: str | None
) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)
    _mutate_manifest(
        manifest_path,
        lambda payload: payload.__setitem__("schema_version", schema_version),
    )

    with pytest.raises(
        InputManifestError,
        match=r"expected 'stateguard3r\.corruption\.v1'",
    ):
        load_input_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda transform: transform.pop("coordinate_reference"),
            "unexpected fields",
        ),
        (
            lambda transform: transform.__setitem__(
                "coordinate_reference", "source_image_before_model_preprocessing"
            ),
            "unsupported coordinate reference",
        ),
    ],
)
def test_v1_requires_exact_rectangle_coordinate_reference(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)

    def mutate_rectangle(payload):
        mutation(payload["frames"][2]["transforms"][0])

    _mutate_manifest(manifest_path, mutate_rectangle)
    with pytest.raises(InputManifestError, match=message):
        load_input_manifest(manifest_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda parameters: parameters.pop("coordinate_reference"),
        lambda parameters: parameters.__setitem__(
            "coordinate_reference", "source_image_before_model_preprocessing"
        ),
    ],
)
def test_v1_requires_label_coordinate_reference(
    tmp_path: Path, mutation
) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)

    def mutate_label(payload):
        mutation(payload["corruptions"][1]["parameters"])

    _mutate_manifest(manifest_path, mutate_label)
    with pytest.raises(InputManifestError, match="unsupported coordinate reference"):
        load_input_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["frames"][1].__setitem__("frame_index", 9), "out of order"),
        (lambda payload: payload["frames"][1].__setitem__("path", "missing.png"), "does not exist"),
        (
            lambda payload: payload["frames"][0]["transforms"][0].__setitem__(
                "type", "unknown_transform"
            ),
            "unknown transform",
        ),
        (lambda payload: payload["frames"][2].__setitem__("transforms", []), "labels"),
        (
            lambda payload: payload["frames"][0]["transforms"][0].__setitem__(
                "replacement_source_index", 4
            ),
            "indices disagree",
        ),
    ],
)
def test_strict_manifest_validation_rejects_misalignment(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)
    _mutate_manifest(manifest_path, mutation)

    with pytest.raises(InputManifestError, match=message):
        load_input_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rectangle", {"x": 0.8, "y": 0.0, "width": 0.3, "height": 0.5}, "bounds"),
        ("rectangle", {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5}, "width"),
        ("fill", [0, 0], "three channels"),
        ("fill", [0, True, 0], "finite number"),
        ("fill", [0, 0, 256], r"\[0, 255\]"),
    ],
)
def test_invalid_rectangle_or_fill_is_rejected(
    tmp_path: Path, field: str, value, message: str
) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)

    def mutation(payload):
        payload["frames"][2]["transforms"][0][field] = value

    _mutate_manifest(manifest_path, mutation)
    with pytest.raises(InputManifestError, match=message):
        load_input_manifest(manifest_path)


def test_source_manifest_hash_and_loader_length_are_enforced(tmp_path: Path) -> None:
    manifest_path, _ = _generated_manifest(tmp_path)
    manifest = load_input_manifest(manifest_path)

    with pytest.raises(InputManifestError, match="returned 1 views"):
        apply_deferred_transforms(
            [{"img": np.zeros((1, 3, 4, 4), dtype=np.float32)}], manifest
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(payload["source_manifest"])
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(InputManifestError, match="SHA-256"):
        load_input_manifest(manifest_path)


def test_input_view_and_source_files_remain_unchanged_on_assignment_failure(
    tmp_path: Path,
) -> None:
    manifest_path, frame_paths = _generated_manifest(tmp_path)
    manifest = load_input_manifest(manifest_path)
    before_files = _file_state(frame_paths)
    source = np.zeros((1, 3, 4, 4), dtype=np.float32)
    views = [{"img": source} for _ in manifest.frames]
    views[2] = {"img": np.zeros((1, 3, 4, 4), dtype=np.int32)}

    with pytest.raises(InputManifestError, match="floating dtype"):
        apply_deferred_transforms(views, manifest)

    np.testing.assert_allclose(source, 0.0)
    assert _file_state(frame_paths) == before_files
