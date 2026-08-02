from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_recal3r_smoke as smoke_script
from scripts.run_recal3r_smoke import (
    PINNED_RECAL3R_COMMIT,
    _assert_module_source,
    _binary_module_provenance,
    _configure_official_recal3r_runtime,
    _input_frame_metadata,
    _input_manifest_metadata,
    _input_timestamps,
    _load_model_with_state_dict_audit,
    _output_health_signals,
    _parser,
    _apply_v2_overlap_to_health,
    _prepare_input_views,
    _runner_provenance,
    _v2_online_visual_overlap,
    _validate_args,
    _validate_model_interface,
    _verify_preflight_inputs,
    _write_json_atomic,
)
from stateguard3r.corruption import generate_corruption_manifest
from stateguard3r.input_manifest import load_input_manifest
from stateguard3r.health import HealthFrame
from stateguard3r.visual_overlap import VisualOverlapResult


BASELINE = Path("/data/wangzheng/Project2/baselines/ReCal3R")


def _args(tmp_path: Path, *, device: str = "cpu") -> argparse.Namespace:
    checkpoint = tmp_path / "cut3r_512_dpt_4_64.pth"
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    checkpoint.write_bytes(b"not-loaded-by-static-validation")
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")
    return argparse.Namespace(
        baseline_root=BASELINE,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        image=[frame_a, frame_b],
        input_manifest=None,
        output_dir=tmp_path / "new-output",
        device=device,
        size=512,
        seed=0,
        beta_base=0.1,
    )


def test_static_validation_freezes_clean_official_baseline(tmp_path: Path) -> None:
    args = _args(tmp_path)

    _validate_args(args, _parser())

    assert args.baseline_root == BASELINE.resolve()
    assert args.baseline_commit == PINNED_RECAL3R_COMMIT
    assert args.checkpoint_size_bytes == len(b"not-loaded-by-static-validation")
    assert all(path.is_absolute() for path in args.image)
    assert not args.output_dir.exists()


def test_v2_profile_is_explicit_and_copies_causal_overlap_without_view_mutation() -> None:
    parser = _parser()
    assert parser.get_default("health_profile") == "v1"

    views = [
        {"img": np.full((1, 3, 8, 10), value, dtype=np.float32)}
        for value in (-0.4, 0.0, 0.4)
    ]
    before = [view["img"].copy() for view in views]

    def fake_series(images, *, config):
        assert config.nfeatures == 2000
        assert config.ratio_threshold == 0.80
        assert config.ransac_pixel_limit == 1.0
        assert len(images) == len(views)
        assert all(
            np.array_equal(image, view["img"]) for image, view in zip(images, views)
        )
        return [
            None,
            VisualOverlapResult(
                score=0.25,
                status="ok",
                keypoints_previous=12,
                keypoints_current=12,
                ratio_matches=12,
                mutual_matches=12,
                inliers=12,
                inlier_ratio=1.0,
                previous_grid_coverage=0.25,
                current_grid_coverage=0.25,
                reference_frame_id=0,
                frame_id=1,
            ),
            VisualOverlapResult(
                score=0.5,
                status="ok",
                keypoints_previous=12,
                keypoints_current=12,
                ratio_matches=12,
                mutual_matches=12,
                inliers=12,
                inlier_ratio=1.0,
                previous_grid_coverage=0.5,
                current_grid_coverage=0.5,
                reference_frame_id=1,
                frame_id=2,
            ),
        ]

    overlaps, metadata = _v2_online_visual_overlap(
        views,
        series_function=fake_series,
        provenance_function=lambda: {"threads": 1, "opencl_enabled": False},
    )

    assert overlaps == [None, 0.25, 0.5]
    assert metadata["input"] == "post_official_loader_post_deferred_transform_normalized_rgb"
    assert len(metadata["model_ready_uint8_rgb_sha256"]) == 3
    assert all(np.array_equal(view["img"], original) for view, original in zip(views, before))
    health = [HealthFrame(frame_id=index) for index in range(3)]
    enriched = _apply_v2_overlap_to_health(health, overlaps)
    assert [record.overlap for record in health] == [None, None, None]
    assert [record.overlap for record in enriched] == overlaps


def test_cuda_validation_requires_one_explicit_visible_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, device="cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())

    args = _args(tmp_path, device="cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_validation_refuses_existing_run_directory(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_atomic_metadata_writer_emits_strict_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "run.json"

    _write_json_atomic(output, {"status": "succeeded", "value": 1.25})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "succeeded",
        "value": 1.25,
    }
    assert list(output.parent.glob(".run.json.*.tmp")) == []


def _corruption_manifest(tmp_path: Path) -> tuple[Path, list[Path]]:
    frames: list[Path] = []
    for index in range(4):
        frame = tmp_path / f"frame_{index}.png"
        frame.write_bytes(f"frame-{index}".encode())
        frames.append(frame)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "sequence": "runner-integration",
                "frames": [
                    {"path": frame.name, "timestamp": index / 10}
                    for index, frame in enumerate(frames)
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "corruption.json"
    generate_corruption_manifest(
        source,
        manifest,
        [
            {
                "type": "low_overlap_jump",
                "start": 0,
                "end": 0,
                "parameters": {"source_start": 3},
            },
            {
                "type": "dynamic_occlusion",
                "start": 1,
                "end": 1,
                "parameters": {
                    "rectangle": {
                        "x": 0.25,
                        "y": 0.25,
                        "width": 0.5,
                        "height": 0.5,
                    },
                    "velocity": {"dx": 0.0, "dy": 0.0},
                    "fill": [255, 0, 0],
                },
            },
            {
                "type": "wrong_order_segment",
                "start": 2,
                "end": 3,
                "parameters": {"mode": "reverse"},
            },
        ],
        check_paths=True,
    )
    return manifest, frames


def test_static_validation_accepts_manifest_and_resolves_final_order(
    tmp_path: Path,
) -> None:
    manifest, frames = _corruption_manifest(tmp_path)
    args = _args(tmp_path)
    args.image = None
    args.input_manifest = manifest

    _validate_args(args, _parser())

    assert args.input_manifest == manifest.resolve()
    assert args.image == [frames[index].resolve() for index in (3, 1, 3, 2)]
    assert len(args.input_manifest_data.frames) == 4


def test_static_validation_rejects_ambiguous_input_modes(tmp_path: Path) -> None:
    manifest, _ = _corruption_manifest(tmp_path)
    args = _args(tmp_path)
    args.input_manifest = manifest

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_manifest_input_materializes_order_and_occlusion_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = _corruption_manifest(tmp_path)
    seen: list[str] = []
    loaded: list[dict[str, np.ndarray]] = []

    def fake_prepare(paths, size, torch):
        assert size == 512
        for path in paths:
            seen.append(Path(path).name)
            source_index = int(Path(path).stem.split("_")[-1])
            loaded.append(
                {"img": np.full((1, 3, 4, 4), source_index / 10, dtype=np.float32)}
            )
        return loaded

    monkeypatch.setattr(smoke_script, "_prepare_views", fake_prepare)
    args = argparse.Namespace(
        input_manifest=manifest,
        input_manifest_data=load_input_manifest(manifest),
        image=[],
        size=512,
    )

    views = _prepare_input_views(args, torch=object())

    assert seen == ["frame_3.png", "frame_1.png", "frame_3.png", "frame_2.png"]
    assert all(view["img"] is not source["img"] for view, source in zip(views, loaded))
    expected = np.full((1, 3, 4, 4), 0.1, dtype=np.float32)
    expected[:, 0, 1:3, 1:3] = 1.0
    expected[:, 1:, 1:3, 1:3] = -1.0
    np.testing.assert_allclose(views[1]["img"], expected)
    np.testing.assert_allclose(loaded[1]["img"], 0.1)


def test_manifest_metadata_records_hashes_and_transform_counts(tmp_path: Path) -> None:
    manifest, _ = _corruption_manifest(tmp_path)
    args = _args(tmp_path)
    args.image = None
    args.input_manifest = manifest
    _validate_args(args, _parser())

    metadata = _input_manifest_metadata(args)

    assert metadata is not None
    assert metadata["path"] == str(manifest.resolve())
    assert len(metadata["sha256"]) == 64
    assert len(metadata["source_manifest_sha256"]) == 64
    assert metadata["transform_counts"] == {
        "source_frame_substitution": 1,
        "rectangle_occlusion": 1,
        "temporal_reorder": 2,
    }
    assert metadata["rectangle_coordinate_reference"] == (
        "model_input_after_resize_and_center_crop"
    )


def test_checkpoint_hash_is_verified_before_model_loading(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.checkpoint_sha256 = "0" * 64

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


@pytest.mark.parametrize(
    ("checkpoint_name", "size"),
    [
        ("cut3r_224_linear_4.pth", 512),
        ("cut3r_512_dpt_4_64.pth", 224),
    ],
)
def test_validation_binds_official_checkpoint_name_to_input_size(
    tmp_path: Path, checkpoint_name: str, size: int
) -> None:
    args = _args(tmp_path)
    checkpoint = tmp_path / checkpoint_name
    if checkpoint != args.checkpoint:
        checkpoint.write_bytes(args.checkpoint.read_bytes())
    args.checkpoint = checkpoint
    args.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args.size = size

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_validation_rejects_unrecognized_checkpoint_filename(tmp_path: Path) -> None:
    args = _args(tmp_path)
    checkpoint = tmp_path / "renamed.pth"
    checkpoint.write_bytes(args.checkpoint.read_bytes())
    args.checkpoint = checkpoint
    args.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_validation_rejects_extension_skipped_by_upstream_loader(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    unsupported = tmp_path / "frame.bmp"
    unsupported.write_bytes(b"not-decoded-during-validation")
    args.image = [args.image[0], unsupported]

    with pytest.raises(SystemExit):
        _validate_args(args, _parser())


def test_changed_manifest_snapshot_is_rejected_before_replay(tmp_path: Path) -> None:
    manifest, _ = _corruption_manifest(tmp_path)
    args = _args(tmp_path)
    args.image = None
    args.input_manifest = manifest
    _validate_args(args, _parser())
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="input manifest changed"):
        _verify_preflight_inputs(args)


def test_manifest_provenance_and_timestamps_follow_final_order(tmp_path: Path) -> None:
    manifest, frames = _corruption_manifest(tmp_path)
    args = _args(tmp_path)
    args.image = None
    args.input_manifest = manifest
    _validate_args(args, _parser())

    provenance = _input_frame_metadata(args)

    assert [frame["source_index"] for frame in provenance] == [3, 1, 3, 2]
    assert [frame["path"] for frame in provenance] == [
        str(frames[index].resolve()) for index in (3, 1, 3, 2)
    ]
    assert [frame["metadata"]["timestamp"] for frame in provenance] == [
        0.3,
        0.1,
        0.3,
        0.2,
    ]
    assert _input_timestamps(args) == [0.3, 0.1, 0.3, 0.2]
    assert [transform["type"] for transform in provenance[0]["transforms"]] == [
        "source_frame_substitution"
    ]


def test_module_source_assertion_rejects_wrong_checkout(tmp_path: Path) -> None:
    expected = tmp_path / "expected.py"
    wrong = tmp_path / "wrong.py"
    expected.write_text("", encoding="utf-8")
    wrong.write_text("", encoding="utf-8")

    assert _assert_module_source(
        SimpleNamespace(__file__=str(expected)), expected, "x"
    ) == str(expected.resolve())
    with pytest.raises(RuntimeError, match="expected frozen source"):
        _assert_module_source(SimpleNamespace(__file__=str(wrong)), expected, "x")


class _CloneableArray:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def clone(self):
        return _CloneableArray(self.value.copy())

    def numpy(self):
        return self.value


def _pose_to_camera(value):
    encoding = value.numpy()
    result = np.repeat(np.eye(4)[None], encoding.shape[0], axis=0)
    result[:, :3, 3] = encoding[:, :3]
    w, x, y, z = np.moveaxis(encoding[:, 3:7], -1, 0)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - z * w)
    result[:, 0, 2] = 2 * (x * z + y * w)
    result[:, 1, 0] = 2 * (x * y + z * w)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - x * w)
    result[:, 2, 0] = 2 * (x * z - y * w)
    result[:, 2, 1] = 2 * (y * z + x * w)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def test_output_health_signals_export_trajectory_pose_jump_and_residual() -> None:
    points = np.array([[[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]])
    predictions = [
        {
            "camera_pose": _CloneableArray([[0, 0, 0, 1, 0, 0, 0]]),
            "pts3d_in_self_view": points,
            "pts3d_in_other_view": points,
        },
        {
            "camera_pose": _CloneableArray([[3, 4, 0, 1, 0, 0, 0]]),
            "pts3d_in_self_view": points,
            "pts3d_in_other_view": points + np.array([3, 4, 0]),
        },
    ]

    jumps, residuals, trajectory = _output_health_signals(
        predictions,
        pose_encoding_to_camera=_pose_to_camera,
        np=np,
    )

    assert jumps == [None, pytest.approx(5.0)]
    assert residuals == [pytest.approx(0.0), pytest.approx(0.0)]
    assert trajectory[0]["pose_jump"] is None
    assert trajectory[1]["translation_jump"] == pytest.approx(5.0)
    assert trajectory[1]["rotation_jump_rad"] == pytest.approx(0.0)
    assert trajectory[1]["camera_to_reference"][0][3] == pytest.approx(3.0)


def test_output_pose_jump_uses_relative_rotation_angle() -> None:
    half_sqrt = np.sqrt(0.5)
    predictions = [
        {
            "camera_pose": _CloneableArray([[0, 0, 0, 1, 0, 0, 0]]),
            "pts3d_in_self_view": np.array([[[[1.0, 0.0, 0.0]]]]),
            "pts3d_in_other_view": np.array([[[[1.0, 0.0, 0.0]]]]),
        },
        {
            "camera_pose": _CloneableArray(
                [[0, 0, 0, half_sqrt, 0, 0, half_sqrt]]
            ),
            "pts3d_in_self_view": np.array([[[[1.0, 0.0, 0.0]]]]),
            "pts3d_in_other_view": np.array([[[[0.0, 1.0, 0.0]]]]),
        },
    ]

    jumps, residuals, trajectory = _output_health_signals(
        predictions,
        pose_encoding_to_camera=_pose_to_camera,
        np=np,
    )

    assert jumps == [None, pytest.approx(np.pi / 2)]
    assert residuals == [pytest.approx(0.0), pytest.approx(0.0, abs=1e-15)]
    assert trajectory[1]["translation_jump"] == pytest.approx(0.0)
    assert trajectory[1]["rotation_jump_rad"] == pytest.approx(np.pi / 2)


def _named_head(name: str, *, has_pose: bool = True, pose_decoder: bool = True):
    head = type(name, (), {})()
    head.has_pose = has_pose
    if pose_decoder:
        head.pose_head = object()
    return head


@pytest.mark.parametrize(
    ("checkpoint_name", "head_type"),
    [
        ("cut3r_224_linear_4.pth", "linear"),
        ("cut3r_512_dpt_4_64.pth", "dpt"),
    ],
)
def test_loaded_model_interface_matches_checkpoint_contract(
    checkpoint_name: str, head_type: str
) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(),
        head_type=head_type,
        pose_head_flag=True,
        output_mode="pts3d+pose",
        downstream_head=_named_head(
            "LinearPts3dPose" if head_type == "linear" else "DPTPts3dPose"
        ),
    )

    assert _validate_model_interface(model, checkpoint_name) == {
        "head_type": head_type,
        "head_type_source": "model.head_type",
        "downstream_head_class": (
            "LinearPts3dPose" if head_type == "linear" else "DPTPts3dPose"
        ),
        "downstream_has_pose": True,
        "independent_cross_pointmap": True,
        "pose_head": True,
        "pose_head_source": "model.pose_head_flag",
        "pose_decoder_present": True,
        "output_mode": "pts3d+pose",
        "output_mode_source": "model.output_mode",
    }


def test_loaded_model_interface_rejects_renamed_or_incompatible_head() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            head_type="dpt",
            pose_head=True,
            output_mode="pts3d+pose",
        ),
        head_type="linear",
        pose_head_flag=True,
        output_mode="pts3d+pose",
        downstream_head=_named_head("LinearPts3dPose"),
    )

    with pytest.raises(RuntimeError, match="expected 'dpt'"):
        _validate_model_interface(model, "cut3r_512_dpt_4_64.pth")


def test_loaded_model_interface_rejects_missing_effective_model_fields() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            head_type="dpt",
            pose_head=True,
            output_mode="pts3d+pose",
        ),
        downstream_head=_named_head("DPTPts3dPose"),
    )

    with pytest.raises(RuntimeError, match="head_type=None"):
        _validate_model_interface(model, "cut3r_512_dpt_4_64.pth")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_mode", "pts3d", "must be 'pts3d\\+pose'"),
        ("pose_head_flag", False, "enabled pose head"),
    ],
)
def test_loaded_model_interface_rejects_disabled_model_contract(
    field: str, value: object, message: str
) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(),
        head_type="dpt",
        pose_head_flag=True,
        output_mode="pts3d+pose",
        downstream_head=_named_head("DPTPts3dPose"),
    )
    setattr(model, field, value)

    with pytest.raises(RuntimeError, match=message):
        _validate_model_interface(model, "cut3r_512_dpt_4_64.pth")


@pytest.mark.parametrize(
    ("head", "message"),
    [
        (_named_head("DPTPts3dPose", has_pose=False), "must enable pose"),
        (_named_head("DPTPts3dPose", pose_decoder=False), "decoder is missing"),
    ],
)
def test_loaded_model_interface_rejects_incomplete_downstream_pose_head(
    head: object, message: str
) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(),
        head_type="dpt",
        pose_head_flag=True,
        output_mode="pts3d+pose",
        downstream_head=head,
    )

    with pytest.raises(RuntimeError, match=message):
        _validate_model_interface(model, "cut3r_512_dpt_4_64.pth")


def test_binary_module_provenance_freezes_expected_ignored_extension(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "curope.cpython-311-x86_64-linux-gnu.so"
    extension.write_bytes(b"compiled-kernel")
    module = SimpleNamespace(__name__="models.curope.curope", __file__=str(extension))

    assert _binary_module_provenance(
        module,
        expected_directory=tmp_path,
        filename_prefix="curope.",
        label="cuRoPE CUDA extension",
    ) == {
        "module": "models.curope.curope",
        "path": str(extension),
        "size_bytes": len(b"compiled-kernel"),
        "sha256": hashlib.sha256(b"compiled-kernel").hexdigest(),
    }


def test_binary_module_provenance_rejects_wrong_directory(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    extension = actual / "curope.test.so"
    extension.write_bytes(b"compiled-kernel")

    with pytest.raises(RuntimeError, match="expected curope"):
        _binary_module_provenance(
            SimpleNamespace(__file__=str(extension)),
            expected_directory=expected,
            filename_prefix="curope.",
            label="cuRoPE CUDA extension",
        )


def test_official_recal3r_runtime_parameters_are_frozen_on_model_and_config() -> None:
    model = SimpleNamespace(config=SimpleNamespace())

    frozen = _configure_official_recal3r_runtime(model, beta_base=0.25)

    assert frozen == {
        "model_update_type": "recal3r",
        "beta_base": 0.25,
        "entropy_eps": 2e-14,
        "entropy_head_reduce": "mean",
        "uncertainty_clamp_max": 1.0,
        "decay": 0.95,
    }
    for name, value in frozen.items():
        assert getattr(model, name) == value
        assert getattr(model.config, name) == value


def test_checkpoint_loader_captures_and_restores_state_dict_audit(tmp_path) -> None:
    class FakeModule:
        def load_state_dict(self, state_dict, strict=True):
            del state_dict
            return SimpleNamespace(
                missing_keys=["missing.weight"],
                unexpected_keys=["legacy.weight"],
            )

    original_load = FakeModule.load_state_dict

    class FakeModel(FakeModule):
        def load_state_dict(self, state_dict, strict=True):
            return super().load_state_dict(state_dict, strict=strict)

        @classmethod
        def from_pretrained(cls, path):
            assert path == str(tmp_path / "checkpoint.pth")
            model = cls()
            model.load_state_dict({}, strict=False)
            return model

    original_model_load = FakeModel.load_state_dict

    model, audit = _load_model_with_state_dict_audit(
        FakeModel,
        tmp_path / "checkpoint.pth",
        SimpleNamespace(nn=SimpleNamespace(Module=FakeModule)),
    )

    assert isinstance(model, FakeModel)
    assert audit == {
        "module_class": "FakeModel",
        "strict": False,
        "missing_keys": ["missing.weight"],
        "unexpected_keys": ["legacy.weight"],
    }
    assert FakeModule.load_state_dict is original_load
    assert FakeModel.load_state_dict is original_model_load


def test_checkpoint_loader_rejects_non_delegating_model_override(tmp_path) -> None:
    class FakeModule:
        def load_state_dict(self, state_dict, strict=True):
            del state_dict, strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class FakeModel(FakeModule):
        def load_state_dict(self, state_dict, strict=True):
            del state_dict, strict
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

        @classmethod
        def from_pretrained(cls, path):
            assert path == str(tmp_path / "checkpoint.pth")
            model = cls()
            model.load_state_dict({}, strict=False)
            return model

    with pytest.raises(RuntimeError, match="observed 0"):
        _load_model_with_state_dict_audit(
            FakeModel,
            tmp_path / "checkpoint.pth",
            SimpleNamespace(nn=SimpleNamespace(Module=FakeModule)),
        )


def test_runner_provenance_requires_clean_tracked_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def clean_git(_repository, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments[:2] == ("status", "--porcelain"):
            return ""
        if arguments[:2] == ("ls-files", "--error-unmatch"):
            return arguments[-1]
        raise AssertionError(arguments)

    monkeypatch.setattr(smoke_script, "_git_output", clean_git)
    provenance = _runner_provenance(Path(sys.prefix).parent)

    assert provenance["commit"] == "a" * 40
    assert provenance["tracked_worktree_clean"] is True
    assert len(provenance["script_sha256"]) == 64
    assert provenance["python_prefix"] == str(Path(sys.prefix).resolve())

    def dirty_git(repository, *arguments):
        if arguments[:2] == ("status", "--porcelain"):
            return " M scripts/run_recal3r_smoke.py"
        return clean_git(repository, *arguments)

    monkeypatch.setattr(smoke_script, "_git_output", dirty_git)
    with pytest.raises(RuntimeError, match="tracked changes"):
        _runner_provenance(Path(sys.prefix).parent)
