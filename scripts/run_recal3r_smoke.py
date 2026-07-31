#!/usr/bin/env python3
"""Run a minimal, instrumented ReCal3R image-sequence smoke test.

Execute this file with the isolated ReCal3R interpreter. It intentionally does
not launch visualization, open a port, download resources, or modify the
upstream baseline checkout.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


PROJECT_ROOT = Path("/data/wangzheng/Project2")
READ_ONLY_ROOT = Path("/data/wangzheng")
PINNED_RECAL3R_COMMIT = "466c7cdf3acd2f589f1d82e5f6391966f19db9ff"
SUPPORTED_INPUT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SUPPORTED_CHECKPOINT_SIZES = {
    "cut3r_224_linear_4.pth": 224,
    "cut3r_512_dpt_4_64.pth": 512,
}
SUPPORTED_CHECKPOINT_HEADS = {
    "cut3r_224_linear_4.pth": "linear",
    "cut3r_512_dpt_4_64.pth": "dpt",
}
SUPPORTED_CHECKPOINT_HEAD_CLASSES = {
    "cut3r_224_linear_4.pth": "LinearPts3dPose",
    "cut3r_512_dpt_4_64.pth": "DPTPts3dPose",
}
OFFICIAL_RECAL3R_ENTROPY_EPS = 2e-14
OFFICIAL_RECAL3R_ENTROPY_HEAD_REDUCE = "mean"
OFFICIAL_RECAL3R_UNCERTAINTY_CLAMP_MAX = 1.0
OFFICIAL_RECAL3R_DECAY = 0.95


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="expected lowercase SHA-256 verified before checkpoint deserialization",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", action="append", type=Path)
    inputs.add_argument(
        "--input-manifest",
        type=Path,
        help=(
            "validated StateGuard3R corruption manifest; final path order is "
            "loaded directly and deferred pixel transforms are applied in memory"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--size", choices=(224, 512), default=512, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--beta-base", default=0.1, type=float)
    return parser


def _inside_project(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents


def _inside_read_only_root(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == READ_ONLY_ROOT or READ_ONLY_ROOT in resolved.parents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _runner_provenance(baseline_root: Path) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    try:
        commit = _git_output(repository, "rev-parse", "HEAD")
        tracked_changes = _git_output(
            repository,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        tracked_script = _git_output(
            repository,
            "ls-files",
            "--error-unmatch",
            str(script_path.relative_to(repository)),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot inspect StateGuard3R runner provenance: {error}") from error
    if tracked_changes:
        raise RuntimeError(
            "StateGuard3R has tracked changes; run the smoke test from a clean commit"
        )
    if tracked_script != str(script_path.relative_to(repository)):
        raise RuntimeError("ReCal3R smoke runner is not tracked by StateGuard3R")

    expected_prefix = (baseline_root / ".venv").resolve(strict=False)
    actual_prefix = Path(sys.prefix).resolve(strict=False)
    if actual_prefix != expected_prefix:
        raise RuntimeError(
            f"runner must use the isolated ReCal3R environment {expected_prefix}; "
            f"got {actual_prefix}"
        )
    return {
        "repository_root": str(repository),
        "commit": commit,
        "tracked_worktree_clean": True,
        "script_path": str(script_path),
        "script_sha256": _sha256(script_path),
        "python_executable": sys.executable,
        "python_executable_resolved": str(
            Path(sys.executable).resolve(strict=False)
        ),
        "python_prefix": str(actual_prefix),
        "python_version": platform.python_version(),
    }


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    baseline_root = args.baseline_root.resolve(strict=False)
    checkpoint = args.checkpoint.resolve(strict=False)
    raw_images = list(getattr(args, "image", None) or [])
    raw_manifest = getattr(args, "input_manifest", None)
    if bool(raw_images) == bool(raw_manifest):
        parser.error("exactly one of --image or --input-manifest is required")
    images = [path.resolve(strict=False) for path in raw_images]
    input_manifest = (
        raw_manifest.resolve(strict=False) if raw_manifest is not None else None
    )
    output_dir = args.output_dir.resolve(strict=False)

    writable_paths = [
        ("baseline root", baseline_root),
        ("output directory", output_dir),
    ]
    read_only_paths = [("checkpoint", checkpoint)]
    if input_manifest is not None:
        read_only_paths.append(("input manifest", input_manifest))
    read_only_paths.extend(("input image", path) for path in images)
    for label, path in writable_paths:
        if not _inside_project(path):
            parser.error(f"{label} must stay inside {PROJECT_ROOT}: {path}")
    for label, path in read_only_paths:
        if not _inside_read_only_root(path):
            parser.error(f"{label} must stay inside {READ_ONLY_ROOT}: {path}")
    if not baseline_root.is_dir():
        parser.error(f"baseline root does not exist: {baseline_root}")
    baseline_src = baseline_root / "src"
    if not baseline_src.is_dir():
        parser.error(f"baseline src directory does not exist: {baseline_src}")
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    expected_size = SUPPORTED_CHECKPOINT_SIZES.get(checkpoint.name)
    if expected_size is None:
        supported = ", ".join(sorted(SUPPORTED_CHECKPOINT_SIZES))
        parser.error(
            f"unsupported checkpoint filename {checkpoint.name!r}; expected one of: "
            f"{supported}"
        )
    if args.size != expected_size:
        parser.error(
            f"checkpoint {checkpoint.name} requires --size {expected_size}, "
            f"not {args.size}"
        )
    expected_checkpoint_sha256 = args.checkpoint_sha256
    if (
        not isinstance(expected_checkpoint_sha256, str)
        or len(expected_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_checkpoint_sha256
        )
    ):
        parser.error("--checkpoint-sha256 must be a lowercase SHA-256")
    actual_checkpoint_sha256 = _sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        parser.error("checkpoint SHA-256 does not match --checkpoint-sha256")
    loaded_manifest = None
    if input_manifest is None:
        if len(images) < 2:
            parser.error("at least two --image arguments are required")
        missing = [str(path) for path in images if not path.is_file()]
        if missing:
            parser.error("input image does not exist: " + ", ".join(missing))
    else:
        if not input_manifest.is_file():
            parser.error(f"input manifest does not exist: {input_manifest}")
        try:
            manifest_snapshot = input_manifest.read_bytes()
            manifest_payload = json.loads(manifest_snapshot.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            parser.error(f"cannot snapshot input manifest: {error}")
        manifest_snapshot_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
        state_src = Path(__file__).resolve().parents[1] / "src"
        if str(state_src) not in sys.path:
            sys.path.insert(0, str(state_src))
        from stateguard3r.input_manifest import InputManifestError, load_input_manifest

        try:
            loaded_manifest = load_input_manifest(input_manifest)
        except InputManifestError as error:
            parser.error(f"invalid input manifest: {error}")
        if _sha256(input_manifest) != manifest_snapshot_sha256:
            parser.error("input manifest changed while it was being validated")
        if len(loaded_manifest.frames) < 2:
            parser.error("input manifest must contain at least two frames")
        protected_paths = (
            loaded_manifest.source_manifest_path,
            *(frame.path for frame in loaded_manifest.frames),
        )
        for path in protected_paths:
            if not _inside_read_only_root(path):
                parser.error(
                    f"manifest source must stay inside {READ_ONLY_ROOT}: {path}"
                )
        source_manifest_sha256 = _sha256(loaded_manifest.source_manifest_path)
        if source_manifest_sha256 != manifest_payload.get("source_manifest_sha256"):
            parser.error("source manifest changed while it was being validated")
        images = list(loaded_manifest.frame_paths)
    unsupported = [
        str(path) for path in images if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES
    ]
    if unsupported:
        parser.error(
            "input image extension is unsupported by load_images_for_eval: "
            + ", ".join(unsupported)
        )
    if output_dir.exists():
        parser.error(
            f"output directory already exists; choose a new run directory: {output_dir}"
        )
    if not math.isfinite(args.beta_base) or not 0.0 <= args.beta_base <= 1.0:
        parser.error("--beta-base must be finite and in [0, 1]")
    if args.device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or not visible.strip():
            parser.error("CUDA runs require an explicit CUDA_VISIBLE_DEVICES value")
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if len(entries) != 1 or not entries[0].isdigit():
            parser.error(
                "smoke runs require exactly one numeric CUDA_VISIBLE_DEVICES entry"
            )

    try:
        baseline_commit = _git_output(baseline_root, "rev-parse", "HEAD")
        tracked_changes = _git_output(
            baseline_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f"cannot inspect baseline Git state: {error}")
    if baseline_commit != PINNED_RECAL3R_COMMIT:
        parser.error(
            "baseline commit does not match the frozen ReCal3R revision: "
            f"{baseline_commit}"
        )
    if tracked_changes:
        parser.error("baseline has tracked changes; refusing an ambiguous smoke run")

    args.baseline_root = baseline_root
    args.baseline_src = baseline_src
    args.checkpoint = checkpoint
    args.checkpoint_sha256 = actual_checkpoint_sha256
    args.checkpoint_size_bytes = checkpoint.stat().st_size
    args.image = images
    args.input_manifest = input_manifest
    args.input_manifest_data = loaded_manifest
    args.input_manifest_sha256 = (
        manifest_snapshot_sha256 if input_manifest is not None else None
    )
    args.source_manifest_sha256 = (
        source_manifest_sha256 if loaded_manifest is not None else None
    )
    args.image_sha256 = {path: _sha256(path) for path in dict.fromkeys(images)}
    args.output_dir = output_dir
    args.baseline_commit = baseline_commit


def _prepare_views(image_paths: Sequence[Path], size: int, torch: Any) -> list[dict[str, Any]]:
    from dust3r.utils.image import load_images_for_eval

    images = load_images_for_eval(
        [str(path) for path in image_paths],
        size=size,
        crop=True,
        verbose=True,
    )
    if len(images) != len(image_paths):
        raise RuntimeError(
            f"image loader returned {len(images)} frames for {len(image_paths)} paths"
        )

    views: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        batch_size = image["img"].shape[0]
        height, width = image["img"].shape[-2:]
        views.append(
            {
                "img": image["img"],
                "ray_map": torch.full(
                    (batch_size, 6, height, width),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(image["true_shape"]),
                "idx": index,
                "instance": str(index),
                "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
        )
    return views


def _prepare_input_views(args: argparse.Namespace, torch: Any) -> list[dict[str, Any]]:
    """Load clean images or replay a corruption manifest without writing frames."""

    if args.input_manifest is None:
        return _prepare_views(args.image, args.size, torch)

    from stateguard3r.input_manifest import apply_deferred_transforms

    manifest = args.input_manifest_data
    if manifest is None:
        raise RuntimeError("validated input manifest snapshot is unavailable")
    loaded = _prepare_views(
        [frame.path for frame in manifest.frames],
        args.size,
        torch,
    )
    return apply_deferred_transforms(loaded, manifest)


def _verify_preflight_inputs(args: argparse.Namespace) -> None:
    """Reject any input changed since static validation."""

    expected: list[tuple[str, Path, str]] = [
        ("checkpoint", args.checkpoint, args.checkpoint_sha256)
    ]
    if args.input_manifest is not None:
        expected.extend(
            [
                (
                    "input manifest",
                    args.input_manifest,
                    args.input_manifest_sha256,
                ),
                (
                    "source manifest",
                    args.input_manifest_data.source_manifest_path,
                    args.source_manifest_sha256,
                ),
            ]
        )
    expected.extend(
        ("input image", path, digest)
        for path, digest in args.image_sha256.items()
    )
    for label, path, digest in expected:
        try:
            actual = _sha256(path)
        except OSError as error:
            raise RuntimeError(
                f"{label} became unreadable after validation: {path}"
            ) from error
        if actual != digest:
            raise RuntimeError(f"{label} changed after validation: {path}")


def _input_timestamps(args: argparse.Namespace) -> list[Any] | None:
    manifest = args.input_manifest_data
    if manifest is None:
        return None
    return [frame.metadata.get("timestamp") for frame in manifest.frames]


def _input_frame_metadata(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = args.input_manifest_data
    if manifest is None:
        return [
            {
                "frame_index": index,
                "source_index": index,
                "path": str(path),
                "sha256": args.image_sha256[path],
                "metadata": {},
                "transforms": [],
            }
            for index, path in enumerate(args.image)
        ]
    return [
        {
            "frame_index": frame.frame_index,
            "source_index": frame.source_index,
            "path": str(frame.path),
            "sha256": args.image_sha256[frame.path],
            "metadata": copy.deepcopy(dict(frame.metadata)),
            "transforms": copy.deepcopy(list(frame.transforms)),
        }
        for frame in manifest.frames
    ]


def _input_manifest_metadata(args: argparse.Namespace) -> dict[str, Any] | None:
    manifest = args.input_manifest_data
    if manifest is None:
        return None
    transform_counts: dict[str, int] = {}
    rectangle_references: set[str] = set()
    for frame in manifest.frames:
        for transform in frame.transforms:
            transform_type = str(transform["type"])
            transform_counts[transform_type] = (
                transform_counts.get(transform_type, 0) + 1
            )
            if transform_type == "rectangle_occlusion":
                rectangle_references.add(str(transform["coordinate_reference"]))
    return {
        "path": str(manifest.path),
        "sha256": args.input_manifest_sha256,
        "source_manifest": str(manifest.source_manifest_path),
        "source_manifest_sha256": args.source_manifest_sha256,
        "materialization": "final_manifest_order_with_in_memory_deferred_transforms",
        "rectangle_coordinate_reference": (
            next(iter(rectangle_references)) if rectangle_references else None
        ),
        "transform_counts": transform_counts,
    }


def _assert_module_source(module: Any, expected_path: Path, label: str) -> str:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError(f"{label} does not expose a source file")
    actual_path = Path(raw_path).resolve(strict=False)
    expected_path = expected_path.resolve(strict=False)
    if actual_path != expected_path:
        raise RuntimeError(
            f"{label} loaded from {actual_path}, expected frozen source {expected_path}"
        )
    return str(actual_path)


def _tensor_summary(value: Any, torch: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach()
    finite = torch.isfinite(tensor) if tensor.is_floating_point() else None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(finite.all().item()) if finite is not None else True,
        "min": (
            float(tensor[finite].min().item())
            if finite is not None and bool(finite.any().item())
            else None
        ),
        "max": (
            float(tensor[finite].max().item())
            if finite is not None and bool(finite.any().item())
            else None
        ),
    }


def _prediction_summary(predictions: Sequence[dict[str, Any]], torch: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        required = {"camera_pose", "pts3d_in_self_view", "pts3d_in_other_view"}
        missing = sorted(required - set(prediction))
        if missing:
            raise RuntimeError(
                f"prediction {index} is missing required output(s): {', '.join(missing)}"
            )
        fields = {
            key: summary
            for key, value in sorted(prediction.items())
            if (summary := _tensor_summary(value, torch)) is not None
        }
        bad_fields = [key for key, summary in fields.items() if not summary["finite"]]
        if bad_fields:
            raise RuntimeError(
                f"prediction {index} contains NaN/Inf in: {', '.join(bad_fields)}"
            )
        summaries.append({"frame_id": index, "tensor_fields": fields})
    return summaries


def _validate_model_interface(model: Any, checkpoint_name: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("loaded checkpoint model does not expose config")
    expected_head = SUPPORTED_CHECKPOINT_HEADS[checkpoint_name]
    actual_head = getattr(config, "head_type", None)
    if actual_head != expected_head:
        raise RuntimeError(
            f"checkpoint {checkpoint_name} loaded head_type={actual_head!r}; "
            f"expected {expected_head!r}"
        )
    pose_head = getattr(config, "pose_head", None)
    if pose_head is not True:
        raise RuntimeError(
            f"checkpoint {checkpoint_name} must expose an enabled pose head"
        )
    output_mode = getattr(config, "output_mode", None)
    if not isinstance(output_mode, str) or "pose" not in output_mode:
        raise RuntimeError(
            f"checkpoint {checkpoint_name} output_mode must include pose, "
            f"got {output_mode!r}"
        )
    downstream_head = getattr(model, "downstream_head", None)
    downstream_head_class = type(downstream_head).__name__
    expected_head_class = SUPPORTED_CHECKPOINT_HEAD_CLASSES[checkpoint_name]
    if downstream_head_class != expected_head_class:
        raise RuntimeError(
            f"checkpoint {checkpoint_name} loaded downstream head "
            f"{downstream_head_class!r}; expected {expected_head_class!r}"
        )
    return {
        "head_type": actual_head,
        "downstream_head_class": downstream_head_class,
        "independent_cross_pointmap": True,
        "pose_head": pose_head,
        "output_mode": output_mode,
    }


def _configure_official_recal3r_runtime(
    model: Any, *, beta_base: float
) -> dict[str, Any]:
    """Apply the runtime parameters frozen by the official relpose launcher."""

    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("loaded checkpoint model does not expose config")
    frozen = {
        "model_update_type": "recal3r",
        "beta_base": float(beta_base),
        "entropy_eps": OFFICIAL_RECAL3R_ENTROPY_EPS,
        "entropy_head_reduce": OFFICIAL_RECAL3R_ENTROPY_HEAD_REDUCE,
        "uncertainty_clamp_max": OFFICIAL_RECAL3R_UNCERTAINTY_CLAMP_MAX,
        "decay": OFFICIAL_RECAL3R_DECAY,
    }
    for target in (model, config):
        for name, value in frozen.items():
            setattr(target, name, value)
    return frozen


def _load_model_with_state_dict_audit(
    model_class: Any,
    checkpoint: Path,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    """Capture the pinned loader's strict=False compatibility result."""

    module_class = torch.nn.Module
    original_load_state_dict = module_class.load_state_dict
    # The pinned ARCroco3DStereo intentionally overrides load_state_dict to
    # normalize legacy keys, then delegates to this base implementation. Audit
    # that delegated compatibility result; a non-delegating override is still
    # rejected below because it produces zero captured calls.
    calls: list[dict[str, Any]] = []

    def audited_load_state_dict(
        module: Any,
        state_dict: Any,
        *call_args: Any,
        **call_kwargs: Any,
    ) -> Any:
        result = original_load_state_dict(
            module,
            state_dict,
            *call_args,
            **call_kwargs,
        )
        strict = (
            call_kwargs.get("strict")
            if "strict" in call_kwargs
            else (call_args[0] if call_args else True)
        )
        calls.append(
            {
                "module_class": type(module).__name__,
                "strict": bool(strict),
                "missing_keys": [str(value) for value in result.missing_keys],
                "unexpected_keys": [
                    str(value) for value in result.unexpected_keys
                ],
            }
        )
        return result

    module_class.load_state_dict = audited_load_state_dict
    try:
        model = model_class.from_pretrained(str(checkpoint))
    finally:
        module_class.load_state_dict = original_load_state_dict

    if len(calls) != 1:
        raise RuntimeError(
            f"expected exactly one checkpoint state_dict load, observed {len(calls)}"
        )
    audit = calls[0]
    return model, audit


def _numpy_value(value: Any, *, name: str, np: Any) -> Any:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        array = np.asarray(current, dtype=np.float64)
    except Exception as error:
        raise RuntimeError(f"{name} cannot be converted to NumPy") from error
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"{name} contains NaN/Inf")
    return array


def _output_health_signals(
    predictions: Sequence[dict[str, Any]],
    *,
    pose_encoding_to_camera: Any,
    np: Any,
) -> tuple[list[float | None], list[float | None], list[dict[str, Any]]]:
    """Derive lightweight output-level signals without retaining pointmaps."""

    pose_jumps: list[float | None] = []
    geometric_residuals: list[float | None] = []
    trajectory: list[dict[str, Any]] = []
    previous_pose: Any | None = None

    for frame_id, prediction in enumerate(predictions):
        pose_encoding = _numpy_value(
            prediction["camera_pose"],
            name=f"prediction {frame_id} camera_pose",
            np=np,
        )
        if pose_encoding.shape != (1, 7):
            raise RuntimeError(
                f"prediction {frame_id} camera_pose must have shape (1, 7), "
                f"got {pose_encoding.shape}"
            )
        try:
            pose_tensor = prediction["camera_pose"].clone()
        except (AttributeError, TypeError) as error:
            raise RuntimeError(
                f"prediction {frame_id} camera_pose must expose clone()"
            ) from error
        camera = _numpy_value(
            pose_encoding_to_camera(pose_tensor),
            name=f"prediction {frame_id} camera matrix",
            np=np,
        )
        if camera.shape != (1, 4, 4):
            raise RuntimeError(
                f"prediction {frame_id} camera matrix must have shape (1, 4, 4), "
                f"got {camera.shape}"
            )
        pose = camera[0]
        if not bool(np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)):
            raise RuntimeError(
                f"prediction {frame_id} camera matrix is not homogeneous"
            )

        translation_jump: float | None = None
        rotation_jump_rad: float | None = None
        pose_jump: float | None = None
        if previous_pose is not None:
            relative_rotation = previous_pose[:3, :3].T @ pose[:3, :3]
            cosine = float((np.trace(relative_rotation) - 1.0) / 2.0)
            rotation_jump_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
            relative_translation = previous_pose[:3, :3].T @ (
                pose[:3, 3] - previous_pose[:3, 3]
            )
            translation_jump = float(np.linalg.norm(relative_translation))
            pose_jump = float(np.hypot(translation_jump, rotation_jump_rad))
            if not math.isfinite(pose_jump):
                raise RuntimeError(
                    f"prediction {frame_id} produced a non-finite pose jump"
                )
        pose_jumps.append(pose_jump)
        previous_pose = pose

        points_self = _numpy_value(
            prediction["pts3d_in_self_view"],
            name=f"prediction {frame_id} pts3d_in_self_view",
            np=np,
        )
        points_other = _numpy_value(
            prediction["pts3d_in_other_view"],
            name=f"prediction {frame_id} pts3d_in_other_view",
            np=np,
        )
        if (
            points_self.shape != points_other.shape
            or points_self.ndim != 4
            or points_self.shape[0] != 1
            or points_self.shape[-1] != 3
        ):
            raise RuntimeError(
                f"prediction {frame_id} pointmaps must share shape (1, H, W, 3); "
                f"got {points_self.shape} and {points_other.shape}"
            )
        transformed_self = points_self @ pose[:3, :3].T + pose[:3, 3]
        point_errors = np.linalg.norm(
            transformed_self - points_other,
            axis=-1,
        )
        point_scales = np.linalg.norm(points_other, axis=-1)
        median_error = float(np.median(point_errors))
        median_scale = float(np.median(point_scales))
        geometric_residual = median_error / max(median_scale, 1e-8)
        if not math.isfinite(geometric_residual):
            raise RuntimeError(
                f"prediction {frame_id} produced a non-finite geometric residual"
            )
        geometric_residuals.append(geometric_residual)
        trajectory.append(
            {
                "frame_id": frame_id,
                "camera_pose_encoding_absT_quaR": pose_encoding[0].tolist(),
                "camera_to_reference": pose.tolist(),
                "translation_jump": translation_jump,
                "rotation_jump_rad": rotation_jump_rad,
                "pose_jump": pose_jump,
                "geometric_residual": geometric_residual,
            }
        )

    return pose_jumps, geometric_residuals, trajectory


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    runner_provenance = _runner_provenance(args.baseline_root)
    run_started_at = datetime.now().astimezone().isoformat()

    state_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(state_src))
    sys.path.insert(0, str(args.baseline_root))

    import numpy as np
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_VISIBLE_DEVICES was set but PyTorch CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    import add_ckpt_path as add_ckpt_path_module

    module_paths = {
        "add_ckpt_path": _assert_module_source(
            add_ckpt_path_module,
            args.baseline_root / "add_ckpt_path.py",
            "add_ckpt_path",
        )
    }
    # The upstream helper is retained, but receives the frozen baseline source
    # directory instead of allowing an arbitrary checkpoint parent to control
    # Python imports.
    add_ckpt_path_module.add_path_to_dust3r(
        str(args.baseline_src / args.checkpoint.name)
    )
    import dust3r.inference as dust3r_inference_module
    import dust3r.model as dust3r_model_module
    import dust3r.utils.camera as dust3r_camera_module
    import dust3r.utils.image as dust3r_image_module

    module_paths.update(
        {
            "dust3r.inference": _assert_module_source(
                dust3r_inference_module,
                args.baseline_src / "dust3r" / "inference.py",
                "dust3r.inference",
            ),
            "dust3r.model": _assert_module_source(
                dust3r_model_module,
                args.baseline_src / "dust3r" / "model.py",
                "dust3r.model",
            ),
            "dust3r.utils.camera": _assert_module_source(
                dust3r_camera_module,
                args.baseline_src / "dust3r" / "utils" / "camera.py",
                "dust3r.utils.camera",
            ),
            "dust3r.utils.image": _assert_module_source(
                dust3r_image_module,
                args.baseline_src / "dust3r" / "utils" / "image.py",
                "dust3r.utils.image",
            ),
        }
    )
    inference_recurrent_lighter = dust3r_inference_module.inference_recurrent_lighter
    ARCroco3DStereo = dust3r_model_module.ARCroco3DStereo
    pose_encoding_to_camera = dust3r_camera_module.pose_encoding_to_camera
    from stateguard3r.health import adapt_recal3r_trace, write_health_jsonl_atomic

    _verify_preflight_inputs(args)
    views = _prepare_input_views(args, torch)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    model, checkpoint_state_dict = _load_model_with_state_dict_audit(
        ARCroco3DStereo,
        args.checkpoint,
        torch,
    )
    _write_json_atomic(
        args.output_dir / "checkpoint-load-audit.json",
        checkpoint_state_dict,
    )
    if checkpoint_state_dict["strict"] is not False:
        raise RuntimeError("pinned ReCal3R loader no longer uses strict=False")
    if (
        checkpoint_state_dict["missing_keys"]
        or checkpoint_state_dict["unexpected_keys"]
    ):
        raise RuntimeError(
            "checkpoint state_dict has missing or unexpected keys; see "
            f"{args.output_dir / 'checkpoint-load-audit.json'}"
        )
    loaded_model_interface = _validate_model_interface(model, args.checkpoint.name)
    model = model.to(args.device)
    recal3r_runtime_config = _configure_official_recal3r_runtime(
        model,
        beta_base=args.beta_base,
    )
    runtime_config_source = args.baseline_root / "eval" / "relpose" / "launch.py"
    runtime_config_source_provenance = {
        "path": str(runtime_config_source),
        "sha256": _sha256(runtime_config_source),
        "size_bytes": runtime_config_source.stat().st_size,
        "pinned_lines": "730-742",
    }
    model.eval()
    model.enable_u_calibration_trace(oracle_window=1)

    update_calls = 0
    original_update = model._compute_recal3r_update_mask

    def counted_update(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal update_calls
        update_calls += 1
        return original_update(*call_args, **call_kwargs)

    model._compute_recal3r_update_mask = counted_update

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    outputs, state_args = inference_recurrent_lighter(
        views,
        model,
        device,
        verbose=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )

    expected_updates = len(views) - 1
    if update_calls != expected_updates:
        raise RuntimeError(
            f"expected {expected_updates} calibrated updates, observed {update_calls}"
        )
    predictions = outputs.get("pred")
    if not isinstance(predictions, list) or len(predictions) != len(views):
        raise RuntimeError("ReCal3R did not return one prediction per input frame")
    prediction_summary = _prediction_summary(predictions, torch)
    pose_jumps, geometric_residuals, trajectory = _output_health_signals(
        predictions,
        pose_encoding_to_camera=pose_encoding_to_camera,
        np=np,
    )

    trace = model.get_u_calibration_trace()
    if not isinstance(trace, dict):
        raise RuntimeError("ReCal3R calibration trace is unavailable")
    health = adapt_recal3r_trace(
        trace,
        all_frame_ids=list(range(len(views))),
        timestamps=_input_timestamps(args),
        batch_size=1,
    )
    if len(health) != len(views):
        raise RuntimeError("health ledger does not align one-to-one with inputs")
    measured_steps = [
        record.frame_id for record in health if record.uncertainty_u is not None
    ]
    expected_steps = list(range(1, len(views)))
    if measured_steps != expected_steps:
        raise RuntimeError(
            f"trace steps {measured_steps} do not match expected steps {expected_steps}"
        )
    if any(health[index].global_state_delta is None for index in expected_steps):
        raise RuntimeError("trace did not expose global-state delta for every update")
    health = [
        replace(
            record,
            pose_jump=pose_jumps[index],
            geometric_residual=geometric_residuals[index],
        )
        for index, record in enumerate(health)
    ]

    _verify_preflight_inputs(args)
    write_health_jsonl_atomic(args.output_dir / "health.jsonl", health)
    _write_json_atomic(args.output_dir / "predictions-summary.json", prediction_summary)
    input_frames = _input_frame_metadata(args)
    reference_frame = input_frames[0]
    _write_json_atomic(
        args.output_dir / "trajectory.json",
        {
            "schema_version": "stateguard3r.trajectory.v0",
            "pose_encoding": "absT_quaR",
            "matrix_convention": "camera_to_first_input_frame_reference",
            "reference_frame": {
                "frame_id": 0,
                "source_index": reference_frame["source_index"],
                "path": reference_frame["path"],
                "sha256": reference_frame["sha256"],
            },
            "pose_jump": "hypot(relative_translation_l2, relative_rotation_angle_rad)",
            "geometric_residual": (
                "median_l2(T_pose(pts3d_in_self_view)-pts3d_in_other_view) / "
                "max(median_l2(pts3d_in_other_view), 1e-8)"
            ),
            "frames": trajectory,
        },
    )
    run_finished_at = datetime.now().astimezone().isoformat()

    metadata = {
        "schema_version": "stateguard3r.recal3r-smoke.v0",
        "status": "succeeded",
        "argv": sys.argv if argv is None else [str(value) for value in argv],
        "baseline_root": str(args.baseline_root),
        "baseline_commit": args.baseline_commit,
        "baseline_tracked_worktree_clean": True,
        "module_paths": module_paths,
        "runner": runner_provenance,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_size_bytes": args.checkpoint_size_bytes,
        "input_mode": "manifest" if args.input_manifest is not None else "images",
        "input_manifest": _input_manifest_metadata(args),
        "images": [
            {"path": frame["path"], "sha256": frame["sha256"]}
            for frame in input_frames
        ],
        "input_frames": input_frames,
        "frame_count": len(views),
        "size": args.size,
        "seed": args.seed,
        "model_update_type": "recal3r",
        "loaded_model_interface": loaded_model_interface,
        "checkpoint_state_dict": checkpoint_state_dict,
        "beta_base": args.beta_base,
        "recal3r_runtime_config": recal3r_runtime_config,
        "recal3r_runtime_config_source": runtime_config_source_provenance,
        "model_eval": not model.training,
        "calibrated_update_calls": update_calls,
        "expected_calibrated_update_calls": expected_updates,
        "trace_frame_steps": measured_steps,
        "health_signal_semantics": {
            "reliability": "1 - ReCal3R uncertainty_u",
            "update_magnitude_source": "global_state_delta",
            "pose_jump": (
                "hypot(relative_translation_l2, relative_rotation_angle_rad)"
            ),
            "geometric_residual": (
                "relative_median_independent_cross_head_pointmap_consistency"
            ),
            "geometric_residual_unavailable_reason": None,
        },
        "state_args_count": len(state_args),
        "runtime_seconds": elapsed,
        "runtime_scope": "inference_recurrent_lighter_only",
        "fps": len(views) / elapsed if elapsed > 0 else None,
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "pid": os.getpid(),
        "output_dir": str(args.output_dir),
        "log_path": None,
        "peak_memory_allocated_mib": peak_memory_mib,
        "peak_memory_scope": (
            "model_resident_plus_inference_after_model_load"
            if device.type == "cuda"
            else "not_applicable_on_cpu"
        ),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    _write_json_atomic(args.output_dir / "run.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
