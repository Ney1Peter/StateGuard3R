#!/usr/bin/env python3
"""Replay Detector v2 visual-overlap candidates on frozen v1 model inputs.

This is a CPU-only *development* checker.  It uses the official ReCal3R image
loader followed by the frozen manifest's deferred transforms, then gives only
adjacent model-ready RGB images to the online correspondence function.  In
particular, depth, ground truth, corruption labels, event intervals, health
ledgers, and model responses are not inputs to the online ledger.

Run this script with ReCal3R's existing virtual environment, which contains
the pinned OpenCV build.  It does not load a checkpoint or run ReCal3R model
inference.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
STATE_SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(STATE_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(STATE_SOURCE_ROOT))

from stateguard3r.input_manifest import InputManifest, load_input_manifest
from stateguard3r.visual_overlap import (
    SCHEMA_VERSION as VISUAL_OVERLAP_SCHEMA_VERSION,
    VisualOverlapConfig,
    VisualOverlapResult,
    normalized_tensor_to_uint8_rgb,
    online_visual_correspondence_series,
    opencv_runtime_provenance,
)


OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"
INPUT_REGISTRY_FILENAME = "formal-pilot-manifest.json"
INPUT_MANIFEST_FILENAME = "input-manifest.json"
REPLAY_SCHEMA_VERSION = "stateguard3r.visual-overlap-replay.v1"
EXPECTED_FRAME_COUNT = 30
EXPECTED_RUN_IDS = frozenset(
    {
        "development-dynamic",
        "development-low",
        "development-wrong",
        "holdout-dynamic",
        "holdout-low",
        "holdout-wrong",
    }
)
RATIO_THRESHOLDS = (0.70, 0.80)
RANSAC_PIXEL_LIMITS = (1.0, 2.0)
RENAME_NOREPLACE = 1
AT_FDCWD = -100


class VisualOverlapReplayError(ValueError):
    """Raised when frozen input replay or immutable publication is unsafe."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == root or root in resolved.parents


def _require_read_only_file(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise VisualOverlapReplayError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise VisualOverlapReplayError(f"{label} must be a regular non-symlink file")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise VisualOverlapReplayError(f"{label} must be frozen at mode 0444")
    try:
        return path.read_bytes()
    except OSError as error:
        raise VisualOverlapReplayError(f"cannot read {label}: {error}") from error


def _require_read_only_directory(path: Path, *, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise VisualOverlapReplayError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise VisualOverlapReplayError(f"{label} must be a directory, not a symlink")
    if stat.S_IMODE(metadata.st_mode) != 0o555:
        raise VisualOverlapReplayError(f"{label} must be frozen at mode 0555")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VisualOverlapReplayError(f"{label} contains non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise VisualOverlapReplayError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, VisualOverlapReplayError) as error:
        raise VisualOverlapReplayError(f"{label} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise VisualOverlapReplayError(f"{label} must contain a JSON object")
    return value


def _manifest_runs(input_root: Path) -> list[tuple[str, str, Path, dict[str, str]]]:
    _require_read_only_directory(input_root, label="frozen v1 input root")
    registry_path = input_root / INPUT_REGISTRY_FILENAME
    registry_bytes = _require_read_only_file(registry_path, label="v1 input registry")
    registry = _strict_json_object(registry_bytes, label="v1 input registry")
    rows = registry.get("runs")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_RUN_IDS):
        raise VisualOverlapReplayError("v1 input registry must declare exactly six runs")
    output: list[tuple[str, str, Path, dict[str, str]]] = []
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise VisualOverlapReplayError("v1 input registry has a non-object run")
        run_id = row.get("run_id")
        split = row.get("dataset_split")
        if not isinstance(run_id, str) or not isinstance(split, str):
            raise VisualOverlapReplayError("v1 input registry run identity is malformed")
        if run_id in observed:
            raise VisualOverlapReplayError(f"v1 input registry repeats {run_id}")
        observed.add(run_id)
        if split not in {"development", "holdout"}:
            raise VisualOverlapReplayError(f"{run_id} has an invalid split")
        run_root = input_root / split / run_id
        _require_read_only_directory(run_root, label=f"frozen input directory {run_id}")
        manifest_path = run_root / INPUT_MANIFEST_FILENAME
        manifest_bytes = _require_read_only_file(
            manifest_path,
            label=f"frozen input manifest {run_id}",
        )
        output.append(
            (
                run_id,
                split,
                manifest_path,
                {
                    "path": str(manifest_path.relative_to(input_root)),
                    "sha256": _sha256_bytes(manifest_bytes),
                    "size_bytes": len(manifest_bytes),
                },
            )
        )
    if observed != EXPECTED_RUN_IDS:
        raise VisualOverlapReplayError("v1 input registry run IDs differ from the frozen six-run pool")
    return sorted(output, key=lambda item: item[0])


def _baseline_provenance(baseline_root: Path) -> dict[str, Any]:
    from scripts import run_recal3r_smoke as runner

    baseline_root = baseline_root.resolve(strict=True)
    if not (baseline_root / "src" / "dust3r" / "utils" / "image.py").is_file():
        raise VisualOverlapReplayError("baseline root lacks ReCal3R's official image loader")
    try:
        commit = runner._git_output(baseline_root, "rev-parse", "HEAD")
        changes = runner._git_output(
            baseline_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except OSError as error:
        raise VisualOverlapReplayError(f"cannot inspect ReCal3R baseline: {error}") from error
    if commit != runner.PINNED_RECAL3R_COMMIT:
        raise VisualOverlapReplayError("ReCal3R commit is not the pinned baseline")
    if changes:
        raise VisualOverlapReplayError("ReCal3R has tracked changes")
    loader = baseline_root / "src" / "dust3r" / "utils" / "image.py"
    return {
        "root": str(baseline_root),
        "commit": commit,
        "official_loader": {
            "path": str(loader),
            "sha256": _sha256_file(loader),
            "size_bytes": loader.stat().st_size,
        },
    }


def _generator_provenance() -> dict[str, Any]:
    try:
        commit = subprocess_run_git(REPOSITORY_ROOT, "rev-parse", "HEAD")
        changes = subprocess_run_git(
            REPOSITORY_ROOT,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    except OSError as error:
        raise VisualOverlapReplayError(f"cannot inspect StateGuard3R worktree: {error}") from error
    if changes:
        raise VisualOverlapReplayError(
            "StateGuard3R has tracked changes; replay must run from a clean commit"
        )
    script = Path(__file__).resolve()
    visual_module = STATE_SOURCE_ROOT / "stateguard3r" / "visual_overlap.py"
    return {
        "state_guard_commit": commit,
        "script": {"path": str(script), "sha256": _sha256_file(script), "size_bytes": script.stat().st_size},
        "visual_overlap_module": {
            "path": str(visual_module),
            "sha256": _sha256_file(visual_module),
            "size_bytes": visual_module.stat().st_size,
        },
    }


def subprocess_run_git(repository: Path, *arguments: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _require_cpu_only(torch_module: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise VisualOverlapReplayError("CPU visual replay requires CUDA_VISIBLE_DEVICES='' ")
    cuda = getattr(torch_module, "cuda", None)
    initialized = getattr(cuda, "is_initialized", None)
    if not callable(initialized) or bool(initialized()):
        raise VisualOverlapReplayError("CUDA was initialized before CPU visual replay")


def _model_ready_sha256(image: Any) -> str:
    return _sha256_bytes(normalized_tensor_to_uint8_rgb(image).tobytes())


def _structural_pair_kind(manifest: InputManifest, index: int) -> str:
    previous = manifest.frames[index - 1]
    current = manifest.frames[index]
    if current.source_index != previous.source_index + 1:
        return "source_discontinuity"
    if previous.transforms or current.transforms:
        return "deferred_pixel_or_order_transform"
    return "clean_contiguous"


def _score_run(
    *,
    run_id: str,
    manifest: InputManifest,
    views: Sequence[Mapping[str, Any]],
    config: VisualOverlapConfig,
    series_function: Callable[..., list[VisualOverlapResult | None]],
) -> dict[str, Any]:
    if len(manifest.frames) != EXPECTED_FRAME_COUNT or len(views) != EXPECTED_FRAME_COUNT:
        raise VisualOverlapReplayError(f"{run_id} must contain exactly {EXPECTED_FRAME_COUNT} frames")
    images = [view["img"] for view in views]
    before_hashes = [_model_ready_sha256(image) for image in images]
    results = series_function(images, config=config)
    after_hashes = [_model_ready_sha256(image) for image in images]
    if before_hashes != after_hashes:
        raise VisualOverlapReplayError(f"{run_id} visual overlap modified a model-ready input")
    if len(results) != EXPECTED_FRAME_COUNT or results[0] is not None:
        raise VisualOverlapReplayError(f"{run_id} online series lacks the required null frame zero")
    ledger: list[dict[str, Any]] = [{"frame_id": 0, "overlap": None}]
    diagnostics: list[dict[str, Any]] = []
    structural_counts = {
        "clean_contiguous": 0,
        "deferred_pixel_or_order_transform": 0,
        "source_discontinuity": 0,
    }
    for frame_id, result in enumerate(results[1:], start=1):
        if not isinstance(result, VisualOverlapResult):
            raise VisualOverlapReplayError(f"{run_id} frame {frame_id} did not yield a visual result")
        if result.frame_id not in {None, frame_id} or result.reference_frame_id not in {None, frame_id - 1}:
            raise VisualOverlapReplayError(f"{run_id} frame {frame_id} has inconsistent causal IDs")
        if not math.isfinite(result.score) or not 0.0 <= result.score <= 1.0:
            raise VisualOverlapReplayError(f"{run_id} frame {frame_id} has non-finite online overlap")
        pair_kind = _structural_pair_kind(manifest, frame_id)
        structural_counts[pair_kind] += 1
        ledger.append({"frame_id": frame_id, "overlap": result.score})
        diagnostics.append({**asdict(result), "structural_pair_kind": pair_kind})
    if any(set(record) != {"frame_id", "overlap"} for record in ledger):
        raise VisualOverlapReplayError("online ledger contains non-online fields")
    return {
        "model_ready_input": "official_loader_post_deferred_transform_normalized_rgb",
        "model_ready_uint8_rgb_sha256": before_hashes,
        "online_ledger": ledger,
        "frame_diagnostics": diagnostics,
        "structural_pair_counts": structural_counts,
    }


def _candidate_payload(
    *,
    run_inputs: Sequence[tuple[str, str, Path, dict[str, str]]],
    size: int,
    torch_module: Any,
    runner_module: Any,
    config: VisualOverlapConfig,
    series_function: Callable[..., list[VisualOverlapResult | None]],
) -> dict[str, Any]:
    from stateguard3r.input_manifest import apply_deferred_transforms

    runs: dict[str, Any] = {}
    for run_id, split, manifest_path, input_artifact in run_inputs:
        manifest = load_input_manifest(manifest_path)
        paths = [frame.path for frame in manifest.frames]
        loaded = runner_module._prepare_views(paths, size, torch_module)
        views = apply_deferred_transforms(loaded, manifest)
        runs[run_id] = {
            "dataset_split": split,
            "input_manifest": input_artifact,
            **_score_run(
                run_id=run_id,
                manifest=manifest,
                views=views,
                config=config,
                series_function=series_function,
            ),
        }
    return {
        "candidate_id": (
            f"orb2000-ratio{config.ratio_threshold:.2f}-ransac{config.ransac_pixel_limit:.1f}"
        ),
        "config": config.to_dict(),
        "runs": runs,
    }


def _directional_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    categories = {"clean_contiguous": [], "source_discontinuity": []}
    for run in candidate["runs"].values():
        for diagnostic in run["frame_diagnostics"]:
            kind = diagnostic["structural_pair_kind"]
            if kind in categories:
                categories[kind].append(float(diagnostic["score"]))
    clean = categories["clean_contiguous"]
    discontinuous = categories["source_discontinuity"]
    if not clean or not discontinuous:
        raise VisualOverlapReplayError("candidate lacks structural clean or discontinuous pairs")
    clean_mean = float(sum(clean) / len(clean))
    discontinuous_mean = float(sum(discontinuous) / len(discontinuous))
    return {
        "clean_contiguous_count": len(clean),
        "clean_contiguous_mean": clean_mean,
        "source_discontinuity_count": len(discontinuous),
        "source_discontinuity_mean": discontinuous_mean,
        "clean_contiguous_strictly_higher": clean_mean > discontinuous_mean,
    }


def _freeze_tree(staging: Path) -> None:
    for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in staging.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    staging.chmod(0o555)


def _rename_noreplace(source: Path, destination: Path) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as error:
        raise VisualOverlapReplayError("atomic no-replace directory publication is unavailable") from error
    result = renameat2(
        ctypes.c_int(AT_FDCWD),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(AT_FDCWD),
        ctypes.c_char_p(os.fsencode(destination)),
        ctypes.c_uint(RENAME_NOREPLACE),
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise VisualOverlapReplayError(f"refusing to replace existing output: {destination}")
    raise VisualOverlapReplayError(
        f"atomic no-replace publication failed for {destination}: {os.strerror(error_number)}"
    )


def _publish(output_dir: Path, files: Mapping[str, bytes]) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise VisualOverlapReplayError(f"output path already exists: {output_dir}")
    if not _inside(output_dir, OUTPUT_ROOT):
        raise VisualOverlapReplayError(f"output must be inside {OUTPUT_ROOT}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        for relative, payload in files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        _freeze_tree(staging)
        _rename_noreplace(staging, output_dir)
        staging = None
        return output_dir
    finally:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            shutil.rmtree(staging)


def _report_markdown(payload: Mapping[str, Any]) -> bytes:
    lines = [
        "# Detector v2 visual-overlap replay",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This CPU-only development replay used only adjacent, model-ready RGB images. "
        "Its online ledger has exactly `frame_id` and `overlap`; it contains no depth, "
        "GT pose, label, event interval, health, or model-response field.",
        "",
        "Visual correspondence coverage is an online proxy, not physical overlap percentage.",
        "",
        "| Candidate | finite frames 1–29 | deterministic replay | clean mean | discontinuity mean | clean > discontinuity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in payload["candidates"]:
        summary = candidate["directional_summary"]
        lines.append(
            "| {id} | {finite} | {deterministic} | {clean:.9f} | {jump:.9f} | {direction} |".format(
                id=candidate["candidate_id"],
                finite="PASS" if candidate["checks"]["all_noninitial_overlap_finite"] else "FAIL",
                deterministic="PASS" if candidate["checks"]["byte_identical_second_scoring"] else "FAIL",
                clean=summary["clean_contiguous_mean"],
                jump=summary["source_discontinuity_mean"],
                direction="PASS" if summary["clean_contiguous_strictly_higher"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "`source_discontinuity` is determined only from adjacent manifest source indices; "
            "it is a structural diagnostic, not a detector input or a corruption label.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def run_visual_overlap_replay(
    input_root: Path,
    output_dir: Path,
    *,
    baseline_root: Path,
    size: int = 512,
    torch_module: Any | None = None,
    runner_module: Any | None = None,
    series_function: Callable[..., list[VisualOverlapResult | None]] = online_visual_correspondence_series,
    provenance_function: Callable[[], dict[str, Any]] = opencv_runtime_provenance,
) -> Path:
    """Score every allowed candidate twice and atomically freeze its evidence."""

    if size != 512:
        raise VisualOverlapReplayError("visual-overlap development replay requires size 512")
    input_root = input_root.resolve(strict=True)
    output_dir = output_dir.resolve(strict=False)
    baseline_root = baseline_root.resolve(strict=True)
    run_inputs = _manifest_runs(input_root)
    if torch_module is None:
        import torch as imported_torch

        torch_module = imported_torch
    _require_cpu_only(torch_module)
    if runner_module is None:
        if str(baseline_root) not in sys.path:
            sys.path.insert(0, str(baseline_root))
        baseline_src = baseline_root / "src"
        if str(baseline_src) not in sys.path:
            sys.path.insert(0, str(baseline_src))
        from scripts import run_recal3r_smoke as imported_runner

        runner_module = imported_runner
    candidates: list[dict[str, Any]] = []
    for ratio_threshold in RATIO_THRESHOLDS:
        for ransac_pixel_limit in RANSAC_PIXEL_LIMITS:
            config = VisualOverlapConfig(
                ratio_threshold=ratio_threshold,
                ransac_pixel_limit=ransac_pixel_limit,
            )
            first = _candidate_payload(
                run_inputs=run_inputs,
                size=size,
                torch_module=torch_module,
                runner_module=runner_module,
                config=config,
                series_function=series_function,
            )
            # Re-score the identical immutable model-ready views through a fresh official
            # loader replay.  Only the candidate payload is compared, so no runtime path
            # or process metadata can hide a non-deterministic correspondence result.
            second = _candidate_payload(
                run_inputs=run_inputs,
                size=size,
                torch_module=torch_module,
                runner_module=runner_module,
                config=config,
                series_function=series_function,
            )
            byte_identical = _canonical_json_bytes(first) == _canonical_json_bytes(second)
            if not byte_identical:
                raise VisualOverlapReplayError(f"{first['candidate_id']} is not byte-deterministic")
            all_finite = all(
                all(
                    entry["overlap"] is None
                    if entry["frame_id"] == 0
                    else isinstance(entry["overlap"], float) and math.isfinite(entry["overlap"])
                    for entry in run["online_ledger"]
                )
                for run in first["runs"].values()
            )
            first["directional_summary"] = _directional_summary(first)
            first["checks"] = {
                "byte_identical_second_scoring": byte_identical,
                "all_noninitial_overlap_finite": all_finite,
                "online_ledger_has_only_frame_id_and_overlap": True,
            }
            candidates.append(first)
    _require_cpu_only(torch_module)
    candidate_passes = [
        candidate
        for candidate in candidates
        if candidate["checks"]["all_noninitial_overlap_finite"]
        and candidate["checks"]["byte_identical_second_scoring"]
        and candidate["directional_summary"]["clean_contiguous_strictly_higher"]
    ]
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "PASS" if candidate_passes else "FAIL",
        "purpose": "disclosed_v1_development_visual_overlap_candidate_replay",
        "online_signal": {
            "schema_version": VISUAL_OVERLAP_SCHEMA_VERSION,
            "definition": "verified_visual_correspondence_coverage_not_physical_overlap_percentage",
            "input": "post_official_loader_post_deferred_transform_normalized_rgb",
            "causal_reference": "immediately_previous_frame_only",
            "forbidden_online_inputs": [
                "depth",
                "groundtruth",
                "gt_pose",
                "corruption_label",
                "event_interval",
                "future_frame",
                "health_ledger",
                "model_response",
            ],
            "ledger_field_names": ["frame_id", "overlap"],
        },
        "input_root": {
            "path": str(input_root),
            "formal_pilot_manifest_sha256": _sha256_file(input_root / INPUT_REGISTRY_FILENAME),
        },
        "baseline": _baseline_provenance(baseline_root),
        "generator": _generator_provenance(),
        "opencv": provenance_function(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frame_count": EXPECTED_FRAME_COUNT,
        "candidates": candidates,
        "checks": {
            "cuda_hidden_and_uninitialized": True,
            "all_six_disclosed_runs_replayed": set(run_id for run_id, _, _, _ in run_inputs) == EXPECTED_RUN_IDS,
            "all_four_candidates_retained": len(candidates) == 4,
            "at_least_one_candidate_has_clean_greater_than_source_discontinuity": bool(candidate_passes),
        },
    }
    report = _report_markdown(payload)
    files = {
        "visual-overlap-replay.json": _canonical_json_bytes(payload),
        "report.md": report,
    }
    _publish(output_dir, files)
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPOSITORY_ROOT.parent / "baselines" / "ReCal3R",
    )
    parser.add_argument("--size", type=int, default=512, choices=(512,))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_visual_overlap_replay(
        args.input_root,
        args.output_dir,
        baseline_root=args.baseline_root,
        size=args.size,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
