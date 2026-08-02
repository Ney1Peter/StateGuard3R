from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.validate_visual_overlap_v2 as replay
from stateguard3r.corruption import generate_corruption_manifest
from stateguard3r.visual_overlap import VisualOverlapResult


class _FakeCuda:
    def __init__(self) -> None:
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeRunner:
    def _prepare_views(self, paths, size: int, torch_module: _FakeTorch):
        assert size == 512
        assert not torch_module.cuda.is_initialized()
        result = []
        for path in paths:
            index = int(Path(path).stem.split("_")[-1])
            result.append({"img": np.full((1, 3, 4, 5), index / 100.0, dtype=np.float32)})
        return result


def _series(frames, *, config):
    output = [None]
    for frame_id, image in enumerate(frames[1:], start=1):
        value = float(image[0, 0, 0, 0])
        score = 0.1 if value >= 0.2 else 0.8
        output.append(
            VisualOverlapResult(
                score=score,
                status="ok",
                keypoints_previous=12,
                keypoints_current=12,
                ratio_matches=12,
                mutual_matches=12,
                inliers=12,
                inlier_ratio=1.0,
                previous_grid_coverage=score,
                current_grid_coverage=score,
                reference_frame_id=frame_id - 1,
                frame_id=frame_id,
            )
        )
    return output


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _input_root(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = []
    for index in range(30):
        frame = raw / f"frame_{index}.png"
        frame.write_bytes(f"frame-{index}".encode())
        frames.append(frame)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "sequence": "visual-overlap-test",
                "frames": [{"path": str(path), "timestamp": index} for index, path in enumerate(frames)],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "frozen-inputs"
    rows = []
    for run_id in sorted(replay.EXPECTED_RUN_IDS):
        split = "development" if run_id.startswith("development-") else "holdout"
        run_root = root / split / run_id
        run_root.mkdir(parents=True)
        source_copy = run_root / "source-manifest.json"
        source_copy.write_bytes(source.read_bytes())
        manifest = run_root / "input-manifest.json"
        generate_corruption_manifest(
            source_copy,
            manifest,
            [
                {
                    "type": "low_overlap_jump",
                    "start": 15,
                    "end": 19,
                    "parameters": {"source_start": 20},
                }
            ],
            check_paths=True,
        )
        rows.append({"run_id": run_id, "dataset_split": split})
    (root / replay.INPUT_REGISTRY_FILENAME).write_text(
        json.dumps({"runs": rows}), encoding="utf-8"
    )
    _freeze_tree(root)
    return root


def _provenance() -> dict[str, object]:
    return {
        "version": "synthetic",
        "module_path": "/synthetic/cv2/__init__.py",
        "module_sha256": hashlib.sha256(b"module").hexdigest(),
        "binary_path": "/synthetic/cv2/cv2.abi3.so",
        "binary_sha256": hashlib.sha256(b"binary").hexdigest(),
        "build_info_sha256": hashlib.sha256(b"build").hexdigest(),
        "threads": 1,
        "opencl_enabled": False,
        "rng_seed": 0,
    }


def test_replay_retains_all_candidates_and_freezes_only_online_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _input_root(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(replay, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(replay, "_baseline_provenance", lambda root: {"root": str(root)})
    monkeypatch.setattr(replay, "_generator_provenance", lambda: {"commit": "test"})
    output = outputs / "candidate-replay"

    actual = replay.run_visual_overlap_replay(
        input_root,
        output,
        baseline_root=tmp_path,
        torch_module=_FakeTorch(),
        runner_module=_FakeRunner(),
        series_function=_series,
        provenance_function=_provenance,
    )

    assert actual == output
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    payload = json.loads((output / "visual-overlap-replay.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["checks"]["all_four_candidates_retained"] is True
    assert len(payload["candidates"]) == 4
    for candidate in payload["candidates"]:
        assert candidate["checks"]["byte_identical_second_scoring"] is True
        assert candidate["directional_summary"]["clean_contiguous_strictly_higher"] is True
        for run in candidate["runs"].values():
            assert all(set(row) == {"frame_id", "overlap"} for row in run["online_ledger"])
            assert run["online_ledger"][0] == {"frame_id": 0, "overlap": None}
            assert all(row["overlap"] is not None for row in run["online_ledger"][1:])

    with pytest.raises(replay.VisualOverlapReplayError, match="already exists"):
        replay.run_visual_overlap_replay(
            input_root,
            output,
            baseline_root=tmp_path,
            torch_module=_FakeTorch(),
            runner_module=_FakeRunner(),
            series_function=_series,
            provenance_function=_provenance,
        )


def test_replay_rejects_visible_or_initialized_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = _input_root(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(replay, "OUTPUT_ROOT", outputs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(replay.VisualOverlapReplayError, match="CUDA_VISIBLE_DEVICES"):
        replay.run_visual_overlap_replay(
            input_root,
            outputs / "not-created",
            baseline_root=tmp_path,
            torch_module=_FakeTorch(),
            runner_module=_FakeRunner(),
            series_function=_series,
            provenance_function=_provenance,
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch_module = _FakeTorch()
    torch_module.cuda.initialized = True
    with pytest.raises(replay.VisualOverlapReplayError, match="initialized"):
        replay.run_visual_overlap_replay(
            input_root,
            outputs / "also-not-created",
            baseline_root=tmp_path,
            torch_module=torch_module,
            runner_module=_FakeRunner(),
            series_function=_series,
            provenance_function=_provenance,
        )
