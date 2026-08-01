from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pytest

import scripts.prepare_formal_pilot_inputs as preparer
import scripts.validate_formal_pilot_inputs as validator
from tests.test_prepare_formal_pilot_inputs import (
    _make_tum_fixture,
    _test_source_policy,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fake_run_commitment(run: Any) -> dict[str, Any]:
    def artifact(payload: bytes) -> dict[str, Any]:
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    return {
        "run_id": run.run_id,
        "dataset_split": run.dataset_split,
        "corruption_type": run.corruption_type,
        "raw_frame_sha256s": list(run.raw_frame_sha256s),
        "artifacts": {
            "source_manifest": artifact(run.source_manifest_json),
            "input_manifest": artifact(run.input_manifest_json),
            "corruption_json": artifact(run.corruption_json),
        },
    }


def _prepare_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(preparer, "make_run_commitment", _fake_run_commitment)
    monkeypatch.setattr(
        preparer,
        "make_split_registry",
        lambda development, holdout: _json_bytes(
            {"development": development, "holdout": holdout}
        ),
    )
    monkeypatch.setattr(
        preparer,
        "make_holdout_commitment",
        lambda holdout: _json_bytes({"holdout": holdout}),
    )
    tum_root = _make_tum_fixture(tmp_path)
    input_root = tmp_path / "formal-inputs"
    preparer.prepare_formal_pilot_inputs(
        tum_root,
        input_root,
        _source_policy=_test_source_policy(tum_root),
    )
    return input_root


def _prepare_contract_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    import stateguard3r.detection_suite as detection_suite

    tum_root = _make_tum_fixture(tmp_path)
    policy = _test_source_policy(tum_root)
    frozen_suite_values = {
        "FORMAL_DATASET_ROOT": str(policy.dataset_root),
        "FORMAL_ARCHIVE_PATH": str(policy.archive_path),
        "FORMAL_ARCHIVE_SIZE_BYTES": policy.archive_size_bytes,
        "FORMAL_ARCHIVE_SHA256": policy.archive_sha256,
        "FORMAL_RAW_MANIFEST_PATH": str(policy.raw_manifest_path),
        "FORMAL_RAW_MANIFEST_SIZE_BYTES": policy.raw_manifest_size_bytes,
        "FORMAL_RAW_MANIFEST_SHA256": policy.raw_manifest_sha256,
    }
    for name, value in frozen_suite_values.items():
        monkeypatch.setattr(detection_suite, name, value)
    input_root = tmp_path / "formal-contract-inputs"
    preparer.prepare_formal_pilot_inputs(
        tum_root,
        input_root,
        _source_policy=policy,
    )
    return input_root


def _provenance() -> dict[str, Any]:
    script = Path(validator.__file__).read_bytes()
    return {
        "repository_commit": "a" * 40,
        "tracked_worktree_clean": True,
        "git_blob_sha256": hashlib.sha256(script).hexdigest(),
        "script": {
            "path": validator.GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": hashlib.sha256(script).hexdigest(),
            "size_bytes": len(script),
        },
    }


def _runtime_provenance() -> dict[str, Any]:
    loader_path = validator.RECAL3R_ROOT / validator.OFFICIAL_LOADER_RELATIVE_PATH
    loader_bytes = loader_path.read_bytes()
    executable = validator.RECAL3R_ROOT / ".venv" / "bin" / "python"
    return {
        "recal3r_repository": {
            "path": str(validator.RECAL3R_ROOT),
            "commit": validator.EXPECTED_RECAL3R_COMMIT,
            "tracked_worktree_clean": True,
        },
        "python": {
            "executable": str(executable),
            "executable_realpath": str(executable.resolve(strict=True)),
            "prefix": str((validator.RECAL3R_ROOT / ".venv").resolve(strict=True)),
        },
        "official_loader": {
            "path": validator.OFFICIAL_LOADER_RELATIVE_PATH.as_posix(),
            "sha256": hashlib.sha256(loader_bytes).hexdigest(),
            "size_bytes": len(loader_bytes),
        },
    }


def _report_path(tmp_path: Path, name: str) -> Path:
    parent = (
        tmp_path
        / "outputs"
        / name
        / validator.PRE_FORWARD_VALIDATION_DIRNAME
    )
    parent.mkdir(parents=True)
    return parent / validator.CPU_VALIDATION_FILENAME


def _rewrite_json(path: Path, value: Any) -> bytes:
    payload = _json_bytes(value)
    path.chmod(0o644)
    path.write_bytes(payload)
    path.chmod(0o444)
    return payload


def _artifact(path_text: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path_text,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _update_run_artifacts(
    input_root: Path,
    run_id: str,
    *,
    source_payload: bytes | None = None,
    input_payload: bytes | None = None,
    groundtruth_source_entry_index: int | None = None,
) -> None:
    registry_path = input_root / validator.INPUT_REGISTRY_FILENAME
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    record = next(row for row in registry["runs"] if row["run_id"] == run_id)
    if source_payload is not None:
        path_text = record["artifacts"]["source_manifest"]["path"]
        record["artifacts"]["source_manifest"] = _artifact(
            path_text, source_payload
        )
        record["suite_run_commitment"]["artifacts"]["source_manifest"] = {
            "sha256": hashlib.sha256(source_payload).hexdigest(),
            "size_bytes": len(source_payload),
        }
    if input_payload is not None:
        path_text = record["artifacts"]["input_manifest"]["path"]
        exact = _artifact(path_text, input_payload)
        record["artifacts"]["input_manifest"] = exact
        record["artifacts"]["corruption_json"] = dict(exact)
        for key in ("input_manifest", "corruption_json"):
            record["suite_run_commitment"]["artifacts"][key] = {
                "sha256": exact["sha256"],
                "size_bytes": exact["size_bytes"],
            }
    if groundtruth_source_entry_index is not None:
        record["consumed_associations"]["groundtruth_source_entry_indices"][
            0
        ] = groundtruth_source_entry_index
    _rewrite_json(registry_path, registry)


def _tamper_dynamic_transform(input_root: Path) -> None:
    spec = validator.FROZEN_RUN_SPECS[0]
    input_path = (
        input_root
        / spec.dataset_split
        / spec.run_id
        / validator.INPUT_MANIFEST_FILENAME
    )
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["frames"][validator.EVENT_START]["transforms"][0]["rectangle"][
        "x"
    ] = 0.30
    input_bytes = _rewrite_json(input_path, payload)
    _update_run_artifacts(
        input_root, spec.run_id, input_payload=input_bytes
    )


def _tamper_groundtruth_association(input_root: Path) -> None:
    spec = validator.FROZEN_RUN_SPECS[0]
    run_root = input_root / spec.dataset_split / spec.run_id
    source_path = run_root / validator.SOURCE_MANIFEST_FILENAME
    input_path = run_root / validator.INPUT_MANIFEST_FILENAME
    source = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    target_index = 100
    gt_path = Path(source["official_lineage"]["dataset_root"]) / "groundtruth.txt"
    data_lines: list[tuple[int, bytes, list[str]]] = []
    for line_number, raw_line in enumerate(
        gt_path.read_bytes().splitlines(keepends=True), start=1
    ):
        tokens = raw_line.decode("utf-8").strip().split()
        if tokens and not tokens[0].startswith("#"):
            data_lines.append((line_number, raw_line, tokens))
    source_line, raw_line, tokens = data_lines[target_index]
    rgb_timestamp = Decimal(source["frames"][0]["timestamp_text"])
    gt_timestamp = Decimal(tokens[0])
    delta = gt_timestamp - rgb_timestamp
    replacement = {
        "source_entry_index": target_index,
        "source_line": source_line,
        "physical_line_sha256": hashlib.sha256(raw_line).hexdigest(),
        "physical_line_size_bytes": len(raw_line),
        "timestamp": float(gt_timestamp),
        "timestamp_text": tokens[0],
        "translation_xyz": [float(value) for value in tokens[1:4]],
        "quaternion_xyzw": [float(value) for value in tokens[4:8]],
        "translation_xyz_text": tokens[1:4],
        "quaternion_xyzw_text": tokens[4:8],
        "delta_from_rgb_seconds": float(delta),
        "absolute_delta_seconds": float(abs(delta)),
    }
    source["frames"][0]["groundtruth"] = replacement
    manifest["frames"][0]["metadata"]["groundtruth"] = replacement
    source_bytes = _rewrite_json(source_path, source)
    manifest["source_manifest_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    input_bytes = _rewrite_json(input_path, manifest)
    _update_run_artifacts(
        input_root,
        spec.run_id,
        source_payload=source_bytes,
        input_payload=input_bytes,
        groundtruth_source_entry_index=target_index,
    )


class _FakeCuda:
    def __init__(self) -> None:
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeRunner:
    def __init__(
        self,
        *,
        initialize_cuda: bool = False,
        mutate_source: bool = False,
        mutate_input_tree: Path | None = None,
        emit_stdout: bool = False,
    ) -> None:
        self.initialize_cuda = initialize_cuda
        self.mutate_source = mutate_source
        self.mutate_input_tree = mutate_input_tree
        self.emit_stdout = emit_stdout
        self.calls: list[tuple[Path, ...]] = []

    def _prepare_views(
        self, image_paths: Sequence[Path], size: int, torch_module: _FakeTorch
    ) -> list[dict[str, np.ndarray]]:
        assert size == 512
        paths = tuple(Path(path) for path in image_paths)
        self.calls.append(paths)
        if self.emit_stdout:
            print("synthetic official-loader progress")
        views: list[dict[str, np.ndarray]] = []
        for path in paths:
            seed = int(hashlib.sha256(path.read_bytes()).hexdigest()[:8], 16)
            image = np.arange(240, dtype=np.float32).reshape(1, 3, 8, 10)
            image = (image + (seed % 997)) / np.float32(1000.0)
            views.append({"img": image})
        if self.initialize_cuda:
            torch_module.cuda.initialized = True
        if self.mutate_source and len(self.calls) == 1:
            target = paths[0]
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b"-changed-during-replay")
            target.chmod(0o444)
        if self.mutate_input_tree is not None and len(self.calls) == 1:
            root = self.mutate_input_tree
            root.chmod(0o755)
            injected = root / "injected-during-loader.json"
            injected.write_text("{}\n", encoding="utf-8")
            injected.chmod(0o444)
            root.chmod(0o555)
        return views


def _run(
    input_root: Path,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner: _FakeRunner | None = None,
    torch_module: _FakeTorch | None = None,
) -> Path:
    runtime_torch = _FakeTorch() if torch_module is None else torch_module
    runtime_runner = _FakeRunner() if runner is None else runner
    registry = json.loads(
        (input_root / validator.INPUT_REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(validator, "OUTPUT_ROOT", output.parents[2])
    monkeypatch.setattr(
        validator, "EXPECTED_TUM_ROOT", Path(registry["official_lineage"]["dataset_root"])
    )
    monkeypatch.setattr(validator, "_capture_generator_provenance", _provenance)
    monkeypatch.setattr(
        validator,
        "_capture_production_runtime_provenance",
        lambda baseline_root, *, require_loader_imported: _runtime_provenance(),
    )
    monkeypatch.setattr(
        validator,
        "_load_production_runtime",
        lambda baseline_root: (runtime_torch, runtime_runner),
    )
    return validator.validate_formal_pilot_inputs(input_root, output)


def test_generates_schema_exact_immutable_report_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    output = _report_path(tmp_path, "successful-validation")
    runner = _FakeRunner(emit_stdout=True)
    torch_module = _FakeTorch()

    result = _run(
        input_root,
        output,
        monkeypatch,
        runner=runner,
        torch_module=torch_module,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("synthetic official-loader progress") == 6

    assert result == output
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    report_bytes = output.read_bytes()
    report = json.loads(report_bytes)
    assert set(report) == {
        "schema_version",
        "status",
        "input_registry",
        "run_ids",
        "environment",
        "source_snapshots_before_sha256",
        "source_snapshots_after_sha256",
        "source_snapshot_artifact_count",
        "input_tree_snapshots_before_sha256",
        "input_tree_snapshots_after_sha256",
        "input_tree_snapshot_entry_count",
        "tum_associations",
        "runs",
        "generator",
        "production_runtime",
        "checks",
    }
    assert report["schema_version"] == validator.CPU_VALIDATION_SCHEMA_VERSION
    assert report["status"] == "PASS"
    assert report["run_ids"] == {
        "development": [
            "development-dynamic",
            "development-wrong",
            "development-low",
        ],
        "holdout": ["holdout-dynamic", "holdout-wrong", "holdout-low"],
    }
    assert report["environment"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "torch_cuda_is_initialized_before": False,
        "torch_cuda_is_initialized_after": False,
    }
    assert report["source_snapshot_artifact_count"] == 385
    assert report["source_snapshots_before_sha256"] == report[
        "source_snapshots_after_sha256"
    ]
    assert report["input_tree_snapshots_before_sha256"] == report[
        "input_tree_snapshots_after_sha256"
    ]
    assert report["input_tree_snapshot_entry_count"] == 24
    assert report["production_runtime"] == _runtime_provenance()
    assert report["generator"] == _provenance()
    assert report["tum_associations"] == {
        "method": "unique_nearest_absolute_timestamp",
        "tie_policy": "reject",
        "max_absolute_delta_seconds": 0.02,
        "allocated_rgb_count": 190,
        "globally_unique_depth_count": 190,
        "globally_unique_groundtruth_count": 190,
        "association_digest_sha256": report["tum_associations"][
            "association_digest_sha256"
        ],
    }
    assert len(report["tum_associations"]["association_digest_sha256"]) == 64
    assert report["checks"] == {
        "strict_manifest_validation": True,
        "cpu_pixel_replay": True,
        "raw_source_snapshot_unchanged": True,
        "input_tree_snapshot_unchanged": True,
        "official_tum_associations": True,
        "production_runtime_provenance": True,
        "cuda_hidden": True,
        "six_run_coverage": True,
    }
    assert [row["run_id"] for row in report["runs"]] == [
        spec.run_id for spec in validator.FROZEN_RUN_SPECS
    ]
    assert all(row["frame_count"] == 30 for row in report["runs"])
    assert len(runner.calls) == 6
    registry = json.loads(
        (input_root / validator.INPUT_REGISTRY_FILENAME).read_text(encoding="utf-8")
    )
    by_id = {record["run_id"]: record for record in registry["runs"]}
    for call, spec in zip(runner.calls, validator.FROZEN_RUN_SPECS, strict=True):
        source = json.loads(
            (
                input_root
                / spec.dataset_split
                / spec.run_id
                / validator.SOURCE_MANIFEST_FILENAME
            ).read_text(encoding="utf-8")
        )
        expected = tuple(
            (
                input_root
                / spec.dataset_split
                / spec.run_id
                / source["frames"][source_index]["path"]
            ).resolve(strict=True)
            for source_index in by_id[spec.run_id][
                "consumed_source_pool_indices_by_output_position"
            ]
        )
        assert call == expected

    with pytest.raises(
        validator.FormalPilotValidationError, match="refusing to overwrite"
    ):
        _run(
            input_root,
            output,
            monkeypatch,
            runner=runner,
            torch_module=torch_module,
        )
    assert output.read_bytes() == report_bytes
    assert len(runner.calls) == 6


def test_real_validator_report_contract_is_accepted_by_orchestration_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_formal_detection_pilot as pilot

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_contract_inputs(tmp_path, monkeypatch)
    output = _report_path(tmp_path, "producer-consumer-contract")
    _run(input_root, output, monkeypatch)

    output_root = output.parents[2]
    monkeypatch.setattr(pilot, "OUTPUT_ROOT", output_root)
    generator_bytes = Path(validator.__file__).read_bytes()
    monkeypatch.setattr(
        pilot,
        "_tracked_file_sha256_at_head",
        lambda repository, relative_path: hashlib.sha256(
            generator_bytes
        ).hexdigest(),
    )
    runner_bytes = (pilot.REPOSITORY_ROOT / pilot.RUNNER_RELATIVE_PATH).read_bytes()
    runner_provenance = {
        "state_guard_repository": {
            "path": str(pilot.REPOSITORY_ROOT),
            "commit": "a" * 40,
            "tracked_worktree_clean": True,
        },
        "runner_script": _artifact(
            pilot.RUNNER_RELATIVE_PATH.as_posix(), runner_bytes
        ),
        "recal3r_repository": {
            "path": str(pilot.RECAL3R_ROOT.resolve(strict=True)),
            "commit": pilot.EXPECTED_RECAL3R_COMMIT,
            "tracked_worktree_clean": True,
        },
    }
    bundle = pilot._load_input_bundle(input_root)
    report_snapshot = pilot._read_snapshot(
        output, name="real validator producer contract report"
    )

    validated = pilot._validate_cpu_validation_report(
        report_snapshot, bundle, runner_provenance
    )

    assert validated["status"] == "PASS"
    assert validated["checks"] == {
        "strict_manifest_validation": True,
        "cpu_pixel_replay": True,
        "raw_source_snapshot_unchanged": True,
        "input_tree_snapshot_unchanged": True,
        "official_tum_associations": True,
        "production_runtime_provenance": True,
        "cuda_hidden": True,
        "six_run_coverage": True,
    }


def test_rejects_inexact_dynamic_pixels_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    output = _report_path(tmp_path, "bad-pixels")

    def omit_deferred_transforms(views: Sequence[dict], manifest: Any) -> list[dict]:
        return [{**view, "img": view["img"].copy()} for view in views]

    import stateguard3r.input_manifest as input_manifest

    monkeypatch.setattr(
        input_manifest, "apply_deferred_transforms", omit_deferred_transforms
    )
    with pytest.raises(
        validator.FormalPilotValidationError, match="replay pixels differ"
    ):
        _run(input_root, output, monkeypatch)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_rejects_dynamic_transform_not_bound_to_frozen_top_level_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    _tamper_dynamic_transform(input_root)
    output = _report_path(tmp_path, "dynamic-transform-drift")

    with pytest.raises(
        validator.FormalPilotValidationError,
        match="dynamic transform is not bound to frozen rectangle/fill",
    ):
        _run(input_root, output, monkeypatch)
    assert not output.exists()


def test_independently_rejects_manifest_groundtruth_association_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    _tamper_groundtruth_association(input_root)
    output = _report_path(tmp_path, "forged-groundtruth")

    with pytest.raises(
        validator.FormalPilotValidationError,
        match="independently parsed TUM physical line",
    ):
        _run(input_root, output, monkeypatch)
    assert not output.exists()


def test_rejects_complete_input_tree_change_adjacent_to_official_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    output = _report_path(tmp_path, "input-tree-toctou")

    with pytest.raises(
        validator.FormalPilotValidationError,
        match="input tree immediately after official loader changed",
    ):
        _run(
            input_root,
            output,
            monkeypatch,
            runner=_FakeRunner(mutate_input_tree=input_root),
        )
    assert not output.exists()


def test_rejects_source_change_or_cuda_initialization_without_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)

    changed_output = _report_path(tmp_path, "changed-source")
    with pytest.raises(
        validator.FormalPilotValidationError,
        match="declaration differs|changed during CPU replay|differs from the frozen raw snapshot",
    ):
        _run(
            input_root,
            changed_output,
            monkeypatch,
            runner=_FakeRunner(mutate_source=True),
        )
    assert not changed_output.exists()

    # Build a fresh tree because the previous adversarial loader intentionally
    # changed one immutable source after its strict-loader pass.
    second_root = tmp_path / "cuda-case"
    second_root.mkdir()
    second_inputs = _prepare_inputs(second_root, monkeypatch)
    cuda_output = _report_path(second_root, "cuda-initialized")
    with pytest.raises(
        validator.FormalPilotValidationError, match="initialized CUDA"
    ):
        _run(
            second_inputs,
            cuda_output,
            monkeypatch,
            runner=_FakeRunner(initialize_cuda=True),
        )
    assert not cuda_output.exists()


def test_requires_explicit_empty_cuda_visibility_and_outputs_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signature = inspect.signature(validator.validate_formal_pilot_inputs)
    assert tuple(signature.parameters) == (
        "input_root",
        "output_report",
        "baseline_root",
    )
    for forbidden_kwargs in (
        {"_runtime": (_FakeTorch(), _FakeRunner())},
        {"_provenance": _provenance()},
        {"_allowed_output_root": tmp_path},
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            validator.validate_formal_pilot_inputs(  # type: ignore[call-arg]
                tmp_path / "input",
                tmp_path / "report",
                **forbidden_kwargs,
            )

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(
        validator.FormalPilotValidationError,
        match="CUDA_VISIBLE_DEVICES must be explicitly set to the empty string",
    ):
        validator.validate_formal_pilot_inputs(
            tmp_path / "missing-inputs",
            tmp_path / "never-created.json",
        )
    assert not (tmp_path / "never-created.json").exists()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    allowed = tmp_path / "outputs"
    allowed.mkdir()
    monkeypatch.setattr(validator, "OUTPUT_ROOT", allowed)
    outside = (
        tmp_path
        / "outside"
        / validator.PRE_FORWARD_VALIDATION_DIRNAME
        / validator.CPU_VALIDATION_FILENAME
    )
    outside.parent.mkdir(parents=True)
    with pytest.raises(
        validator.FormalPilotValidationError, match="must stay inside"
    ):
        validator.validate_formal_pilot_inputs(
            tmp_path / "missing-inputs",
            outside,
        )

    wrong_parent = allowed / "job" / validator.CPU_VALIDATION_FILENAME
    wrong_parent.parent.mkdir()
    with pytest.raises(
        validator.FormalPilotValidationError, match="direct parent"
    ):
        validator.validate_formal_pilot_inputs(
            tmp_path / "missing-inputs", wrong_parent
        )

    forbidden = (
        allowed
        / "job"
        / "runs"
        / validator.PRE_FORWARD_VALIDATION_DIRNAME
        / validator.CPU_VALIDATION_FILENAME
    )
    forbidden.parent.mkdir(parents=True)
    with pytest.raises(
        validator.FormalPilotValidationError, match="run/split output trees"
    ):
        validator.validate_formal_pilot_inputs(
            tmp_path / "missing-inputs", forbidden
        )


def test_generator_and_official_loader_provenance_are_derived_from_fixed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_bytes = Path(validator.__file__).read_bytes()

    def fake_git_output(repository: Path, *arguments: str) -> str:
        if arguments[:2] == ("rev-parse", "HEAD"):
            return "c" * 40
        if arguments and arguments[0] == "status":
            return ""
        if arguments and arguments[0] == "ls-files":
            return validator.GENERATOR_RELATIVE_PATH.as_posix()
        raise AssertionError(arguments)

    monkeypatch.setattr(validator, "_git_output", fake_git_output)
    monkeypatch.setattr(
        validator, "_git_bytes", lambda repository, *arguments: script_bytes
    )
    generator = validator._capture_generator_provenance()
    assert generator == {
        "repository_commit": "c" * 40,
        "tracked_worktree_clean": True,
        "git_blob_sha256": hashlib.sha256(script_bytes).hexdigest(),
        "script": {
            "path": validator.GENERATOR_RELATIVE_PATH.as_posix(),
            "sha256": hashlib.sha256(script_bytes).hexdigest(),
            "size_bytes": len(script_bytes),
        },
    }
    monkeypatch.setattr(
        validator, "_git_bytes", lambda repository, *arguments: b"forged blob"
    )
    with pytest.raises(
        validator.FormalPilotValidationError, match="tracked HEAD blob"
    ):
        validator._capture_generator_provenance()

    baseline = tmp_path / "ReCal3R"
    loader = baseline / validator.OFFICIAL_LOADER_RELATIVE_PATH
    loader.parent.mkdir(parents=True)
    loader.write_bytes(b"official-loader-bytes")
    monkeypatch.setattr(
        validator, "_validate_baseline", lambda baseline_root: baseline
    )
    wrong_loader = tmp_path / "wrong-loader.py"
    wrong_loader.write_bytes(b"wrong")
    monkeypatch.setitem(
        sys.modules,
        "dust3r.utils.image",
        SimpleNamespace(__file__=str(wrong_loader)),
    )
    with pytest.raises(
        validator.FormalPilotValidationError, match="fixed ReCal3R module"
    ):
        validator._capture_production_runtime_provenance(
            baseline, require_loader_imported=True
        )
    monkeypatch.setitem(
        sys.modules,
        "dust3r.utils.image",
        SimpleNamespace(__file__=str(loader)),
    )
    runtime = validator._capture_production_runtime_provenance(
        baseline, require_loader_imported=True
    )
    assert runtime["official_loader"] == {
        "path": validator.OFFICIAL_LOADER_RELATIVE_PATH.as_posix(),
        "sha256": hashlib.sha256(b"official-loader-bytes").hexdigest(),
        "size_bytes": len(b"official-loader-bytes"),
    }
    assert runtime["recal3r_repository"]["commit"] == (
        validator.EXPECTED_RECAL3R_COMMIT
    )
    assert runtime["python"]["executable_realpath"] == str(
        Path(sys.executable).resolve(strict=True)
    )


def test_atomic_publication_rolls_back_on_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    input_root = _prepare_inputs(tmp_path, monkeypatch)
    output = _report_path(tmp_path, "rolled-back")

    def fail_fsync(path: Path) -> None:
        raise OSError("synthetic parent fsync failure")

    monkeypatch.setattr(validator, "_fsync_directory", fail_fsync)
    with pytest.raises(
        validator.FormalPilotValidationError, match="publication rolled back"
    ):
        _run(input_root, output, monkeypatch)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))
