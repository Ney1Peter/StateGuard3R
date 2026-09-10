from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts import acquire_state_triage_v2_stage0p8_scene as scene
from scripts import acquire_tum_v3_holdout as tum


def test_scene_binding_is_exact_and_restores_shared_validator_constants() -> None:
    original = (
        tum.DATASET_NAME,
        tum.CANONICAL_URL,
        tum.EXPECTED_ARCHIVE_BYTES,
        tum.EXPECTED_TOP_LEVEL,
        tum.MAX_ARCHIVE_AND_RAW_TREE_BYTES,
    )
    with scene._scene_binding():
        assert tum.DATASET_NAME == scene.DATASET_NAME
        assert tum.CANONICAL_URL == scene.CANONICAL_URL
        assert tum.EXPECTED_ARCHIVE_BYTES == 782_381_450
        assert tum.EXPECTED_TOP_LEVEL == scene.DATASET_NAME
        assert tum.MAX_ARCHIVE_AND_RAW_TREE_BYTES == 5 * 1024 * 1024 * 1024
    assert (
        tum.DATASET_NAME,
        tum.CANONICAL_URL,
        tum.EXPECTED_ARCHIVE_BYTES,
        tum.EXPECTED_TOP_LEVEL,
        tum.MAX_ARCHIVE_AND_RAW_TREE_BYTES,
    ) == original


def test_stage0p8_output_names_are_fixed() -> None:
    assert scene.PREFLIGHT_OUTPUT_NAME == "state-triage-v2-stage0p8-data-preflight-0001"
    assert scene.ACQUISITION_OUTPUT_NAME == "state-triage-v2-stage0p8-data-0001"
    assert scene.CANONICAL_URL.endswith("/freiburg1/rgbd_dataset_freiburg1_room.tgz")


def test_scene_script_is_directly_importable_outside_repository(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "acquire_state_triage_v2_stage0p8_scene.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "preflight" in completed.stdout


def test_acquire_reuses_verified_archive_after_prepublication_failure(tmp_path: Path, monkeypatch) -> None:
    tum_root = tmp_path / "tum"
    outputs = tmp_path / "outputs"
    tum_root.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(scene.tum, "TUM_ROOT", tum_root)
    monkeypatch.setattr(scene.tum, "OUTPUT_ROOT", outputs)
    monkeypatch.setattr(scene, "EXPECTED_ARCHIVE_BYTES", len(b"fixture"))
    monkeypatch.setattr(scene, "MINIMUM_DATA_BUDGET_BYTES", 1)
    archive = tum_root / f"{scene.DATASET_NAME}.tgz"
    archive.write_bytes(b"fixture")
    archive.chmod(0o444)
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    preflight_record = {
        "schema_version": f"{scene.SCHEMA_PREFIX}-preflight.v1",
        "status": "PASS",
        "dataset": scene.DATASET_NAME,
        "expected_archive_bytes": len(b"fixture"),
        "archive_target": str(archive),
        "raw_target": str(tum_root / scene.DATASET_NAME),
        "model_response_search_hits": [],
        "official_source": {"canonical_url": scene.CANONICAL_URL},
    }
    (preflight / "preflight.json").write_text(json.dumps(preflight_record), encoding="utf-8")
    staging_parent = tmp_path / "staging"
    staged_root = staging_parent / scene.DATASET_NAME
    staged_root.mkdir(parents=True)
    monkeypatch.setattr(scene.tum, "_validate_archive", lambda _archive: {"fixture_archive_validation": True})
    monkeypatch.setattr(scene.tum, "_tree_manifest", lambda _root: {"fixture_raw_manifest": True})
    monkeypatch.setattr(scene.tum, "_extract_to_staging", lambda _archive: (staging_parent, staged_root))
    monkeypatch.setattr(scene.tum, "_freeze_raw_tree", lambda _root, *, freeze_root: None)
    monkeypatch.setattr(scene.tum, "_download_resume", lambda: (_ for _ in ()).throw(AssertionError("must not re-download verified archive")))

    report = scene.acquire(preflight)

    assert report["download"]["reused_verified_archive"] is True
    assert (tum_root / scene.DATASET_NAME).is_dir()
    assert not staging_parent.exists()
