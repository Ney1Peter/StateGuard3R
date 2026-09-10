from __future__ import annotations

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
