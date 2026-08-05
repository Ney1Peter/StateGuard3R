from __future__ import annotations

from pathlib import Path

from scripts import acquire_recovery_quality_scene_b as scene_b
from scripts import acquire_tum_v2_holdout as tum


def test_scene_binding_is_exact_and_restores_shared_validator_constants() -> None:
    original = (tum.DATASET_NAME, tum.CANONICAL_URL, tum.EXPECTED_ARCHIVE_BYTES, tum.EXPECTED_TOP_LEVEL, tum.MINIMUM_DATA_BUDGET_BYTES)
    with scene_b._scene_binding():
        assert tum.DATASET_NAME == scene_b.DATASET_NAME
        assert tum.CANONICAL_URL == scene_b.CANONICAL_URL
        assert tum.EXPECTED_ARCHIVE_BYTES == 527_550_055
        assert tum.EXPECTED_TOP_LEVEL == scene_b.DATASET_NAME
        assert tum.MINIMUM_DATA_BUDGET_BYTES == 5 * 1024 * 1024 * 1024
    assert (tum.DATASET_NAME, tum.CANONICAL_URL, tum.EXPECTED_ARCHIVE_BYTES, tum.EXPECTED_TOP_LEVEL, tum.MINIMUM_DATA_BUDGET_BYTES) == original


def test_scene_b_cli_accepts_preflight_and_acquire_paths(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Path] = {}

    def fake_preflight(path: Path) -> dict[str, str]:
        captured["preflight"] = path
        return {"status": "PASS"}

    def fake_acquire(preflight: Path, output: Path) -> dict[str, str]:
        captured["acquire_preflight"] = preflight
        captured["acquire_output"] = output
        return {"status": "PASS"}

    monkeypatch.setattr(scene_b, "preflight", fake_preflight)
    assert scene_b.main(["preflight", str(tmp_path / "pre")]) == 0
    monkeypatch.setattr(scene_b, "acquire", fake_acquire)
    assert scene_b.main(["acquire", str(tmp_path / "pre"), str(tmp_path / "out")]) == 0
    assert captured["acquire_output"] == tmp_path / "out"
