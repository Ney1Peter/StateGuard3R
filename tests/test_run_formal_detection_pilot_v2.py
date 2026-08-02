from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_formal_detection_pilot_v2 as pilot
from stateguard3r.health import HealthFrame


def _frozen_run(root: Path, run_id: str) -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    for filename in pilot.RUN_FILES.values():
        (directory / filename).write_text("{}\n", encoding="utf-8")
        (directory / filename).chmod(0o444)
    directory.chmod(0o555)


def _health_snapshot(overlap_at_one: object = 0.5) -> pilot.Snapshot:
    records = []
    for frame_id in range(30):
        record = HealthFrame(frame_id=frame_id, overlap=None if frame_id == 0 else 0.5).to_dict()
        records.append(record)
    records[1]["overlap"] = overlap_at_one
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode()
    return pilot.Snapshot(Path("health.jsonl"), payload)


def test_blind_preflight_is_metadata_only_and_requires_all_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = [
        pilot.RunSpec("holdout-dynamic", "holdout", "dynamic_occlusion", tmp_path, pilot.Snapshot(tmp_path / "s", b"{}"), pilot.Snapshot(tmp_path / "i", b"{}")),
        pilot.RunSpec("holdout-wrong", "holdout", "wrong_order_segment", tmp_path, pilot.Snapshot(tmp_path / "s", b"{}"), pilot.Snapshot(tmp_path / "i", b"{}")),
        pilot.RunSpec("holdout-low", "holdout", "low_overlap_jump", tmp_path, pilot.Snapshot(tmp_path / "s", b"{}"), pilot.Snapshot(tmp_path / "i", b"{}")),
    ]
    _frozen_run(tmp_path, "holdout-dynamic")
    _frozen_run(tmp_path, "holdout-wrong")
    with pytest.raises(pilot.FormalV2Error, match="all three blind outputs"):
        pilot._preflight_holdout_outputs(tmp_path, specs)
    _frozen_run(tmp_path, "holdout-low")

    def fail_if_payload_is_read(self: Path) -> bytes:
        raise AssertionError(f"preflight must not read {self}")

    monkeypatch.setattr(Path, "read_bytes", fail_if_payload_is_read)
    pilot._preflight_holdout_outputs(tmp_path, specs)


@pytest.mark.parametrize("bad_overlap", [None, float("nan"), 2.0])
def test_v2_health_rejects_missing_nonfinite_or_out_of_range_overlap(bad_overlap: object) -> None:
    with pytest.raises(pilot.FormalV2Error, match="non-finite JSON|invalid online overlap"):
        pilot._health(_health_snapshot(bad_overlap), run_id="holdout-dynamic")


def test_immutable_publish_rejects_overwrite_and_freezes_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pilot, "OUTPUTS", tmp_path)
    output = tmp_path / "formal-output"
    published = pilot._publish(output, {"nested/evidence.json": b"{}\n"})

    assert published == output
    assert (output / "nested" / "evidence.json").stat().st_mode & 0o777 == 0o444
    assert output.stat().st_mode & 0o777 == 0o555
    with pytest.raises(pilot.FormalV2Error, match="already exists"):
        pilot._publish(output, {"different.json": b"{}\n"})
