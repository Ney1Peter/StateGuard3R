from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_quarantine_pilot_v2 as quarantine


def test_policy_drops_fixed_buffer_without_labels_or_future_outside_buffer() -> None:
    scores = [0.0] * 30
    scores[15:18] = [3.0, 2.5, 2.2]
    deltas = [None] + [0.1] * 29

    action = quarantine._policy(scores, deltas)

    assert action["triggered"] is True
    assert action["alarm_position"] == 15
    assert action["buffer_positions"] == [15, 16, 17]
    assert action["decision"] == "drop"
    assert action["withheld_update_mass_proxy"] == pytest.approx(0.3)


def test_policy_releases_only_when_final_two_buffered_scores_are_safe() -> None:
    scores = [0.0] * 30
    scores[15:18] = [3.0, 0.2, 0.3]
    action = quarantine._policy(scores, [None] + [0.2] * 29)

    assert action["decision"] == "release"
    assert action["withheld_update_mass_proxy"] == 0.0


def test_quarantine_publish_is_atomic_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quarantine, "OUTPUTS", tmp_path)
    output = tmp_path / "quarantine-output"

    quarantine._publish(output, {"evidence.json": b"{}\n"})

    assert output.stat().st_mode & 0o777 == 0o555
    assert (output / "evidence.json").stat().st_mode & 0o777 == 0o444
    with pytest.raises(quarantine.QuarantineV2Error, match="output must"):
        quarantine._publish(output, {"second.json": b"{}\n"})
