from __future__ import annotations

from decimal import Decimal

from scripts.prepare_formal_pilot_v2_inputs import Entry, _corruption, _nearest


def _entry(index: int, timestamp: str) -> Entry:
    return Entry(index, Decimal(timestamp), timestamp, (timestamp, "rgb/x.png"))


def test_nearest_rejects_tied_or_distant_associations() -> None:
    reference = _entry(0, "1.0")
    assert _nearest(reference, [_entry(1, "0.99"), _entry(2, "1.02")], "depth").index == 1
    assert _nearest(reference, [_entry(1, "0.99"), _entry(2, "1.01")], "depth") is None
    assert _nearest(reference, [_entry(1, "1.03")], "depth") is None


def test_corruption_layout_keeps_event_after_v2_warmup() -> None:
    assert _corruption("dynamic_occlusion")["start"] == 15
    assert _corruption("wrong_order_segment")["end"] == 18
    assert _corruption("low_overlap_jump")["parameters"]["source_start"] == 30
