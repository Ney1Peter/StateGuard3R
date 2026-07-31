import json
import os
import stat

import numpy as np
import pytest

import stateguard3r.health as health_module
from stateguard3r.health import (
    HealthFrame,
    HealthJSONLError,
    HealthValidationError,
    adapt_recal3r_trace,
    append_health_jsonl,
    iter_health_jsonl,
    read_health_jsonl,
    write_health_jsonl_atomic,
)


def test_atomic_jsonl_roundtrip_preserves_nulls(tmp_path):
    path = tmp_path / "health.jsonl"
    records = [
        HealthFrame(
            frame_id=0,
            timestamp=1.25,
            overlap=0.8,
            uncertainty_u=0.2,
            candidate_beta=0.4,
            final_beta=0.25,
            decision="update",
        ),
        HealthFrame(frame_id=1, pose_jump=0.03, decision="observe"),
    ]

    assert write_health_jsonl_atomic(path, records) == path
    assert read_health_jsonl(path) == records
    assert list(iter_health_jsonl(path)) == records

    raw_records = [json.loads(line) for line in path.read_text().splitlines()]
    assert raw_records[0]["reliability"] == pytest.approx(0.8)
    assert raw_records[1]["uncertainty_u"] is None
    assert raw_records[1]["reliability"] is None
    assert raw_records[1]["geometric_residual"] is None
    assert raw_records[1]["local_mem_delta"] is None


def test_streaming_append_and_iteration(tmp_path):
    path = tmp_path / "stream.jsonl"
    append_health_jsonl(path, [HealthFrame(frame_id=3)])
    append_health_jsonl(
        path,
        (HealthFrame(frame_id=frame_id) for frame_id in (4, 5)),
        fsync=True,
    )

    iterator = iter_health_jsonl(path)
    assert next(iterator).frame_id == 3
    assert [record.frame_id for record in iterator] == [4, 5]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestamp", float("nan")),
        ("timestamp", float("inf")),
        ("overlap", -0.01),
        ("overlap", 1.01),
        ("pose_jump", -1.0),
        ("geometric_residual", -0.1),
        ("update_magnitude", -0.1),
        ("candidate_beta", 1.1),
        ("final_beta", -0.1),
        ("global_state_delta", -0.1),
        ("local_mem_delta", float("nan")),
        ("risk_score", float("inf")),
    ],
)
def test_bad_numeric_values_are_rejected(field, value):
    with pytest.raises(HealthValidationError):
        HealthFrame(frame_id=0, **{field: value})


def test_finite_signed_risk_score_is_allowed():
    record = HealthFrame(frame_id=0, risk_score=-0.25)

    assert record.risk_score == pytest.approx(-0.25)


@pytest.mark.parametrize("frame_id", [-1, 1.5, True, "1"])
def test_bad_frame_ids_are_rejected(frame_id):
    with pytest.raises(HealthValidationError):
        HealthFrame(frame_id=frame_id)


def test_uncertainty_reliability_relation_is_explicit():
    from_u = HealthFrame(frame_id=0, uncertainty_u=np.float32(0.25))
    assert from_u.uncertainty_u == pytest.approx(0.25)
    assert from_u.reliability == pytest.approx(0.75)

    from_reliability = HealthFrame(frame_id=1, reliability=0.9)
    assert from_reliability.uncertainty_u == pytest.approx(0.1)

    with pytest.raises(HealthValidationError, match="must equal"):
        HealthFrame(frame_id=2, uncertainty_u=0.2, reliability=0.2)

    normalized = HealthFrame(
        frame_id=3, uncertainty_u=0.2, reliability=0.8000000001
    )
    assert normalized.reliability == 1.0 - normalized.uncertainty_u


def test_summary_bounds_have_complementary_reliability_bounds():
    record = HealthFrame(
        frame_id=5,
        uncertainty_u=0.3,
        uncertainty_u_min=0.1,
        uncertainty_u_max=0.6,
    )
    assert record.reliability == pytest.approx(0.7)
    assert record.reliability_min == pytest.approx(0.4)
    assert record.reliability_max == pytest.approx(0.9)

    with pytest.raises(HealthValidationError, match="min <= mean <= max"):
        HealthFrame(
            frame_id=6,
            uncertainty_u=0.3,
            uncertainty_u_min=0.4,
            uncertainty_u_max=0.6,
        )


def test_atomic_write_does_not_replace_good_file_on_bad_record(tmp_path):
    path = tmp_path / "atomic.jsonl"
    original = HealthFrame(frame_id=0, decision="observe")
    write_health_jsonl_atomic(path, [original])
    before = path.read_bytes()

    def records():
        yield HealthFrame(frame_id=1)
        yield {"frame_id": 2, "overlap": float("nan")}

    with pytest.raises(HealthValidationError):
        write_health_jsonl_atomic(path, records())
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".atomic.jsonl.*.tmp"))


def test_atomic_write_tolerates_unsupported_directory_fsync(tmp_path, monkeypatch):
    path = tmp_path / "atomic.jsonl"
    real_fsync = os.fsync

    def reject_directory_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("directory fsync is unsupported")
        real_fsync(descriptor)

    monkeypatch.setattr(health_module.os, "fsync", reject_directory_fsync)

    write_health_jsonl_atomic(path, [HealthFrame(frame_id=0)])
    assert read_health_jsonl(path) == [HealthFrame(frame_id=0)]


def test_invalid_jsonl_reports_line_without_accepting_bad_values(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"frame_id":0}\n{"frame_id":1,"overlap":2}\n')

    iterator = iter_health_jsonl(path)
    assert next(iterator).frame_id == 0
    with pytest.raises(HealthJSONLError, match=r"bad\.jsonl:2"):
        next(iterator)


def test_recal3r_trace_adapter_maps_u_to_uncertainty_and_nulls_unknowns():
    trace = {
        "frame_step": [np.array([1]), np.array([2])],
        "frame_u_mean": [np.array([0.2]), np.array([0.7])],
        "frame_u_min": [np.array([0.1]), np.array([0.5])],
        "frame_u_max": [np.array([0.4]), np.array([0.9])],
        "frame_h_mean": [np.array([0.3]), np.array([0.6])],
        "frame_h_min": [np.array([0.2]), np.array([0.4])],
        "frame_h_max": [np.array([0.5]), np.array([0.8])],
        "delta_norm": [np.array([1.0, 3.0]), np.array([2.0, 4.0])],
        "frame_idx": [np.array([1, 1]), np.array([2, 2])],
    }

    records = adapt_recal3r_trace(trace, timestamps={1: 10.0, 2: 11.0})

    assert [record.frame_id for record in records] == [1, 2]
    assert records[0].timestamp == 10.0
    assert records[0].uncertainty_u == pytest.approx(0.2)
    assert records[0].reliability == pytest.approx(0.8)
    assert records[0].reliability_min == pytest.approx(0.6)
    assert records[0].reliability_max == pytest.approx(0.9)
    assert records[0].attention_entropy_mean == pytest.approx(0.3)
    assert records[0].global_state_delta == pytest.approx(2.0)

    # ReCal3R's trace does not contain these values. They must remain null
    # rather than being inferred from unrelated quantities.
    assert records[0].overlap is None
    assert records[0].pose_jump is None
    assert records[0].geometric_residual is None
    assert records[0].update_magnitude is None
    assert records[0].candidate_beta is None
    assert records[0].final_beta is None
    assert records[0].local_mem_delta is None
    assert records[0].decision is None


def test_recal3r_trace_adapter_can_emit_frames_without_summaries():
    trace = {
        # ReCal3R first computes its calibration mask at i=1, so frame 0 has
        # no U/H summary even though it is part of the input stream.
        "frame_step": [np.array([1])],
        "frame_u_mean": [np.array([0.25])],
        "frame_u_min": [np.array([0.1])],
        "frame_u_max": [np.array([0.5])],
    }

    records = adapt_recal3r_trace(
        trace,
        all_frame_ids=[0, 1],
        timestamps=[10.0, 11.0],
    )

    assert [record.frame_id for record in records] == [0, 1]
    assert records[0].timestamp == 10.0
    assert records[0].uncertainty_u is None
    assert records[0].reliability is None
    assert records[0].global_state_delta is None
    assert records[1].timestamp == 11.0
    assert records[1].uncertainty_u == pytest.approx(0.25)
    assert records[1].reliability == pytest.approx(0.75)


def test_recal3r_trace_adapter_joins_delayed_delta_by_frame_id():
    trace = {
        "frame_step": [np.array([1]), np.array([2])],
        "frame_u_mean": [np.array([0.2]), np.array([0.3])],
        # With oracle_window > 1, the delta for an earlier frame appears only
        # after a later summary. frame_idx preserves the original frame ID.
        "delta_norm": [np.array([2.0, 4.0])],
        "frame_idx": [np.array([1, 1])],
    }

    records = adapt_recal3r_trace(trace)

    assert records[0].frame_id == 1
    assert records[0].global_state_delta == pytest.approx(3.0)
    assert records[1].frame_id == 2
    assert records[1].global_state_delta is None


def test_recal3r_trace_adapter_accepts_detached_cpu_tensor_duck_type():
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.was_detached = False
            self.was_moved_to_cpu = False

        def detach(self):
            self.was_detached = True
            return self

        def cpu(self):
            self.was_moved_to_cpu = True
            return self

        def __array__(self, dtype=None):
            return np.asarray(self.values, dtype=dtype)

    frame_step = FakeTensor([[1]])
    frame_u_mean = FakeTensor([[0.2]])

    records = adapt_recal3r_trace(
        {"frame_step": frame_step, "frame_u_mean": frame_u_mean}
    )

    assert records[0].frame_id == 1
    assert records[0].uncertainty_u == pytest.approx(0.2)
    assert frame_step.was_detached and frame_step.was_moved_to_cpu
    assert frame_u_mean.was_detached and frame_u_mean.was_moved_to_cpu


def test_recal3r_trace_adapter_rejects_missing_or_inconsistent_summaries():
    with pytest.raises(HealthValidationError, match="frame_u_mean"):
        adapt_recal3r_trace({"frame_step": [1]})

    with pytest.raises(HealthValidationError, match="expected 2"):
        adapt_recal3r_trace(
            {
                "frame_step": [1, 2],
                "frame_u_mean": [0.1, 0.2],
                "frame_u_min": [0.0],
            }
        )

    with pytest.raises(HealthValidationError, match="disagree"):
        adapt_recal3r_trace(
            {"frame_step": [1], "frame_u_mean": [0.1]}, frame_ids=[2]
        )


def test_recal3r_trace_adapter_rejects_delta_without_frame_indices():
    with pytest.raises(HealthValidationError, match="delta_norm requires frame_idx"):
        adapt_recal3r_trace(
            {
                "frame_step": [1],
                "frame_u_mean": [0.2],
                "delta_norm": [np.array([1.0, 2.0])],
            }
        )


def test_recal3r_trace_adapter_rejects_ambiguous_multi_batch_trace():
    trace = {"frame_step": [1], "frame_u_mean": [0.2]}

    with pytest.raises(HealthValidationError, match="only supports batch_size=1"):
        adapt_recal3r_trace(trace, batch_size=2)

    with pytest.raises(HealthValidationError, match="only supports batch_size=1"):
        adapt_recal3r_trace({**trace, "batch_size": np.int64(2)})


def test_empty_recal3r_trace_adapts_to_no_records():
    assert adapt_recal3r_trace({"frame_u_mean": []}) == []


def test_empty_recal3r_trace_can_emit_null_records_for_known_input_frames():
    records = adapt_recal3r_trace(
        {"frame_u_mean": []},
        all_frame_ids=[0],
        timestamps={0: 12.5},
    )

    assert records == [HealthFrame(frame_id=0, timestamp=12.5)]
