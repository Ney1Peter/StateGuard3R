from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stateguard3r.detection import (
    _write_metrics_json_atomic,
    auroc,
    compute_baseline_scores,
    detection_delay,
    evaluate_detection,
    evaluate_scores,
    frame_labels_from_corruption,
    main,
    robust_z,
    trailing_rolling_median_mad,
)


def test_trailing_reference_uses_only_preceding_window() -> None:
    medians, mads = trailing_rolling_median_mad([1.0, 2.0, 100.0], window=2)

    np.testing.assert_allclose(medians, [1.0, 1.0, 1.5])
    np.testing.assert_allclose(mads, [0.0, 0.0, 0.5])


def test_constant_series_has_zero_robust_z() -> None:
    scores = robust_z(np.full(12, 7.5), window=5)

    assert np.all(np.isfinite(scores))
    np.testing.assert_allclose(scores, 0.0)


def test_robust_z_uses_mad_plus_epsilon_without_sigma_calibration() -> None:
    scores = robust_z([0.0, 2.0, 4.0], window=2, epsilon=1.0)

    assert scores[-1] == pytest.approx(1.5)


def test_robust_z_is_absolute_for_negative_deviations() -> None:
    scores = robust_z([1.0, 1.0, 1.0, 0.0], window=3)

    assert scores[-1] == pytest.approx(1_000_000.0)


def test_robust_z_has_no_future_lookahead() -> None:
    prefix = [1.0, 1.1, 0.9, 1.0, 1.2]
    first = robust_z(prefix + [1.0, 1.0], window=3)
    second = robust_z(prefix + [1_000.0, -1_000.0], window=3)

    np.testing.assert_allclose(first[: len(prefix)], second[: len(prefix)])


def test_combined_score_has_no_future_lookahead() -> None:
    common = [
        {
            "frame_id": index,
            "update_magnitude": value,
            "reliability_mean": 0.9,
        }
        for index, value in enumerate([1.0, 1.0, 1.1, 0.9, 1.0])
    ]
    first_records = common + [
        {"frame_id": 5, "update_magnitude": 1.0, "reliability_mean": 0.9}
    ]
    second_records = common + [
        {"frame_id": 5, "update_magnitude": 1_000.0, "reliability_mean": 0.0}
    ]

    first = compute_baseline_scores(first_records, window=3)["combined"]["scores"]
    second = compute_baseline_scores(second_records, window=3)["combined"]["scores"]

    np.testing.assert_allclose(first[: len(common)], second[: len(common)])


def test_corruption_intervals_expand_to_inclusive_frame_labels() -> None:
    labels = frame_labels_from_corruption(
        [10, 20, 30, 40, 50],
        {
            "corruptions": [
                {"type": "jump", "start_frame": 20, "end_frame": 40},
            ]
        },
    )

    np.testing.assert_array_equal(labels, [0, 1, 1, 1, 0])


def test_corruption_index_intervals_and_disabled_entries() -> None:
    labels = frame_labels_from_corruption(
        ["a", "b", "c", "d"],
        {
            "corruptions": [
                {"start_index": 1, "end_index": 2},
                {
                    "start_frame": "a",
                    "end_frame": "d",
                    "enabled": False,
                },
            ]
        },
    )

    np.testing.assert_array_equal(labels, [0, 1, 1, 0])


def test_corruption_manifest_start_end_are_stream_positions() -> None:
    labels = frame_labels_from_corruption(
        [100, 200, 300, 400],
        {"corruptions": [{"type": "wrong_order_segment", "start": 1, "end": 2}]},
    )

    np.testing.assert_array_equal(labels, [0, 1, 1, 0])


@pytest.mark.parametrize(
    ("start_key", "end_key"),
    [
        ("start", "end"),
        ("start_index", "end_index"),
        ("start_frame", "end_frame"),
    ],
)
@pytest.mark.parametrize("invalid_start", ["1", 1.0, True])
def test_corruption_interval_boundaries_require_integers(
    start_key: str,
    end_key: str,
    invalid_start,
) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        frame_labels_from_corruption(
            [0, 1, 2],
            {"corruptions": [{start_key: invalid_start, end_key: 2}]},
        )


def test_missing_signals_are_excluded_and_reported() -> None:
    records = [
        {"frame_id": 0, "update_magnitude": 0.0},
        {"frame_id": 1, "update_magnitude": 0.1, "reliability_mean": None},
        {"frame_id": 2, "update_magnitude": 0.2},
    ]

    methods = compute_baseline_scores(records, window=2)

    assert methods["update_magnitude_only"]["used_signals"] == ["update_magnitude"]
    assert methods["reliability_only"]["used_signals"] == []
    np.testing.assert_allclose(methods["reliability_only"]["scores"], 0.0)
    assert methods["combined"]["used_signals"] == ["update_magnitude"]
    assert methods["combined"]["signal_sources"][0]["field"] == "update_magnitude"


def test_uncertainty_alias_is_a_reliability_only_risk_signal() -> None:
    records = [
        {"frame_id": index, "uncertainty_u": value}
        for index, value in enumerate([0.1, 0.1, 0.1, 0.9])
    ]

    result = compute_baseline_scores(records, window=3)
    method = result["reliability_only"]

    assert method["used_signals"] == ["reliability"]
    assert method["signal_sources"][0]["field"] == "uncertainty_u"
    assert method["signal_sources"][0]["direction"] == "high"
    assert method["signal_sources"][0]["transformation"] == "value_minus_one"
    np.testing.assert_allclose(method["scores"], [-0.9, -0.9, -0.9, -0.1])


def test_combined_score_sums_components_instead_of_averaging() -> None:
    records = [
        {
            "frame_id": index,
            "pose_jump": 10.0 if index == 3 else 0.0,
            "update_magnitude": 10.0 if index == 3 else 0.0,
        }
        for index in range(4)
    ]

    methods = compute_baseline_scores(records, window=3)
    pose_peak = 10_000_000.0
    update_peak = methods["update_magnitude_only"]["scores"][-1]
    combined_peak = methods["combined"]["scores"][-1]

    assert update_peak == pytest.approx(10_000_000.0)
    assert combined_peak == pytest.approx(pose_peak + update_peak)
    assert combined_peak != pytest.approx((pose_peak + update_peak) / 2.0)


def test_robust_components_use_absolute_two_sided_z() -> None:
    upward = [
        {"frame_id": index, "update_magnitude": value}
        for index, value in enumerate([1.0, 1.0, 1.0, 2.0])
    ]
    downward = [
        {"frame_id": index, "update_magnitude": value}
        for index, value in enumerate([1.0, 1.0, 1.0, 0.0])
    ]

    upward_score = compute_baseline_scores(upward, window=3)[
        "update_magnitude_only"
    ]["scores"][-1]
    downward_score = compute_baseline_scores(downward, window=3)[
        "update_magnitude_only"
    ]["scores"][-1]

    assert upward_score == pytest.approx(1_000_000.0)
    assert downward_score == pytest.approx(1_000_000.0)


def test_overlap_and_reliability_use_direct_deficits() -> None:
    records = [
        {"frame_id": 0, "overlap_score": 0.8, "reliability": 0.9},
        {"frame_id": 1, "overlap_score": 0.2, "reliability": 0.25},
    ]

    methods = compute_baseline_scores(records, window=1)

    np.testing.assert_allclose(methods["reliability_only"]["scores"], [-0.9, -0.25])
    np.testing.assert_allclose(methods["combined"]["scores"], [-0.7, 0.55])
    sources = {item["signal"]: item for item in methods["combined"]["signal_sources"]}
    assert sources["overlap"]["transformation"] == "one_minus_value"
    assert sources["reliability"]["transformation"] == "negated_value"


def test_seeded_random_baseline_is_reproducible() -> None:
    records = [{"frame_id": index} for index in range(20)]

    first = compute_baseline_scores(records, seed=42)["random"]["scores"]
    second = compute_baseline_scores(records, seed=42)["random"]["scores"]
    different = compute_baseline_scores(records, seed=43)["random"]["scores"]

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_known_update_peak_is_detected_with_perfect_metrics() -> None:
    records = [
        {
            "frame_id": index,
            "update_magnitude": 10.0 if index == 5 else 0.0,
        }
        for index in range(10)
    ]
    result = evaluate_detection(
        records,
        {"corruptions": [{"start_frame": 5, "end_frame": 5}]},
        window=4,
        threshold=3.0,
        seed=7,
    )

    update = result["methods"]["update_magnitude_only"]
    assert update["scores"][5] == pytest.approx(10_000_000.0)
    assert update["metrics"]["auroc"] == pytest.approx(1.0)
    assert update["metrics"]["f1"] == pytest.approx(1.0)
    assert update["metrics"]["false_positive_rate"] == pytest.approx(0.0)
    assert (
        update["metrics"]["detection_delay"]["mean_delay_frames"]
        == pytest.approx(0.0)
    )


def test_each_method_uses_its_own_frozen_threshold() -> None:
    records = [
        {
            "frame_id": index,
            "update_magnitude": 10.0 if index == 3 else 0.0,
        }
        for index in range(6)
    ]
    result = evaluate_detection(
        records,
        {"corruptions": [{"start_frame": 3, "end_frame": 3}]},
        window=3,
        threshold=3.0,
        thresholds={"update_magnitude_only": 21.0, "combined": 19.0},
        max_z=20.0,
    )

    assert result["config"]["threshold_policy"] == "frozen_per_method"
    assert result["config"]["thresholds"] == {
        "random": 3.0,
        "update_magnitude_only": 21.0,
        "reliability_only": 3.0,
        "combined": 19.0,
    }
    assert result["methods"]["update_magnitude_only"]["metrics"]["f1"] == 0.0
    assert result["methods"]["combined"]["metrics"]["f1"] == 1.0


def test_metrics_are_grouped_by_corruption_type() -> None:
    records = [
        {
            "frame_id": index,
            "update_magnitude": 10.0 if index in {1, 4, 7} else 0.0,
        }
        for index in range(10)
    ]
    result = evaluate_detection(
        records,
        {
            "corruptions": [
                {"type": "low_overlap_jump", "start": 1, "end": 1},
                {"type": "dynamic_occlusion", "start": 4, "end": 4},
                {"type": "low_overlap_jump", "start": 7, "end": 7},
                {
                    "type": "wrong_order_segment",
                    "start": 8,
                    "end": 9,
                    "enabled": False,
                },
            ]
        },
        window=3,
        threshold=3.0,
    )

    grouped = result["by_corruption_type"]
    assert set(grouped) == {"low_overlap_jump", "dynamic_occlusion"}
    assert grouped["low_overlap_jump"]["labels"] == [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    assert grouped["dynamic_occlusion"]["labels"] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert grouped["low_overlap_jump"]["positive_frames"] == 2
    assert set(grouped["low_overlap_jump"]["methods"]) == {
        "random",
        "update_magnitude_only",
        "reliability_only",
        "combined",
    }
    update_metrics = grouped["low_overlap_jump"]["methods"][
        "update_magnitude_only"
    ]
    assert update_metrics["threshold"] == pytest.approx(3.0)
    assert update_metrics["detection_delay"]["interval_count"] == 2


def test_auroc_is_tie_aware() -> None:
    assert auroc([0, 1], [0.0, 0.0]) == pytest.approx(0.5)
    assert auroc([0, 0, 1, 1], [0.0, 0.1, 0.8, 1.0]) == pytest.approx(1.0)


def test_no_positive_class_returns_null_undefined_metrics() -> None:
    result = evaluate_scores(
        [0, 0, 0],
        [0.0, 1.0, 0.0],
        threshold=0.5,
    )

    assert result["auroc"] is None
    assert result["f1"] is None
    assert result["false_positive_rate"] == pytest.approx(1.0 / 3.0)
    assert result["detection_delay"]["interval_count"] == 0
    assert result["detection_delay"]["mean_delay_frames"] is None


def test_no_negative_class_returns_null_auroc_and_fpr() -> None:
    result = evaluate_scores(
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        threshold=0.5,
    )

    assert result["auroc"] is None
    assert result["false_positive_rate"] is None
    assert result["f1"] == pytest.approx(1.0)
    assert result["detection_delay"]["mean_delay_frames"] == pytest.approx(0.0)


def test_detection_delay_uses_first_crossing_inside_interval() -> None:
    result = detection_delay(
        [0, 1, 1, 1, 0],
        [9.0, 0.0, 0.5, 4.0, 0.0],
        threshold=3.0,
        frame_ids=[100, 200, 300, 400, 500],
    )

    assert result["interval_count"] == 1
    assert result["detected_intervals"] == 1
    assert result["mean_delay_frames"] == pytest.approx(2.0)
    assert result["intervals"][0]["detected_frame_id"] == 400


def test_cli_reads_jsonl_and_writes_strict_metrics_json(tmp_path) -> None:
    health_path = tmp_path / "health.jsonl"
    corruption_path = tmp_path / "corruption.json"
    output_path = tmp_path / "nested" / "metrics.json"
    records = [
        {
            "frame_id": index,
            "update_magnitude": 8.0 if index == 3 else 0.0,
            "reliability": 0.1 if index == 3 else 0.9,
        }
        for index in range(7)
    ]
    health_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    corruption_path.write_text(
        json.dumps(
            {
                "sequence": "tiny",
                "corruptions": [
                    {
                        "type": "known_peak",
                        "start_frame": 3,
                        "end_frame": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(health_path),
            str(corruption_path),
            "--output",
            str(output_path),
            "--window",
            "3",
            "--threshold",
            "3",
            "--method-threshold",
            "random=0.5",
            "--seed",
            "11",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "detection-only-v0"
    assert payload["config"]["execution_mode"] == "exploratory"
    assert payload["config"]["formal"] is False
    assert payload["config"]["frozen_from_development"] is False
    assert payload["config"]["frozen_config_match_verified"] is False
    assert payload["config"]["development_calibration_provenance"] is None
    assert payload["input_provenance"]["snapshot_policy"] == (
        "single_read_bytes_used_for_parse_and_sha256"
    )
    assert payload["config"]["thresholds"]["random"] == pytest.approx(0.5)
    assert payload["labels"] == [0, 0, 0, 1, 0, 0, 0]
    assert set(payload["methods"]) == {
        "random",
        "update_magnitude_only",
        "reliability_only",
        "combined",
    }
    assert payload["methods"]["combined"]["used_signals"] == [
        "update_magnitude",
        "reliability",
    ]
    assert payload["methods"]["combined"]["metrics"]["auroc"] == pytest.approx(1.0)


def _development_calibration_config() -> dict:
    return {
        "schema_version": "stateguard3r.detection-calibration.v0",
        "development_run_id": "DEV-0001",
        "dataset_split": "development",
        "selection_rule": "maximize F1, then minimize FPR",
        "frozen_detection_config": {
            "window": 15,
            "seed": 0,
            "epsilon": 1e-6,
            "max_z": None,
            "thresholds": {
                "random": 0.5,
                "update_magnitude_only": 3.0,
                "reliability_only": -0.5,
                "combined": 3.0,
            },
        },
    }


def _write_development_calibration(path, config: dict | None = None) -> None:
    payload = _development_calibration_config() if config is None else config
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _formal_cli_arguments(
    calibration_path,
    *,
    health_path="health.jsonl",
    corruption_path="corruption.json",
    output_path="metrics.json",
    extra=(),
) -> list[str]:
    return [
        str(health_path),
        str(corruption_path),
        "--output",
        str(output_path),
        "--formal",
        "--dev-calibration-config",
        str(calibration_path),
        *extra,
        "--method-threshold",
        "random=0.5",
        "--method-threshold",
        "update_magnitude_only=3.0",
        "--method-threshold",
        "reliability_only=-0.5",
        "--method-threshold",
        "combined=3.0",
    ]


def test_formal_cli_requires_and_records_frozen_thresholds_and_provenance(
    tmp_path,
) -> None:
    health_path = tmp_path / "health.jsonl"
    corruption_path = tmp_path / "corruption.json"
    calibration_path = tmp_path / "development-calibration.json"
    output_path = tmp_path / "metrics.json"
    health_path.write_text(
        "".join(
            json.dumps({"frame_id": index, "update_magnitude": float(index)})
            + "\n"
            for index in range(4)
        ),
        encoding="utf-8",
    )
    corruption_path.write_text(
        json.dumps({"corruptions": [{"start": 2, "end": 2}]}),
        encoding="utf-8",
    )
    calibration = _development_calibration_config()
    calibration["frozen_detection_config"].update(
        {"window": 3, "seed": 11, "epsilon": 0.25, "max_z": 5.0}
    )
    _write_development_calibration(calibration_path, calibration)

    exit_code = main(
        _formal_cli_arguments(
            calibration_path,
            health_path=health_path,
            corruption_path=corruption_path,
            output_path=output_path,
            extra=(
                "--window",
                "3",
                "--seed",
                "11",
                "--epsilon",
                "0.25",
                "--max-z",
                "5.0",
            ),
        )
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["execution_mode"] == "formal"
    assert config["formal"] is True
    assert config["frozen_from_development"] is True
    assert config["frozen_config_match_verified"] is True
    assert config["threshold_policy"] == "formal_frozen_per_method"
    assert config["window"] == 3
    assert config["seed"] == 11
    assert config["epsilon"] == pytest.approx(0.25)
    assert config["max_z"] == pytest.approx(5.0)
    assert config["explicit_method_thresholds"] == [
        "random",
        "update_magnitude_only",
        "reliability_only",
        "combined",
    ]
    assert config["fallback_threshold_methods"] == []
    provenance = config["development_calibration_provenance"]
    assert provenance["config_path"] == str(calibration_path.resolve())
    assert provenance["config_sha256"] == hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    assert provenance["config"]["development_run_id"] == "DEV-0001"
    assert provenance["config"]["schema_version"] == (
        "stateguard3r.detection-calibration.v0"
    )
    assert provenance["config"]["dataset_split"] == "development"
    assert provenance["config"]["frozen_detection_config"] == (
        calibration["frozen_detection_config"]
    )
    input_provenance = payload["input_provenance"]
    assert input_provenance["snapshot_policy"] == (
        "single_read_bytes_used_for_parse_and_sha256"
    )
    assert input_provenance["health_jsonl"] == {
        "absolute_path": str(health_path.resolve()),
        "sha256": hashlib.sha256(health_path.read_bytes()).hexdigest(),
        "size_bytes": health_path.stat().st_size,
    }
    assert input_provenance["corruption_json"] == {
        "absolute_path": str(corruption_path.resolve()),
        "sha256": hashlib.sha256(corruption_path.read_bytes()).hexdigest(),
        "size_bytes": corruption_path.stat().st_size,
    }


def test_formal_cli_parses_and_hashes_each_input_from_one_snapshot(
    tmp_path, monkeypatch
) -> None:
    health_path = tmp_path / "health.jsonl"
    corruption_path = tmp_path / "corruption.json"
    calibration_path = tmp_path / "development-calibration.json"
    output_path = tmp_path / "metrics.json"
    health_before = b'{"frame_id": 0}\n{"frame_id": 1}\n'
    health_after = b'{"frame_id": 8}\n{"frame_id": 9}\n{"frame_id": 10}\n'
    corruption_before = b'{"corruptions": [{"start": 1, "end": 1}]}'
    corruption_after = b'{"corruptions": [{"start": 0, "end": 0}]}'
    health_path.write_bytes(health_before)
    corruption_path.write_bytes(corruption_before)
    _write_development_calibration(calibration_path)

    original_read_bytes = Path.read_bytes
    read_counts = {health_path: 0, corruption_path: 0}

    def read_then_modify(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if path == health_path:
            read_counts[path] += 1
            path.write_bytes(health_after)
        elif path == corruption_path:
            read_counts[path] += 1
            path.write_bytes(corruption_after)
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_modify)

    assert (
        main(
            _formal_cli_arguments(
                calibration_path,
                health_path=health_path,
                corruption_path=corruption_path,
                output_path=output_path,
            )
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["frame_ids"] == [0, 1]
    assert payload["labels"] == [0, 1]
    assert read_counts == {health_path: 1, corruption_path: 1}
    assert payload["input_provenance"]["health_jsonl"]["sha256"] == (
        hashlib.sha256(health_before).hexdigest()
    )
    assert payload["input_provenance"]["corruption_json"]["sha256"] == (
        hashlib.sha256(corruption_before).hexdigest()
    )
    assert original_read_bytes(health_path) == health_after
    assert original_read_bytes(corruption_path) == corruption_after


def test_formal_cli_rejects_a_missing_method_threshold(tmp_path, capsys) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    _write_development_calibration(calibration_path)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "health.jsonl",
                "corruption.json",
                "--output",
                "metrics.json",
                "--formal",
                "--dev-calibration-config",
                str(calibration_path),
                "--method-threshold",
                "random=0.5",
                "--method-threshold",
                "update_magnitude_only=3.0",
                "--method-threshold",
                "combined=3.0",
            ]
        )

    assert "missing: reliability_only" in capsys.readouterr().err


def test_formal_cli_rejects_duplicate_method_thresholds(tmp_path, capsys) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    _write_development_calibration(calibration_path)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "health.jsonl",
                "corruption.json",
                "--output",
                "metrics.json",
                "--formal",
                "--dev-calibration-config",
                str(calibration_path),
                "--method-threshold",
                "random=0.5",
                "--method-threshold",
                "random=0.6",
                "--method-threshold",
                "update_magnitude_only=3.0",
                "--method-threshold",
                "reliability_only=-0.5",
                "--method-threshold",
                "combined=3.0",
            ]
        )

    assert "duplicate --method-threshold method(s): random" in capsys.readouterr().err


def test_formal_cli_requires_development_calibration_config(capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "health.jsonl",
                "corruption.json",
                "--output",
                "metrics.json",
                "--formal",
                "--method-threshold",
                "random=0.5",
                "--method-threshold",
                "update_magnitude_only=3.0",
                "--method-threshold",
                "reliability_only=-0.5",
                "--method-threshold",
                "combined=3.0",
            ]
        )

    assert "--formal requires --dev-calibration-config" in capsys.readouterr().err


def test_formal_cli_rejects_calibration_without_development_run_id(
    tmp_path, capsys
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    del calibration["development_run_id"]
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "health.jsonl",
                "corruption.json",
                "--output",
                "metrics.json",
                "--formal",
                "--dev-calibration-config",
                str(calibration_path),
                "--method-threshold",
                "random=0.5",
                "--method-threshold",
                "update_magnitude_only=3.0",
                "--method-threshold",
                "reliability_only=-0.5",
                "--method-threshold",
                "combined=3.0",
            ]
        )

    assert "non-empty development_run_id string" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "frozen_value"),
    [
        ("window", 16),
        ("seed", 1),
        ("epsilon", 2e-6),
        ("max_z", 5.0),
    ],
)
def test_formal_cli_rejects_each_runtime_config_mismatch(
    tmp_path, capsys, field, frozen_value
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    calibration["frozen_detection_config"][field] = frozen_value
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    error = capsys.readouterr().err
    assert f"formal CLI {field}=" in error
    assert f"frozen_detection_config.{field}=" in error


def test_formal_cli_rejects_threshold_mismatch(tmp_path, capsys) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    calibration["frozen_detection_config"]["thresholds"]["combined"] = 4.0
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    error = capsys.readouterr().err
    assert "formal CLI threshold for combined=3.0" in error
    assert "frozen_detection_config.thresholds['combined']=4.0" in error


@pytest.mark.parametrize(
    "missing_field",
    ["window", "seed", "epsilon", "max_z", "thresholds"],
)
def test_formal_cli_rejects_each_missing_frozen_config_field(
    tmp_path, capsys, missing_field
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    del calibration["frozen_detection_config"][missing_field]
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    assert (
        f"frozen_detection_config is missing required field(s): {missing_field}"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("missing", "missing method(s): reliability_only"),
        ("unknown", "unknown method(s): oracle"),
    ],
)
def test_formal_cli_rejects_inexact_frozen_threshold_methods(
    tmp_path, capsys, mutation, expected_error
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    thresholds = calibration["frozen_detection_config"]["thresholds"]
    if mutation == "missing":
        del thresholds["reliability_only"]
    else:
        thresholds["oracle"] = 1.0
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    assert expected_error in capsys.readouterr().err


def test_formal_cli_rejects_duplicate_calibration_json_keys(
    tmp_path, capsys
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    serialized = json.dumps(_development_calibration_config(), sort_keys=True)
    serialized = serialized.replace(
        '"random": 0.5', '"random": 0.4, "random": 0.5', 1
    )
    calibration_path.write_text(serialized, encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    assert "duplicate JSON object key 'random'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    [
        (
            "schema_version",
            "stateguard3r.detection-calibration.v1",
            "schema_version must be 'stateguard3r.detection-calibration.v0'",
        ),
        (
            "dataset_split",
            "holdout",
            "dataset_split must be 'development'",
        ),
    ],
)
def test_formal_cli_requires_schema_and_development_split(
    tmp_path, capsys, field, invalid_value, expected_error
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    calibration[field] = invalid_value
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    assert expected_error in capsys.readouterr().err


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("schema_version", "schema_version must be"),
        ("dataset_split", "dataset_split must be 'development'"),
        ("frozen_detection_config", "must contain a frozen_detection_config"),
    ],
)
def test_formal_cli_rejects_missing_calibration_contract_fields(
    tmp_path, capsys, missing_field, expected_error
) -> None:
    calibration_path = tmp_path / "development-calibration.json"
    calibration = _development_calibration_config()
    del calibration[missing_field]
    _write_development_calibration(calibration_path, calibration)

    with pytest.raises(SystemExit, match="2"):
        main(_formal_cli_arguments(calibration_path))

    assert expected_error in capsys.readouterr().err


def test_atomic_metrics_write_preserves_existing_output_on_failure(tmp_path) -> None:
    output_path = tmp_path / "metrics.json"
    output_path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(TypeError):
        _write_metrics_json_atomic(output_path, {"bad": object()})

    assert output_path.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.glob(".metrics.json.*.tmp")) == []
