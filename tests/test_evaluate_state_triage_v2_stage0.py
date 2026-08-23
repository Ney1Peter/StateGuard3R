from __future__ import annotations

from scripts.evaluate_state_triage_v2_stage0 import _gates


def _metrics(normal_recall: float, false_reject: float, macro: float, unsafe: float) -> dict:
    return {
        "macro_f1": macro,
        "per_cause": {"normal_novelty": {"recall": normal_recall}},
        "unsafe_confusions": {
            "registration_as_transient": unsafe,
            "bad_observation_as_novelty": unsafe,
            "transient_as_registration": unsafe,
            "normal_novelty_false_reject": false_reject,
        },
    }


def test_gate_e_requires_all_effect_safety_and_resource_conditions() -> None:
    metrics = {"S0": _metrics(0.8, 0.05, 0.30, 0.05), "T0": _metrics(0.8, 0.10, 0.45, 0.10)}
    bootstrap = {"lower_95": 0.01}
    resource = {"control_parity": True, "median_observer_to_vanilla_ratio": 0.19, "max_control_gpu_peak_delta_mib": 10.0}
    limits = {
        "macro_f1_margin": 0.10,
        "bootstrap_lower_95_min": 0.0,
        "normal_novelty_false_reject_max": 0.10,
        "unsafe_confusion_increase_max": 0.05,
        "observer_overhead_ratio_max": 0.20,
        "gpu_peak_delta_mib_max": 512.0,
    }
    assert all(_gates(metrics, bootstrap, resource, "S0", limits).values())
    metrics["T0"]["unsafe_confusions"]["bad_observation_as_novelty"] = 0.11
    assert not _gates(metrics, bootstrap, resource, "S0", limits)["unsafe_confusions"]
