# Detection-only v0 protocol

Status: implementation frozen; real ReCal3R evaluation pending the official
checkpoint and clean-smoke gates.

This document fixes the score semantics before any holdout result is observed.
Synthetic and unit-test results validate only the implementation; they are not
evidence that ReCal3R corruption is detectable.

## Causal reference

For frame position `t`, the reference contains finite observations only from
`[max(0, t-K), t)`. The current and future frames never enter their own
reference. If a signal has no finite history, its first finite observation uses
itself as the median and therefore receives a zero robust score.

The protocol uses the research-plan definition exactly:

```text
robust_z(x_t) = abs(x_t - median(x_history)) / (MAD(x_history) + epsilon)
```

There is no z-score cap by default. An experiment may set one explicitly, but
must record it and use the same value for development and holdout data.

## Combined score

The default weights are all one:

```text
risk_t =
    robust_z(geometric_residual_t)
  + robust_z(pose_jump_t)
  + robust_z(update_magnitude_t)
  + (1 - overlap_score_t)
  - reliability_t
```

Components are summed, not averaged. Missing components contribute nothing and
the output records every field alias actually used. A method with no usable
signal emits a neutral zero series and reports an empty signal list; it must not
be described as a successful detector.

ReCal3R trace `U` is an uncertainty/fallback weight. When `uncertainty_u` is the
available alias, the reliability term is computed as `U - 1`, which is exactly
`-reliability` because `reliability = 1 - U`.

## Required comparisons

The evaluator reports four score series:

1. seeded random;
2. update-magnitude only;
3. reliability only;
4. combined health score.

For each series it reports AUROC, F1, false-positive rate, and detection delay.
It also repeats those metrics one corruption type versus all other frames. The
one-vs-all convention is recorded in the output because frames belonging to a
different corruption type count as negatives for that grouped view.

## Threshold freeze

F1, false-positive rate, and delay require a threshold for each method. Window,
epsilon, optional cap, random seed, and all four thresholds must be selected on
the development sequence and written to the run configuration before the
holdout command is launched. The holdout result may be recomputed with that
frozen configuration, but it must not be used to retune it.

The CLI's single fallback threshold exists for API convenience only. A formal
run must pass and record four explicit per-method thresholds. AUROC does not use
those thresholds.

### Formal CLI gate

Use `--formal` for any holdout result that may inform a Go/No-Go decision. This
mode requires exactly one explicit `--method-threshold METHOD=VALUE` entry for
each of `random`, `update_magnitude_only`, `reliability_only`, and `combined`.
Missing entries and repeated method names are errors; the single `--threshold`
fallback is therefore never used by a formal run.

A formal run also requires `--dev-calibration-config PATH`. `PATH` must already
exist and contain a strict development-calibration JSON object. It must declare
`schema_version` as `stateguard3r.detection-calibration.v0`, set
`dataset_split` to `development`, provide a non-empty string
`development_run_id`, and contain `frozen_detection_config`. The frozen object
must explicitly contain `window`, `seed`, `epsilon`, `max_z` (JSON `null` means
uncapped), and `thresholds`. Its threshold map must contain all four and only the
four required method names.

The formal gate compares the parsed CLI values for every frozen field and every
method threshold against this object. Numerical equality after JSON/CLI parsing
is required. Duplicate JSON keys, a missing field, an unknown threshold method,
or a mismatch is an error before the health ledger or corruption manifest is
loaded. For example:

```json
{
  "schema_version": "stateguard3r.detection-calibration.v0",
  "development_run_id": "DEV-0001",
  "dataset_split": "development",
  "source_metrics": "outputs/dev-0001/metrics.json",
  "selection_rule": "maximize F1, then minimize FPR",
  "frozen_detection_config": {
    "window": 15,
    "seed": 0,
    "epsilon": 0.000001,
    "max_z": null,
    "thresholds": {
      "random": 0.5,
      "update_magnitude_only": 3.0,
      "reliability_only": -0.5,
      "combined": 3.0
    }
  }
}
```

The metrics output records `execution_mode`, `formal`,
`frozen_from_development`, `frozen_config_match_verified`, all explicit and
fallback threshold method names, and `development_calibration_provenance`. The
provenance contains the resolved config path, SHA-256 of its exact bytes, and its
full parsed JSON content. These fields make later substitution or omission
visible; they do not independently prove that the supplied metadata is
scientifically valid.

```bash
uv run python -m stateguard3r.detection \
  health.jsonl corruption.json --output metrics.json \
  --formal --dev-calibration-config development-calibration.json \
  --method-threshold random=0.5 \
  --method-threshold update_magnitude_only=3.0 \
  --method-threshold reliability_only=-0.5 \
  --method-threshold combined=3.0
```

Without `--formal`, the CLI remains an exploratory/smoke interface: omitted
per-method values still inherit `--threshold`, and the Python API retains the
same fallback behavior. Passing `--dev-calibration-config` without `--formal`
is rejected so an exploratory result cannot accidentally advertise formal
development provenance.

### CLI input snapshots

For both formal and exploratory CLI runs, the health JSONL and corruption JSON
are each read into memory exactly once. Parsing and SHA-256 hashing use that same
immutable byte snapshot; neither file is reopened for evaluation. This prevents
a path change during launch from producing metrics from one file version and a
hash from another. The result's `input_provenance` records the snapshot policy
and, for each input, its resolved absolute path, raw-byte SHA-256, and byte size.
The standalone Python loader functions keep their existing interfaces.

## Current signal boundary

The upstream ReCal3R trace directly supplies uncertainty/reliability, attention
entropy, and—when enabled—global-state delta. It does not directly supply
overlap, pose jump, or a geometric residual in the current wrapper. Until those
signals are measured by a real logger, their ledger fields remain JSON `null`;
no proxy value is fabricated.
