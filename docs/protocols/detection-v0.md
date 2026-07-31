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

## Current signal boundary

The upstream ReCal3R trace directly supplies uncertainty/reliability, attention
entropy, and—when enabled—global-state delta. It does not directly supply
overlap, pose jump, or a geometric residual in the current wrapper. Until those
signals are measured by a real logger, their ledger fields remain JSON `null`;
no proxy value is fabricated.
