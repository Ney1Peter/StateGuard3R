# StateTriage3R v2 Stage 0.9 calibration audit

Date: 2026-09-11
Status: **hashed-tile recipe calibration accepted; final benchmark deliberately not run**

## Completion decision

The sealed Stage 0.9 calibration completed with all 12 responses valid and
immutable, and with 11 activated first-event rows.  This exceeds the
pre-registered minimum of 10/12, so the terminal calibration result is
`P0.9_HASHED_TILE_CALIBRATION_ACCEPTED`.

The accepted result is narrow: it establishes that the fixed, non-periodic
hashed-tile local-content recipe can satisfy this calibration gate on its
fresh calibration windows.  It is not a detector GO, final benchmark score,
generalization claim, natural-event result, action/recovery/write/rollback
claim, or an ATE/RPE result.  No final matrix was run.

## Frozen execution and artifacts

The recipe replaced a local 75%-area rectangle on event frames with a
deterministic black-or-white, non-periodic 16-model-ready-pixel tile field
(seed `1909`).  The twelve fresh 30-frame `rgbd_dataset_freiburg2_desk`
capsules had event frames 12--17 and source starts
`110, 270, 730, 870, 1230, 1430, 1770, 1950, 2380, 2630, 2760, 2850`.

| Artifact | SHA-256 |
| --- | --- |
| `outputs/state-triage-v2-stage0p9-inputs-0001/protocol.json` | `dd25fccdd77b9e324fa9c6d054837aaf17ff565a562e6e7bf54453c4419b69b3` |
| `docs/protocols/state-triage-v2-stage0p9-calibration-gate-b.json` | `2dc7e72ef2f7516fb2f837eb6f715b6e94ea1e405bbf7eea5bff58b0c76f3aac` |
| `outputs/state-triage-v2-stage0p9-calibration-evaluation-0001/result.json` | `54533660f9b776770ac007a7d91fb273a9524941e7926feb0e2a327d9b1781e9` |
| `logs/state-triage-v2-stage0p9-calibration-0001.log` at audit time | `1ec1da7bcc5e3946596a3f9408249f8e5fb78e5840de74a39d7b10c6e81e51a9` |

The serial observer-on forwards ran from 01:27:17 to 01:35:20 +08:00 in the
`stateguard` tmux session on physical GPU 6 with `CUDA_VISIBLE_DEVICES=6`.
Each reported `6,366.72 MiB` peak allocated memory.  After the `12/12` done
sentinel, the evaluator ran once and immutably published its result at
01:35:47 +08:00.  The evaluator confirmed the frozen runner and observer
source hashes, input hashes, immutable modes, provenance, and strict
observer-prefix causality for every response.

## Pre-registered gate result

Activation required a first event row with sharpness `> .05`, clipped
fraction `>= .20`, and coverage ratio `< .25`.  The independent evaluator
reported `all_12_valid_events = true`, `input_and_output_provenance = true`,
and `activation = true` (11/12).

| Source start | Sharpness | Clipped fraction | Coverage ratio | Activated | T0.8 / C0.8 / S0.8 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0110 | 2.6917 | 0.75098 | 0.39583 | no | local / local / registration-or-order |
| 0270 | 2.7335 | 0.76562 | 0.00000 | yes | local / novelty / novelty |
| 0730 | 2.9244 | 0.75391 | 0.00000 | yes | local / novelty / novelty |
| 0870 | 2.8901 | 0.75781 | 0.01042 | yes | local / novelty / novelty |
| 1230 | 2.7310 | 0.76270 | 0.00521 | yes | local / novelty / novelty |
| 1430 | 2.7315 | 0.76855 | 0.00521 | yes | local / novelty / novelty |
| 1770 | 2.9005 | 0.76660 | 0.01562 | yes | local / novelty / novelty |
| 1950 | 2.8696 | 0.75195 | 0.00000 | yes | local / novelty / novelty |
| 2380 | 2.6995 | 0.75000 | 0.01042 | yes | local / novelty / novelty |
| 2630 | 2.6953 | 0.75391 | 0.00000 | yes | local / novelty / novelty |
| 2760 | 2.8686 | 0.75098 | 0.00000 | yes | local / novelty / novelty |
| 2850 | 2.9371 | 0.75879 | 0.00521 | yes | local / novelty / novelty |

All rows passed the sharpness and clipped-fraction parts of the gate.  The
sole non-activation was start 0110, whose coverage ratio was `.39583`; the
strict coverage condition correctly rejects it.  Stage 0.8 had four
coverage-gate failures, whereas this independently sealed Stage 0.9 set has
one.  Because the windows and recipe both differ, this is feasibility evidence
for the new recipe, not a controlled estimate that texture alone caused the
difference.  The typed T0.8 diagnostic labels all twelve rows as local
content, while the coverage-first and scalar ablations label the eleven
low-coverage rows as novelty.  Those diagnostic labels are reported for
mechanism inspection only; the sealed acceptance decision is the feature gate
above, not a classification-accuracy estimate.

## Why the study stops here

The frozen inventory replay found only 23 pairwise-disjoint fresh 30-frame
windows after treating all Stage 0--0.8 calibration and final-reserved
capsules as permanently occupied.  Stage 0.9 has now consumed 12 of those
windows and was explicitly calibration-only.  The remaining existing windows
cannot provide a valid 31-window final-only matrix; reusing a calibration,
historic, or final-reserved response would be post-hoc leakage.

Therefore the smallest next authorized data requirement is one official,
unaltered new source containing at least 31 pairwise-disjoint 30-frame
windows, disjoint from every Stage 0--0.9 source window.  Before any such
download or forward, a final-only protocol must freeze that source inventory,
the accepted hashed-tile recipe without further changes, the score and
decision contract, hashes, output names, and a single-run rule.  Until the
user authorizes that data acquisition, no additional forward, retry, threshold
change, texture tuning, or final-result claim is valid.

## Verification performed

The Stage 0.9 focused regression suite was run after completion:

```text
./.venv/bin/python -m pytest -q \
  tests/test_state_triage_capsule_v2.py \
  tests/test_build_state_triage_v2_stage0p9_capsules.py \
  tests/test_state_triage_v2_stage0p9_calibration.py
13 passed in 0.37s
```

The dispatcher test now explicitly checks the intended post-run safety
property: once any sealed Stage 0.9 output exists it must refuse a dry-run
that could become a retry; before execution it still checks that the frozen
protocol yields exactly twelve records.  No frozen runner or observer source
was changed after protocol publication.
