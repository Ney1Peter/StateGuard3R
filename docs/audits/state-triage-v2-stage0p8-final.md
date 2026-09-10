# StateTriage3R v2 Stage 0.8 final audit

Date: 2026-09-11
Status: **documented design infeasibility; no Stage 0.8 final forward was run**

## Completion decision

The sealed calibration route is complete, valid, and rejected. It produced
8 activated first-event rows, below the pre-registered minimum of 10/12.
The rejected calibration, together with the locked no-new-download budget and
the remaining fresh-window inventory, makes another valid Stage 0.8 calibration
route impossible without violating the protocol. This closes Stage 0.8 as a
design infeasibility, not as a final benchmark GO and not as a claim about an
action, recovery, write, rollback, natural semantic event, generalization,
ATE, or RPE.

The 31 reserved-final capsules were not run, scored, inspected for selection,
or reused as calibration data. A final score would be invalid after the
calibration rejection and is deliberately absent.

## Frozen inputs and execution

| Artifact | SHA-256 |
| --- | --- |
| `outputs/state-triage-v2-stage0p8-inputs-0001/protocol.json` | `1e28ccf1a38534c58e9db7317f081c43afee1c5c0de2a79bdefed89a790bb218` |
| `docs/protocols/state-triage-v2-stage0p8-calibration-gate-b.json` | `f6e951ba84e8e1e072db20056a61a29dfaf4b011f3755f3b1adf1a3652e1af3c` |
| `outputs/state-triage-v2-stage0p8-calibration-evaluation-0001/result.json` | `1abb9f952d2dfd4502cf9bd2f471053794cd1dc938c952ab9d7ef8e4cb9f0490` |
| `logs/state-triage-v2-stage0p8-calibration-0001.log` at audit time | `8fdb967f99280a1634f268df11aae89fc22d223a9434d234026cbb98b3d30125` |

The 12 serial observer-on forwards ran in `tmux` session `stateguard`, window
`stage0p8-calibration`, on physical GPU 6 with
`CUDA_VISIBLE_DEVICES=6`. The dispatch log records 12/12 immutable outputs
from 00:37:06 through 00:46:18 +08:00. The separate evaluator window ran
once after its 12/12 done sentinel and published a read-only result at
00:46:36 +08:00. Every run used the source hashes, `EvidenceConfig`, strict
observer-prefix causality statement, and output names frozen in the protocol.

## Calibration result

`P0.8_CALIBRATION_ACCEPTED` required all 12 valid rows and at least ten first
event rows with sharpness `> .05`, clipped fraction `>= .20`, and coverage
ratio `< .25`. All 12 responses passed provenance and integrity validation;
the activation gate failed exactly at 8/12.

| Source start | Sharpness | Clipped fraction | Coverage ratio | Activated |
| ---: | ---: | ---: | ---: | --- |
| 0000 | 9.7093 | 0.78223 | 0.75521 | no |
| 0150 | 9.6735 | 0.75000 | 0.25000 | no |
| 0240 | 9.7387 | 0.75977 | 0.00000 | yes |
| 0420 | 9.7490 | 0.80762 | 0.00000 | yes |
| 0510 | 9.6196 | 0.75000 | 0.43229 | no |
| 0660 | 9.6865 | 0.75293 | 0.00521 | yes |
| 0750 | 9.7608 | 0.76270 | 0.00521 | yes |
| 0900 | 9.6927 | 0.77246 | 0.17188 | yes |
| 1020 | 9.6809 | 0.79199 | 0.67708 | no |
| 1170 | 9.6847 | 0.87207 | 0.02604 | yes |
| 1260 | 9.7176 | 0.79004 | 0.01042 | yes |
| 1320 | 9.7428 | 0.75684 | 0.02604 | yes |

The textured 75%-area mechanism did avoid the prior quality-collapse failure:
all 12 sharpness values were far above `.05`, and all clipped fractions were
above `.20`. Its limiting mechanism was coverage: four rows retained coverage
at or above `.25` (including one exactly `.25`, which correctly fails the
frozen strict inequality). The primary typed diagnostic labelled all 12 as
local content; the coverage-first ablation differs on precisely the eight
activated rows. These diagnostics establish neither a final score nor a method
GO, because the calibration acceptance gate is pre-registered and was not met.

## Why no retry or final test is allowed

The new `fr1_room` scene has 45 pairwise-disjoint 30-frame slots. The input
inventory reserved 43 before its first forward: 12 calibration slots plus 31
final-only slots. The only untouched slots are starts 330 and 990, two fewer
than the 12 required for a fresh calibration route. The Stage 0.8 plan binds
this route to that new scene, forbids altering/retrying published calibration
responses, and forbids touching the final set after calibration rejection.

Under the active constraint against downloading another scene, there is
therefore no legal independent 12-capsule calibration route and no legal path
to a 31-capsule final matrix. Lowering the threshold, changing the texture,
or using the 31 reserved capsules after seeing this result would be post-hoc
tuning and is excluded.

## A possible future study, outside Stage 0.8

A future numbered study would require explicit approval for a further small,
official data acquisition with at least 43 fresh pairwise-disjoint 30-frame
windows, a newly pre-registered local-content mechanism and calibration gate,
and a newly sealed calibration/final split. It must not relabel, reuse, or
revise any Stage 0.8 response or final-reserved capsule.
