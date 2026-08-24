# StateTriage3R v2 Stage 0.7 final audit — robust typed performance, but formal interaction-activation NO-GO

Date: 2026-08-24  
Status: **`STATE_TRIAGE_V2_STAGE0P7_WINDOW_NO_GO`**  
Scope: a new, pre-registered 80-window development-proxy robustness and
interaction benchmark.  This audit neither changes nor reruns any frozen Stage
0, 0.5, or 0.6 artifact.

## Result in plain language

The primary typed rule is accurate on this substantially larger fresh matrix:
it has event Macro-F1 **0.9495** (versus **0.2481** for the scalar baseline),
recalls of 1.00/1.00/1.00/0.80 for bad observation, normal novelty,
registration/order fault, and transient local content, and no registered
unsafe confusions.  The fixed coverage-first ablation is materially weaker on
local content (9/20 correct versus 16/20 for the typed rule).

The intended ordering interaction really did occur on seven fresh local
windows.  On every activated window the typed rule selected transient local
content (7/7), while the coverage-first ablation selected normal novelty (0/7
local recall).  This is a clear observed **+1.00** local-recall difference in
the activated stratum.

Nevertheless, this benchmark is formally a **NO-GO**.  The pre-registered
protocol required at least eight activated local-coverage events; the immutable
responses contain only **7**.  Every other gate passed.  The shortfall is by
one event, but it must not be rounded up, repaired by post-hoc recipe choice,
or offset by the strong aggregate result.  Stage 0.7 therefore establishes
useful mechanism evidence, not the pre-specified strong robustness GO.

## Frozen protocol and execution integrity

- Protocol: `docs/protocols/state-triage-v2-stage0p7-window-gate-b.json`,
  SHA-256 `b389bad5938e6e4f2a2b362d2d43c1fae39e4d3cae72c00b4d77e3c380f732ae`.
- Inputs: 80 fresh, mutually disjoint immutable 30-frame TUM RGB capsules;
  input inventory `outputs/state-triage-v2-stage0p7-inputs-0001/protocol.json`,
  SHA-256 `fda5b19ba14ce825f215aff2c4bd15644bab4ba54307d4988eab4cb04ece58fb`.
  Source allocation was 54 `freiburg2_desk`, 15 `freiburg1_desk`, 9
  `freiburg3_walking_xyz`, and 2 `freiburg3_walking_static` capsules, with
  20 fixed-cause capsules per class.
- Source hashes were verified by the evaluator: runner
  `scripts/run_state_triage_v2_stage0.py` SHA-256
  `75b8c7d066889d4f28623fd68b9f3a36a9a5897c8b525475b99df09c0b834723`, and
  observer `src/stateguard3r/state_triage_v2.py` SHA-256
  `6977a00ca12dec1ebe905cf3d16fcd0d2b5279bee8bd98044e40bcd4f732c4a7`.
  Their match is the condition under which the frozen Stage 0.6 clean-control
  audit was permitted to be reused.
- All 80 observer-only forwards succeeded once, serially, in tmux window
  `stateguard:stage0p7-window`, on the shared NVIDIA L20 selected through
  `CUDA_VISIBLE_DEVICES=1`.  Every output was published read-only; none was
  overwritten or retried.  The dispatcher wrote `ALL_DONE 80/80` before the
  evaluator started.
- The sole completed evaluator invocation wrote
  `outputs/state-triage-v2-stage0p7-window-evaluation-0001/result.json`,
  SHA-256 `331eef09043a29ca356f89293bcb852aad27550bfd9afd5dbfbc81ca8381dc95`.
  It verified capsule hashes, direct output locations, read-only permissions,
  observer mode, 30-frame response length, causal observer evidence, and the
  frozen source/configuration contracts before scoring.
- Aggregate official recurrent-forward time was 292.12 seconds (mean 3.65
  seconds/capsule); aggregate observer wall time was 34.30 seconds (mean 0.43
  seconds/capsule).  Peak allocation ranged from 6,364.01 to 6,366.72 MiB.
  These are descriptive resource observations, not performance results.

## Immutable event outcome

| Method | Macro-F1 | Bad-observation recall | Normal recall | Registration recall | Local-content recall | Normal false reject | Unsafe confusions | Unresolved rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0.7 scalar | 0.2481 | 0/20 | 20/20 | 8/20 | 0/20 | 0.00 | 30 | 0.15 |
| C0.7 coverage-first | 0.8452 | 20/20 | 20/20 | 20/20 | 9/20 | 0.00 | 0 | 0.00 |
| T0.7 typed local-first | 0.9495 | 20/20 | 20/20 | 20/20 | 16/20 | 0.00 | 0 | 0.00 |

T0.7's advantage over S0.7 is **+0.7014** Macro-F1.  Every T0.7 decision was
made at the first event frame (median and maximum delay 0 frames).  The four
local-content misses were each routed to the frozen bad-observation branch
because first-event sharpness was below or equal to 0.05 (0.04190, 0.03275,
0.03775, and 0.03527).  This is evidence about the fixed precedence contract;
it is not authorization to retune it on these responses.

## Activated local-before-coverage interaction

The pre-registered activated stratum requires local truth plus first-event
`sharpness > .05`, `clipped_fraction >= .20`, and `coverage_ratio < .25`.
Seven immutable capsules meet it, not the required eight:

| Quantity | Result |
| --- | ---: |
| Activated local-coverage events | 7 / required 8 |
| T0.7 local recall in activated stratum | 7/7 = 1.00 |
| C0.7 local recall in activated stratum | 0/7 = 0.00 |
| T0.7 minus C0.7 activated local recall | +1.00 |

Thus the central causal contrast is observed rather than merely assumed: on
these seven rows moving clipped-area evidence before coverage changes the
prediction from novelty to the correct local-content label.  The activation
count gate fails solely because the intentionally broad fresh recipe mix
yielded seven qualifying rows.  Nine other local rows retained coverage at or
above 0.25 and therefore do not test this order difference; four had low
sharpness and took the common quality-collapse branch.

## Pre-registered gate decision

| Gate | Outcome | Pass? |
| --- | --- | --- |
| 80 valid immutable, causal, provenance-checked events | 80 | yes |
| T0.7 Macro-F1 >= 0.85 | 0.9495 | yes |
| T0.7 minus S0.7 Macro-F1 >= 0.30 | +0.7014 | yes |
| Each T0.7 cause recall >= 0.75 | 1.00, 1.00, 1.00, 0.80 | yes |
| Normal-novelty false reject <= 0.10 | 0.00 | yes |
| Each unsafe confusion <= 2 | 0 each | yes |
| Activated local-coverage events >= 8 | 7 | **no** |
| T0.7 activated local recall >= 0.75 | 1.00 | yes |
| T0.7 activated recall minus C0.7 >= 0.75 | +1.00 | yes |
| Input/output/source integrity | verified | yes |

The formal status is therefore
**`STATE_TRIAGE_V2_STAGE0P7_WINDOW_NO_GO`**.  The benchmark is closed: no
response may be overwritten, relabeled, replayed, or used to tune a threshold
or recipe.

## What this supports, and the only justified next study

Stage 0.7 supports a limited, development-proxy conclusion: the frozen typed
screen is much stronger than its scalar comparator on this 80-window matrix,
and its local-before-coverage ordering has a large observed effect when the
pre-registered interaction conditions occur.  It does **not** support an
action policy, state write, recovery, rollback, upstream ReCal3R modification,
natural-semantic generalization, persistent-change claim, or ATE/RPE claim.

The only justified follow-up is a separately pre-registered fresh
interaction-replication protocol.  It must lock an input construction that
delivers at least the required number of high-sharpness, high-clipped,
low-coverage local rows before forwards; retain independent normal, quality,
and registration controls; and preserve the current responses strictly as
closed evaluation evidence.  It may not relax the Stage 0.7 gate after seeing
these seven rows.
