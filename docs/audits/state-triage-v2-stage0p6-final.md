# StateTriage3R v2 Stage 0.6 final audit — screening GO, interaction not activated

Date: 2026-08-24  
Status: **`STATE_TRIAGE_V2_STAGE0P6_PILOT_GO`**  
Scope: an independent, pre-registered development screening pilot.  It is not
a revision or a rerun of Stage 0 or Stage 0.5.

## Result in plain language

The fixed Stage 0.6 typed rule passed every pre-registered gate on 12 fresh,
non-overlapping deterministic input proxies.  It classified all four causes
correctly (3/3 each), with event Macro-F1 **1.0000**, zero unresolved events,
zero normal-novelty false rejects, and zero registered unsafe confusions.  The
fixed scalar comparator had Macro-F1 **0.2381**, so the typed advantage was
**+0.7619**.

This is a useful positive screening result: the full frozen rule is viable on
this fresh proxy matrix.  It is deliberately not a claim of real dynamic-object
semantics, model-state safety, recovery benefit, generalization, or ATE/RPE
improvement.

There is an important limitation.  Although Stage 0.6 pre-registered local
clip evidence before coverage, its three fresh local-overlay first rows had
coverage 0.2604, 0.2969, and 0.2760—none met `coverage < 0.25`.  A read-only,
post-hoc application of the already-frozen Stage 0.5 ordering to the Stage 0.6
rows produced the same twelve labels.  Thus Stage 0.6 establishes a GO for the
full rule on new proxies, but **does not independently activate or prove the
local-before-coverage interaction**.  That interaction was isolated by the
Stage 0.5 failure and needs a separate, purpose-built fresh interaction test.

No action policy, recurrent-state write, hold/replay, recovery, rollback,
upstream ReCal3R change, or safety claim is authorized by this screening GO.

## Frozen protocol and execution integrity

- Protocol: `docs/protocols/state-triage-v2-stage0p6-gate-b.json`, SHA-256
  `74642f6235f676c8cb2b585f30cd04a5158f7a0ab9fa3118de44efa330ad4c9e`.
- Inputs: 12 fresh mutually disjoint 30-frame TUM RGB capsules, no downloads;
  immutable input-inventory SHA-256
  `b62b9347aff4c73c0134e8195ea6ebeeaf548701ac7ed5359d2d0e4325812c59`.
- The evaluator verified every capsule hash, direct output name, read-only
  permission, frame count, `observer=on`, causal evidence/configuration, and
  unchanged runner/observer source hash before scoring.  The frozen clean
  source-control parity audit was reused only under those unchanged hashes.
- All 12 observer-only forwards succeeded once, serially, in tmux window
  `stateguard:stage0p6-pilot`, on the shared NVIDIA L20 selected by
  `CUDA_VISIBLE_DEVICES=1`.  Each had 30 frames and 29 native updates; peak
  allocation was 6,364.01–6,366.72 MiB.
- The aggregate official recurrent-forward time was 44.48 seconds (mean 3.71
  seconds/capsule); observer wall time was 5.58 seconds total (mean 0.47
  seconds/capsule).  These are descriptive resource observations only.
- The first evaluator invocation exposed a path-composition bug before it read
  the first run output, emitted any score, or created an evaluation directory.
  Commit `da2fadc` fixed that one path.  The 12 forwards were not rerun; the
  subsequent invocation is the only completed evaluation output.
- Result: `outputs/state-triage-v2-stage0p6-evaluation-0001/result.json`,
  SHA-256 `242dda1b98b5aedc1024c9292c59780331d632d207e8699b848c954577cb3f2a`.

## Frozen event outcome and gates

| Method | Macro-F1 | Registration recall | Bad-observation recall | Transient recall | Normal recall | Normal false reject | Unsafe confusion count | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0.6 scalar | 0.2381 | 1/3 | 0/3 | 0/3 | 3/3 | 0 | 6 | 2/12 |
| T0.6 typed | 1.0000 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0/12 |

| Pre-registered gate | Outcome | Pass? |
| --- | --- | --- |
| 12 valid causal/provenance-checked events | 12 | yes |
| T0.6 Macro-F1 >= 0.75 | 1.0000 | yes |
| T0.6 minus S0.6 >= 0.25 | +0.7619 | yes |
| every cause recall >= 2/3 | 3/3 each | yes |
| normal-novelty false reject <= 1/3 | 0 | yes |
| each unsafe confusion <= 1 event | 0 each | yes |
| immutable input/output provenance | verified | yes |

The formal pre-registered status is therefore
`STATE_TRIAGE_V2_STAGE0P6_PILOT_GO`.  Stage 0.6 is closed: no result may be
overwritten, rerun, relabeled, or threshold-tuned.

## What this does and does not establish

The full typed decision made its first decision at the event start for every
Stage 0.6 event (median and maximum delay 0 frames).  For the three local
overlays, clipped fraction was 0.5137, 0.5176, and 0.5127, so T0.6 selected the
new local-content branch correctly.  Because coverage never fell below 0.25,
the old coverage-before-local branch would also have reached its clip condition
on these particular rows.  The post-hoc identical-label check is explanatory
only and was not used to change a gate, threshold, label, or result.

The justified next research action is a separately pre-registered fresh
**interaction-activation** benchmark: it must include local overlays whose
first event rows are expected to cross the coverage contract while retaining
high clipped fraction, alongside clean novelty controls.  It must test whether
local-before-coverage changes a decision when that interaction is actually
present.  It must not reuse Stage 0.6 responses as new training, tuning, or
evaluation data.
