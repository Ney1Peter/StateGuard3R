# StateTriage3R v2 Stage 0.5 — ordered structural-cue replication pilot

Date: 2026-08-24  
Status: **pre-forward plan; no Stage 0.5 event response exists yet**  
Scope: a small, independent development pilot, not a revision of Stage 0.

## 1. Why a new pilot is justified

The completed Stage 0 is irrevocably `NO_GO`.  Its typed rule ran
`coverage -> quality -> global/local geometry`; a full-frame observation
corruption therefore often destroyed current geometric coverage and was called
`normal_novelty` before the quality evidence could be examined.

The Stage 0 result is used here only for **hypothesis generation**, not for
threshold selection or re-scoring.  Its threshold-free diagnostic found a
different causal ordering worth falsifying on fresh, non-overlapping windows:

```text
global information collapse
  -> new-region coverage
     -> localized artificial content
        -> temporal-order / registration proxy
```

This pilot has a deliberately narrower claim: can simple, semantically fixed
structural cues separate these four deterministic input proxies on fresh local
windows?  It does not claim real dynamic-object semantics, persistent change,
state recovery, ATE/RPE improvement, generalization, or an action policy.

## 2. Single hypothesis and fixed rule

`T0.5` operates on a current observer row and its strict observer prefix only:

1. `sharpness <= 0.05` -> `bad_observation`.
   A spatially constant full-frame input has zero discrete-Laplacian variance;
   the threshold is an absolute numerical-contract tolerance, not fitted to a
   pilot label.
2. Otherwise `coverage_ratio < 0.25` -> `normal_novelty`.
3. Otherwise `clipped_fraction >= 0.20` -> `transient_local_content`.
   This is a local, high-chroma/clip-area proxy, not an object identifier.
4. Otherwise `pose_jump_z >= 3.0` or
   `native_geometric_residual_z >= 3.0` -> `registration_or_order_fault`.
   Three robust deviations is the pre-existing extractor's standard robust
   multiplier.
5. Otherwise -> `unresolved`.

The scalar comparator `S0.5` is fixed before forward: coverage below 0.25 is
`normal_novelty`; otherwise native geometric-residual z at least 3.0 is
`registration_or_order_fault`; otherwise `unresolved`.

The priority is part of the hypothesis and may not be changed after any pilot
row exists.  Neither classifier receives a capsule cause, recipe, source
sequence/index, raw path, GT, future row, or any old Stage 0 decision.

## 3. Fresh input matrix

There are twelve 30-frame capsules: three per cause and one fresh source window
in each of `fr2_desk`, `fr3_walking_static`, and `fr3_walking_xyz`.  Every
window is disjoint from every Stage 0 evaluation and control source group.

| Cause | fr2_desk | fr3_walking_static | fr3_walking_xyz |
| --- | ---: | ---: | ---: |
| normal novelty | 0520 | 0330 | 0100 |
| registration/order proxy | 0900 | 0550 | 0250 |
| full-frame bad observation | 1600 | 0220 | 0430 |
| local transient-content proxy | 2350 | 0700 | 0600 |

Events occupy local frames 12–17, except normal novelty (0–4).  Registration
uses the already-audited local temporal reversal.  Bad observation is a
full-frame constant image, with deterministic grey/black/white variants across
the three sequences.  Transient content is a 50% rectangular local overlay,
with deterministic colour/location variants.  The variants prevent the pilot
from being a one-colour recipe check while preserving the intended global vs
local spatial distinction.

All inputs must be built into a new `0444` capsule directory and validated
before any GPU task.  No new raw data, weight, dependency, or source sequence
is downloaded.

## 4. Execution and safeguards

1. Implement capsule builder, pure fixed decision function and evaluator with
   unit tests.  The ReCal3R observer and runner remain unchanged.
2. Freeze the input inventory, exact 12-output mapping, rule, comparator,
   metrics and gates in a tracked protocol commit.
3. Reuse Stage 0's four clean-control parity results only because the observer
   and runner source hashes are unchanged.  If either changes, stop and rerun
   independent paired controls before any event forward.
4. On GPU 2 in a named tmux window, run the 12 observer-only capsules serially.
   Each capsule may create exactly one successful response.  A failure before
   output may be diagnosed with a distinct attempt ID; a response is never
   overwritten or rerun.
5. Run the evaluator once after all 12 output directories are immutable, then
   write a final `GO` or `NO_GO` audit and stop this pilot.

## 5. Pre-registered pilot gates

This is a screening result, so it uses no generalization claim or threshold
search.  It is `STATE_TRIAGE_V2_STAGE0P5_PILOT_GO` only if all hold:

- all 12 frozen capsules produce valid causal observer rows;
- T0.5 event Macro-F1 is at least 0.75 and exceeds S0.5 by at least 0.25;
- each of the four causes has recall at least 2/3;
- normal-novelty false reject is at most 1/3;
- none of the three unsafe confusions occurs more than once;
- all output provenance and immutable input checks pass.

Any other outcome is `STATE_TRIAGE_V2_STAGE0P5_PILOT_NO_GO`.  A pilot GO only
justifies a later, separately planned robustness benchmark with varied
photometric and local-content processes; it does not authorize state actions.
