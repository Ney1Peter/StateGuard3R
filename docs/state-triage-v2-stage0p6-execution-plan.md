# StateTriage3R v2 Stage 0.6 — local-before-coverage replication pilot

Date: 2026-08-24  
Status: **pre-forward plan; no Stage 0.6 event response exists**

## Objective and hypothesis

Stage 0.5 was closed as a mechanism-specific NO-GO.  Its frozen fresh results
showed that two 50%-area local overlays had `clipped_fraction > 0.50`, but were
called novelty because `coverage < 0.25` was evaluated first.  This is only a
hypothesis generator.  Stage 0.6 independently falsifies one fixed ordering on
fresh, non-overlapping existing TUM RGB windows:

```text
global information collapse -> local clipped content -> coverage novelty -> registration/order conflict
```

`T0.6` must use only the current observer row and strict observer prefix:

1. `sharpness <= 0.05` -> `bad_observation`.
2. Otherwise `clipped_fraction >= 0.20` -> `transient_local_content`.
3. Otherwise `coverage_ratio < 0.25` -> `normal_novelty`.
4. Otherwise `pose_jump_z >= 3` or `native_geometric_residual_z >= 3` ->
   `registration_or_order_fault`.
5. Otherwise `unresolved`.

The fixed scalar comparator `S0.6` is unchanged from S0.5: coverage below
0.25 is novelty; otherwise residual z at least 3 is registration; otherwise
unresolved.  These constants are already numerical/extractor contracts, not
tuned on a Stage 0.6 response.  No decision sees a label, recipe, source path,
source index, raw RGB, GT, future row, model state, recovery action, or a prior
stage's response.

## Fresh input matrix

There are 12 new 30-frame capsules: one per cause x source sequence.  Every
new source window is disjoint from Stage 0 evaluation/control, Stage 0.5, and
every other Stage 0.6 capsule.

| Cause | fr2_desk | fr3_walking_static | fr3_walking_xyz |
| --- | ---: | ---: | ---: |
| normal novelty | 1200 | 0030 | 0020 |
| registration/order proxy | 0300 | 0150 | 0400 |
| full-frame bad observation | 2000 | 0250 | 0470 |
| local transient-content proxy | 2600 | 0580 | 0750 |

Events use local frames 12–17, except normal novelty at frames 0–4.
Registration is the pre-existing local temporal reversal.  Bad observations
are full-frame constant colour variants.  Transient content is a deterministic
50%-area rectangle with three new colour/location variants.  These deterministic
proxies are development-only input recipes, not natural dynamic-object labels.

## Pre-forward safeguards and gates

1. Build an immutable `0444` input directory and validate every capsule.
2. Freeze its manifest hash, 12 output names, rules, comparator, gates, and
   unchanged runner/observer source hashes in a tracked protocol commit.
3. Reuse the four clean parity controls only if those two source hashes remain
   identical; otherwise stop and run new controls before any event response.
4. Run each observer-only capsule serially once in a named tmux window.  A
   failed validation before an output directory is not a response; no published
   response may be overwritten or retried.
5. Evaluate once after all 12 outputs are immutable, then close Stage 0.6 with
   a GO or NO-GO audit.

`STATE_TRIAGE_V2_STAGE0P6_PILOT_GO` requires all 12 valid causal rows,
T0.6 Macro-F1 >= 0.75, T0.6 minus S0.6 Macro-F1 >= 0.25, recall >= 2/3 for
every cause, normal-novelty false reject <= 1/3, no unsafe confusion type more
than once, and full immutable provenance.  Any failure is
`STATE_TRIAGE_V2_STAGE0P6_PILOT_NO_GO`.

A GO only supports a later robustness benchmark.  It does not authorize an
action policy, persistent state, recovery, rollback, upstream modification,
generalization claim, or ATE/RPE claim.
