# StateTriage3R v2 Stage 0.7 — 80-window robustness and interaction benchmark

Date: 2026-08-24  
Status: **pre-forward plan; no Stage 0.7 response exists**

## Research objective

Stage 0.6 was a screening GO, but its fresh local-overlay rows never reached
`coverage < 0.25`; it therefore did not activate the ordering interaction it
was meant to test.  Stage 0.7 is deliberately larger than a smoke test: it is
an 80-capsule, multi-window robustness benchmark that simultaneously asks:

1. Does the fixed typed rule remain accurate across many fresh windows and
   recipe variants?
2. When a partial local overlay has both high clip area and low current
   coverage, does moving the local cue ahead of coverage make the intended
   causal decision better than a fixed coverage-first ablation?

All Stage 0/0.5/0.6 inputs, outputs, gates, and conclusions remain frozen.
Stage 0.7 is a new development-only proxy benchmark, not a claim about natural
object semantics, persistent change, recovery, model quality, or generalization.

## Frozen methods

Each method sees only one current observer row plus the observer's strict
prefix.  It never receives an event cause, recipe, source path/index, GT,
future row, model state, or any previous-stage response.

```text
T0.7 (primary):
  sharpness <= .05                         -> bad observation
  clipped_fraction >= .20                  -> transient local content
  coverage_ratio < .25                     -> normal novelty
  pose_jump_z >= 3 or residual_z >= 3      -> registration/order fault
  otherwise                                 -> unresolved

C0.7 (structural ablation):
  sharpness <= .05                         -> bad observation
  coverage_ratio < .25                     -> normal novelty
  clipped_fraction >= .20                  -> transient local content
  pose_jump_z >= 3 or residual_z >= 3      -> registration/order fault
  otherwise                                 -> unresolved

S0.7 (scalar baseline):
  coverage_ratio < .25                     -> normal novelty
  residual_z >= 3                          -> registration/order fault
  otherwise                                 -> unresolved
```

The constants are unchanged numerical contracts.  C0.7 is pre-registered only
as an explanatory fixed ablation; it is not a response-time fallback or action
policy.

## Window matrix and input recipes

The benchmark has 80 immutable 30-frame capsules, 20 for each deterministic
cause.  Events are at local frames 12–17 except normal novelty at 0–4.  All
source windows are selected by the static round-robin allocation in
`build_state_triage_v2_stage0p7_capsules.py`: it uses only prelisted 30-frame
slots disjoint from all prior Stage 0, 0.5 and 0.6 capsule windows.

| Source sequence | Normal | Registration | Bad observation | Local content | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fr2_desk` | 13 | 14 | 14 | 13 | 54 |
| `fr1_desk` | 4 | 4 | 4 | 3 | 15 |
| `fr3_walking_xyz` | 2 | 2 | 2 | 3 | 9 |
| `fr3_walking_static` | 1 | 0 | 0 | 1 | 2 |
| **Total** | **20** | **20** | **20** | **20** | **80** |

Normal novelty uses no transform; registration is the audited six-frame local
temporal reversal; bad observation is a full-frame constant-colour transform.
Local content uses deterministic high-chroma rectangles.  Twelve of its twenty
variants leave one quarter of the image unmodified (75%-area partial-overlay
stress recipes); the remaining eight cover half the image.  All remain partial
image transforms, not full-frame bad-observation recipes.  Their locations and
colours are varied deterministically.

## Interaction definition and pre-registered gates

An event belongs to the *activated local-coverage stratum* only if its truth is
`transient_local_content` and its first event row has all of:

```text
sharpness > .05, clipped_fraction >= .20, coverage_ratio < .25
```

For such a row T0.7 reaches the local branch while C0.7 deterministically
reaches novelty.  This stratum is calculated only after immutable outputs
exist; it does not select inputs, thresholds, or labels.

`STATE_TRIAGE_V2_STAGE0P7_WINDOW_GO` requires every condition below:

- all 80 pre-registered outputs valid and immutable;
- T0.7 Macro-F1 >= 0.85 and T0.7 minus S0.7 >= 0.30;
- every cause T0.7 recall >= 0.75;
- normal-novelty false reject <= 0.10;
- no unsafe confusion type occurs more than twice;
- at least 8 activated local-coverage events;
- within that activated stratum T0.7 local recall >= 0.75 and exceeds C0.7
  local recall by at least 0.75;
- all source, input, output and causality checks pass.

Anything else is `STATE_TRIAGE_V2_STAGE0P7_WINDOW_NO_GO`.  A GO only supports a
future, separately planned robustness study; it cannot authorize state writes,
recovery, rollback, upstream modification, an action policy, or a performance
claim beyond these deterministic development proxies.

## Execution discipline

1. Implement pure decision/evaluation code, builder and tests; test the full
   80-window allocation against all frozen prior windows.
2. Build a new `0444` input inventory and commit a protocol that records its
   hash, exact 80 output names, sources, methods and gates before any forward.
3. Reuse clean-control parity only if runner and observer source hashes are
   unchanged; otherwise stop for new paired controls.
4. In named tmux, run the 80 observer-only capsules serially.  Each successful
   output is immutable and must never be overwritten or retried.
5. Run the evaluator once after all 80 outputs exist, write a final audit, then
   close this benchmark and push completed source/audit commits.
