# StateTriage3R v2 Stage 0.5 final audit — mechanism isolated, pilot NO-GO

Date: 2026-08-24  
Status: **`STATE_TRIAGE_V2_STAGE0P5_PILOT_NO_GO`**  
Scope: independent development screening pilot only; not a revision or rerun
of the frozen Stage 0 result.

## Result in plain language

The quality-first change worked for the failure it targeted: it separated all
three full-frame bad-observation proxies from novelty, whereas the scalar
comparator called all three novelty.  It also identified all three temporal
order proxies and all three normal-novelty controls.  The typed pilot therefore
increased Macro-F1 from **0.3030** to **0.8125** (a pre-registered advantage of
**+0.5095**).

It is nevertheless a pilot **NO-GO**.  Only one of the three local-content
proxies was called `transient_local_content` (recall 1/3), below the frozen
minimum of 2/3.  The new result is still useful and falsifiable: it isolates a
second priority-order interaction—coverage was checked before the local clip
cue, so two local overlays with low initial coverage were called novelty.

No detector action, state write, hold/replay, recovery, rollback, model change,
generalization, or ATE/RPE claim is authorized by this result.

## Frozen protocol and execution integrity

- Pre-forward protocol: `docs/protocols/state-triage-v2-stage0p5-gate-b.json`,
  SHA-256 `34db87233d4adc565266c4ac6628934b64a9d307aa0a7a12510dc045cf2733fe`.
- Inputs: 12 fresh, mutually disjoint 30-frame windows from already-present TUM
  RGB data; their immutable inventory hash is
  `c324ca920b1368293c59508b1e456c2015c430c45b92859a2f6c327de3044d8e`.
- The evaluator verified the frozen runner and observer source hashes, the
  inherited four clean-control parity audit hash, every input capsule hash,
  `observer=on`, frame count, evidence causality/configuration, and all output
  read-only permissions before it computed any decision.
- All 12 observer-only forwards succeeded exactly once, serially, with the
  committed dispatcher at `0099960`.  Each used `CUDA_VISIBLE_DEVICES=3`,
  NVIDIA L20, 30 frames, 29 native calibrated updates, and a 6,366.72 MiB peak.
  GPU 3 is the pre-forward resource-placement amendment recorded in
  `state-triage-v2-stage0p5-pre-forward-resource-amendment.md`; GPU 2 had only
  5,166 MiB free versus the known 6,366.72 MiB peak.
- One earlier tmux command had a shell-quoting error and failed argument
  validation with an empty sample name before any model forward or output
  directory.  It created no response and was superseded by the committed,
  dry-run-checked dispatcher; it is not an event retry.
- Evaluation was run once after all 12 immutable outputs existed.  Its result
  is `outputs/state-triage-v2-stage0p5-evaluation-0001/result.json`, SHA-256
  `3faa8efa6af910db33a318cf07c319acef45c8f8ddecb43eaf811c6a68558572`.

The aggregate official recurrent-forward runtime was 52.31 seconds (mean 4.36
seconds/capsule); observer wall time was 6.00 seconds total (mean 0.50
seconds/capsule).  These resource observations are descriptive only.

## Frozen event-level outcome

| Method | Macro-F1 | Registration recall | Bad-observation recall | Transient recall | Normal recall | Normal false reject | Unsafe confusion count | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0.5 scalar | 0.3030 | 2/3 | 0/3 | 0/3 | 3/3 | 0 | 4 | 1/12 |
| T0.5 typed | 0.8125 | 3/3 | 3/3 | 1/3 | 3/3 | 0 | 0 | 0/12 |

For S0.5, the four unsafe events are three `bad_observation -> normal_novelty`
and one `transient_local_content -> registration_or_order_fault`.  T0.5 had no
registered unsafe confusion, but its two
`transient_local_content -> normal_novelty` outcomes remain failures and are
visible in the normal confusion column rather than being treated as correct.

## Gate decision

| Pre-registered gate | Outcome | Pass? |
| --- | --- | --- |
| 12 valid causal/provenance-checked events | 12 | yes |
| T0.5 Macro-F1 >= 0.75 | 0.8125 | yes |
| T0.5 minus S0.5 >= 0.25 | +0.5095 | yes |
| every cause recall >= 2/3 | transient = 1/3 | **no** |
| normal-novelty false reject <= 1/3 | 0 | yes |
| each unsafe confusion <= 1 event | 0 each | yes |
| immutable input/output provenance | verified | yes |

Because a single pre-registered gate failed, the status is unambiguously
`STATE_TRIAGE_V2_STAGE0P5_PILOT_NO_GO`.  This pilot is closed and must not be
rerun, relabeled, threshold-tuned, or reinterpreted as Stage 0.

## The isolated mechanism

The T0.5 branch was deliberately:

```text
global quality collapse -> coverage -> local clipped content -> registration
```

At the first event row for each 50%-area local overlay, the clip cue was strong
in all three capsules, but coverage differed:

| Fresh capsule | First-row coverage | First-row clipped fraction | T0.5 outcome |
| --- | ---: | ---: | --- |
| `fr2_desk-2350` | 0.4427 | 0.5508 | transient local content |
| `fr3_walking_static-0700` | 0.2448 | 0.5303 | normal novelty |
| `fr3_walking_xyz-0600` | 0.1927 | 0.5078 | normal novelty |

The static and xyz cases crossed `coverage < 0.25` by tiny/large margins
respectively before the `clipped_fraction >= 0.20` branch could be reached.
Their later rows also retained clipped fractions above 0.50, but the frozen
event aggregation must use the first non-unresolved causal decision and cannot
go back to revise it.  This directly supports the mechanism, rather than an
unbounded search over thresholds or labels: for local overlays, the remaining
error is the coverage-before-local order, not lack of a clip signal.

## Constrained next research step (not run here)

The justified follow-on is a separately planned Stage 0.6 fresh-window pilot
that tests one *new, pre-registered* ordering:

```text
global information collapse -> local clipped content -> coverage novelty -> registration
```

It must use new disjoint windows and inputs, retain the unmodified observer and
runner controls, include normal windows to test whether clip-before-coverage
harms novelty specificity, and set its full thresholds/gates before any forward.
It must not reuse these 12 responses to tune a threshold, claim generalization,
or enable a state action.
