# StateTriage3R v2 Stage 0 — final audit

Date: 2026-08-23  
Decision: **`STATE_TRIAGE_V2_STAGE0_NO_GO`**  
Scope: one frozen 20-event, development-only evaluation; no new data,
weights, dependencies, GT, recovery action, or state write was used.

## Decision

Stage 0 does not establish the prerequisite for cause-matched actions.  The
frozen coverage-conditioned typed rule (`T0`) is worse than the preselected
one-dimensional native-health baseline (`S0`) at event-level cause triage:

| Measure | S0 scalar | T0 typed | Required result |
| --- | ---: | ---: | --- |
| Macro-F1 | 0.2917 | 0.1667 | T0 at least S0 + 0.10 |
| T0 minus S0 Macro-F1 | — | -0.1250 | >= +0.10 |
| Sequence-bootstrap 95% lower bound | — | -0.2000 | > 0 |
| Normal-novelty recall | 1.0000 | 1.0000 | no decrease |
| Normal-novelty false reject | 0.0000 | 0.0000 | <= 0.10 |

The three hard failures are the negative effect size, its negative bootstrap
lower bound, and unsafe confusion.  T0 classified 60% of
registration/order-proxy events as `transient_local_content` (S0: 0%), and it
did not reduce bad-observation-as-novelty (both are 100%).  The offline
`T0+6` diagnostic also has Macro-F1 0.1667 and zero transient recall, so a
six-frame future confirmation window does not rescue this particular evidence
schema.

The clearest diagnostic is structural: under these deterministic proxies the
coverage-first veto frequently calls a covered corruption `normal_novelty`.
When the veto does not fire, the quality branch absorbs all transient proxies,
and the global/local branches do not separate registration from transient
content.  This is evidence against the proposed route, not a reason to tune
the frozen thresholds after looking at the matrix.

## What did pass

- All 20 of 20 event capsules produced one successful, immutable observer
  response.  No capsule was overwritten or rerun.
- Four independent vanilla/observer clean-control pairs (`fr1_desk`,
  `fr2_desk`, `fr3_walking_static`, `fr3_walking_xyz`) have byte-identical
  checkpoint-audit, prediction-summary, trajectory and native-health files.
- The median observer overhead is 0.1195 of vanilla per-frame runtime, below
  the 0.20 budget; paired peak GPU-memory delta is 0 MiB, below 512 MiB.
- The observer remained read-only and causal; all evaluation runs record
  StateGuard3R commit `38f1d9d35cd5dacd363f8f728921a2219912ccc3`.

Thus the experiment cleanly rules out the current *typed causal evidence and
proxy recipe*, rather than being invalidated by observer intrusion or runtime.

## Frozen evidence

- Gate-B protocol: `docs/protocols/state-triage-v2-stage0-gate-b.json`
  (committed before event outputs).
- Evaluator result:
  `outputs/state-triage-v2-stage0-evaluation-0001/result.json`, SHA-256
  `1a4d7f85904ce7b645363562995e8fc26b7be67d6c5cb9af4bad4fd35afd6816`.
- Attempt seal:
  `outputs/state-triage-v2-stage0-evaluation-0001/attempt-seal.json`, SHA-256
  `75de1d2f656963f316c6cbaaf1128f15b9e0f75efbf418333d21ea98460d08ab`.

Both artifacts are read-only (`0444`) in a read-only directory (`0555`).  The
result includes every method's confusion matrix, per-cause P/R/F1, NLL/ECE,
unresolved rate, delay, unsafe-confusion table, bootstrap samples and resource
ledger.

## Consequence

Do not enter the planned Stage 1 action simulator, provisional branch,
versioning, rollback, or second adapter based on this result.  Those stages
depend on typed triage being demonstrably better than scalar health, which this
frozen development protocol falsified.

The only scientifically honest follow-up, if separately planned, is narrower:
study coverage-conditioned conflict *detection* and calibrated abstention with
a benchmark whose labels and interventions do not make coverage collapse under
observation corruption.  It must use a new protocol; it may not reuse this
completed Stage-0 matrix to tune the failed typed rule.  This NO-GO is only for
already-disclosed local TUM development proxies and does not claim anything
about real persistent scene change, unseen data, or ReCal3R reconstruction
quality.
