# StateTriage3R v2 Stage 0 — Gate B/C freeze

Date: 2026-08-23  
Status: **PASS — inputs, controls, thresholds, evaluator and matrix are frozen before any evaluation response.**

## Frozen scope

- Development inputs: `state-triage-v2-stage0-inputs-0001`, SHA-256
  `c1ef23a57e8413b8020d81649b95e0347dcb67c907c3a7808555bd4ad83a4984`.
- Evaluation matrix: 20 event capsules, five per cause, every cause spanning
  `fr2_desk`, `fr3_walking_static`, and `fr3_walking_xyz`.  Their exact paths,
  digests and one permitted observer output name are in the JSON protocol.
- Clean controls: one 30-frame capsule for each source sequence.  The three
  new control windows are disjoint from every evaluation window and are
  explicitly excluded from event metrics.
- No data, weights, dependencies, GT, future frame, old detector response,
  source index, state write, recovery action or persistent-change label enters
  the online observer or T0 decision.

## Gate C result

The paired vanilla/observer control passed for all four source sequences.  In
each pair, checkpoint audit, prediction summary, trajectory and native health
artifacts have identical SHA-256 digests.  The full digest table is frozen in
the protocol.

The sparse observer's calibration wall time per frame is 0.01141, 0.01590 and
0.01745 seconds; the corresponding four-control vanilla median is 0.13312
seconds per frame.  Thus the median conservative ratio is below the 0.20
budget.  All paired peak-memory deltas are 0 MiB.

## Decision contract

`S0` is the only primary scalar baseline: native geometric-residual z-score.
`S1` (raw robust geometric residual) is retained as an ablation and is not
selected after event results.  `T0` applies coverage first, then fixed quality,
global-registration and local-transient branches.  All numerical constants are
the `fr1_desk` clean-prefix median ± 3 MAD values, except for the predeclared
coverage novelty threshold of 0.25.  `T0+6` is explicitly offline-only: it can
confirm a tentative transient only after six later rows, and cannot be used in
the real-time claim.

The frozen evaluator reports every method's full confusion matrix, per-cause
precision/recall/F1, NLL/ECE diagnostic values, unresolved rate, event delay,
unsafe confusions, resource ledger and a 10,000-resample sequence bootstrap.
It emits `GO` only if all thresholds in the JSON protocol hold; otherwise it
emits one terminal `NO-GO`.

## Irreversible rule

From this commit onward, no threshold, rule priority, sample, recipe, output
name, scalar selection or statistic may change in response to evaluation
evidence.  Each evaluation capsule allows exactly one successful observer
forward.  A failed attempt may only be replaced if it produced no response,
with its failure log retained and a new output ID.
