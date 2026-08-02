# Minimal quarantine pilot following formal Detector v2 GO

Status: **pre-registered exploratory ledger-level pilot.** This protocol is
created only because `formal-v2-evaluation-0001` returned Detector v2 **GO**.
It is a bounded feasibility experiment, not a second detector evaluation and
not evidence that ReCal3R geometry or trajectory recovery works.

## Scope and evidence status

- Bind the immutable formal-v2 commitment 0002, calibration 0002, evaluation
  0001, and all nine frozen run outputs exactly by SHA-256.
- Use only the already disclosed three new-scene holdout ledgers. No model
  forward, dataset download, ReCal3R source change, checkpoint change, or
  detector threshold change is allowed.
- The scene is no longer blind after formal evaluation. Results are therefore
  explicitly **exploratory** and cannot establish generalization.
- The experiment never changes the saved ReCal3R recurrent state, predictions,
  trajectory, Health Ledger, or baseline output. It reports a counterfactual
  *ledger-level committed-update-magnitude proxy* only.
- No result from this pilot authorizes rollback, safe replay, a deployment
  claim, or a claim that reconstruction quality improved.

## Fixed detector trigger and action

The trigger is the already frozen combined Detector v2 score and threshold
`2.0686323694270046`, exactly as recorded in the formal evaluation. It sees
only the pre-existing online health ledger and is not recalibrated. On the
first score at or above threshold, the policy starts once per run:

1. buffer that frame and the next two positions (maximum buffer length 3);
2. at the decision point, release the buffered ledger magnitudes only when the
   final two buffered combined scores are both strictly below the same frozen
   threshold; otherwise drop all buffered magnitudes;
3. do not trigger again in that run.

The policy input is solely fixed scores and `global_state_delta` from the
already generated online ledger. It has no runtime access to GT pose/depth,
event labels, corruption type, source-pool metadata, future frames beyond its
three-position fixed buffer, or any recovery oracle. Labels may be used only
after the action to report exploratory clean/event placement; they must not
alter the action.

## Metrics and interpretation

For every action, record alarm position, buffered positions, release/drop
decision, absolute `global_state_delta` mass in the buffer, and the proxy mass
that would remain committed. The primary proxy quantity is:

```text
withheld update-magnitude proxy = baseline buffered absolute-delta mass
                                  - counterfactual committed mass
```

Report run timelines, action count, release/drop count, proxy reduction,
buffer length, CPU runtime, and evaluation-only clean/event action placement.
This proxy can show that a fixed detector policy would withhold update mass;
it cannot show a different ReCal3R output or reduced persistent geometric
error because no alternate recurrent forward is performed.

The exploratory feasibility condition is: formal GO binding and immutable
provenance pass; policy inputs prove no label/GT fields; no baseline artifact
changes; and at least one positive proxy reduction under the predeclared
policy. Regardless of this condition, the final decision string is
`EXPLORATORY_PROXY_ONLY_NO_ROLLBACK`.

## Commitment, uniqueness, and completion

Before evaluation, a separate commitment freezes this protocol, current
StateGuard3R commit, policy values, formal-v2 artifact hashes, and output
paths. The evaluation output is atomic, no-overwrite, and uses directories
`0555` and files `0444`. It is executed exactly once. Completion requires its
metrics, timelines, manifest, limitations statement, CPU tests, repository
checks, and GPU/process cleanup; it does not require nor permit a further
recovery experiment.
