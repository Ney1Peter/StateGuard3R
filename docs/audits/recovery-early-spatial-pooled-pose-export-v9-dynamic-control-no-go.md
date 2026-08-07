# v9 dynamic always-control: availability/runtime no-go

- Date: 2026-08-07
- Decision: **EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_AVAILABILITY_OR_RUNTIME_NO_GO**
- Bound run: `recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001`

The sole preregistered Gate-B dynamic always-control was dispatched once from
the `stateguard` tmux session on GPU 2.  Its child loaded the pinned checkpoint
strictly, then terminated with exit code 1 before producing an output tree:

```
EarlySpatialPooledPoseRunnerError: pinned early spatial sequence shape changed
```

The failure occurs in
`recal3r_early_spatial_pooled_pose_runner_v9.py:168`, during the runner's
shape guard.  This means the v9 frozen source assumption is not available in
the actual pinned GPU recurrent execution.  It is a runtime availability
failure, not a quality measurement.

## Immutable evidence

- Driver result: exit 1, child PID 62313, start ticks 595666877.
- Original preflight, main, transcript, driver, result and postflight are
  regular NUL-free files frozen `0444` under `logs/` with the bound run ID.
- The independent CPU-only validator is frozen `0444` and reports
  `CHILD_NONZERO_EVIDENCE_COMPLETE`.
- Postflight records no Project2 process remaining on GPU 2.
- No v9 output directory exists, so no partial result is interpreted.

## Required stop

Do not run the dynamic candidate, wrong/low controls or candidates, GT/quality
evaluation, or a new v9 ID.  Do not alter the v9 source, shape guard, pool,
head, precision, detector, dispatcher, or logging and retry.  Any successor
must be a mechanism-distinct v10 proposal with a fresh Gate-A contract.
