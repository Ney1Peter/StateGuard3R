# v20 post-commit native-scalar state hold: Gate-A NO-GO

- Date: 2026-08-14
- Decision: **`V20_IMPLEMENTATION_OR_INPUT_NO_GO` — terminal before CUDA.**
- Scope: the committed v20 execution plan is the complete authority for this
  version.  No v20 runtime capability, tmux window, CUDA process, control,
  candidate, wrong/low condition, GT read, download or upstream modification
  occurred.

## What was established

The v20 source and CUDA-hidden tests establish the narrow local mechanism
without exercising it on a GPU.  The official pinned ReCal3R lightweight loop
has this relevant order:

```text
_downstream_head -> native _compute_recal3r_update_mask
-> global-state write with update_mask1
-> native memory write with update_mask2
-> _maybe_record_u_calibration_step -> CPU prediction export
```

The v20 instance wrapper calls the native mask first and can replace only its
returned state mask with an all-zero same-interface clone on one previously
armed eligible frame.  It leaves the separate native memory mask untouched.
Its detector row is limited to current scalar uncertainty/reliability,
post-commit state-delta, raw-pose jump and geometric residual.  The
state-delta is reduced inside the permitted post-commit calibration callback;
it does not use ReCal3R's `oracle_window` or `final_state` trace modes, which
would introduce a future/final-state route.  Reset timing was also aligned to
the official previous-frame reset control flow, and a consumed hold requires
exact GPU `torch.equal(state_before, state_after)`.

With `CUDA_VISIBLE_DEVICES=''`, the v20 targeted suite passed **20 tests in
0.20 seconds**.  It covers closed scalar schema, frozen constants, arm/clear/
last-frame/reset/one-consume behavior, native-mask-first replacement, exact
held-state identity in a fake official loop, exception restoration, dynamic
and frame-zero capability isolation, and an AST source/import audit.  The
v20 modules also passed `compileall` and `git diff --check`.

## Falsified release premise

Gate B in the committed plan requires that the one direct-native control have
all four protected files' SHA-256 values **"unchanged"**:

```text
checkpoint-load-audit.json
health.jsonl
predictions-summary.json
trajectory.json
```

However, the pre-registration supplies no expected SHA-256 for any of these
files, no frozen v20 reference artifact, and no deterministic formal serializer
whose bytes are pre-bound elsewhere in the plan.  The only SHA-256 values in
the plan are for the raw RGB listing, the ordered RGB path list and checkpoint.
Its hard exclusion separately forbids a v20 runtime import/read path to any
v1--v19 output, runner, capability, log or artifact.

Thus a v20 dispatcher cannot mechanically decide Gate-B PASS or FAIL:

```text
expected protected hashes: absent
allowed historical reference read: forbidden
permitted extra baseline/control to derive hashes: absent
```

Executing the declared one control and then treating its newly produced hashes
as the missing expectation would be a post-hoc calibration, not a test of an
already registered equivalence criterion.  Taking hashes from historical
outputs would violate v20's own isolation rule.  Either choice destroys the
specified one-use decision rule.  This is an implementation/protocol failure
at Gate A, so the conservative outcome is an early terminal NO-GO rather than
an unregistered GPU release.

## Preservation and stop rule

The ReCal3R worktree remains clean at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`; its checkpoint and raw data were
only read by the already frozen CPU probe.  No dependency, data or weight was
downloaded.  The pre-existing v14/v15 untracked files remain untouched and
were not staged.

V20 is terminal.  Do not add the missing hashes after this decision, derive
them from a new or historical run, create its capability, start CUDA/tmux,
run either release ID, or weaken the source-isolation rule.  Any further
research route must use a fresh version and explicitly pre-register a
deterministic serializer plus all protected expected hashes (or a different,
fully specified control criterion) before Gate A.
