# v12 patch-embed global-pooled pose export: pre-execution availability NO-GO

- Date: 2026-08-13
- Decision:
  **`PATCH_EMBED_GLOBAL_POOLED_POSE_EXPORT_V12_IMPLEMENTATION_OR_AVAILABILITY_OR_RUNTIME_NO_GO`**
- Bound control ID:
  `recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001`

## What happened

The committed Gate-A PASS authorized exactly one dynamic always-commit control,
and only through the committed tmux waiting wrapper.  On 2026-08-13 that
wrapper was started in the existing `stateguard` tmux session with no v12
control/candidate output, lease, or command JSON already present.  GPU 2 had
the required UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250` and reported
`45586 MiB` free both before and after the waiter's 30-second confirmation.
The wrapper recorded that external gate PASS and then terminated before it
wrote the one-use command JSON or invoked the dispatcher.

The exact committed `require_clean_tracked_worktrees` function is:

```bash
if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    return 1
fi
if [[ -n "$(git -C "${RECAL3R_ROOT}" status --porcelain)" ]]; then
    return 1
fi
```

Both worktrees were clean.  Consequently the second false `if` is the last
command in the function and Bash returns status 1 from the function.  With
the wrapper's committed `set -e`, its top-level invocation exits immediately.
This is a launcher pre-execution failure, not a GPU-capacity, dataset,
checkpoint, model, detector, or pose-decoding result.

## Immutable boundary and observed evidence

The ignored but preserved waiter log is
`logs/recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001-wait.log`.
Its final complete records are:

```text
V12_WAIT_START ... min_free_mib=12288 ...
V12_WAIT_EXTERNAL_GATE_PASS ... first_free_mib=45586 second_free_mib=45586 ...
```

There is no subsequent `V12_WAIT_DISPATCH` record.  Read-only checks after
termination found all of the following absent:

- `tmp/recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001-command.json`;
- `tmp/recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001.owner-lease`;
- `outputs/recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001/`;
- all v12 control primary terminal artifacts, result JSON, postflight, and
  validator.

There was no remaining v12 waiter, dispatcher, runner, or model process and
GPU 2 still reported `45586 MiB` free.  Thus no CUDA forward, checkpoint
load, model child, raw/GT/future-frame read, dynamic candidate, wrong/low
condition, or quality evaluation occurred.  The two tracked worktrees were
clean at the failure boundary; StateGuard3R was at `0eddf03` and ReCal3R at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.

## Why this terminates v12

The frozen Gate-A audit requires that the control be dispatched only by the
committed waiter and dispatcher.  The execution plan states that *any control
failure* is terminal v12 NO-GO and explicitly prohibits changing the launcher
or retrying v12.  Adding an explicit `return 0`, calling the dispatcher by
hand, recreating the command JSON, or launching a new v12 ID would repair and
retry the failed version rather than test the pre-registered version.

Accordingly, do not run the v12 candidate, wrong/low controls or candidates,
GT/quality work, or any modified/retried v12 launcher.  Preserve the two
frozen CPU probe artifacts and the waiter log.  A successor must use a newly
pre-registered, mechanism-distinct recovery route with its own Gate-A and
one-use execution boundary.
