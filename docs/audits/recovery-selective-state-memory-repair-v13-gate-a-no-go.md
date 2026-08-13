# v13 selective state-memory repair: Gate-A implementation/CPU-contract NO-GO

- Date: 2026-08-13
- Decision:
  **`SELECTIVE_STATE_MEMORY_REPAIR_V13_IMPLEMENTATION_OR_CPU_CONTRACT_NO_GO`**
- Scope: the pre-CUDA Gate-A required by the frozen
  [v13 execution plan](../recovery-selective-state-memory-repair-v13-execution-plan.md).

## Decision

The v13 source violates an immutable mechanism boundary before any CUDA work is
authorized.  Its production runner directly imports the earlier v2 recovery
runner:

```python
# scripts/run_recal3r_selective_state_memory_repair_v13.py:29--32
import scripts.run_recal3r_online_quarantine_v2 as v2
smoke = v2.smoke
```

The same old recovery route is also imported by the v13 recurrent module for
`QuarantineWatchdog`:

```python
# src/stateguard3r/recal3r_selective_state_memory_repair_runner_v13.py:11
from .recal3r_online_quarantine_v2 import QuarantineWatchdog
```

Section 3.2 of the frozen plan permits reuse only of low-level state
snapshot/witness, Detector-v3 and terminal-evidence interfaces, and explicitly
forbids importing or calling any v3--v12 recovery runner, dispatcher or
launcher.  These imports are direct v2 recovery-runner dependencies, not any
of the three allowed interface classes.  They also make the v13 source audit
incomplete: `audit_v13_runner_contract` checks the inner recurrent loop, but
does not inspect the production runner's import graph, so its local PASS cannot
establish the plan's required rejection of previous recovery imports.

There is a second independently disqualifying input-boundary violation.
`_validate_args` in the production runner calls `json.loads` on the whole
development input manifest before running a model.  Those manifests embed
`metadata.groundtruth`.  The plan forbids reading GT before all six development
outputs are frozen.  The Gate-A test invokes this validator for the three
development manifests; a subsequent read-only validation query during this
audit also printed the embedded metadata.  No GT was used to select state or
memory rows, evaluate a candidate, tune a parameter, or report a quality
metric, but the access itself is outside the pre-registered boundary and is
recorded rather than hidden.

The plan's Section 4 states that **any** Gate-A failure is terminal and forbids
every v13 CUDA run and later evaluation.  Removing these imports, introducing
a manifest reader that avoids the GT field, broadening the allow-list, or
rerunning the tests would modify and retry v13 after its Gate-A failure.  None
is authorized under this version.

## Independent one-use control preflight failure

The control was released before this audit was reconstructed.  Its committed
tmux waiter recorded the following immutable sequence in
`logs/recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001-wait.log`:

```text
V13_WAIT_START ... at=2026-08-13T21:55:42+08:00
V13_WAIT_EXTERNAL_GATE_PASS ... first_free_mib=45586 second_free_mib=45586 ...
V13_WAIT_DISPATCH ... at=2026-08-13T21:56:13+08:00
```

It wrote the exact canonical command JSON at
`tmp/recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001-command.json`
and then the dispatcher atomically created the empty mode-`0700` owner lease
at `tmp/recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001.owner-lease`.
There is no preflight, main log, tmux transcript, driver, result, postflight,
validator or output directory, and no v13 process remains.

This pins the stop to the dispatcher prefix before the first preflight artifact:
after `_acquire_lease`, the dispatcher still performs child-contract validation
and component provenance validation before it can take GPU snapshots, create a
tmux pane, or create a model child.  The wait log does not capture the
dispatcher's stderr, so the evidence cannot distinguish those early failure
branches.  It would be inaccurate to infer a particular exception from the
later worktree state.  That missing terminal reason means the one-use control
cannot prove its required preflight or control identity, and is itself a
fail-closed availability/evidence failure.  Its one-use lease means the
registered control ID cannot be reissued.  The plan independently declares any
control failure terminal, so this is a second terminal v13 boundary; it does
not authorize a retry.

## What did pass, and why it is insufficient

The implementation did provide useful but non-authorizing mechanical evidence:

- `recal3r_selective_state_memory_repair_runner_v13.py` statically places a
  candidate repair after native state/memory commit and calibration, after the
  raw detector decision, and before CPU transfer.  Its local audit reports
  `after_native_state_memory_update_and_calibration_before_cpu_transfer`.
- The CUDA-disabled targeted Gate-A suite passed **25 tests** in 10.16 seconds:
  the fixed rowwise RMS/top-eighth operator, fixed `96`/`32` selections,
  tie/nonfinite rejection, no in-place pre-state mutation, candidate/control
  ordering, raw-pose export witness, CPU interface probe, one-use dispatcher,
  and the tmux waiter's clean-worktree success branch.
- The complete CUDA-disabled project CPU suite passed **692 tests** in 81.33
  seconds.  `python -m compileall -q src scripts tests`, `git diff --check`,
  and `bash -n scripts/wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.sh`
  also passed.
- The immutable CPU-only interface probe is present at
  `logs/recovery-selective-state-memory-repair-v13-dynamic-frame0-cpu-interface-probe-0001.json`
  (mode `0444`, SHA-256
  `e099a93bc6aff5d4e9ea367a15c930305f0b7c749486e4abd0947c0ec8df8380`).
  It loaded the existing checkpoint on CPU without a model forward and bound
  `state_feat=(1,768,768)` and `mem=(1,256,1536)`.

Those facts show that the proposed state/memory operator is mechanically
testable.  They do **not** cure an immutable provenance/input-boundary failure
and do not constitute a v13 Gate-A PASS.

## Execution boundary and provenance

No GPU forward was launched.  The unique control's waiter observed the pinned
GPU 2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250` at `45586 MiB` free
twice; its dispatcher then stopped before its own GPU preflight or any child.
There is no v13 control/candidate output, terminal evidence, wrong/low output,
quality evaluation, data download or ReCal3R modification.  The command JSON
and empty owner lease are preserved one-use failure artifacts, not evidence of
a model execution.

The pinned ReCal3R worktree remains clean at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`; the checkpoint remains
`src/cut3r_512_dpt_4_64.pth`, SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
The source used for the v13 implementation is recorded by StateGuard3R commits
`e48bd28`, `7fb5f0d`, `4f03932`, and `bb10363`; the v13 waiter and its test
are committed in `bb10363`.  The earlier committed Gate-A PASS audit
`5072e60` is superseded by this later boundary audit; it remains in history as
the contemporaneous record rather than being rewritten after the fact.

## Required stop

Do not start the v13 tmux waiter, dispatcher, dynamic always-control or
candidate, wrong/low matrix, GT/quality evaluation, or any v13 retry.  In
particular, do not remove the existing one-use lease/command JSON or create a
second control ID.  Do not alter the v13 one-eighth operator, detector,
launcher, import boundary or manifest reader in order to reopen this version.

Any later proposal must use a new version and new pre-registration.  It must
be mechanism-distinct from v13, own its allowed utilities rather than importing
a prior recovery runner, and prove a GT-safe manifest/header interface before
any non-CUDA or CUDA execution.
