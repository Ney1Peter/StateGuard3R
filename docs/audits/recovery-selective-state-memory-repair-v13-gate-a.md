# v13 selective persistent state/memory repair: Gate-A audit

- Date: 2026-08-13
- Decision: **PASS — authorizes only the pre-registered v13 dynamic
  always-commit control.**
- Scope: v13 source, causal contract, CUDA-disabled unit tests, one real
  checkpoint/interface probe, provenance and CPU regression. **No v13 CUDA
  forward, dynamic output, lease, command JSON, dispatcher invocation,
  candidate, wrong/low run, GT read, quality evaluation, data acquisition or
  download occurred.**

## Bound mechanism

After native `forward_recurrent_lighter` has computed the current raw result,
the ReCal3R update mask, and the native state/memory proposals, v13 lets the
frozen Detector-v3 inspect raw current health.  An alarm then restores exactly
the largest current row-RMS changes from the two pre-update persistent tensors:

```text
state_feat: (1,768,768), 96 / 768 rows restored
mem:        (1,256,1536), 32 / 256 rows restored
```

The current `camera_pose` is neither supplied to the repair operator nor
replaced; its GPU digest must be unchanged through the repair.  The intervention
affects only a later recurrent frame.  The runner uses the native masked/blended
proposals—not unmasked `new_state` or `new_mem`—and fixes order as:

```text
prestate reference → native rollout/memory proposal/output → native masks
→ sequence age + calibration record → raw detector decision → repair → CPU transfer
```

The control has `observer=None`, hence constructs no observer, selector,
candidate tensor clone, delta, or repair.  Candidate repair fails closed for a
non-CUDA state/memory tensor, nonfinite values, shape/dtype/device mismatch,
zero effective repair, a tie crossing the fixed top-eighth boundary, an
in-place mutation of a prestate reference, or a noninitial reset.

## Real interface evidence

The immutable [CPU interface probe](../../logs/recovery-selective-state-memory-repair-v13-dynamic-frame0-cpu-interface-probe-0001.json)
ran with `CUDA_VISIBLE_DEVICES=''` from committed source `4f03932`. It loaded
the existing dynamic development frame 0 only to bind the manifest/image
boundary and loaded the existing checkpoint to bind model interfaces. It did
not execute a model forward, `_encode_image`, RoPE, recurrence, detector,
repair, GT, future frame, or CUDA initialization.

| Bound interface | Observed |
| --- | --- |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| checkpoint load | no missing/unexpected keys |
| state register | CPU `float32` `(768,1024)` |
| state projection | `1024 → 768`, yielding `(1,768,768)` |
| pose memory | CPU `float32` `(1,256,1536)` |
| values serialized | false |
| CUDA initialized | false |

The checkpoint loader omits constructor fields equal to upstream defaults;
therefore `state_size/local_mem_size/state_pe/pose_head` are bound by the
actual instantiated register/memory/module interfaces, rather than falsely
rejecting valid omitted defaults.

## Gate-A test record

The targeted Torch-enabled CPU command was:

```text
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/tmp" \
PYTHONPATH="$PWD:$PWD/src:/data/wangzheng/Project2/baselines/ReCal3R/.venv/lib/python3.11/site-packages" \
.venv/bin/python -m pytest -q \
  tests/test_probe_recal3r_selective_state_memory_interface_v13.py \
  tests/test_selective_state_memory_repair_v13.py \
  tests/test_recal3r_selective_state_memory_repair_runner_v13.py \
  tests/test_run_recal3r_selective_state_memory_repair_v13.py \
  tests/test_dispatch_recal3r_selective_state_memory_repair_v13.py \
  tests/test_wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.py
```

Result: **29 passed**. It covers fixed `96/32` independent selection, exact
selected/unselected copy semantics, malformed/nonfinite/dtype/device/tie/no-op
failures, first-frame eligibility, raw current output preservation, later-frame
causality, no CPU calibration-mirror read, candidate-only repair, full-byte
token release, one-use lease, PID/start-tick ownership, two GPU snapshots,
postflight freezing, candidate semantic validation, and the waiter’s actual
clean-worktree success branch. The latter specifically covers the successful
`return 0` that v12 lacked.

The complete project CPU suite, with CUDA disabled, passed **617 passed, 67
skipped in 51.44 s**. The existing project virtualenv lacks Torch; all v13
Torch mathematics is instead covered by the targeted command above using the
already-installed ReCal3R site packages. `compileall` and `git diff --check`
passed.

## One-use dispatch boundary

Committed dispatcher/waiter source is `bb10363`. It pins the exact v13 runner,
operator, runner, dispatcher, waiter, baseline commit, checkpoint, source,
three manifests and frozen Detector-v3 configuration. It requires two GPU-2
UUID snapshots with at least `12288 MiB` free and no Project2 GPU-2 process;
it uses the existing `stateguard` tmux session, a one-use owner lease, pipe-ready
ordering, a full-byte random release token, PID/start-tick postflight proof,
artifact freeze, and a fresh CPU validator.

The validator is semantic as well as terminal. The control must produce 30 raw
direct commits with no repair and byte-identical protected outputs relative to
the frozen v1 dynamic always-control. A candidate must prove a real alarm,
the fixed `96/32` GPU repair evidence, no fallback, unchanged raw pose digest,
and a later trajectory change; an exit status alone cannot pass it.

## Authorized next action and stop rule

Only this exact command is authorized next, through the committed waiter in
the existing `stateguard` tmux session:

```text
scripts/wait_and_dispatch_recal3r_selective_state_memory_repair_v13_control.sh
```

It may release only:

```text
recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001
```

If that control fails any terminal, provenance, identity, output-identity, or
runtime gate, v13 is terminal
`SELECTIVE_STATE_MEMORY_REPAIR_V13_IMPLEMENTATION_OR_AVAILABILITY_OR_RUNTIME_NO_GO`.
Do not retry, edit the fixed v13 route, run the v13 candidate, wrong/low, GT,
or quality work. Only a frozen control PASS can authorize the single v13
dynamic candidate.
