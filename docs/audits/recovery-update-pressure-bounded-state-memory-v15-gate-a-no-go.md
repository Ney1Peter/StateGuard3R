# v15 bounded update-pressure: Gate-A interface-capability NO-GO

- Date: 2026-08-13
- Decision: **`V15_IMPLEMENTATION_OR_INPUT_NO_GO`**
- Scope: the pre-CUDA Gate A of the frozen
  [v15 execution plan](../recovery-update-pressure-bounded-state-memory-v15-rgb-list-execution-plan.md).

## Decision

Gate A does not authorize a v15 CUDA release.  Its required one-use,
CUDA-hidden interface probe was specified to load only frame zero through the
v15 capsule and checkpoint, and to bind the three persistent interfaces
without a forward.  The actual probe did load only one RGB image and did not
initialize CUDA or execute a forward, but its capability path was wider than
that probe boundary.

The probe calls `_frame_zero_from_capsule()` at
`scripts/probe_recal3r_bounded_update_pressure_interface_v15.py:245`, which
calls `load_rgb_listing_capsule_v15(CAPSULE)` at line 104.  The general loader
calls `_listed_rgb_paths(RGB_LISTING)` at
`src/stateguard3r/rgb_listing_capsule_v15.py:216`.  That helper verifies,
hashes and reads the raw `rgb.txt` listing (lines 170--180) before returning
the selected RGB paths.  Thus the actual probe capability was:

```text
capsule → raw rgb.txt → frame-zero RGB → checkpoint
```

not the pre-registered probe capability:

```text
capsule → frame-zero RGB → checkpoint
```

This is a hard input-boundary failure, not a harmless redundant integrity
check.  A later change that gave the probe a narrower capsule-only parser, or
a re-execution of the same evidence/run ID, would repair and retry a version
after its Gate-A failure.  Section 5 of the plan instead requires the terminal
decision above.

## Frozen evidence and what it does establish

The first invocation created the one-use immutable evidence file:

| artifact | SHA-256 | mode |
| --- | --- | --- |
| `logs/recovery-update-pressure-v15-dynamic-frame0-cpu-interface-probe-0001.json` | `e35f5493ffd636973e3141c29526b1be5e8c7d785c6b1b2868e1c17402359ac4` | `0444` |

It records ReCal3R commit
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, checkpoint hash
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`,
the frozen v15 capsule hash
`c15bb580ea036b78784d339c3f1a5d3cafa7823aa3ccb548c5d0fdc8a7d95d40`,
and `cuda_visible_devices=""`, `cuda_initialized=false`,
`model_forward_executed=false`.  It also records the intended interfaces:

| interface | observed shape |
| --- | --- |
| recurrent state feature | `(1,768,768)` |
| pose memory | `(1,256,1536)` |
| native update pressure | `(1,768,1)` |

Those observations remain useful forensic evidence, but the file's internal
`status="passed"` describes only its local assertions.  It cannot override the
plan-level capability violation above.  A deliberate second invocation was
refused because the `O_EXCL` evidence path was already occupied; it neither
changed this file nor opened CUDA.

## Execution boundary

Before this decision, the targeted CPU tests for the then-built v15 capsule,
operator, recurrence, runner and probe passed (`18 passed`).  The interface
probe ran with `CUDA_VISIBLE_DEVICES=''`; an independent check reported CUDA
unavailable and uninitialized.  Gate-A item 5 nevertheless fails, so a full
Gate-A PASS is impossible.

No v15 dispatcher/waiter was invoked.  There is no v15 owner lease, GPU
preflight, tmux child/window, CUDA checkpoint forward, control or candidate
output, wrong/low capsule, GT evaluation, data/dependency/weight download, or
modification of ReCal3R, the checkpoint, raw RGB or `rgb.txt`.

## Required stop

Do not run the v15 waiter, dispatcher, dynamic control, candidate, wrong/low
matrix or GT evaluator.  Do not edit/rebuild/rerun the v15 capsule, interface
probe, input parser, operator, runner or one-use protocol to reopen this
version.  Preserve its untracked v15 implementation/tests and frozen evidence
for audit.

Any successor must be a newly numbered, independently pre-registered route.
Before any CUDA work, it must give the interface probe a genuinely
capsule-only frame-zero reader and prove, with an instrumented file-access
test, that no raw listing is read transitively by that probe.
