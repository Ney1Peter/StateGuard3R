# v14 bounded update-pressure: Gate-A input-boundary NO-GO

- Date: 2026-08-13
- Decision: **`V14_IMPLEMENTATION_OR_INPUT_NO_GO`**
- Scope: the pre-CUDA Gate A required by the frozen
  [v14 execution plan](../recovery-update-pressure-bounded-state-memory-v14-execution-plan.md).

## Decision

Gate A does not authorize a v14 CUDA release.  The one-time dynamic capsule
builder violates the plan's hard quarantined-archive boundary before it reaches
its narrow `images` projection.  Specifically,
[`build_dynamic_capsule_v14`](../../scripts/build_recal3r_dynamic_input_capsule_v14.py)
calls `sha256_file(archive)` before `extract_image_projection_v14`:

```python
# scripts/build_recal3r_dynamic_input_capsule_v14.py:163--167
archive_metadata = _regular_0444(archive, label="quarantined v1 archive")
if sha256_file(archive) != ARCHIVE_SHA256:
    raise CapsuleBuildV14Error("quarantined v1 archive hash differs")
projection = extract_image_projection_v14(archive)
```

`sha256_file` opens the archive and repeatedly reads it to EOF.  This is a
whole-file read, not an opaque external attestation: it necessarily reads the
quarantined suffix.  The fixed historical archive is `113519` bytes; its
top-level `images` array ends at byte offset `8491`, and the next top-level
key, `input_frames`, begins at offset `8496`.  The suffix contains the legacy
metadata that the v14 plan explicitly quarantines.

Section 3 requires the builder to stop at the closing `images` delimiter and
states that a poisoned test must fail on **any** read beyond that delimiter.
The full-file hash is therefore directly incompatible with the pre-registered
input construction.  It is not enough that the subsequent projection function
returns an array without suffix fields: the prohibited read has already
occurred.  Gate A item 2 consequently fails, and Section 5 requires the
terminal decision above and forbids v14 CUDA.

Changing the builder to omit/redefine the archive hash, weakening the poison
requirement, rebuilding the capsule, or rerunning this v14 route would modify
and retry a version after its Gate-A failure.  None is authorized here.

## Evidence and execution boundary

The generated RGB-only capsule remains frozen at
`outputs/recovery-update-pressure-v14-dynamic-input-capsule-0001.json`
(mode `0444`, SHA-256
`75542fe1e990cdf06f2333571ca5a9f7f877ac51fdac7e721f54d765053234e2`).
It is not evidence that the archival construction complied with the required
read boundary.

The sole v14 runtime-adjacent evidence is the frozen CPU interface probe at
`logs/recovery-update-pressure-v14-dynamic-frame0-cpu-interface-probe-0001.json`
(mode `0444`, SHA-256
`d166770a73d796e98dc124a8144a7a5665585d608c66165e039f06a90d90b52c`).
It records `status="passed"`, `cuda_visible_devices=""`,
`cuda_initialized=false`, no model forward, and the required interfaces
`state_feat=(1,768,768)`, `mem=(1,256,1536)`, and
`update_pressure=(1,768,1)`.  This limited CPU observation does not cure the
independent capsule-builder violation.

No v14 dispatcher or waiter was invoked; there is no v14 owner lease,
preflight, tmux child, CUDA checkpoint load/forward, control or candidate
output, wrong/low run, GT quality evaluation, download, or modification of
ReCal3R, the checkpoint, or raw RGB.  The pinned ReCal3R commit and checkpoint
remain the values specified in the plan.

## Required stop

Do not start the v14 tmux waiter, dispatcher, dynamic control, candidate,
wrong/low matrix, or GT evaluator.  Do not alter the v14 capsule builder,
capsule, operator, runner, dispatcher, tests, or plan thresholds in an attempt
to reopen this version.

Any later proposal must use a new version and a new pre-registration.  Before
any model work, it must choose an archive-provenance scheme that is compatible
with a demonstrable no-read-past-`images` boundary, and must prove that scheme
with an instrumented poisoned-reader test.
