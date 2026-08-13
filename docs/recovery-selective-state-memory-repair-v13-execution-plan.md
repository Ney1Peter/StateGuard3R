# Recovery selective state-memory repair v13: execution plan

- Date: 2026-08-13
- Status: **terminal Gate-A NO-GO —
  `SELECTIVE_STATE_MEMORY_REPAIR_V13_IMPLEMENTATION_OR_CPU_CONTRACT_NO_GO`.
  The unique dynamic always-control was released through its committed tmux
  waiter, but its dispatcher aborted before a preflight, child, checkpoint
  load, CUDA forward or output.  No v13 control/candidate output, wrong/low
  matrix, quality evaluation, data acquisition, download or modification of
  ReCal3R occurred.**  See
  [the Gate-A NO-GO audit](audits/recovery-selective-state-memory-repair-v13-gate-a-no-go.md).
- Long-horizon objective (12–18 hours of active work): determine whether a
  causal, bounded **partial repair of persistent ReCal3R state and pose memory**
  can suppress an alarm frame's largest state changes while preserving its raw
  current output, baseline control semantics, provenance, runtime and frozen
  development quality gates.  A failed gate is a terminal v13 result, never a
  license to tune or retry v13.

## 1. Motivation and mechanism boundary

v1/v2 rollback proved that a full recurrent-state restore is causal but harms
quality: it rejects all information from an alarm frame.  v3--v12 instead
tried to repair the *exported camera pose* with an anchor, geometry, pointmap,
retriever/decoder/encoder latent, or patch embedding.  Those routes either
failed the fixed quality/runtime gate or stopped at availability/provenance.

v13 is neither a full rollback nor any pose-export mechanism.  It leaves every
frame's raw `camera_pose` export untouched.  After the existing current-frame
Detector-v3 has raised an alarm, it makes a bounded, in-place **persistent
state repair**:

```text
pre_state_feat, pre_mem -- native frame update --> proposed_state_feat, proposed_mem
                                               -- per-row delta ranking --> selected rows
committed_state_feat[selected] = pre_state_feat[selected]
committed_mem[selected]         = pre_mem[selected]
all unselected rows             = proposed rows
current prediction/export       = raw native prediction
```

The ranking signal is the current candidate's rowwise root-mean-square delta,
not RGB, pose, pointmap, decoder feature, GT, a future frame, an anchor, a
baseline alarm file, or previous exported pose.  It is evaluated only after
the frozen detector has consumed raw current health and after the ordinary
ReCal3R update has been computed.  The repaired tensors influence only later
native recurrent frames.

This is therefore a token-level local-repair hypothesis for the actual
persistent state, rather than a correction/replacement for a camera output.

## 2. Fixed v13 operator

For finite floating tensors with batch size one, define

```text
d_state[j] = RMS(proposed_state_feat[0,j,:] - pre_state_feat[0,j,:])
d_mem[j]   = RMS(proposed_mem[0,j,:] - pre_mem[0,j,:])
```

For each tensor independently, choose the largest exactly one eighth of rows:
`k = row_count // 8`.  The current pinned shapes consequently require
`state_feat=(1,768,768), k_state=96` and `mem=(1,256,1536), k_mem=32`; the
implementation also rejects any nonpositive `k`, malformed dimensions,
dtype/device mismatch, nonfinite value/delta, or a tie crossing a selected
cut boundary.  It records indices and non-sensitive delta summary statistics,
never state or memory values.  The selected rows are copied from the
pre-frame tensors; all other rows must be byte/value-identical to the proposed
tensors.  No blend, threshold, weight, learned parameter, retry, fallback or
second selection is permitted.

`state_pos`, `init_state_feat`, `init_mem`, reset state, ReCal3R calibration
trace, update-pressure accumulator, model reference state, RNG and detector
state are **not** restored by v13.  This is deliberate: v13's sole intervention
is the two persistent tensors named above, while ReCal3R's native calibration
side state remains causally advanced.  Any attempt to broaden that set is a
different version.

The current prediction is produced before repair and its `camera_pose` is not
read, encoded, decoded, replaced, used as a template, or supplied to the
operator.  Each alarm must prove a GPU-side raw-pose digest is unchanged across
the repair boundary.  Clear frames use the exact original raw transfer.

## 3. Immutable common boundaries

1. Reuse only the three existing disclosed development manifests, the present
   checkpoint, pinned ReCal3R commit, and frozen Detector-v3 configuration.
   Do not download data, weights or dependencies; modify ReCal3R, checkpoint,
   raw RGB, GT, corruption manifests or Detector-v3 config; or read GT/blind
   data before all six development outputs are frozen.
2. v13 may reuse only low-level state snapshot/witness, Detector-v3 and
   terminal-evidence interfaces.  It must not import/call any v3--v12 recovery
   pose primitive, exporter, runner, dispatcher or launcher, and must not
   construct a pose alternative.
3. The detector consumes only current/past raw candidate health, prior/current
   model-ready RGB overlap and raw capture timestamps, as in frozen v3.  It is
   called before a repair; repaired state/output never feeds the detector for
   that frame.
4. No future input, GT/depth/label/event/source index, baseline alarm artifact,
   anchor, history pose, external registration, pointmap, encoder/decoder/
   patch latent, raw pose value, or CPU-side state solver may enter selection.
5. v13 runs only on GPU 2 UUID
   `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`; it never stops or reserves
   another user's process.  Before every dispatch, two UUID-identical snapshots
   must each report at least 12288 MiB free and no Project2 GPU-2 process.

## 4. Gate A: implementation and non-CUDA contract

Before CUDA work, all of the following must pass and be committed in a Gate-A
audit:

1. Source audit binds the pinned ReCal3R lighter update order: native update
   mask, state/memory proposal, calibration-step recording, then v13's
   candidate-only post-update repair before CPU transfer; it rejects previous
   recovery imports and every prohibited input.
2. CUDA-disabled Torch tests prove rowwise RMS/top-eighth selection, strict
   selection boundary, copying only selected state/memory rows, exact retention
   of unselected proposal rows, malformed/nonfinite/dtype/device/tie failure,
   raw-pose non-access, and no token values in evidence.
3. Synthetic runner tests prove candidate-only execution, all-clear baseline
   identity, detector-before-repair order, raw-current-export preservation,
   no reset crossing, and causal use of repaired state on a later frame.
4. A non-CUDA one-frame real manifest probe may load only the frame's image and
   checkpoint as needed to bind source/interface metadata; it must never enter
   `_encode_image`/RoPE, recurrent rollout, detector, GT or a v13 state repair.
   Since the pinned native CPU encoder is conclusively unavailable (v11), the
   real availability proof for state tensors belongs to the single GPU dynamic
   candidate below, not a substituted CPU encoder.
5. The v13-only dispatcher and tmux waiter must have non-CUDA child tests for
   exact run ID/GPU UUID, two memory snapshots, clean worktrees, command
   construction, one-use lease, pipe-ready ordering, PID/start-ticks/full-byte
   release token, postflight, freeze and independent validator.  The waiter
   test must execute the **clean-worktree-success** branch so the v12 missing
   success return cannot recur silently.
6. Run targeted tests with CUDA disabled, the full project CPU suite,
   `compileall`, `git diff --check`, permission/hash checks and clean
   StateGuard3R/ReCal3R provenance.

Any Gate-A failure is
`SELECTIVE_STATE_MEMORY_REPAIR_V13_IMPLEMENTATION_OR_CPU_CONTRACT_NO_GO` and
forbids every v13 CUDA run or later evaluation.

## 5. Gate B: one dynamic control and one dynamic candidate

Only a frozen Gate-A PASS authorizes this exactly-once control:

```text
recovery-selective-state-memory-repair-v13-dynamic-always-commit-0001
```

The control has `observer=None`: it must not construct a selector, compute a
state/memory delta, clone a candidate tensor, call a detector or repair helper.
It must exit zero with complete terminal and CPU-validator evidence, no
remaining Project2 GPU-2 process, 30 direct commits/pending zero, runtime/v1
`<=1.20`, and byte-identical frozen-v1 dynamic
`checkpoint-load-audit.json`, `health.jsonl`, `predictions-summary.json` and
`trajectory.json`.  Any failure terminates v13.

Only that frozen PASS authorizes this exactly-once candidate:

```text
recovery-selective-state-memory-repair-v13-dynamic-candidate-0001
```

For every real candidate alarm, its immutable timeline must prove:

- `action=selective_state_memory_repair`, pending transaction count zero, and
  no full rollback/fallback;
- detector decision/health occurred before the repair;
- state and memory proposed/pre/committed shape, dtype/device and GPU-only
  fingerprints; `96` and `32` selected rows; tie-free selection; finite deltas;
- selected committed rows equal their pre-frame rows, unselected committed rows
  equal their proposal rows, and at least one repaired state and memory row;
- raw `camera_pose` GPU digest is unchanged over the repair and
  `export_action=export_real_camera_pose`;
- no raw pose, GT/future, anchor, pointmap, encoder/decoder/patch latent, or
  external fallback provenance; and candidate runtime/v1 `<=1.20`.

At least one alarm with complete evidence and a future-frame trajectory change
relative to the control are required to show a non-no-op state intervention.
Any control/candidate error, no-op, missing proof, or runtime failure is
`SELECTIVE_STATE_MEMORY_REPAIR_V13_IMPLEMENTATION_OR_AVAILABILITY_OR_RUNTIME_NO_GO`.
It forbids retrying v13, changing the one-eighth fraction/operator/detector/
launcher, and all wrong/low/GT/quality work.

## 6. Gate C: complete frozen development matrix and decision

Only frozen dynamic candidate PASS authorizes exactly one always-control and
one candidate for each existing `wrong` and `low` manifest, using the same
operator and terminal protocol.  The six outputs must all be frozen before the
existing evaluator reads the disclosed GT.

Candidate readiness requires all three controls byte-identical to their v1
baselines, zero causal/provenance failure, at least two of three conditions
improving both tail ATE and translation RPE, both median improvements at least
`+5%`, and median candidate runtime ratio `<=1.20`.  PASS writes an unexecuted
new-data acquisition plan only.  Any failure writes a v13 feasibility NO-GO
and does not reopen this matrix.

## 7. Terminal execution record

Gate A is closed as a NO-GO.  The independent audit found that the v13
production runner directly imports the old v2 recovery runner, contrary to the
immutable v13 mechanism boundary.  It also found that its argument validator
deserializes the complete development manifest, including embedded GT
metadata, before the six development outputs are frozen.  These are not
permitted to be repaired or retried within v13.  Separately, the unique dynamic
always-control had already reached the one-use dispatcher but aborted before
its preflight.  Its implementation did not retain stderr for the preflight
exception, so the exact early failure is not reproducible; this makes the
one-use control evidence incomplete as well as failed.  The only real v13
forward-adjacent evidence is the frozen non-CUDA interface probe and the
immutable waiter record; no GPU forward or later-stage output exists.
