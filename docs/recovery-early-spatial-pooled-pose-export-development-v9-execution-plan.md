# Recovery early-spatial-pooled pose export development v9: execution plan

- Date: 2026-08-07
- Status: **pre-registered; source-corrected before implementation or GPU execution**
- Long-horizon objective: within 14--20 hours, run this mechanism through every
  applicable hard gate with immutable evidence. The research programme advances only
  when one clean, candidate-ready mechanism satisfies causal safety, normal-path
  equivalence, per-condition runtime, and frozen quality gates. A v9 failure is a
  truthful terminal result and requires a mechanism-distinct v10 plan, never a v9 retry.
- Single v9 objective: determine whether a fixed early spatial-token arithmetic mean can
  replace an alarm-frame exported pose after rollback without raw pose reuse, fallback,
  new downloads, or more than 20% recurrent-policy runtime overhead.

### Source correction before Gate A

The initial v9 wording incorrectly treated `dec[0]` as a 768-wide sequence containing a
pose token. Frozen ReCal3R source shows that it is instead the 576-token, 1024-wide image
feature sequence before `decoder_embed`; the dedicated pose token is introduced only after
that frozen projection. This was found by source inspection before a v9 component, CPU
model check, or GPU forward existed. It is not a measured v9 result and authorizes no
output reuse. The only compatible v9 latent is therefore the fixed arithmetic mean of all
576 spatial tokens followed once by the frozen official projection, as specified below.

## 1. Frozen prior state and distinct hypothesis

v1--v8 are frozen. v4 external 3D registration and v5 pointmap consensus exceeded the
runtime budget. v6 pre-rollout pose retriever query failed the wrong-condition control
runtime gate. v8 early layer-0 pose token never reached a valid candidate: its Gate-B
terminal evidence was incomplete and an unauthorized 0001 forward occurred; see
docs/audits/recovery-early-decoder-pose-export-v8-provenance-no-go.md.

v9 tests a different latent source, not a changed index, precision, detector, or retry
of v8. After the complete current-frame recurrent rollout, `dec[0]` is the early
pre-projection spatial image sequence `(1,576,1024)`. The frozen official
`model.decoder_embed` maps the fixed mean of precisely those 576 spatial tokens from
`(1,1,1024)` to `(1,1,768)` before any special pose token is concatenated. The fixed mean
retains current-image geometry while never selecting a dedicated pose token. A frozen
official pose head might decode that pooled `(1,1,768)` latent into a usable alarm export.

This remains current-image- and model-latent-dependent. It is not latent-independent
recovery, an independently trained pose estimator, or a claim that it is disconnected
from the model shared computation.

## 2. Immutable v9 mechanism and causal boundary

1. Keep the frozen Detector-v3 decision, raw post-rollout health, complete current-frame
   structural rollback/witness, watchdog 8, checkpoint, ReCal3R commit
   466c7cdf3acd2f589f1d82e5f6391966f19db9ff, and the three existing development
   manifests. The detector always consumes raw prediction; export never feeds model,
   detector, memory, state, or a later frame.
2. Candidate only: after _recurrent_rollout returns dec, after a guard confirms exactly
   `dec[0] == (1,576,1024)`, and before update_mem with dec[-1][:,0:1], calculate exactly
   early_spatial_pooled_pose_token =
   model.decoder_embed(dec[0].mean(dim=1,keepdim=True)).detach().clone() when observer is
   present, otherwise None. It has one batch, one token, and width 768. The pool is an
   unweighted arithmetic mean over all 576 pre-projection spatial tokens, followed once
   by the already-pinned official 1024-to-768 decoder projection. No special pose token,
   new learned projection, sampling, mask, history, cache, or adaptive constant is
   permitted. Always-control makes no projection/token/mean/observer/export-wrapper call
   and retains direct raw to_cpu(res).
3. Clear frames commit and export their normal raw camera pose. At an alarm, first execute
   frozen full rollback/witness; only then decode the already captured pooled latent with
   frozen model.downstream_head.pose_head, official postprocess_pose, and official camera
   decoder. It replaces only exported camera_pose.
4. Decoder accepts only the pooled latent, frozen torch/head/postprocess/mode/camera
   decoder. It rejects malformed/nonfinite/nonfloating (1,1,768) tokens, device mismatch,
   malformed/nonfinite (1,7) head output, malformed/nonfinite camera, reflection,
   non-homogeneous row, or determinant/orthogonality error above 1e-8. The head stays in
   model dtype; only its produced seven values are promoted to float64 on the same GPU
   before official postprocess/camera. Every failure is
   EARLY_SPATIAL_POOLED_POSE_UNAVAILABLE_FAIL_CLOSED; no raw pose, hold, motion, anchor,
   pointmap, retry, or fallback is exported.
5. Alarm evidence records layer 0, source
   decoder_layer_0_preprojection_spatial_mean_after_frozen_decoder_embed_before_update_mem,
   the frozen decoder-projection SHA/interface, arithmetic mean, exact spatial-token count
   576, shape/dtype/device/GPU digest, source hashes,
   postprocess/proper-SO3 values, raw/exported pose digests, rollback witness and
   no-fallback status. Each alarm uses only its own current pooled token.

## 3. Data, source, and evaluation boundaries

- Read only the existing manifests:
  outputs/formal-v1-inputs-0001/development/development-dynamic/input-manifest.json
  outputs/formal-v1-inputs-0001/development/development-wrong/input-manifest.json
  outputs/formal-v1-inputs-0001/development/development-low/input-manifest.json
  Do not download data, checkpoint, dependency, or create a second dataset copy.
- Do not change ReCal3R, checkpoint, raw RGB, GT, corruption manifests, Detector-v3
  configuration, formal calibration, or v1--v8 output/log artifacts. Do not read GT or
  quality artifacts until every v9 control/candidate output is immutable.
- Add independent v9 primitive/export/runner/script and tests. Do not import or delegate
  to v1--v8 recovery runners/exporters, ORB/RANSAC, anchor/motion/pointmap solver, or the
  v6 pose retriever query. Low-level frozen detector, rollback witness and non-policy
  device utilities may be reused.
- Pin and hash dust3r/model.py, dpt_head.py, postprocess.py, v9 executable components,
  checkpoint, manifest and Detector-v3 configuration. Any drift fails closed.

## 4. Terminal-evidence protocol: new v9 launcher

The v8 log failure is a process failure, so v9 has one tested, single-owner launcher.
It must be implemented and tested before any v9 GPU child.

1. Validate a safe nonempty run ID and its exact direct `outputs/<run-id>` and canonical
   `logs/<run-id>-{preflight,main,postflight,tmux-transcript,driver,result}` layout;
   reject aliases, symlinks, nested/staging paths and every existing owner artifact. Then
   atomically acquire a same-ID owner lease with `mkdir` or `O_CREAT|O_EXCL` under the
   repository `tmp/` before any inspection or pane creation. The later independent
   validator report is deliberately *not* an initial owner artifact: it is a new
   `logs/<run-id>-validator-0001.log` created exactly once with `O_CREAT|O_EXCL` only after
   the owner has sealed its child output and terminal evidence.
2. In the owner process, record exactly two GPU-2 UUID
   GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250 preflight snapshots. Each must report at least
   12288 MiB free. Verify preflight is NUL-free, then freeze it 0444.
3. Create a fresh pane only in existing stateguard tmux session. Establish a live pipe-pane
   to the new transcript, emit and observe `V9_PIPE_READY`, and only then send the driver.
   Driver must write `V9_DRIVER_START`, an exact shlex-quoted command and command SHA that
   were already frozen in preflight, `V9_DRIVER_DISPATCHED`, child PID plus process start
   time, and zero/nonzero exit markers in one ordered stream. It writes result only after
   wait reaps that exact child.
4. Same owner waits for driver completion, detaches/closes pipe-pane, and verifies the
   transcript has drained and cannot change before it performs any terminal check. It then
   collects a timestamped postflight snapshot after child reaping, proves the PID/start-time
   pair is absent (PID-reuse safe), records result exit, GPU snapshot, source/worktree and
   output modes. It verifies all newly written preflight/main/postflight/transcript/driver/
   result bytes are regular, NUL-free and correctly ordered, then freezes each 0444. Every
   output directory is 0555, every output file 0444, and no output path is a symlink.
5. Separate one-use, CPU-only validator recomputes every source hash, command/hash/order,
   snapshot, result, path and mode instead of trusting launcher flags. It writes its own
   report only under logs, verifies it NUL-free, and freezes it 0444. Missing/ill-ordered
   markers, snapshot before child finish, stale/reused PID, bad permission/NUL status,
   mutable transcript, path reuse, or missing postflight is terminal v9 provenance
   failure. It cannot be repaired by manual reconstruction or a recovery ID.

Every GPU forward uses CUDA_VISIBLE_DEVICES=2, begins from stateguard, records
command/PID/GPU/log/output/start/finish, and checks no project process remains on GPU 2.
No other user process is modified.

## 5. Gate A: code, source, and causal contract

Before GPU execution, separately commit implementation, tests, and Gate-A audit. Require:

- synthetic identity/translation/rotation pooled-token decode tests; bad token/head/camera,
  reflection, non-SO3, device and finite failures fail closed;
- source and AST mutation tests that reject dec[-1], unprojected 1024-wide input,
  any special-pose-token selection, a non-mean reducer, pooling after update_mem, a
  control-path projection/clone/observer, raw camera pose/prediction,
  RGB, pointmap, confidence, anchor/history, GT/future and decoder/export cpu transfer;
- runner tests proving candidate pool relation, rollback before export, finalization before
  CPU transfer, raw detector health, watchdog/reset/no leakage and untouched control path;
- actual existing-checkpoint CPU test of pinned DPT pose head, official postprocess and
  float64 proper-SO3 contract;
- unit tests of v9 launcher with non-CUDA child: fresh-path refusal, two preflights,
  atomic same-ID lease and concurrent-owner refusal, pipe-ready-before-dispatch ordering,
  driver/result ordering, pipe-close-and-drain before freeze, child-PID/start-time-absent
  postflight, NUL/alias/symlink rejection, recursive freeze modes, one-use validator
  report, and no manual postflight API;
- ReCal3R-Torch targeted tests, full CPU suite with repository-local tmp basetemp,
  compileall, diff check, and clean StateGuard3R/ReCal3R worktrees.

Any Gate-A failure yields EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_IMPLEMENTATION_NO_GO; no v9
GPU forward is allowed.

## 6. Gate B: dynamic short circuit

1. Once only run recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001. Its four
   protected files must be byte-identical to frozen v1 dynamic baseline; status succeeds;
   timeline is final/direct; policy is always-commit with 30 transactions and zero pending;
   all provenance/permissions pass; runtime/baseline is at most 1.20.
2. Only after independent control validation passes, once only run
   recovery-early-spatial-pooled-pose-v9-dynamic-candidate-0001. Every real alarm must
   have full rollback witness, pooled-token export action, matching current-frame
   source/count evidence, no fallback and valid SO3; runtime/baseline is at most 1.20.
3. A control, candidate, availability, runtime, or terminal-evidence failure is
   EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_AVAILABILITY_OR_RUNTIME_NO_GO. Do not run
   wrong/low/GT and do not change v9 pool/source/head/precision/detector/logging then retry.

## 7. Gate C: frozen cross-condition matrix and blind quality selection

Only Gate-B PASS permits exactly: wrong control, wrong candidate, low control, low
candidate. Each uses a new ID, full terminal-evidence validation and same runtime at most
1.20 gate. All six v9 outputs freeze before CPU evaluator reads logical-base GT.

Evaluator uses frozen prefix frames 0--14 for Sim(3), tail 19--29 for ATE/RPE, and
(baseline-candidate)/baseline. Candidate-ready requires every control byte-equivalent,
all alarms complete, zero restore/query/watchdog/leakage failure, at least two of three
conditions with both ATE and translation-RPE improvement, both medians at least +5%,
candidate runtime median at most 1.20, and complete immutable provenance. PASS writes
only unexecuted blind acquisition plan. Otherwise write
EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_FEASIBILITY_NO_GO and stop v9 without retuning.
