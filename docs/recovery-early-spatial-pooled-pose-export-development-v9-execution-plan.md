# Recovery early-spatial-pooled pose export development v9: execution plan

- Date: 2026-08-07
- Status: **pre-registered; implementation and GPU execution have not started**
- Long-horizon objective: within 14--20 hours, run this mechanism through every
  applicable hard gate with immutable evidence. The research programme advances only
  when one clean, candidate-ready mechanism satisfies causal safety, normal-path
  equivalence, per-condition runtime, and frozen quality gates. A v9 failure is a
  truthful terminal result and requires a mechanism-distinct v10 plan, never a v9 retry.
- Single v9 objective: determine whether a fixed early spatial-token arithmetic mean can
  replace an alarm-frame exported pose after rollback without raw pose reuse, fallback,
  new downloads, or more than 20% recurrent-policy runtime overhead.

## 1. Frozen prior state and distinct hypothesis

v1--v8 are frozen. v4 external 3D registration and v5 pointmap consensus exceeded the
runtime budget. v6 pre-rollout pose retriever query failed the wrong-condition control
runtime gate. v8 early layer-0 pose token never reached a valid candidate: its Gate-B
terminal evidence was incomplete and an unauthorized 0001 forward occurred; see
docs/audits/recovery-early-decoder-pose-export-v8-provenance-no-go.md.

v9 tests a different latent source, not a changed index, precision, detector, or retry
of v8. After the complete current-frame recurrent rollout, dedicated layer-0 pose token
dec[0][:,0:1] may already carry the pose/state corruption path. The same layer non-pose
spatial tokens dec[0][:,1:], pooled by one fixed arithmetic mean, retain current-image
geometry while not reusing the dedicated pose token. A frozen official pose head might
decode that pooled (1,1,768) latent into a usable alarm export.

This remains current-image- and model-latent-dependent. It is not latent-independent
recovery, an independently trained pose estimator, or a claim that it is disconnected
from the model shared computation.

## 2. Immutable v9 mechanism and causal boundary

1. Keep the frozen Detector-v3 decision, raw post-rollout health, complete current-frame
   structural rollback/witness, watchdog 8, checkpoint, ReCal3R commit
   466c7cdf3acd2f589f1d82e5f6391966f19db9ff, and the three existing development
   manifests. The detector always consumes raw prediction; export never feeds model,
   detector, memory, state, or a later frame.
2. Candidate only: after _recurrent_rollout returns dec, after a guard confirms that
   dec[0] has more than one token, and before update_mem with dec[-1][:,0:1], calculate
   exactly early_spatial_pooled_pose_token =
   dec[0][:,1:].mean(dim=1,keepdim=True).detach().clone() when observer is present,
   otherwise None. It has one batch, one token, and width 768. The pool is an unweighted
   arithmetic mean over every non-pose token in layer 0; no sampling, learned projection,
   mask, history, cache, or adaptive constant is permitted. Always-control makes no
   token, mean, observer, or export-wrapper call and retains direct raw to_cpu(res).
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
   decoder_layer_0_nonpose_spatial_mean_after_rollout_before_update_mem, arithmetic mean,
   spatial-token count, shape/dtype/device/GPU digest, source hashes,
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

1. Given a one-use run ID, atomically refuse if its direct output, preflight, main,
   postflight, transcript, driver, result, or validator path already exists. Reject nested
   or staging output paths.
2. In the owner process, record exactly two GPU-2 UUID
   GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250 preflight snapshots. Each must report at least
   12288 MiB free. Verify preflight is NUL-free, then freeze it 0444.
3. Create a fresh pane only in existing stateguard tmux session. Establish live pipe-pane
   to new transcript before dispatch. Pane driver writes V9_DRIVER_START, exact command,
   child PID and zero/nonzero exit marker; it emits V9_DRIVER_DISPATCHED immediately
   before CUDA child and writes result only after wait observes child exit.
4. Same owner, after result and before freezing terminal artifacts, collects timestamped
   postflight GPU snapshot, records expected child PID absent, result exit, output file
   modes and clean worktrees. It verifies all newly written
   preflight/main/postflight/transcript/driver/result bytes NUL-free, then freezes each
   0444; output root is 0555 and all files 0444.
5. Separate one-use validator rechecks all evidence, not launcher flags. Missing/ill-ordered
   markers, snapshot before child finish, stale PID, bad permission/NUL status, path reuse,
   or missing postflight is terminal v9 provenance failure. It cannot be repaired by manual
   reconstruction or a recovery ID.

Every GPU forward uses CUDA_VISIBLE_DEVICES=2, begins from stateguard, records
command/PID/GPU/log/output/start/finish, and checks no project process remains on GPU 2.
No other user process is modified.

## 5. Gate A: code, source, and causal contract

Before GPU execution, separately commit implementation, tests, and Gate-A audit. Require:

- synthetic identity/translation/rotation pooled-token decode tests; bad token/head/camera,
  reflection, non-SO3, device and finite failures fail closed;
- source and AST mutation tests that reject dec[-1], token zero, a non-mean reducer,
  pooling after update_mem, a control-path clone/observer, raw camera pose/prediction,
  RGB, pointmap, confidence, anchor/history, GT/future and decoder/export cpu transfer;
- runner tests proving candidate pool relation, rollback before export, finalization before
  CPU transfer, raw detector health, watchdog/reset/no leakage and untouched control path;
- actual existing-checkpoint CPU test of pinned DPT pose head, official postprocess and
  float64 proper-SO3 contract;
- unit tests of v9 launcher with non-CUDA child: fresh-path refusal, two preflights,
  pipe-before-dispatch ordering, child result, child-PID-absent postflight, NUL rejection,
  freeze ordering, and no manual postflight API;
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
