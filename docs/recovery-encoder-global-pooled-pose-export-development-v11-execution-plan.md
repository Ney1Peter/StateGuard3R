# Recovery encoder-global pooled pose export v11: execution plan

- Date: 2026-08-07
- Status: **TERMINAL GATE-A CPU AVAILABILITY NO-GO; DO NOT EXECUTE.**  The
  real-frame evidence is frozen in
  `docs/audits/recovery-encoder-global-pooled-pose-export-v11-cpu-probe-no-go.md`.
- Long-horizon objective: determine, with immutable evidence and without new
  data, weights, downloads, GT, future frames, raw-pose reuse, or fallback,
  whether a frozen current-frame encoder-global latent can provide a safe
  alarm export. Continue only through the stated gates; a failure is a terminal
  result, not a reason to retry the same v11 ID.

## Mechanism and distinction

v9 failed because its post-rollout `dec[0]` guard literally required 576
tokens. The production loader (`load_images_for_eval`) resizes the long side to
512 then center-crops to 16-pixel geometry; the prior v9 terminal evidence
recorded 512×384 and its 16-pixel patch grid therefore had 768 tokens. That is
source-plus-v9 evidence, not yet a v11 runtime measurement. The rejected v10
design only changed that count and is documented separately as a pre-Gate-A
mechanism-boundary no-go.

v11 never reads `dec[0]` and does not wait for recurrent rollout.  Candidate
policy only captures the frozen model's native current-image global feature
after feature selection, but before pose retrieval, recurrence and memory:

```python
global_img_feat_i = model._get_img_level_feat(feat_i)
encoder_global_pose_token = model.decoder_embed(global_img_feat_i).detach().clone()
```

`_get_img_level_feat` is the official current-frame spatial aggregation already
used by the base model. It must yield finite floating `(1,1,1024)` for any
valid image grid; the pinned 1024→768 `decoder_embed` must yield finite
floating `(1,1,768)`. The actual input grid length is evidence, never a new
eligibility constant. This is an encoder-global, pre-retrieval mechanism, not
an aspect-ratio adjustment to the v9 decoder-global mechanism. It selects no
pose token, queries no memory and passes no raw prediction/camera pose, RGB,
pointmap, confidence, history, anchor, GT or future value to the recovery
decoder.

At an alarm, first apply the existing full structural rollback/witness, then
decode only the captured token with the pinned official pose head,
postprocess and camera decoder; replace only exported `camera_pose`.  Bad
shape/dtype/device/finite/head/camera/SO(3) values fail closed with no raw,
hold, motion, anchor or retry fallback.  Clear frames retain raw export.
Always-control may execute its ordinary base-model `_get_img_level_feat`, but
must not project, clone, observe or export any v11 token.

## Immutable boundaries and provenance

Reuse only the current three development manifests, checkpoint, Detector-v3
config and pinned ReCal3R revision.  No download or ReCal3R modification is
allowed.  A distinct v11 primitive/export/runner/CLI/dispatcher must be pinned
and hashed together with `model.py`, `patch_embed.py`, `image.py`, DPT head,
postprocess, checkpoint, manifests and detector config; v9 files and artifacts
remain untouched. Per-frame evidence records the true loaded image geometry,
encoder spatial count, token shapes/dtypes/devices and GPU-only digests. It
does not serialize token values or read GT.

The always-commit control must make no v11 `decoder_embed` projection, clone,
observer or export-wrapper call. Its ordinary frozen `_get_img_level_feat`
call remains because that is baseline recurrence behavior. Candidate capture
must not feed detector, model state, memory, a later frame, or the ordinary
raw prediction path.

## Gate A: implementation and CPU contract

Before CUDA, require all of the following. The required real-frame probe
failed on the pinned CPU encoder/RoPE path, so Gate A did not pass and v11
cannot advance to any CUDA work:

1. Unit-test the encoder-global primitive: only a finite floating
   `(1,1,1024)` current-model output projects to `(1,1,768)`; wrong
   shape/dtype/device/nonfinite values fail closed. Test the official
   pose-head/postprocess/camera and proper-SO(3) contract on the existing
   checkpoint. In addition, a CUDA-disabled real-frame probe must load only
   dynamic-manifest frame 0 through `load_images_for_eval`, call only
   `_encode_image` → `_get_img_level_feat` → `decoder_embed` → official
   pose-head/postprocess/camera, record the loaded geometry and encoder count,
   and prove that it called no rollout, pose-retriever/memory, detector, GT or
   future-frame path. The observed count is evidence, never a v11 guard.
2. Add AST/source mutation tests proving candidate capture is after feature
   selection and before pose retrieval/rollout/memory, uses
   `_get_img_level_feat` then `decoder_embed` exactly once, and forbids
   `dec[0]`, special-pose-token selection, pose-retriever query, raw
   camera-pose/prediction inputs, CPU copies, GT/future/history/anchor/fallback
   data at the decoder/export boundary.
3. Test runner/control separation, rollback-before-export, detector raw-health
   input, watchdog/reset handling, no feedback into model state, and audit
   evidence for arbitrary valid spatial lengths (including 768), without a
   hard-coded image-token count.
4. Test a distinct v11 dispatcher with non-CUDA children for atomic lease
   isolation, two UUID/memory preflights, pipe-ready before wrapper launch,
   PID/start-time plus full-byte random gate, normal freeze/validator, and a
   no-result `PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE` terminal.
5. Run the v11 Torch-sensitive suite with CUDA disabled, the complete CPU
   suite in the repository Python 3.11 environment, compileall, diff check,
   and clean StateGuard3R/ReCal3R provenance. A separate Gate-A audit may
   authorize only the control below.

## Gate B: one dynamic short circuit

Only a committed Gate-A PASS authorizes exactly one tmux/GPU-2 run, after two
UUID-identical snapshots each with at least 12288 MiB free:
`recovery-encoder-global-pooled-pose-v11-dynamic-always-commit-0001`.
The control needs complete terminal evidence and CPU-validator PASS, no project
GPU process at postflight, four protected files byte-identical to frozen v1
dynamic (`checkpoint-load-audit.json`, `health.jsonl`,
`predictions-summary.json`, `trajectory.json`), 30 final direct commits, zero
pending transactions and runtime/v1 ≤1.20.

Only that PASS authorizes one dynamic candidate.  Its alarm exports must have
current encoder-global evidence, full rollback/witness, valid SO(3), no
fallback and runtime/v1 ≤1.20.  Any failure stops v11: no candidate retry,
wrong/low, GT or quality evaluation, and no edit of source/projection/head,
detector or launcher can create another v11 run.

## Gate C and final decision

Only a frozen dynamic candidate PASS permits one always-control and one
candidate for each wrong and low development condition. After all six outputs
are immutable, the existing blind evaluator may read frozen quality data.
Candidate readiness requires all controls byte-identical, all alarms complete,
zero causal/provenance failures, at least two of three conditions improving
both ATE and translation RPE, both medians at least +5%, and candidate runtime
median ≤1.20. PASS writes only the unexecuted acquisition plan; otherwise a
v11 feasibility NO-GO is final.
