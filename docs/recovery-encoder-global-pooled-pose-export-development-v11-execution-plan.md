# Recovery encoder-global pooled pose export v11: execution plan

- Date: 2026-08-07
- Status: **pre-registered, not implemented and not GPU-authorized**
- Objective: test whether a frozen, current-frame encoder-global latent can
  replace an alarm export without new data, weights, GT, future frames, raw
  pose reuse or fallback.  Only the hard gates below can advance the work.

## Mechanism and distinction

v9 failed because its post-rollout `dec[0]` shape was literally fixed to 576
tokens while the pinned 512×384 loader creates a 768-token grid.  The rejected
v10 design only changed that count and is documented separately as a
preimplementation no-go.

v11 never reads `dec[0]` and does not wait for recurrent rollout.  Candidate
policy only captures the frozen model's native current-image global feature
after feature selection, but before pose retrieval, recurrence and memory:

```python
global_img_feat_i = model._get_img_level_feat(feat_i)
encoder_global_pose_token = model.decoder_embed(global_img_feat_i).detach().clone()
```

`_get_img_level_feat` is the official current-frame spatial aggregation already
used by the base model.  It must yield finite `(1,1,1024)` for any valid image
grid; the pinned 1024→768 `decoder_embed` must yield finite `(1,1,768)`.  This
is an encoder-global, pre-retrieval mechanism, not an aspect-ratio adjustment
to the v9 decoder-global mechanism.  It selects no pose token, queries no
memory and passes no raw prediction/camera pose, RGB, pointmap, confidence,
history, anchor, GT or future value to the recovery decoder.

At an alarm, first apply the existing full structural rollback/witness, then
decode only the captured token with the pinned official pose head,
postprocess and camera decoder; replace only exported `camera_pose`.  Bad
shape/dtype/device/finite/head/camera/SO(3) values fail closed with no raw,
hold, motion, anchor or retry fallback.  Clear frames retain raw export.
Always-control may execute its ordinary base-model `_get_img_level_feat`, but
must not project, clone, observe or export any v11 token.

## Immutable boundaries and Gate A

Reuse only the current three development manifests, checkpoint, Detector-v3
config and pinned ReCal3R revision.  No download or ReCal3R modification is
allowed.  A distinct v11 primitive/export/runner/CLI/dispatcher must be pinned
and hashed; v9 files and artifacts remain untouched.

Before CUDA, require all of the following:

1. CPU primitive tests for finite `(1,1,1024)` encoder-global inputs, frozen
   projection to `(1,1,768)`, official pose-head/postprocess/camera and proper
   SO(3), using only the existing checkpoint.
2. AST/source mutation tests proving capture is after `feat_i` selection and
   before `pose_retriever.inquire`, `_recurrent_rollout` and `update_mem`; they
   must reject `dec[0]`, pose token/query, raw-pose/prediction input, CPU copy,
   GT/future/history/anchor/fallback and a control-path projection.
3. Runner tests for arbitrary valid encoder spatial lengths (including 768),
   rollback-before-export, raw-health detector input, watchdog/reset/no
   feedback and evidence containing actual image/grid/token metadata.
4. A separately pinned copy of the two-stage owner/PID/start-time/full-byte
   gate dispatcher, tested for normal freeze/validation and immediate
   no-result `PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE` terminal behavior.
5. CUDA-disabled v11 Torch tests, full CPU regression in repository Python
   3.11, compileall/diff check and clean StateGuard3R/ReCal3R worktrees.  A
   fresh Gate-A audit may authorize only the control below.

## Gate B/C stop rules

Only Gate-A PASS authorizes once, from `stateguard` tmux and GPU 2 after two
snapshots each ≥12288 MiB:
`recovery-encoder-global-pooled-pose-v11-dynamic-always-commit-0001`.
The control needs complete terminal evidence/CPU validator PASS, no project
GPU process at postflight, four protected files byte-identical to frozen v1
dynamic, 30 final direct commits, zero pending, and runtime/v1 ≤1.20.

Only that PASS authorizes one dynamic candidate.  Its alarm exports must have
current encoder-global evidence, full rollback/witness, valid SO(3), no
fallback and runtime/v1 ≤1.20.  Any failure stops v11: no candidate retry,
wrong/low, GT or quality evaluation.  Only a frozen dynamic candidate PASS
permits wrong/low control/candidate and the existing blind quality gate, which
still requires all controls equivalent, complete alarms, two-of-three ATE and
translation-RPE improvement, +5% medians and runtime median ≤1.20.
