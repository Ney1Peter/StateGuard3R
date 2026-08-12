# Recovery patch-embedding global pooled pose export v12: execution plan

- Date: 2026-08-07
- Status: **Gate A PASS at commit `a472c6a`; no v12 CUDA run, development
  output, candidate, quality evaluation or new data acquisition exists.**  The
  only currently authorized forward is the dynamic always-control, pending its
  strict GPU-2 availability gate.  See
  [`audits/recovery-patch-embed-global-pooled-pose-export-v12-gate-a.md`](audits/recovery-patch-embed-global-pooled-pose-export-v12-gate-a.md).
- Long-horizon objective (12–18 hours of active work): determine whether a
  current-frame, pre-RoPE patch-embedding global latent can safely replace an
  alarm-frame exported camera pose, while preserving baseline state semantics,
  runtime, provenance and frozen development quality gates.  A failed gate is
  a terminal v12 result, never a license to tune or retry this version.

## 1. Why v12 is a distinct next mechanism

v11 is terminal at its mandatory real-frame CPU Gate A: frozen ReCal3R
attention forces query/key tensors to `float16` before native RoPE, while the
pinned CPU CuRoPE implementation accepts `float` only.  Its full
`_encode_image` path therefore cannot execute on CPU without replacing the
frozen upstream path.  The evidence is frozen in
[`audits/recovery-encoder-global-pooled-pose-export-v11-cpu-probe-no-go.md`](audits/recovery-encoder-global-pooled-pose-export-v11-cpu-probe-no-go.md).

v12 does not capture v11's full encoder feature, does not call
`_get_img_level_feat`, and does not need an encoder block or RoPE for its CPU
availability proof.  Its sole candidate token is:

```python
patch_tokens_i, _patch_pos_i = model.patch_embed(selected_imgs, true_shape=selected_shapes)
patch_global_i = torch.mean(patch_tokens_i, dim=1, keepdim=True)
patch_global_pose_token = model.decoder_embed(patch_global_i).detach().clone()
```

This is a current-image, **pre-attention/pre-RoPE patch embedding** mean.  It
is captured before the ordinary full `_encode_image`, pose retrieval,
recurrent rollout and memory update.  In contrast, v11 is a post-24-block
full-encoder mean (`_get_img_level_feat(feat_i)`).  v12 neither changes a
v9/v10 decoder grid constant nor reads `dec[0]`, a pose token, a retriever
query, model memory, raw prediction/camera pose, anchor, GT or a future frame.

The patch embedder necessarily consumes the current loaded image to construct
the frozen native patch feature.  That RGB value is quarantined at feature
construction: the decoder/export primitive receives only the detached
`(1,1,768)` projected token and has no RGB or prediction argument.

## 2. Immutable boundaries

1. Reuse only the existing three development manifests, existing checkpoint,
   pinned ReCal3R revision and frozen Detector-v3 config.  Do not download
   data, weights or dependencies, modify ReCal3R/checkpoint/raw RGB/GT/
   corruption manifests/Detector-v3 config, or read GT or blind data before
   all six development outputs are frozen.
2. New v12 modules, runner, CLI and dispatcher must be independent and must
   not import/call v3 anchor, v4 registration, v5 pointmap, v6 query, v8/v9
   decoder, v10, or v11 recovery primitive/export/runner/dispatcher.
   Reusing frozen low-level structural witness, quarantine state and detector
   interfaces is allowed.
3. Clear frames execute the frozen base path and export the original
   `camera_pose`.  Candidate code performs its additional patch-embed/mean/
   projection only when an observer exists; no v12 projection, clone,
   observer or export wrapper may run in always-control.
4. On an alarm, apply the existing complete pre-state rollback and structural
   witness first.  Decode only the captured token with the frozen official
   pose head, official postprocess and official camera decoder, and overwrite
   only exported `camera_pose`.  There is no raw/hold/motion/anchor fallback,
   retry, cache, detector feedback or model-state feedback.
5. The actual patch-token count is per-frame evidence, not an eligibility
   constant.  Valid count is any positive `N`; v12 has no `576`, `768` or
   aspect-ratio gate.  Evidence records shapes/dtypes/devices and GPU-only
   digests, never token values.

## 3. Fixed v12 contracts

- Native patch feature: finite floating `(1,N,1024)`, `N > 0`, current-frame
  `model.patch_embed` output before the encoder blocks.
- Global aggregate: finite mean `(1,1,1024)`, computed once across `N` by
  `torch.mean(..., dim=1, keepdim=True)`.
- Projection: exactly frozen `torch.nn.Linear(1024,768)` at matching device
  and dtype; detached cloned token shape `(1,1,768)`.
- Decode: input only that token; official frozen pose head receives
  `token[:,0]`; exactly seven resulting pose values may be promoted to
  float64 for official postprocess/camera decoding.  The `(1,7)` pose and
  `(1,4,4)` camera must be finite, homogeneous and proper SO(3), with
  determinant, orthonormality and homogeneous-row errors each at most `1e-8`.
  Any violation is `PATCH_EMBED_GLOBAL_POSE_UNAVAILABLE_FAIL_CLOSED`.
- Detector: sees the ordinary raw post-rollout health/prediction only.  The
  patch token may not be supplied to detector state, watchdog, model state,
  memory or later frames.

## 4. Gate A: source, CPU and causal contract

Before any CUDA work, all of the following must pass:

1. Torch tests for patch feature/mean/projection correctness; arbitrary
   positive `N` including five and 768; shape/dtype/device/nonfinite/
   projection failures; token-only decoding; proper-SO(3); no raw-pose
   fallback; and evidence without serialized token values.
2. A CUDA-disabled real-frame probe reads only dynamic manifest frame 0 using
   `load_images_for_eval`, calls only
   `model.patch_embed -> mean -> decoder_embed -> official pose_head ->
   postprocess_pose -> pose_encoding_to_camera`, records loaded geometry and
   actual `N`, and hard-rejects `_encode_image`, all encoder blocks/RoPE,
   retriever/memory/rollout, detector, GT and future frame paths.  It must
   use the unmodified pinned upstream patch embedder, strictly load the
   existing checkpoint, and never initialize CUDA.
3. AST/source mutation tests bind capture after current feature selection but
   before the ordinary full encoder/retriever/rollout/update.  They require
   exactly one candidate `model.patch_embed`, exactly one mean and projection,
   and reject `dec[...]`, `model.pose_token`, retriever query, raw
   prediction/camera, CPU copies, GT/future/history/anchor/fallback values at
   the capture/decode/export boundary.  They also prove control separation,
   rollback-before-export, raw-health detector input, watchdog/reset handling,
   no state feedback and no hard-coded patch count.
4. A v12-only dispatcher must be tested with non-CUDA children for atomic
   lease isolation, two UUID/memory preflights, pipe-ready before wrapper
   release, PID/start-ticks plus full-byte random gate, freeze/validator, and
   a no-result `PREEXEC_IDENTITY_NO_GO_EVIDENCE_COMPLETE` terminal.
5. Run the targeted Torch suite with CUDA disabled, project Python-3.11 CPU
   suite, compileall, diff check and clean StateGuard3R/ReCal3R provenance.
   A committed Gate-A audit may authorize **only** the dynamic always-control.

Any Gate-A failure is `PATCH_EMBED_GLOBAL_POOLED_POSE_EXPORT_V12_IMPLEMENTATION_OR_CPU_AVAILABILITY_NO_GO`.  It forbids all v12 CUDA,
candidate/wrong/low/GT work and requires a new distinct mechanism.

## 5. Gate B: exactly one dynamic short circuit

Only a frozen Gate-A PASS authorizes one tmux/GPU-2 run after two
UUID-identical snapshots each with at least 12288 MiB free:

```text
recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001
```

It must have complete terminal and CPU-validator evidence, no project GPU
process at postflight, 30 direct commits/pending zero, runtime/v1 `<= 1.20`,
and byte-identical frozen-v1 dynamic
`checkpoint-load-audit.json`, `health.jsonl`, `predictions-summary.json` and
`trajectory.json`.  Failure terminates v12.

Only that PASS authorizes the one candidate:

```text
recovery-patch-embed-global-pooled-pose-v12-dynamic-candidate-0001
```

Every alarm must show current patch evidence, rollback/witness before export,
proper SO(3), no fallback and candidate runtime/v1 `<=1.20`.  Control or
candidate failure terminates v12 and forbids retry or changes to patch source,
mean/projection/head/detector/launcher.

## 6. Gate C and decision

Only frozen dynamic candidate PASS permits one always-control and one
candidate for each of the existing wrong and low development conditions.  Only
after the six immutable outputs exist may the existing evaluator read frozen
quality data.  Candidate readiness requires all controls byte-identical, all
alarm provenance complete, zero causal/provenance failures, at least two of
three conditions improving both ATE and translation RPE, both medians at least
`+5%`, and candidate runtime median `<=1.20`.

PASS writes only an unexecuted acquisition plan.  Any failure writes a v12
feasibility NO-GO and does not reopen this disclosed development matrix.
