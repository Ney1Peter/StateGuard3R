# v12 patch-embedding global pooled pose export: Gate-A audit

- Date: 2026-08-12
- Decision: **PASS — authorizes only the pre-registered v12 dynamic
  always-commit control.**
- Scope: v12 source, causal contract, CUDA-disabled unit tests, real single
  dynamic-frame CPU availability probe, provenance and CPU regression only.
  **No v12 CUDA forward, dispatcher invocation, development output, candidate,
  wrong/low condition, GT read or quality evaluation was run.**

## Mechanism and execution boundary

v12's candidate-only token is the native current-image patch-embedding mean
before the encoder blocks and RoPE:

```python
patch_tokens_i, _patch_pos_i = model.patch_embed(
    selected_imgs, true_shape=selected_shapes
)
patch_global_i = torch.mean(patch_tokens_i, dim=1, keepdim=True)
patch_global_pose_token = model.decoder_embed(patch_global_i).detach().clone()
```

The actual patch count is runtime evidence; it is not an eligibility constant.
The implementation accepts every positive `N` with native feature shape
`(1,N,1024)`, then projects exactly once to `(1,1,768)`.  Its decoder accepts
only that token, passes `token[:,0]` to the frozen official pose head, promotes
only the seven head outputs to on-device float64 for official postprocess and
camera decoding, and rejects a non-finite, non-homogeneous, non-orthonormal,
or improper-SO(3) result.

Unlike terminal v11, this route neither calls `_encode_image` nor enters an
encoder attention block/RoPE during the availability proof.  It is also
mechanistically independent of the prior v1--v11 recovery primitives: it does
not read `dec`, a pose token, query, memory, raw camera-pose numeric value,
anchor, history, point-map, GT or future frame.

The candidate runner captures this token only when an observer exists, after
current image selection and before the ordinary full encoder/retrieval/rollout/
memory update.  The always-control executes the legacy raw path: no additional
v12 `patch_embed`, mean/projection, clone, observer, or export wrapper.  On an
alarm, the complete pre-state structural rollback and witness occur before
`observer.finalize`; the export wrapper filters `camera_pose` before copying
the remaining mapping entries, so it does not numerically read the raw pose.
Clear frames return the original raw mapping.

## Pinned implementation and provenance

| Item | Value |
| --- | --- |
| StateGuard3R implementation/probe source commit | `d0ffade640c7a106357c85aea11c8aa60c4d5a5d` |
| Post-probe fixed run-ID hardening commit | `5f67a6fa9564255376f8771cc5a3c350c678a42d` |
| Post-audit evidence-target hardening commit | `b70300e` (this amendment's parent) |
| tmux waiter / dispatcher-provenance commits | `53a6d9a`, `8e09a42` |
| waiter static-contract test commit | `a061491` |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| `dust3r/model.py` SHA-256 | `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1` |
| `dust3r/patch_embed.py` SHA-256 | `c0dae36e876d2124f0e31403a3a306db7e6d5bd69d8e9c6f16bbfb81c82eef3d` |
| DPT head SHA-256 | `c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e` |
| pose postprocess SHA-256 | `ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7` |

The source auditor binds the checkpoint loader's explicit rewrite from
`ManyAR_PatchEmbed` to the real `PatchEmbedDust3R`, its native BCHW-to-BNC
projection/norm path, the long-side-512 cropped `load_images_for_eval` loader,
and runner ordering.  Mutations replacing the native patch call, introducing
decoder/pose/retriever data at capture, moving capture after the encoder, or
contaminating the control branch are rejected.

Both StateGuard3R and ReCal3R tracked worktrees were clean after the committed
test/probe checks.  No data, checkpoint or dependency was downloaded and
ReCal3R was not modified.

## Test and CPU availability evidence

The Torch-enabled targeted command was:

```text
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/tmp" \
PYTHONPATH="$PWD/src:/data/wangzheng/Project2/baselines/ReCal3R/.venv/lib/python3.11/site-packages" \
.venv/bin/python -m pytest -q \
  tests/test_patch_embed_global_pooled_pose_v12.py \
  tests/test_patch_embed_global_pooled_pose_export_v12.py \
  tests/test_recal3r_patch_embed_global_pooled_pose_runner_v12.py \
  tests/test_run_recal3r_patch_embed_global_pooled_pose_export_v12.py \
  tests/test_probe_recal3r_patch_embed_global_pose_v12.py \
  tests/test_dispatch_recal3r_patch_embed_global_pooled_pose_v12.py
```

Result: **42 passed**.  It covers arbitrary `N=5,768`, mean/projection
equivalence and malformed/dtype/device/nonfinite rejection; token-only
decode, output-only float64 promotion and proper SO(3); hostile mapping proof
that raw `camera_pose` cannot be read by alarm export; non-sensitive evidence;
source-audit mutations; candidate/control separation and rollback-before-
finalize; fixed CLI/manifests; and a v12-only non-CUDA dispatcher simulation
covering atomic lease, UUID/memory snapshots, pipe-ready ordering, full-byte
release token, PID/start-time ownership, freeze, validator and pre-execution
identity NO-GO terminal evidence.

`compileall` and `git diff --check` passed. The final full project CPU suite
ran with the project virtualenv and CUDA disabled: **593 passed, 62 skipped in
58.45 s**. Its skips are the existing no-Torch cases; the v12 Torch tests are
covered by the targeted invocation above.

The Gate-A real-frame availability proof ran from committed implementation
source `d0ffade` with `CUDA_VISIBLE_DEVICES=''`, existing dynamic-development
manifest frame 0 and the existing checkpoint. Its immutable, mode-0444
evidence is
[`recovery-patch-embed-global-pooled-pose-v12-dynamic-frame0-cpu-probe-0002.json`](../../logs/recovery-patch-embed-global-pooled-pose-v12-dynamic-frame0-cpu-probe-0002.json).
It records:

| Contract | Observed result |
| --- | --- |
| CUDA | visible devices `""`; initialized `false` |
| loader | dynamic frame 0 only; `img=(1,3,384,512)`, `true_shape=(1,2)` |
| checkpoint | strict compatibility, zero missing/unexpected keys |
| pinned native embedder | `PatchEmbedDust3R` |
| patch feature | CPU float32 `(1,768,1024)` |
| aggregate / projection | `(1,1,1024)` / `(1,1,768)` |
| official output | finite pose `(1,7)`, camera `(1,4,4)` |
| SO(3) | determinant `1.0`; orthonormality error `1.11e-16`; homogeneous error `0.0` |
| serialization | no latent, pose or camera values |

The probe hard-blocks `_encode_image`, encoder blocks/RoPE, retriever
inquire/update/memory, recurrence, detector, GT and future-frame paths.
Therefore it proves the required native pre-RoPE availability path rather than
an in-memory substitute or a shortened full inference.

### Evidence-target hardening amendment

The initial `0001` probe was a preliminary run made before the implementation
files were committed; it is retained under the two-phase evidence-preservation
rule but is **not** used to authorize Gate B.  The CPU-probe implementation at
`d0ffade` fixed the payload `run_id`, but accepted an arbitrary new direct
`--evidence-json` path.  The committed-source availability proof was therefore
written at the noncanonical `0002` filename while its payload truthfully
retained the fixed logical run ID `0001`.  The v12 Gate-A plan has no
exactly-once CPU-forward rule; this is the single committed-source proof used
for authorization, not a candidate or a development-matrix experiment. No
GPU, candidate, GT, extra frame, download or modified upstream source was
involved, and it did not read either prior probe output.

Commit `b70300e` closes that artifact-path ambiguity before any Gate-B work:
the probe now rejects any evidence path other than the canonical
`logs/<RUN_ID>.json`, before importing torch or touching data. The committed
source `0002` artifact remains the Gate-A runtime evidence; the preliminary
`0001` file and noncanonical filename are retained only under the repository's
two-phase evidence-preservation rule and must not be interpreted as separate
authorized experiments.

## Authorized next action and irreversible stop rules

Only this single forward is now authorized:

```text
recovery-patch-embed-global-pooled-pose-v12-dynamic-always-commit-0001
```

It must be dispatched only by
`scripts/dispatch_recal3r_patch_embed_global_pooled_pose_v12.py` from the
existing `stateguard` tmux session, on GPU 2 UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`, after **two** matching snapshots
with at least 12288 MiB free and no project process.  The independent CPU
validator must prove 30 direct commits/pending zero, runtime/v1 `<=1.20`, and
byte-identical v1 dynamic `checkpoint-load-audit.json`, `health.jsonl`,
`predictions-summary.json` and `trajectory.json`.

`scripts/wait_and_dispatch_recal3r_patch_embed_global_pooled_pose_v12_control.sh`
is the only permitted non-model waiting wrapper: it merely polls, waits 30 s,
then requires the same free-memory threshold again before calling the
dispatcher.  Its blob is included in the dispatcher's pinned component
provenance, so any later modification makes Gate B fail closed.  It does not
reserve a GPU or override the dispatcher's own two snapshots.

After adding that waiter, its isolated shell/route test passed (`1 passed`):
it checks shell syntax, fixed run/GPU IDs, both free-memory reads, clean
worktree checks, canonical always-control argv and sole dispatcher entrypoint.
An additional CPU-only preflight against the final HEAD recomputed the six
pinned component hashes (including the waiter) and accepted the exact
production control argv (`v12_dispatcher_production_contract=PASS`).

Only a frozen PASS permits exactly one v12 dynamic candidate.  Any control or
candidate failure is terminal
`PATCH_EMBED_GLOBAL_POOLED_POSE_EXPORT_V12_IMPLEMENTATION_OR_AVAILABILITY_OR_RUNTIME_NO_GO`:
do not retry v12, alter its patch source/mean/projection/head/detector/launcher,
or run wrong/low/GT/quality work.  Only a candidate PASS may authorize the
pre-registered Gate-C matrix.
