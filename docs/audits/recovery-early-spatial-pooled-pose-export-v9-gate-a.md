# v9 early-spatial-pooled pose export: Gate A audit

- Date: 2026-08-07
- Scope: code/source/causality/CPU gate only. **No v9 CUDA forward was run.**
- Decision: **PASS — authorize only the one-use v9 dynamic always-commit
  control named in the v9 execution plan.** This is not a quality result, does
  not authorize a candidate yet, and does not authorize any v8 activity.

## Mechanism bound before execution

The initial v9 description was corrected and committed before implementation:
the frozen ReCal3R dec[0] is (1,576,1024), not a 768-wide sequence with a pose
token. v9 therefore captures only the following candidate-only latent, after
_recurrent_rollout and before update_mem:

    model.decoder_embed(dec[0].mean(dim=1, keepdim=True)).detach().clone()

It is a fixed arithmetic mean over all 576 layer-0 pre-projection spatial
tokens, followed once by frozen decoder_embed (1024 to 768). It never selects
a special pose token. The decoder accepts only the resulting (1,1,768) token
and sends it through the pinned official pose head, postprocess, and camera
decoder. The head stays in native dtype; only its seven outputs are promoted
to float64 on-device for the required proper-SO(3) check.

The candidate path rolls back its complete current structural state before it
calls the export policy. The policy receives no numerical raw camera_pose,
prediction pointmap, RGB, confidence, anchor/history, future, GT, fallback,
or CPU copy. Clear frames retain direct raw export. The always-commit path
does not construct a pooled token, call decoder_embed, create an observer, or
call an export wrapper.

## Pinned state

| Item | Value |
| --- | --- |
| StateGuard3R implementation commit | 2eda68d0babeb27f9d5727e8e70ecf1935eca7d4 |
| Gate-A tests/dispatcher commit | 872c748539102ec2a572f15d7070d874253f2f32 |
| ReCal3R commit | 466c7cdf3acd2f589f1d82e5f6391966f19db9ff |
| Checkpoint SHA-256 | 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 |
| ReCal3R model SHA-256 | 32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1 |
| DPT head SHA-256 | c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e |
| Pose postprocess SHA-256 | ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7 |

The two worktrees were clean at the Gate-A check, and no direct *v9* output,
log, or tmp terminal artifact existed. Thus no v9 output is reused or
interpreted as a failed/partial forward.

## Test evidence

The Torch-enabled targeted invocation was:

    CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/tmp" \
    PYTHONPATH="src:/data/wangzheng/Project2/baselines/ReCal3R/.venv/lib/python3.11/site-packages" \
    .venv/bin/python -m pytest -q [five v9 test files]

Result: **22 passed in 22.80 s** (a repeat after dispatcher hardening was
22 passed in 23.51 s). It covers:

- mean/projection equivalence; rejection of unprojected 1024-wide input;
  nonfinite, malformed, device, reflection, non-orthogonal, and homogeneous
  camera failures;
- AST/source checks for a token-only decoder and an alarm exporter that never
  numerically loads the raw prediction pose;
- runner mutation checks rejecting special-token selection and moving the
  pooling operation after memory update; a synthetic execution check proves
  capture before memory mutation and zero candidate-policy projection calls in
  the control path;
- a CPU load of the existing official checkpoint with zero missing/unexpected
  keys, official DPT head/postprocess, float64 output contract, and
  proper-SO(3) camera result;
- v9 CLI/provenance/direct-output constraints; and
- a non-CUDA-child dispatcher simulation that proves fresh-ID refusal, two
  preflight snapshots, pipe-pane before driver dispatch, result-before-
  postflight, PID-absent postflight, NUL rejection, and freeze ordering.

The complete StateGuard3R CPU regression was also run with the repository
virtualenv:

    521 passed, 38 skipped in 48.41s

The skips are the existing no-Torch tests in that virtualenv; every new
Torch-sensitive v9 check ran in the separately stated Torch-enabled
invocation.

## New terminal-evidence owner

scripts/dispatch_recal3r_early_spatial_pooled_pose_v9.py is the only permitted
v9 GPU launcher. Before creating a child it requires fresh direct paths, clean
StateGuard3R/ReCal3R worktrees, committed component blobs, and two GPU-2
UUID/free-memory snapshots. It attaches pipe-pane before the driver, records
dispatch/PID/exit in original main and live transcript files, writes
postflight only after its result and absent-PID check, verifies NUL-free
artifacts, and freezes every terminal artifact plus the output tree. It has no
recovery/post-hoc postflight API.

## Exact next action and stop rules

The only authorized forward is
recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001.

It must use the single-owner dispatcher, GPU 2 UUID
GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250, the dynamic development manifest,
and the fixed always-commit policy. A separate validator must require the four
protected files to be byte-identical to frozen v1 dynamic baseline, 30 direct
final transactions with zero pending, runtime ratio at most 1.20, and complete
original ordered terminal evidence. Any failure is a v9 terminal
availability/runtime/provenance NO-GO: do not run candidate/wrong/low/GT and
do not retry v9 under a new ID.
