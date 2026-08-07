# v11 encoder-global source forensics

- Date: 2026-08-07
- Status: source-only evidence; **not a v11 GPU result or authorization**.

## Purpose

This audit separates observed source facts from inferences before any v11
implementation. It rejects the invalid v10 count-only successor and records
why v11's source and causal time are distinct from v9.

## Pinned source facts

The pinned ReCal3R source hashes are:

| File | SHA-256 |
| --- | --- |
| `src/dust3r/model.py` | `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1` |
| `src/dust3r/heads/dpt_head.py` | `c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e` |
| `src/dust3r/heads/postprocess.py` | `ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7` |
| `src/dust3r/patch_embed.py` | `c0dae36e876d2124f0e31403a3a306db7e6d5bd69d8e9c6f16bbfb81c82eef3d` |
| `src/dust3r/utils/image.py` | `a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96` |

The production runner calls `load_images_for_eval`, which resizes the long
side to 512 and center-crops the result to its 16-pixel input geometry.
`ManyAR_PatchEmbed.forward` enforces 16-pixel patch divisibility and flattens
its spatial patch grid. The prior v9 terminal evidence recorded 512×384, for
which the arithmetic grid is 32×24=768 image tokens. A v11 real-frame CPU
probe must log the actual geometry and count rather than turn that historical
observation into a new fixed guard.

In `model.py`, `_get_img_level_feat(feat)` is the official
`torch.mean(feat, dim=1, keepdim=True)`. The normal recurrent path calculates
it immediately after selecting `feat_i`; it then uses it as the key to
`pose_retriever.inquire`, performs `_recurrent_rollout`, and later
`pose_retriever.update_mem`. `decoder_embed` is the frozen `Linear(1024,768)`.

The official DPT head takes the final decoder's first token, runs
`pose_head`, then `postprocess_pose`; the existing StateGuard camera decoder
validates the resulting 7-vector as a proper homogeneous SO(3) camera. V11
uses the same head/postprocess/camera contract but gives it only a captured
encoder-global projected token.

## Causal comparison

| Version | Candidate input | Capture time | Why it cannot proceed |
| --- | --- | --- | --- |
| v9 | `decoder_embed(mean(dec[0]))` | after recurrent rollout | Literal 576-token availability guard failed on its only dynamic control. |
| v10 | same as v9 with 768 guard | same as v9 | Count-only correction, hence not mechanism-distinct. |
| v11 | `decoder_embed(_get_img_level_feat(feat_i))` | before query, rollout and memory | Not yet tested; must pass Gate A first. |

V11's decoder boundary excludes raw prediction/camera pose, `dec`, pose token,
pose-retriever query, memory, RGB, point map, confidence, history, anchor, GT
and future frames. The candidate's current encoder-global token is captured
once, detached and cloned. At an alarm it is decoded only after rollback; clear
frames retain direct raw export. This does not claim that the encoder latent is
independent of model computation. It establishes only the intended causal and
provenance boundary for a testable recovery export.

## Consequence

Source evidence leaves v11 technically testable, but does not establish pose
quality, normal-path equivalence, runtime, or GPU availability. GPU use remains
prohibited until v11 Gate A has a committed PASS audit.
