# Detector v2 visual-overlap audit

- Audit date: 2026-08-02
- Scope: disclosed formal-v1 inputs used only as Detector v2 development data
- Result: **PASS** for causal online-signal availability, deterministic replay, and the structural direction check. This is not a Detector v2 readiness result and does not select the final scoring candidate.

## Signal meaning and boundary

`online_visual_correspondence_coverage` is a **verified visual-correspondence coverage proxy**, not a physical overlap percentage. At frame `t`, it receives only the official loader's post-resize/crop, post-deferred-transform normalized RGB tensors for frames `t-1` and `t`. It copies them into independent RGB `uint8` arrays, then uses ORB, bidirectional ratio-filtered mutual Hamming matching, RANSAC fundamental-matrix verification, and minimum endpoint coverage of an `8 x 6` grid, multiplied by inlier ratio.

The online ledger contains exactly `{frame_id, overlap}`. Frame 0 is `null`; every later frame is finite in `[0, 1]`. The scorer API neither accepts nor reads depth, GT pose, corruption label, event interval, future frame, health ledger, or model response. The input-manifest parser is used solely to obtain the final official RGB path order and deferred pixel transforms. The separate `gt_depth_reprojection_overlap` helper remains offline-only: it may be used for future donor/severity construction but cannot enter an online ledger or score.

`source_discontinuity` below is also only a post-hoc structural diagnostic: adjacent manifest source indices fail to increase by one. It is not supplied to the online correspondence function and is not a detector feature or label.

## Frozen real-input replay

The checker was run CPU-only with `CUDA_VISIBLE_DEVICES=''`. It did not load a checkpoint or call ReCal3R inference. It replayed all four pre-specified candidates over all six disclosed v1 manifests, twice per candidate through a fresh official-loader replay, and rejected any changed model-ready input.

| Candidate | Clean contiguous pairs | Clean mean | Source-discontinuity pairs | Discontinuity mean | Direction | Pair statuses |
|---|---:|---:|---:|---:|---|---|
| ORB-2000, ratio 0.70, RANSAC 1.0 | 140 | 0.436254748 | 14 | 0.313497432 | PASS | 170 `ok`, 4 `match_insufficient` |
| ORB-2000, ratio 0.70, RANSAC 2.0 | 140 | 0.549721729 | 14 | 0.397442251 | PASS | 170 `ok`, 4 `match_insufficient` |
| ORB-2000, ratio 0.80, RANSAC 1.0 | 140 | 0.442298506 | 14 | 0.313096553 | PASS | 170 `ok`, 4 `match_insufficient` |
| ORB-2000, ratio 0.80, RANSAC 2.0 | 140 | 0.563442002 | 14 | 0.410097360 | PASS | 170 `ok`, 4 `match_insufficient` |

All four retain the required fixed values: 2,000 ORB features, 12 minimum matches, `8 x 6` grid, single OpenCV thread, OpenCL disabled, and RANSAC RNG seed 0. A `match_insufficient` result is deliberately recorded as finite coverage `0`, rather than a missing observation or dropped frame.

## Reproducibility and provenance

The first frozen output is `outputs/detector-v2-derived-development-0001/`:

- `visual-overlap-replay.json`: SHA-256 `f2cbbff03a00baeb4ec93b731b003c61b72fa0d95776b6df819fd5e41dabff41`
- `report.md`: SHA-256 `7bc1f21000724a9ec1dc9f0e0b3c5eca80ccd0ff7d02ed9f212429a2784b4ee9`

A second independent process wrote `outputs/detector-v2-derived-development-0001-replay/`; both files are byte-identical to their counterparts above. Both trees are mode `0555`, and their files are mode `0444`.

The replay binds the frozen v1 input registry SHA-256 `1de55c281a780002d8961fd5a05cb507fd1168804aaf1c493c2e48b459d77932`, the unchanged ReCal3R commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, and the official loader SHA-256 `a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96`. It used OpenCV `4.11.0`; its extension binary SHA-256 is `188d38ace4716a74534ddf4dd4f7d77a06641bd0908bf12eb218a3c2fef35f38`, and the build-information SHA-256 is `f3f58b60f58039df358dceb98f617583025ea7b080b4b3b5c84cc15ef13edf09`.

The frozen generator was StateGuard3R commit `cc13dbcd9dbd1bb8d36f848941050bc104b9d5d4`; its checker SHA-256 is `a755b68d5789894172a67aedf5aa2af6c244a0bd5715e444be4a18e81328db68` and its visual-overlap module SHA-256 is `e50c6b1fe26be2ece5edc494316266f8d75c158e70e61786fb5bb4d080c616bc`.

## Consequence for the next phase

The visual signal has met its Phase 2 availability and causal-provenance requirements on the disclosed pool. It has **not** been selected by these means alone: the plan requires retaining all candidates and comparing them in the subsequent development scoring search, including the constrained threshold, cross-validation, and readiness criteria. No new dataset download, GPU forward, formal-v2 commitment, or formal-v2 holdout access is authorized at this point.
