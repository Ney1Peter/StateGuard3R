# Formal Detector v2 cross-scene pilot protocol

Status: **pre-registered for formal commitment; no new-scene ReCal3R model response, response hash, or runner main-log byte may be read before the tracked protocol, v2 orchestrator, and commitment have been frozen.** This document defines one confirmatory Detector v2 evaluation. It does not make a recovery claim and it does not authorize a second threshold-selection pass on the new scene.

## Scope, evidence roles, and exclusions

- Detector v1's within-sequence pilot was formal **NO-GO**. Its artifacts are immutable and are not modified or relabelled here.
- Every old `fr1_desk` run is disclosed development data for v2, including the three runs formerly called v1 holdout. None of those six runs is blind or confirmatory evidence for v2.
- The only blind v2 responses are the three frozen `rgbd_dataset_freiburg3_walking_static` inputs named below. The project had not inspected a ReCal3R response from that scene when these inputs were generated.
- The protocol permits no additional dataset archive. It does not train or fine-tune ReCal3R, modify the ReCal3R repository, use ATE/RPE/depth accuracy as a detector outcome, or execute quarantine/rollback before the formal decision.
- Online detector inputs are the model-ready RGB sequence and ReCal3R's own health fields only. Ground-truth pose, depth, labels, event interval, source-pool role, and future frames must not enter an online component or score.

## Frozen code, runner, checkpoint, and development selection

The commitment snapshots the exact StateGuard3R and ReCal3R commits, tracked runner and orchestrator bytes, protocol bytes, interpreter/OpenCV provenance, and checkpoint. The fixed checkpoint is `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`, SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`. The ReCal3R revision must be `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` and its tracked worktree must be clean. The production invocation is the tracked `scripts/run_recal3r_smoke.py` with `--health-profile v2`, size 512, batch one, seed 0, and `beta-base 0.1`.

The disclosed development search is immutable at `outputs/detector-v2-development-search-0001/search.json`, SHA-256 `45287f2eeeee33c8c0eee9cf7586f6996589d9963683775fcb8c93e61250c129`. It selected the following before the new-scene response exists:

| Item | Fixed value |
| --- | --- |
| visual feature | ORB, `nfeatures=2000` |
| visual geometry | grid 8x6, minimum matches 12, ratio 0.80, RANSAC 1.0 pixel |
| OpenCV determinism | RNG 0, one thread, OpenCL disabled |
| scorer | causal window 10, min-history 5, max-z 10, master seed 0 |
| aggregation | arithmetic mean of finite direction-aware components |
| floors | geometric `4.948262292669166e-05`; overlap `0.006875120234255727`; pose `0.000806030222234299`; reliability `0.000527346134185791`; update `0.0074473662806365136` |
| random threshold | `0.8757517985798549` |
| update-only threshold | `1.2113849262902454` |
| reliability-only threshold | `5.485329360937135` |
| combined threshold | `2.0686323694270046` |

`geometric_residual`, `pose_jump`, and `update_magnitude` are high-risk; `overlap` and `reliability` are low-risk. A component at frame *t* sees only strictly earlier finite values in its window. During min-history warm-up its finite neutral value is zero and it remains in the metric denominator. Every component is capped at 10. Frame zero has `overlap=null`; every later frame must have a finite visual-correspondence coverage in `[0,1]`. A missing or non-finite post-startup overlap is a formal failure, not a zero substitution.

Visual correspondence coverage is an ORB/RANSAC/grid proxy measured from consecutive post-loader, post-deferred-transform RGB tensors. It is neither a physical overlap percentage nor a ground-truth signal. The model inputs are hashed before and after its computation to establish no mutation.

## New-scene allocation and GT-only donor construction

The sole new archive is `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg3_walking_static.tgz`, 474074105 bytes, SHA-256 `8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1`. Raw acquisition passed at `outputs/formal-v2-data-0002/acquisition.json`, SHA-256 `c04e7e835edf861b234ecd8fbeae3f9f24842a3d0151d11e7b1a3660b0e0b391`. The raw archive and tree are read-only. The input registry is `outputs/formal-v2-inputs-0001/formal-v2-manifest.json`; its commitment and the exact source/input manifest bytes are revalidated and bound by the formal commitment.

The three 30-frame allocations are cross-scene disjoint from `rgbd_dataset_freiburg1_desk`:

| Run | Corruption | Base indices | donor indices |
| --- | --- | --- | --- |
| `holdout-dynamic` | `dynamic_occlusion` | 3--32 | none |
| `holdout-wrong` | `wrong_order_segment` | 50--79 | none |
| `holdout-low` | `low_overlap_jump` | 104--133 | 436--440 |

RGB, depth, and GT associations use unique nearest absolute timestamp with a 0.02-second maximum and reject ties. The offline donor ranking uses TUM intrinsics `(fx=535.4, fy=539.2, cx=320.1, cy=247.6)`, depth/GT reprojection at stride 8, and selected donor start 436 with mean score `0.13875603825269686` (five scores: `0.14054726368159204`, `0.12731196054254007`, `0.12327082692898862`, `0.13685179502915004`, `0.1657983450812136`). This value is used only to choose and describe the low-overlap donor; it is never written to Health Ledger, component, threshold, or online score.

## Fixed events and metric policy

Each run has one enabled corruption beginning at position 15:

- `dynamic_occlusion`: positions 15--19 inclusive; normalized rectangle `(0.25, 0.25, 0.5, 0.5)`, velocity `(0.04, 0.025)`, RGB fill `[255,0,0]`.
- `wrong_order_segment`: positions 15--18 inclusive; reverse order.
- `low_overlap_jump`: positions 15--19 inclusive; explicit donor source-pool positions 30--34.

Seed 0 applies to construction and the master random baseline. Primary positives are exactly the enabled interval. The five positions immediately after the interval are washout and are ignored in primary denominators but are still forwarded and scored; they break primary false-positive streaks. Startup positions 0--14 are evaluated negatives. Per run, primary evaluation therefore has 25 positions. The three-run split has 14 positives, 61 primary negatives, and 15 ignored washout positions. The evaluator reports threshold metrics, pooled and equal-corruption macro AUROC, event detection/delay, per-run score timeline, clean-prefix FPR/streak, and boundary/washout alarms.

## Nine-run registry and execution/read-lock rules

The formal commitment creates one registry with the following exact order:

```text
development-dynamic  (formal-v1-inputs-0001/development/development-dynamic)
development-wrong    (formal-v1-inputs-0001/development/development-wrong)
development-low      (formal-v1-inputs-0001/development/development-low)
development2-dynamic (formal-v1-inputs-0001/holdout/holdout-dynamic)
development2-wrong   (formal-v1-inputs-0001/holdout/holdout-wrong)
development2-low     (formal-v1-inputs-0001/holdout/holdout-low)
holdout-dynamic      (formal-v2-inputs-0001/holdout-dynamic)
holdout-wrong        (formal-v2-inputs-0001/holdout-wrong)
holdout-low          (formal-v2-inputs-0001/holdout-low)
```

The six first rows are the disclosed development pool. The three last rows are blind and cannot be opened, statted for content, hashed, parsed, or have their main log read by calibration. Calibration can only snapshot those six development output directories. It deterministically recomputes the fixed v2 scores and fixed thresholds from the freshly generated six development ledgers, publishes immutable calibration artifacts, and atomically creates a separate holdout-unlock artifact.

Development forwards run serially in the first six-row order. Only after the unlock do blind forwards run serially `dynamic -> wrong -> low`. Each forward requires two fresh, adjacent `nvidia-smi`/process snapshots, an explicitly chosen physical `CUDA_VISIBLE_DEVICES` GPU with at least 12288 MiB free in both, the command, PID, GPU UUID, output directory, log path, start/finish, runtime, and post-exit release record. The GPU may be shared if this capacity condition holds; no other user's process is altered. Every successful run is first structurally checked, then frozen with run directory mode `0555` and all artifacts mode `0444`.

The formal run root contains exactly the runner's five artifacts per run: `checkpoint-load-audit.json`, `health.jsonl`, `predictions-summary.json`, `trajectory.json`, and `run.json`. A failed development forward may be operationally retried only before holdout unlock and only with the same input and command in a new formal version. A failed blind forward stops the formal path without substituting an input or output directory. During blind forwarding, the launcher may only `lstat`/list and freeze its own completed tree; it must not open, parse, or hash any response or main-log byte. Before all three blind directories exist, are complete, and are frozen, evaluate rejects them without reading any response or main-log byte. Evaluation snapshots all three only after that preflight and is protected by a no-overwrite output directory, so it can execute exactly once.

## Formal decision: all nine conditions are required

The one evaluation is **GO** only when every condition holds:

1. combined holdout equal-corruption macro-AUROC is strictly greater than 0.75;
2. combined minus seeded-random macro-AUROC is at least 0.10;
3. combined is no more than 0.02 below the best single real signal;
4. development and holdout share at least two detected corruption types;
5. combined holdout pooled primary FPR is at most 0.20;
6. maximum per-run primary false-positive streak is at most 3;
7. provenance, runtime, finite/replay/reproducibility, and real-signal checks all pass;
8. frames 1--29 have finite online visual coverage, combined uses it, and no GT/label leakage is present; and
9. holdout clean-prefix pooled FPR is at most 0.15 and startup maximum streak is at most 2.

The decision is a conjunction, not a selection rule. **GO** supports only a separately pre-registered minimum quarantine pilot. **NO-GO** immediately discloses this holdout, stops before quarantine/rollback, and requires a full failure report; neither decision allows tuning and rerunning this gate.

## Immutable artifacts and completion

All formal directories publish atomically under `outputs/` and use directory mode `0555` and file mode `0444`: commitment, run registry, development calibration, holdout unlock, and unique evaluation. Every manifest records relative path, byte size, SHA-256, schema, source commit, upstream binding, and execution timeline. Outputs, logs, raw data, archives, and checkpoints are never committed to Git. The project closes only after the GO quarantine path or the NO-GO failure path completes its report, full CPU verification, repository/permission checks, and process/GPU cleanup.
