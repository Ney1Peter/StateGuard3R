# v17 one-shot native beta-base floor: execution plan

- Date: 2026-08-14
- Status: **pre-registration — no v17 CUDA work is authorized until every Gate-A item has passed, is audited, and is committed.**
- Long-horizon objective (12--18 active hours): conduct one falsifiable, GT-free test of whether a raw-health alarm can make exactly one *subsequent* native ReCal3R global-state write more conservative. Completion means a frozen Gate-E result or a terminal Gate-A/B/C NO-GO; elapsed time never grants permission to skip a gate.

## 1. Hypothesis, novelty and hard exclusions

v16 is terminal `V16_AVAILABILITY_OR_RUNTIME_NO_GO`: its one permitted control completed but failed protected-output identity and runtime. v17 is neither a repair, rerun, re-export nor parameter adjustment of v16. It owns new source, capability schemas, builders, probe, runner, observer, dispatcher, waiter, tests, logs and IDs.

The candidate is an alarm-triggered, one-shot floor change in ReCal3R's existing native state-update equation:

```text
Detector commits current raw health at t and, only on alarm, arms one later eligible update.
At t+1, native beta_base is temporarily changed from 0.1 to 0.0 while the native
update mask is computed; it is restored to 0.1 immediately after that computation.
```

The independently pinned ReCal3R source computes `beta_final = (1-R) * beta_trust + R * beta_base` at `model.py:1199--1251`, and the lighter loop uses that mask only to commit `state_feat` at `model.py:1803--1811`; `mem` remains a native full update. Given `R in [0,1]`, changing `0.1` to `0.0` makes the relevant mask non-increasing and strictly lower where `R > 0`. The constant follows from this algebraic monotonicity and is not selected from a prior outcome.

The v17 operator receives only an alarm boolean, one arm bit, reset eligibility, and constants `0.1`/`0.0`. It may not receive/read/write RGB, prediction, pose, state/memory tensor, latent, attention, pointmap, timestamp, index, GT/depth/label/event/future data, `update_pressure`, state position, init state/memory, model reference state, calibration trace or output/export. It has no rollback/replay, tensor repair, pose replacement, fallback, row selection, learned parameter or threshold tuning.

No v17 code may import, call, read, hash, parse or inspect any v1--v16 recovery/input/operator/runner/probe/dispatcher/output/log/archive/manifest. The transitive audit rejects prior-version components and names containing `recovery`, `manifest`, `archive`, `quarantine`, `pressure`, `pointmap`, `anchor`, `registration`, `prerollout`, `decoder`, `spatial`, `encoder`, `patch`, `repair`, or `rollback`.

## 2. Causal contract

The only legal per-frame order is:

```text
previous/current model-ready RGB overlap
-> native rollout and raw current prediction
-> native mask, state/memory commit, sequence age, calibration and reset
-> frozen Detector-v3 observes and commits raw current health
-> alarm only: arm one future eligible update
-> raw current prediction export
-> next eligible frame: temporary beta_base=0.0 during native mask calculation
-> beta_base restored to 0.1 before later computation/export.
```

An alarm frame must retain its native raw prediction, state/memory commit, pressure and beta base. A clear frame arms nothing. An armed reset cancels the arm. An arm is consumed no more than once; a later alarm can create a new arm only after the old one is consumed/cancelled. Candidate availability requires a genuine alarm, an actual arm consumption, a strict native mask reduction witness, complete scalar restoration, and a changed later raw prediction or trajectory. Evidence never serializes tensors: it records only frame IDs, booleans, scalar base values, shapes/dtypes/devices and GPU-only fixed summaries/digests.

## 3. Data, pins and one-use IDs

No historic output is an input. Fresh v17 builders may read only raw mode-`0444` `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt` (SHA-256 `d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`) and selected raw RGB files. Each may inspect only its own absent target before atomically publishing an `O_EXCL`, mode-`0444` capability:

| capability | ID | consumer data reads |
| --- | --- | --- |
| frame-zero probe | `recovery-beta-floor-v17-dynamic-frame0-probe-capsule-0001` | probe capsule, frame-zero raw RGB, checkpoint |
| dynamic runtime | `recovery-beta-floor-v17-dynamic-rgb-list-capsule-0001` | runtime capsule, raw `rgb.txt`, its 30 RGB files, frozen detector config, checkpoint |

The runtime selector finds `rgb/1305031461.059662.png` then takes 30 paths, requiring ordered-list SHA-256 `de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`. It binds regular/non-symlink/mode-`0444` path, inode, mtime, size and SHA-256. It uses only `load_images_for_eval(size=512,crop=true,square_ok=false)`, clone then the fixed red rectangles: frames 15--19, `(x,y)=(.25+.04k,.25+.025k)`, width/height `.5/.5`, RGB `(255,0,0)`, floor/ceil limits. The probe capsule contains one path/snapshot and no listing, transform, index or timestamp; its parser cannot import dynamic-capability code.

Pins are clean ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, checkpoint `src/cut3r_512_dpt_4_64.pth` SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`, frozen mode-`0444` Detector-v3 config `outputs/formal-v3-calibration-0001/formal-config.json`, and GPU-2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`. No data, weight or dependency download, or modification of ReCal3R/checkpoint/raw RGB/listing is allowed.

The control acceptance constants are formal-baseline constants, never runtime files to read: checkpoint audit `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d`; health `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe`; prediction summary `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed`; trajectory `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b`; reference runtime `5.660642675124109 s`, limit `6.79277121014893 s`.

The only dynamic IDs are `recovery-beta-floor-v17-dynamic-always-commit-0001` and `recovery-beta-floor-v17-dynamic-candidate-0001`. Each GPU release needs two fresh GPU-2 UUID snapshots with at least `12288 MiB` free and no Project2 GPU-2 process, and may run only in a new window of the existing `stateguard` tmux session.

## 4. Gate A: committed non-CUDA proof

1. A source audit binds the model equation/lighter spans and walks all local imports, proving the only candidate mutation is the short beta-base scalar scope.
2. Independent builders have poisoned-reader tests: duplicate key, nonfinite value, path escape, symlink, wrong order/hash/stat, altered transform, mutable input and post-create output reread fail closed.
3. Operator tests prove clear identity, one arm/consume, cancellation at reset, restoration on exception, no cross-alarm carry, mask monotonicity/strictness and rejection of tensors/forbidden input/serialization. Fake recurrence proves the stated causal order and unchanged alarm-frame commit/export.
4. A one-use CUDA-hidden probe runs with `CUDA_VISIBLE_DEVICES=''`; its instrumented ledger has exactly capsule, frame-zero RGB and checkpoint, and blocks listing/dynamic capsule, forward, encoder, RoPE, detector and operator. It binds `(1,768,768)`, `(1,256,1536)`, `(1,768,1)`, never initializes CUDA and emits immutable `O_EXCL`/`0444` evidence.
5. The control branch reproduces the formal health/trajectory serialization contract directly, without old recovery code or historical output reads. Targeted tests prove its four protected outputs and runtime accounting are genuinely comparable.
6. A v17-only dispatcher/waiter covers fake child success/nonzero, dirty source, early/pipe failure, missing pre-exec PID, PID reuse, exception, inventory/freeze and a separate CUDA-hidden validator. It owns an `O_EXCL` journal/logs, pipe-pane readiness, random 256-bit token and verified PID/start-tick lifecycle.
7. Targeted tests, full CUDA-hidden CPU suite, `compileall`, shell syntax, `git diff --check`, pins and permissions pass. The only allowed StateGuard3R untracked paths are the exact preserved v14/v15 inventory; all v17 release bytes are committed HEAD bytes and ReCal3R is clean.

Any failure is `V17_IMPLEMENTATION_OR_INPUT_NO_GO`: no v17 CUDA, no repair and no retry of the version.

## 5. Gates B--E

The dispatcher first freezes preflights, creates one detached `stateguard` pane, waits for `V17_PIPE_READY`, releases one token-gated child, drains/freezes streams, proves the PID/start pair absent, rechecks source/GPU, freezes/inventories output and invokes a fresh CUDA-hidden validator.

- Gate B launches exactly one dynamic always-control: 30 direct native commits, no detector/operator/arm evidence, all four protected hashes identical, and runtime within the fixed limit. Any failure is terminal v17.
- Gate C is allowed only after frozen B PASS: candidate must have a real alarm, consumed arm, strict mask-reduction and restoration witnesses, unchanged alarm-frame raw state/memory/pose and changed later output, within the same runtime limit. Any failure is `V17_AVAILABILITY_OR_RUNTIME_NO_GO`; wrong/low/GT are forbidden.
- Gate D only follows C PASS: independently build wrong/low v17 capabilities and run one control/candidate for each, freezing all six outputs.
- Gate E only follows D: a fresh isolated CPU evaluator may read GT. Candidate readiness requires at least two conditions with both positive tail ATE and translation RPE, both medians at least `+5%`, and median runtime at most `1.20x`; otherwise freeze `V17_FEASIBILITY_NO_GO` without tuning.
