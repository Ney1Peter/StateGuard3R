# v16 bounded update-pressure: Gate-A audit

- Date: 2026-08-14
- Decision: **PASS — authorizes only the pre-registered v16 dynamic
  always-commit control.**
- Scope: split-capability source, causal contract, CPU tests, one real
  CUDA-hidden interface probe, provenance and frozen inputs. **No v16 CUDA
  forward, tmux window, dispatcher invocation, control/candidate output,
  wrong/low condition, GT read, download, or upstream modification occurred.**

## Bound mechanism

After ReCal3R has completed the current recurrent rollout, native update mask,
state/memory commits, calibration record and reset, the frozen Detector-v3
commits its raw current health decision.  Only an alarm then writes the native
CUDA update-pressure tensor:

```text
P <- clamp(P + 0.25, 0.0, 1.0)
```

`P` is `(1,768,1)`.  A reset bootstrap creates a CUDA tensor filled with
`0.25`; a clear frame is exact identity.  The operator accepts only a plain
boolean, the existing pressure (if any), fixed shape/device/dtype metadata and
the fixed constants.  It rejects CPU, nonfloating, nonfinite, wrong-shape and
wrong-dtype tensors.  The recurrent-contract audit fixes this order:

```text
previous/current RGB overlap → native rollout/raw result → native update
→ state and memory commits → calibration → reset → detector commit
→ alarm-only pressure write → raw-result CPU export
```

The candidate-only write has witnesses that the current raw state, memory and
pose fingerprints are unchanged.  A later native update is the only allowed
causal effect.  The always-control constructs no observer, detector, pressure
or candidate evidence.

## Split input capabilities and immutable probe

The independent builders created the following fresh frozen capabilities; both
are mode `0444` and were built only after their targets were verified absent.

| artifact | SHA-256 | purpose |
| --- | --- | --- |
| `outputs/recovery-update-pressure-v16-dynamic-frame0-probe-capsule-0001.json` | `43c418fca7c85f9c3bdeffa4eafce3c2afa3bd60f683e7fa46a6334eb36a15c5` | Gate-A frame-zero probe only |
| `outputs/recovery-update-pressure-v16-dynamic-rgb-list-capsule-0001.json` | `c734a8af297f3d4f5ede70a1d7fba36809b87423338ea8c3f332ddca4fb12792` | later 30-frame runtime only |

The frame-zero parser physically contains neither raw-list selector nor
builder functionality.  Its one-use probe ran with `CUDA_VISIBLE_DEVICES=''`
and produced immutable mode-`0444`
[`recovery-update-pressure-v16-dynamic-frame0-cpu-interface-probe-0001.json`](../../logs/recovery-update-pressure-v16-dynamic-frame0-cpu-interface-probe-0001.json)
(SHA-256 `ee6d3a99dd323bacf4fcbb454a63db8af12be1d540e6cac54fff97595048c69b`).

Its instrumented ledger has exactly these three resource classes, with the
official image and checkpoint loaders wrapped at their actual open points:

```text
probe capsule → frame-zero RGB → checkpoint
```

It neither reads `rgb.txt` nor the dynamic capability.  It loaded one RGB as
CPU `(1,3,384,512)`, loaded the checkpoint with all keys matched, bound native
state `(1,768,768)`, pose memory `(1,256,1536)` and pressure `(1,768,1)`, did
not initialize CUDA, and hard-blocked forward/encoder/RoPE/recurrent/detector/
operator routes.  It did not serialize image or tensor values.

## Provenance and regression record

| item | bound value |
| --- | --- |
| StateGuard3R source commit | `36fafbcf1dacbffba1df38fe07f10359f3a748ea` |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19dbff` |
| checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| raw `rgb.txt` SHA-256 | `d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284` |
| runtime import graph | seven local modules; no v1–v15, recovery/input, GT or offline artifact route |

The ReCal3R tracked worktree was clean and at the pinned commit.  The source
audit recursively visited `bounded_update_pressure_v16`,
`recal3r_bounded_update_pressure_runner_v16`, `online_detector_v16`,
`online_visual_overlap_v16`, `timestamp_order_v16`,
`dynamic_rgb_capability_v16`, and the existing GT-free `health` adapter.  It
reported both legacy/import and offline-artifact flags false.  The only
remaining StateGuard3R untracked paths are the exact preserved terminal
v14/v15 inventory; all v16 release components are committed HEAD bytes.

Using the pinned ReCal3R Python environment (Torch `2.4.0+cu121`) without GPU
visibility, the targeted Gate-A suite passed **49 tests**.  It covers poisoned
reader builders; malformed capability inputs; exact add/cap/saturation/reset
operator behavior; causal fake recurrence; source/import mutation checks;
detector, overlap and timestamp boundaries; CUDA-hidden runner rejection;
dispatcher fake-child/identity/pipe/freeze paths; and waiter syntax/one-use
log paths.  `compileall`, shell syntax and `git diff --check` also passed.

## Authorized next action and stop rule

Only this one-use release is now authorized, and only through the committed
dispatcher/waiter in a new window of the existing `stateguard` tmux session:

```text
recovery-update-pressure-v16-dynamic-always-commit-0001
```

It requires two matching GPU-2 UUID snapshots, at least `12288 MiB` free each,
and no Project2 process on that GPU.  Its control must produce 30 direct
native commits, no detector/operator/pressure evidence, the pre-registered
protected hashes and runtime at most `1.20 × 5.660642675124109` seconds.
Any failure is a terminal v16 availability/runtime/provenance NO-GO: do not
retry v16, run its candidate, wrong/low conditions, GT or quality evaluation.
Only a frozen control PASS can authorize the single v16 candidate.
