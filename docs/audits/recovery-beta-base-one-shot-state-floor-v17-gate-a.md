# v17 native one-shot beta-base floor: Gate-A audit

- Date: 2026-08-14
- Decision: **PASS — authorizes only the pre-registered v17 dynamic
  always-commit Gate-B control.**
- Scope: committed non-CUDA source, CPU contracts, frozen frame-zero probe,
  provenance and release preflight.  **No v17 CUDA forward, dynamic runtime
  capability, tmux pane, dispatcher, control/candidate output, wrong/low
  condition, GT evaluation, download, or upstream modification has occurred.**

## Bound mechanism and causal scope

The candidate is a deliberately small, one-shot change to the native ReCal3R
state-mask calculation.  After the current recurrent rollout, native state and
memory commits, calibration and reset have completed, Detector-v3 commits the
raw current health.  An alarm may arm exactly one later eligible update.  On
that later update only, `model.beta_base` is temporarily `0.0` while the native
mask is calculated, then restored to `0.1` before any later work or export.

The pinned upstream equation is in
`baselines/ReCal3R/src/dust3r/model.py:1199-1251`:

```text
beta_final = (1 - R) * beta_trust + R * beta_base
```

The lighter loop consumes this mask only in its native global `state_feat`
commit at `model.py:1803-1811`; `mem` remains the native full update.  Thus, for
native `R` in `[0,1]`, the fixed change from `0.1` to `0.0` cannot increase the
mask and is strictly lower where `R > 0`.  It is not tuned from a prior result.

The source audit binds this legal order:

```text
previous/current RGB overlap -> native rollout/raw prediction -> native mask,
state/memory commit, sequence age, calibration and reset -> detector raw-health
commit -> alarm-only future arm -> raw prediction export -> next eligible,
temporary beta_base=0.0 native mask -> restoration to 0.1.
```

It reports the sole candidate model mutation as
`temporary_model_beta_base_scalar_scope`, requires the GPU-resident strict-mask
witness, and rejects CPU calibration-state reads.  The production import audit
visited exactly the seven closed v17 modules (including the runner and its
beta-floor operator); all historical-local, offline-artifact and historical
production-artifact flags are false.  The production script SHA-256 is
`099bc73d53bdcd0219ffaea87430eeee2189df9d57797bdda976b6173f442922` and the
runner SHA-256 is
`2060cde475433c3de812d79052b372177e7747ee6f3e380b5ff70dc17cfff04f`.

## Frozen frame-zero evidence

The independent one-use builder and CUDA-hidden probe have already created the
following immutable evidence.  Both paths are non-symlink regular files with
mode `0444`.

| artifact | SHA-256 | result |
| --- | --- | --- |
| `outputs/recovery-beta-floor-v17-dynamic-frame0-probe-capsule-0001.json` | `220702e858fd6ef00a52990e625bcedb7a198396918ab2b1f7a91e02d8bc85f8` | one frame-zero capability |
| `logs/recovery-beta-floor-v17-dynamic-frame0-cpu-interface-probe-0001.json` | `d61a253a5e6f2d2bf5327bec8940dbba115ebf19c780bea8b89e5b8a7120e31c` | passed CPU interface probe |

The probe used `CUDA_VISIBLE_DEVICES=''`; CUDA was not initialized and model
forward did not execute.  Its instrumented access ledger has exactly three
resource classes: probe capsule, frame-zero RGB and checkpoint.  It did not
read `rgb.txt`, the dynamic capability, detector or operator.  It bound the
native CPU interfaces:

```text
state feature     (1, 768, 768) float32
pose memory       (1, 256, 1536) float32
native update mask (1, 768, 1) float32
```

The checkpoint audit is fixed to SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`; the raw
listing pin is SHA-256
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`.

## Regression, provenance and release hygiene

StateGuard3R source was tested at
`658b6b7e2bb8b31d4570359f9bb28077c15b8f22`.  With CUDA hidden, the following
command passed **52 tests in 3.98 seconds**:

```bash
PYTHONPATH="$PWD/src:/data/wangzheng/Project2/baselines/ReCal3R/.venv/lib/python3.11/site-packages" \
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  --basetemp "$PWD/tmp/pytest-v17-gate-a-0001" \
  tests/test_beta_base_floor_v17.py \
  tests/test_dynamic_rgb_capability_v17.py \
  tests/test_frame_zero_probe_capability_v17.py \
  tests/test_online_v17_components.py \
  tests/test_probe_recal3r_beta_base_floor_interface_v17.py \
  tests/test_recal3r_beta_base_floor_runner_v17.py \
  tests/test_run_recal3r_beta_base_one_shot_v17.py \
  tests/test_v17_source_import_audit.py \
  tests/test_dispatch_recal3r_beta_base_one_shot_v17.py \
  tests/test_wait_and_dispatch_recal3r_beta_base_one_shot_v17_control.py
```

This covers poisoned capability readers; immutable-input and transform checks;
one-arm, consume, cancellation, restoration, identity and strictness operator
rules; fake native recurrence ordering; CUDA-hidden interface blocking;
independent formal-compatible control serialization; source/import graph;
one-use dispatcher child/PID/pipe/freeze paths; and waiter syntax/release
paths.  `compileall`, `bash -n` on the v17 waiter, `git diff --check` and
`git diff --cached --check` all passed.

All v17 release components equal committed HEAD bytes.  The StateGuard3R work
tree has no tracked modifications; its only untracked paths are the exact 28
preserved terminal v14/v15 paths, which were neither edited nor staged.
ReCal3R is clean and pinned at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.  No new data, weight or dependency
was downloaded.

## Authorized next action and stop rule

After this audit is committed, and only after a fresh release preflight, build
the one-use dynamic capability
`recovery-beta-floor-v17-dynamic-rgb-list-capsule-0001`.  Then the committed
v17 dispatcher/waiter may create one new pane in existing tmux session
`stateguard` and run only:

```text
recovery-beta-floor-v17-dynamic-always-commit-0001
```

GPU-2 must match UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250` in two fresh
snapshots, each with at least `12288 MiB` free and no Project2 GPU process.
Gate B passes only if the four protected output hashes equal the pre-registered
values and runtime is at most `6.79277121014893 s`.  Any Gate-B failure is a
terminal v17 NO-GO: do not retry control, run candidate/wrong/low, or evaluate
GT.  Only a frozen Gate-B PASS authorizes the one v17 candidate.
