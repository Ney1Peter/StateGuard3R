# v18 one-shot native update-hold: execution plan

- Date: 2026-08-14
- Status: **pre-registration.**  This is a new version, not a repair, rerun,
  export or parameter change of v17.  No v18 CUDA execution is permitted until
  Gate A is complete, audited and committed.
- Objective (12--18 active hours): establish, with a finite one-use control and
  candidate, whether a current raw-health alarm can safely suppress exactly one
  *subsequent* native ReCal3R state-and-memory write while preserving the alarm
  frame.  Completion is either a frozen Gate-E result or terminal Gate-A/B/C
  NO-GO; elapsed time is never a bypass.

## 1. Why v18 is a separate experiment

v17 is frozen `V17_AVAILABILITY_OR_RUNTIME_NO_GO`.  Its control was exact, but
the one permitted candidate ran out of CUDA memory in the required native mask
calculation before any health decision or intervention.  V18 does not rerun
that candidate, reuse its capability/output/log/evidence, alter its beta-base
floor, or read its artifacts.

The v18 hypothesis is different:

```text
Once raw current health is committed after a native ReCal3R frame, an alarm can
hold exactly the immediately following eligible native state-and-memory update
by setting that raw view's existing update gate to false.  The official native
recurrent loop then computes its normal prediction and commits zero masks.
```

This is a one-frame `update`-eligibility hold, not a beta weighting change.  It
uses the upstream `forward_recurrent_lighter` path rather than a copied
recurrent implementation.  The design response to v17 is falsifiable: if the
official-loop hook is still unavailable, v18 is availability NO-GO; it does not
convert v17's OOM into a success claim.

The ReCal3R source at pinned `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
computes the raw gate at `model.py:1780-1808` and applies it to `state_feat`
and `mem` at `1808-1813`.  With batch-one `img_mask=true` and injected
`update=false`, the native product `img_mask & update` is false, so both native
writes are exact identities.  The control never constructs detector, arm, hook
policy or operator.

## 2. Causal and capability contract

The only legal candidate order is:

```text
model-ready current/previous RGB overlap
-> official native current recurrent rollout and raw prediction
-> official native update/state+memory commit, calibration and reset
-> frozen Detector-v3 observes and commits current scalar raw health
-> alarm: arm only the immediately following raw view
-> before that following native rollout, and only if it is not reset:
   replace its existing update eligibility by false
-> official native prediction and zero state-and-memory writes on that frame
-> gate returns to native for every later frame.
```

The detector may receive only scalar current health, a causal overlap scalar,
timestamp-order boolean and raw current prediction-derived scalars.  It cannot
receive future frames, GT, labels, depth, state/memory tensor values, attention
values or mutable model access.  The hook is temporary, instance-local and may
only wrap the official downstream-result and post-commit calibration methods;
it may not replace the recurrent loop, encoder, decoder, state equation,
prediction head, pose, checkpoint or export.

The v18 policy accepts only an alarm boolean, one arm bit, immediate-next reset
eligibility and fixed booleans.  It has no learned value, threshold tuning,
rollback, replay, pose replacement, row selection, latent write, input image
change or fallback.  A reset cancels the arm.  An alarm frame retains its
native prediction and state-and-memory write.  Candidate availability requires a real
alarm, exactly one consumed arm, a verified false native update gate, equality
of pre/post state-and-memory GPU summaries on the held frame, complete hook
restoration, and a changed later prediction or trajectory.

## 3. Inputs, pins and one-use IDs

No v1--v17 output, manifest, log, capability, archive, result, run registry or
recovery component may be read/imported by v18 runtime code.  Fresh v18
builders may read only raw mode-`0444`
`baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt`
(SHA-256 `d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`)
and their selected raw RGB files.  They independently select the 30 paths
starting at `rgb/1305031461.059662.png`, require ordered-list SHA-256
`de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`, and use
the official `load_images_for_eval(size=512,crop=true,square_ok=false)` plus
the fixed red rectangles on frames 15--19.  Each published target is O_EXCL
and mode `0444`.

Pins: clean ReCal3R commit above; checkpoint
`src/cut3r_512_dpt_4_64.pth` SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`; frozen
Detector-v3 config `outputs/formal-v3-calibration-0001/formal-config.json`
mode `0444`; GPU-2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`.  No new
data, weight, dependency or upstream change is allowed.

One-use IDs:

| purpose | ID |
| --- | --- |
| CPU interface capability | `state-hold-v18-frame0-probe-0001` |
| runtime capability | `state-hold-v18-dynamic-rgb-0001` |
| Gate-B control | `state-hold-v18-always-native-0001` |
| Gate-C candidate | `state-hold-v18-one-shot-0001` |

## 4. Gate A: committed non-CUDA proof

1. Audit the pinned state-gate equation and official lighter-loop ordering;
   statically reject an owned recurrent/encoder/decoder implementation, beta
   mutation, prior-version local import and historical artifact literal.
2. Test independent poisoned-reader capability builder/parser: duplicate key,
   path escape, symlink, wrong order/hash/stat, modified rectangle, mutable
   input and output reread must fail closed.
3. Test arm/consume/reset/clear identity, exact `false` gate injection, hook
   restoration on exception and fake official-loop causality.  Prove an alarm
   cannot alter the current frame and that only its immediate later raw view is
   changed.
4. Run a CUDA-hidden frame-zero probe.  Its ledger can read only its own
   capsule, one raw RGB and checkpoint; it must bind state `(1,768,768)`, memory
   `(1,256,1536)` and update `(1,768,1)`, execute no forward and initialize no
   CUDA.
5. Bind an independent control serializer to the four formal protected hashes;
   test the official `inference_recurrent_lighter` call and reject detector or
   hook construction in control.
6. Implement a v18-only O_EXCL dispatcher with two fresh GPU snapshots, new
   pane in existing `stateguard` tmux session, token-gated child, PID/start-tick
   proof, frozen stream/output inventory and a fresh CUDA-hidden validator.
7. Pass targeted CUDA-hidden CPU suite, `compileall`, shell syntax,
   `git diff --check`, pins and exact untracked-inventory check.  Commit Gate-A
   audit before any capability build or CUDA/tmux release.

Any failure is terminal `V18_IMPLEMENTATION_OR_INPUT_NO_GO`; no v18 CUDA work
or repair/retry is allowed.

## 5. Gates B--E and stop rules

After Gate-A commit, build the one dynamic capability and run exactly one
control in a new `stateguard` pane.  It needs two fresh GPU-2 snapshots, each
with at least `12288 MiB` free and no Project2 GPU-2 process.  Gate B must make
30 direct official native commits, construct neither detector nor hook, match
the fixed four hashes (`3e12875a…`, `9de0c0b6…`, `cb11415b…`, `fdba3312…`) and
finish within `6.79277121014893 s`.  Failure is terminal v18 NO-GO.

Only frozen Gate-B PASS permits the one candidate.  Gate C requires a real
alarm, one held immediate next update, false update-gate witness, unchanged
alarm-frame raw state/memory/pose, held-frame state-and-memory identity, full restoration
and a changed later raw output, within the same time bound.  Failure is
terminal `V18_AVAILABILITY_OR_RUNTIME_NO_GO`; do not retry or run wrong/low/GT.

Only Gate-C PASS permits independently built wrong/low capabilities (Gate D),
followed by isolated GT evaluation (Gate E).  Gate-E readiness requires at
least two conditions with positive tail ATE and translation RPE, both median
improvements at least `+5%`, and median runtime at most `1.20x`; otherwise
freeze `V18_FEASIBILITY_NO_GO` with no tuning.
