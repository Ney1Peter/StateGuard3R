# v20 post-commit native-scalar state hold: execution plan

- Date: 2026-08-14
- Status: **pre-registration.** No v20 CUDA, tmux pane, capability, control or
  candidate is allowed until Gate A is passed, audited and committed.
- Objective (12--18 active hours): test once whether a detector that observes
  only post-commit native/current scalar health can alarm and make exactly one
  immediately following official ReCal3R *global-state* write an identity.
  Completion is frozen Gate-E evidence or terminal Gate-A/B/C NO-GO.

## 1. New, testable causal contract

V17 is terminal for copied-loop candidate OOM; v18 is terminal because its
state-and-memory witness was unavailable; v19 is terminal because its required
pre-rollout RGB overlap had no permitted hook.  None may be rerun, repaired or
read by v20.

V20 makes a different, narrower claim.  It uses the official
`@torch.no_grad()` `inference_recurrent_lighter` loop and an instance-local
wrapper around only `_downstream_head`, `_compute_recal3r_update_mask` and
`_maybe_record_u_calibration_step`.  The detector has no RGB/overlap input.  It
receives only current raw scalar health *after* the official native state and
memory commits and calibration: native uncertainty/reliability/state-delta plus
raw-prediction pose-jump and geometric residual.  Its current decision then
arms the immediately following eligible frame.

On that following frame, v20 always calls the native mask method first and,
only if the old arm is eligible, returns an all-zero same-interface final state
mask.  The official code then performs an identity global-state update while
its independent `update_mask2` retains the native memory update.  Beta, RGB,
raw update gate, output, pose, memory, weights and upstream source are never
altered.

```text
official current rollout/raw prediction -> native mask -> official state/memory
commit -> native calibration -> current scalar health and detector commit
-> alarm arms one next frame -> next native mask runs -> zero final state mask
only -> official state identity + native memory write -> detector commit.
```

The alarm frame retains native state and pose across detector observation.  A
clear arms nothing; reset cancels an arm; the final frame cannot arm; an arm is
consumed once; an old-consume/current-new-alarm fails closed.  Candidate
availability requires genuine alarm, one consumption, native-mask-first proof,
zero final-mask proof, held global-state GPU identity, native memory unchanged
rule, restoration and changed later output/trajectory.

## 2. Hard exclusions

The detector cannot receive image/RGB, model object, state/memory tensor,
attention, pointmap, timestamp, index, GT/depth/label/event/future data,
offline artifact or output.  It has no learned parameter, threshold tune,
rollback/replay, fallback, pose/output replacement or latent write.  V20 local
runtime imports/read paths may not contain v1--v19 recovery/input/operator/
runner/capability/output/log/archive/manifest code or artifacts.

## 3. Inputs, pins and one-use releases

No downloads.  Fresh v20 builders read only raw mode-`0444` RGB listing
`baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt` SHA-256
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284` and 30
selected raw RGB paths starting at `rgb/1305031461.059662.png`, ordered-list
SHA-256 `de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`.
Use official loader size 512/crop true/square false and fixed red rectangles on
frames 15--19.  Publish O_EXCL/mode-`0444` only.

Pins: clean ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`; checkpoint
SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`;
GPU-2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`.  The fixed detector has
only these preregistered constants: history `5`, minimum history `3`, robust
scale floor `1e-6`, max score `20`, combined threshold `2.0`.

| purpose | one-use ID |
| --- | --- |
| CPU interface capability | `native-scalar-hold-v20-frame0-probe-0001` |
| runtime capability | `native-scalar-hold-v20-dynamic-rgb-0001` |
| Gate-B control | `native-scalar-hold-v20-always-native-0001` |
| Gate-C candidate | `native-scalar-hold-v20-one-shot-0001` |

## 4. Gate A

1. Audit official inference/no-grad, native-mask-first, state/memory update and
   calibration order; reject copied loop, additional encode/device hook,
   beta/update/RGB mutation, direct state/memory assignment and prior imports.
2. Test scalar detector and policy: malformed/nonfinite/tensor/model inputs
   fail closed; arm/clear/reset/last/one-consume/same-frame behavior; fake
   official loop proves causal order, state hold, alarm-frame preservation,
   native-memory rule and restoration.
3. Independently build/poison-test v20 capability and CUDA-hidden frame-zero
   probe; bind `(1,768,768)`, `(1,256,1536)`, `(1,768,1)` without forward/CUDA.
4. Test direct official always-control serializer/hashes and v20-only O_EXCL
   dispatcher/waiter lifecycle (two snapshots, new `stateguard` window, PID,
   token, pipe drain, frozen inventory, CUDA-hidden validation).
5. Run full targeted CPU suite with CUDA hidden, compile/shell/diff/pin/hygiene
   checks.  Audit and commit before capability or GPU use.

Gate-A failure is terminal `V20_IMPLEMENTATION_OR_INPUT_NO_GO`.

## 5. Gates B--E

After committed A, build exactly one v20 capability.  Each GPU release needs
two fresh GPU-2 UUID snapshots, >=`12288 MiB` free, no Project2 GPU process and
a new `stateguard` window.  Gate B is one official direct-native control, 30
commits/no detector-policy-wrapper, four protected formal hashes unchanged and
runtime <=`6.79277121014893 s`.  Any failure is terminal.

Only frozen B PASS permits one candidate.  C requires the availability evidence
above and same limit; any failure is terminal
`V20_AVAILABILITY_OR_RUNTIME_NO_GO`, with no retry/wrong/low/GT.  Only C PASS
permits independently built wrong/low conditions and then isolated GT evaluation
with the frozen positive-tail ATE/RPE, `+5%` median, `1.20x` runtime rule.
