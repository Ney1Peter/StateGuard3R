# v19 official native final-mask hold: execution plan

- Date: 2026-08-14
- Status: **pre-registration.** No v19 CUDA, tmux pane, dynamic capability,
  control or candidate may exist before Gate-A is passed, audited and committed.
- Objective (12--18 active hours): run one falsifiable GT-free experiment of
  whether a current scalar raw-health alarm can make exactly one immediately
  subsequent ReCal3R **global-state** write an identity, while the native local
  memory write remains unchanged.  Finish only with frozen Gate-E evidence or
  a terminal Gate-A/B/C NO-GO.

## 1. New mechanism and relationship to terminal versions

V17 is terminal because its copied candidate loop ran with autograd enabled and
OOMed in the required native mask call.  It must not be repaired/rerun.  V18 is
terminal at Gate A because its stricter state-and-memory hold required a memory
witness unavailable from its official-loop callback boundary.  It must not be
weakened or run.

V19 changes both the mechanism and runtime topology.  It runs the official,
`@torch.no_grad()` `inference_recurrent_lighter` path.  An instance-local,
pre-registered wrapper calls the native `_compute_recal3r_update_mask` exactly
once as normal, then, only on a previously armed immediate-next eligible frame,
returns `zeros_like(native_mask)`.  Thus the official loop executes its normal
encoder, decoder, output head, raw native mask calculation, calibration and
memory update; only the final mask that the official code applies to
`state_feat` is held.  It does not mutate beta-base, input RGB, native update
eligibility, prediction, pose, memory, model weights or upstream source.

At pinned ReCal3R `model.py:1803-1813`, the returned `update_mask1` is used in
the state write whereas `update_mask2 = update_mask` continues to update `mem`.
For batch one, a returned all-zero `update_mask1` makes
`state_feat_post = state_feat_prev` exactly.  This state-only claim is
explicit; V19 makes no claim that the memory update is held.

## 2. Causal contract and exclusions

The only candidate order is:

```text
causal previous/current RGB overlap
-> official current native rollout and raw prediction
-> native update-mask calculation
-> candidate only: return zero final state mask if old arm is eligible
-> official state write, native memory write, calibration and reset
-> scalar health adapter observes current committed state trace and raw-output
   scalars, then frozen Detector-v3 commits it
-> current alarm arms only the next raw eligible frame.
```

The alarm frame must use its native final state mask and retain its raw state,
memory and pose after detector observation.  A clear arms nothing; an armed
reset cancels the arm; the final frame cannot arm; an arm is consumed once;
same-frame old-consume/new-alarm fails closed.  Candidate readiness requires a
genuine alarm, exactly one zero-mask return *after* a native mask calculation,
state GPU identity on that held frame, full wrapper restoration, native-memory
update confirmation, alarm-frame preservation, and changed later prediction or
trajectory.

The scalar detector obtains only overlap, pose jump, geometric residual, native
state delta/reliability/uncertainty and timestamp-order boolean.  It cannot see
RGB arrays, prediction/model/tensor objects, attention, state/memory values,
GT, depth, label, event, future data, output or calibration trace objects.  No
rollback/replay, row selection, threshold tune, learned parameter, fallback or
output replacement is allowed.  No v1--v18 recovery/operator/runner/capability/
manifest/log/output may be imported/read by v19 code.

## 3. Pins, data and single-use IDs

No new data, weights or dependencies.  Fresh v19 builders can read only raw
mode-`0444` RGB listing
`baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt` SHA-256
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284` and their
30 raw RGB files.  They select paths from `rgb/1305031461.059662.png`, require
ordered-list SHA-256 `de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`,
load with `load_images_for_eval(size=512,crop=true,square_ok=false)`, clone then
apply fixed red rectangles only on frames 15--19.  Capability targets use
O_EXCL and mode `0444`.

Pins are clean ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, checkpoint
SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`,
frozen Detector-v3 config `outputs/formal-v3-calibration-0001/formal-config.json`
and GPU-2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`.

| purpose | one-use ID |
| --- | --- |
| frame-zero probe | `official-mask-hold-v19-frame0-probe-0001` |
| runtime capability | `official-mask-hold-v19-dynamic-rgb-0001` |
| Gate-B control | `official-mask-hold-v19-always-native-0001` |
| Gate-C candidate | `official-mask-hold-v19-one-shot-0001` |

## 4. Gate A: non-CUDA proof, audit and commit

1. Source audit binds the official `@torch.no_grad` inference call, original
   native mask call, the post-mask state write and independent memory write.
   It rejects copied loop/encoder/decoder, beta mutation, direct state/memory
   assignment, input-gate edit, unapproved method wrapper and old imports.
2. Test scalar policy arm/clear/last/reset/one-consume, native-mask-first order,
   zero-mask only after native call, exception restoration, same-frame fail
   closed, alarm-frame preservation, held-state identity and native-memory
   continuation using fake official-loop methods.
3. Test fresh capability builder/parser poisoned readers and CUDA-hidden
   frame-zero probe (only capsule/frame-zero RGB/checkpoint; no forward/CUDA;
   interfaces state `(1,768,768)`, memory `(1,256,1536)`, mask `(1,768,1)`).
4. Test direct official control branch/serializer against the four formal
   protected hashes and ensure it cannot construct detector/policy/wrapper.
5. Test v19-only O_EXCL dispatcher/waiter: source/pin/inventory preflight,
   two GPU snapshots, new `stateguard` pane, pipe readiness, 256-bit token,
   PID/start proof, drain/freeze and fresh CUDA-hidden validator.
6. Pass full targeted CUDA-hidden CPU tests, `compileall`, shell syntax,
   diff checks, permissions and source hygiene.  Record and commit Gate-A audit.

Any failure is `V19_IMPLEMENTATION_OR_INPUT_NO_GO`: no v19 CUDA, repair or
retry.

## 5. Gates B--E

After Gate-A commit, build only the v19 runtime capability.  Each GPU release
needs two fresh matching GPU-2 UUID snapshots, >=`12288 MiB` free and no
Project2 GPU-2 process; it runs only in a new `stateguard` window.

Gate B runs exactly one 30-frame direct-native control.  It must construct no
detector/policy/wrapper; its checkpoint/health/predictions/trajectory hashes
must equal respectively `3e12875a…`, `9de0c0b6…`, `cb11415b…`, `fdba3312…`, and
runtime must be <=`6.79277121014893 s`.  Any failure is terminal v19 NO-GO.

Only frozen B PASS authorizes one candidate.  C must satisfy the causal
readiness conditions above and same runtime limit.  Any C failure is terminal
`V19_AVAILABILITY_OR_RUNTIME_NO_GO`: do not retry or execute wrong/low/GT.
Only C PASS permits independently built wrong/low controls/candidates (D),
then a fresh isolated GT evaluation (E) with the pre-registered positive-tail
ATE/RPE and `+5%` median-improvement / `1.20x` median-runtime criteria.
