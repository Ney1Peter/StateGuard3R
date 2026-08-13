# v15 bounded update-pressure from an independent RGB listing: execution plan

- Date: 2026-08-13
- Status: **terminal — `V15_IMPLEMENTATION_OR_INPUT_NO_GO`; no v15 CUDA run
  is authorized.**
- Explicit objective (12--18 hours of active work): make one falsifiable,
  GT-free test of whether a Detector-v3 alarm can causally make *only later*
  ReCal3R global-state writes more conservative by adding a fixed bounded
  value to the model's native update-pressure.  The objective is complete only
  when this version has a frozen feasibility decision or a terminal
  pre-CUDA/runtime NO-GO.  A failed gate is evidence, not permission to tune
  or retry the same v15 run ID.

> **Terminal Gate-A decision (2026-08-13).** The one-use CUDA-hidden interface
> probe called the general v15 capsule loader in order to select frame zero.
> That loader revalidated and read the raw `rgb.txt` listing.  Gate A item 5
> limited that probe to the capsule, frame-zero RGB and checkpoint, so the
> evidence exceeds its own pre-registered capability boundary.  The probe's
> immutable output must not be reinterpreted as PASS.  Per Section 5, v15 is
> terminal; do not alter, rerun, dispatch, or otherwise reopen its IDs.  The
> full record is in
> [the v15 Gate-A NO-GO audit](audits/recovery-update-pressure-bounded-state-memory-v15-gate-a-no-go.md).

## 1. Why v15 exists and what it does not reuse

v14 is terminal `V14_IMPLEMENTATION_OR_INPUT_NO_GO`: its capsule builder
computed a full SHA-256 of a quarantined legacy run archive and therefore read
past the permitted top-level `images` array.  See the immutable
[v14 Gate-A audit](audits/recovery-update-pressure-bounded-state-memory-v14-gate-a-no-go.md).

v15 does not repair, import, call, read, hash, parse, or otherwise use the
v14 capsule, legacy run archive, legacy manifest, source manifest, v14
runner/operator/dispatcher, or any v1--v14 recovery component.  It has its
own code graph, capsule, tests, launcher, evidence and run IDs.  The only
unchanged scientific constants are the independently specified native model
equation and the fixed intervention below; they are not selected from v14's
unexecuted response.

## 2. Fixed mechanism and causal boundary

The mechanism is the same narrow hypothesis that v14 never reached: after a
raw current-frame Detector-v3 alarm, and only after ReCal3R's native update,
state/memory commit, calibration record and native reset, write

```text
P <- clamp(P + 0.25, 0.0, 1.0).
```

`P` is only ReCal3R's existing CUDA `update_pressure`; the scalar add/clamp is
elementwise.  If native reset removes it, an alarm creates exactly a CUDA
floating `(1,768,1)` tensor of `0.25`; a clear reset leaves it absent.  The
operator receives only a plain current alarm boolean, existing pressure (or
fixed shape/device/dtype metadata after reset), and fixed constants.  It may
not receive or inspect RGB, prediction, pose, state/memory values, latent,
pointmap, decoder, GT/depth/label/event, source index, timestamps, prior
export, anchor, future data, detector history or CPU tensor values.

The mandatory per-frame order is:

```text
model-ready current/previous RGB overlap
→ native recurrent rollout and raw current prediction
→ native ReCal3R pressure/update mask
→ native state_feat and mem commits
→ native sequence age and calibration record
→ native reset
→ frozen Detector-v3 observes raw current health and commits it
→ candidate alarm only: bounded pressure injection
→ CPU export of unchanged raw current prediction.
```

The alarm frame's raw prediction, raw pose, state feature and memory must have
identical CUDA fingerprints before/after injection.  A later raw prediction or
trajectory difference versus the frozen control is required for candidate
availability.  There is no rollback, replay, row selection, repair, pose
replacement, reset/watchdog, fallback, learned parameter or response tuning.

## 3. Independent GT-free input capability

No quarantined historical artifact is an input at any stage.  The sole source
for v15's dynamic first release is the ordinary read-only raw dataset listing

```text
baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt
```

whose SHA-256 is
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`.
It is neither a manifest nor an archive and contains only RGB timestamps and
relative RGB paths.  An isolated, non-CUDA builder may read this listing in
order, find the exact first permitted path
`rgb/1305031461.059662.png`, and take its next 30 contiguous listed RGB paths.
It must validate the full fixed ordered path-list SHA-256
`de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`,
verify and hash the 30 regular mode-`0444` RGB files, then atomically create
the new mode-`0444` capsule.  It must never read anything under `outputs/`
except its fresh target absence check, and it may not materialize a legacy
archive/manifest or any GT/depth/label/event field.

The capsule strict whitelist is only schema/version/id, dataset root, RGB root,
raw `rgb.txt` identity, loader contract, 30 RGB relative paths and stat/hash
snapshots, and the following hard-coded transform data.  It may not contain a
source index, archive path/hash, manifest, GT, depth, timestamp, condition,
event, label, corruption or future field.

| frames | rectangle x/y | width/height | fill |
| --- | --- | --- | --- |
| 15--19 | `(0.25+.04k, 0.25+.025k)`, `k=0..4` | `.5/.5` | RGB `(255,0,0)` |

Runtime may read only the v15 capsule, its 30 RGB files, the same raw
`rgb.txt`, frozen Detector-v3 configuration and checkpoint.  It loads RGB
only with `dust3r.utils.image.load_images_for_eval(size=512, crop=true,
square_ok=false)`, clones images, and applies the rectangles after resize/crop
using floor-start/ceil-end rounding.  Timestamp ordering is obtained only
through `timestamp_order_v3.capture_timestamp_records(ordered RGB paths,
rgb.txt, dataset root)`.

Gate-B's protected historical trajectory schema includes a legacy
`reference_frame.source_index: 0` field.  v15 may emit that exact literal
*only as a fixed output-schema compatibility constant*: it is not a capsule
field, runtime input, lookup, parsed listing value or operator input.  Tests
must prove the runtime cannot read a source-index capability and that the
literal is required solely for byte identity of the unchanged native control.

## 4. Pins, run IDs and release preconditions

- ReCal3R: clean commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.
- checkpoint: `src/cut3r_512_dpt_4_64.pth`, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
- Detector-v3: `outputs/formal-v3-calibration-0001/formal-config.json`,
  immutable and unchanged.
- GPU: only physical GPU 2 UUID
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`.
- Every release needs two fresh identical-UUID snapshots, at least `12288 MiB`
  free, and no Project2 GPU-2 process.  No data, dependency or weight download
  and no ReCal3R/raw-RGB/checkpoint modification is allowed.

The fixed dynamic one-use IDs are:

```text
recovery-update-pressure-v15-dynamic-always-commit-0001
recovery-update-pressure-v15-dynamic-candidate-0001
```

The control is the only initial release.  Its validated PASS alone authorizes
the candidate.  Candidate failure consumes v15 and forbids wrong/low/GT.  Only
candidate PASS authorizes building independent wrong/low RGB-only capsules;
only after all six outputs freeze can a new isolated CPU evaluator read GT.

## 5. Gate A: required committed non-CUDA proof

Before any CUDA release, all of the following must pass and be committed:

1. Source/import graph audit binds the exact pinned ReCal3R lighter span and
   rejects direct/transitive v1--v14 recovery, legacy-manifest/archive, GT and
   forbidden-operator-input paths.
2. An RGB-list-only builder proof verifies all reads originate from raw
   `rgb.txt`/raw RGB; an instrumented poison fixture makes attempts to access
   any `outputs/` source or a non-whitelisted field fail.  Schema duplicate-key,
   nonfinite, symlink, mutable path, wrong list/order/hash, altered transform,
   path escape and changed RGB/listing tests fail closed.
3. Operator tests prove exact add/cap, four-alarm saturation, absent-pressure
   bootstrap, clear identity, finite floating CUDA shape/dtype/device checks,
   and no CPU tensor serialization.
4. Fake recurrence proves the required order, unconditional clean native
   pressure start for both policies, detector raw-health commit before
   injection, unchanged alarm-frame raw state/memory/pose, reset semantics,
   clear identity and strictly more conservative later native update.
5. A fresh CUDA-hidden interface probe loads only frame zero through the v15
   capsule and checkpoint, blocks all forward/encoder/RoPE/detector/operator
   paths, and records `(1,768,768)`, `(1,256,1536)`, `(1,768,1)`.
6. Its own dispatcher/waiter test matrix covers normal fake child, nonzero
   child, dirty provenance, early failure, pipe failure, missing pre-exec PID,
   PID reuse-safe timeout, dispatcher exception, postflight/inventory and
   fresh CUDA-hidden validator.  The waiter owns an `O_EXCL` terminal log and
   captures both dispatcher streams and exit status.
7. All targeted tests, complete CUDA-hidden CPU suite, `compileall`, shell
   syntax, `git diff --check`, permission/provenance checks pass.  Gate-A PASS
   is committed before an eligible tmux release.

Any failure is `V15_IMPLEMENTATION_OR_INPUT_NO_GO`, forbids all v15 CUDA, and
requires a new version rather than a modified retry.

## 6. One-use tmux protocol and gates B--E

The v15 dispatcher acquires an exclusive owner lease then immediately creates
an `O_EXCL` journal.  It validates clean committed source/capsule/pins, writes
and freezes preflight with two snapshots, opens one detached window in the
existing `stateguard` session, attaches `pipe-pane`, waits for
`V15_PIPE_READY`, and runs a token-gated driver.  The driver publishes its PID
and `/proc` start ticks before it receives one random 256-bit `O_EXCL` go
token; it validates the exact token before setting `CUDA_VISIBLE_DEVICES=2`
and `exec`ing the runner.  The dispatcher drains/freeze streams only after pane
exit, proves the PID/start pair absent without signalling an unverified PID,
rechecks source and GPU state, freezes/output-inventories results, and invokes
a separate fresh `CUDA_VISIBLE_DEVICES=''` validator.  There is no rerun,
repair or postflight-only CLI.

- **Gate B:** one dynamic always-control.  It makes 30 direct native commits,
  constructs no detector/operator/pressure evidence, has no pending work,
  matches v1 byte-for-byte in `checkpoint-load-audit.json`, `health.jsonl`,
  `predictions-summary.json` and `trajectory.json`, and runs no slower than
  `1.20x` v1 (`5.660642675124109` s).  Any failure ends v15.
- **Gate C:** one candidate.  It requires a real alarm; raw-current fingerprints
  unchanged across every pressure injection; detector-before-injection and
  always raw-health commit witnesses; no fallback/reset/watchdog/provenance
  failure; a changed later raw trajectory/prediction; and runtime at most
  `1.20x`.  Failure is `V15_AVAILABILITY_OR_RUNTIME_NO_GO` and forbids
  wrong/low/GT.
- **Gate D:** only after C PASS, one control/candidate for each independently
  built wrong and low RGB-only condition, with all six outputs frozen.
- **Gate E:** only then an isolated CPU evaluator may read GT.  It passes only
  if two of three conditions jointly show positive tail ATE and translation
  RPE, both medians improve at least five percent, and median candidate runtime
  is at most `1.20x`; otherwise freeze `V15_FEASIBILITY_NO_GO` with no tuning.
