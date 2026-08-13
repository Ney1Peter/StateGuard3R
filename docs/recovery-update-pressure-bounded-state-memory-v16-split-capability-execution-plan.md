# v16 bounded update-pressure with split RGB capabilities: execution plan

- Date: 2026-08-13
- Status: **pre-registration — no v16 CUDA run is authorized until a committed
  Gate-A PASS.**
- Explicit objective (12--18 hours of active work): make one falsifiable,
  GT-free test of whether a Detector-v3 alarm can make *only later* ReCal3R
  global-state writes more conservative by a fixed bounded native-pressure
  increment.  This version is complete only with a frozen feasibility decision
  or a terminal pre-CUDA/runtime NO-GO.  A failed gate consumes its ID; it is
  never permission to tune, repair, or rerun that ID.

## 1. Why v16 is independent

v15 is terminal `V15_IMPLEMENTATION_OR_INPUT_NO_GO`: its one-use interface
probe reached the general RGB-list capsule loader, which re-read `rgb.txt`.
That exceeded the probe's pre-registered `capsule -> frame-zero RGB ->
checkpoint` capability boundary.  See the immutable
[v15 Gate-A NO-GO audit](audits/recovery-update-pressure-bounded-state-memory-v15-gate-a-no-go.md).

v16 is not a repair or rerun of v15.  It owns new code, schemas, builders,
capsules, probe, runner, dispatcher, tests, logs and IDs.  It must not import,
call, read, hash, parse, inspect or use any v1--v15 recovery/input component,
run, manifest, archive, capsule, probe evidence, dispatcher, output or source
provenance.  In particular, no v16 code may import a module whose name contains
`v15`, `v14`, `recovery`, `manifest`, `archive`, `quarantine`, `pointmap`,
`anchor`, `registration`, `prerollout`, `decoder`, `spatial`, `encoder` or
`patch`.  The only repeated scientific constants are independently specified
native ReCal3R interfaces and a fixed equation; neither was selected using an
unexecuted v15 outcome.

## 2. Fixed hypothesis, operator and causal order

After a raw current-frame Detector-v3 alarm, and only after native recurrent
work is complete, write elementwise on CUDA:

```text
P <- clamp(P + 0.25, 0.0, 1.0)
```

`P` is ReCal3R's existing `(1,768,1)` floating `update_pressure`.  If native
reset removed it, an alarm creates exactly an all-`0.25` CUDA tensor of that
shape, device and native dtype.  A clear frame returns exact identity and
creates nothing.  The operator may receive only a plain current boolean,
existing pressure (or fixed shape/device/dtype metadata after reset), and
`0.25`/`1.0`; it must reject non-CUDA/nonfinite/wrong-shape/wrong-dtype input
and cannot receive RGB, raw prediction, pose, state/memory values, latent,
pointmap, detector history, timestamps, index, GT/depth/label/event/future
data or a CPU tensor value.

For every frame the required order is:

```text
model-ready current/previous RGB overlap
→ native recurrent rollout and raw current prediction
→ native ReCal3R pressure/update mask
→ native state_feat and mem commits
→ native sequence age and calibration record
→ native reset
→ frozen Detector-v3 observes and commits raw current health
→ candidate alarm only: bounded pressure injection
→ CPU export of unchanged raw current prediction.
```

The alarm frame's raw prediction, state, memory and pose must have identical
CUDA fingerprints before/after injection.  Candidate availability requires a
real alarm and a changed later raw prediction or trajectory compared with the
frozen control.  There is no rollback/replay, row selection, repair, pose
replacement, reset/watchdog/fallback, learned parameter or threshold tuning.

## 3. Two non-interchangeable RGB capabilities

No historical experiment artifact is an input at any stage.  Both v16
builders may read only the ordinary read-only raw listing

```text
baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt
```

whose SHA-256 is
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`,
and the raw RGB files selected by that listing.  They may inspect only their
own fresh output target for absence, never any existing `outputs/` content.
They create these separate, `0444` one-use files:

| capability | fixed ID | permitted consumer and file reads |
| --- | --- | --- |
| frame-zero probe capability | `recovery-update-pressure-v16-dynamic-frame0-probe-capsule-0001` | only the Gate-A probe; its data reads must be exactly this capsule, raw frame-zero RGB, and checkpoint |
| dynamic runtime capability | `recovery-update-pressure-v16-dynamic-rgb-list-capsule-0001` | only dynamic runner; capsule, 30 listed RGB files, same raw `rgb.txt`, Detector-v3 config and checkpoint |

The raw selector is independently fixed: find
`rgb/1305031461.059662.png` and take its next 30 listed RGB paths.  It verifies
the ordered path-list SHA-256
`de0d7506a0578704410f7b4c25ad760d4260f20533648aee17cd36545fdfe120`; all
selected files must be regular, non-symlink, mode `0444`, and are bound by
size/inode/mtime/SHA-256.  The runtime capability contains only schema/ID,
dataset/RGB roots, raw-list identity, loader contract, selected relative paths
and file snapshots, and the exact transforms below.  It cannot contain an
index, historical path/hash, manifest/archive, GT/depth/timestamp/condition,
event/label/corruption/future field.

The distinct probe capability contains only schema/ID, dataset/RGB roots,
loader contract, *one* `frame_id: 0` RGB relative path and that raw file's
stat/hash snapshot.  It deliberately contains no listing location/identity,
no 30-frame list, no transform, timestamp, index or hidden fallback.  Its
parser must not import the dynamic-capability module or read a raw listing.

| frames | rectangle x/y | width/height | fill |
| --- | --- | --- | --- |
| 15--19 | `(0.25+.04k, 0.25+.025k)`, `k=0..4` | `.5/.5` | RGB `(255,0,0)` |

Runtime uses only `dust3r.utils.image.load_images_for_eval(size=512,
crop=true, square_ok=false)`, clones RGB tensors, then applies the rectangles
after resize/crop with floor-start/ceil-end rounding.  It derives timestamp
ordering only from its own narrow v16 raw-listing parser, whose sole inputs are
the ordered runtime RGB paths, `rgb.txt` and dataset root.  The fixed literal
`reference_frame.source_index: 0` is allowed only as output-schema
compatibility text; it cannot be a capsule/runtime/operator input.

## 4. Pins and one-use run IDs

- ReCal3R: clean commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.
- checkpoint: `src/cut3r_512_dpt_4_64.pth`, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
- frozen Detector-v3 config:
  `outputs/formal-v3-calibration-0001/formal-config.json` unchanged.
- GPU: only physical GPU 2 UUID
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`.
- No data, weight or dependency download; no ReCal3R/checkpoint/raw-RGB/listing
  modification.

Every actual release needs two fresh matching UUID snapshots, at least
`12288 MiB` free in each and no Project2 GPU-2 process.  Long GPU work can
exist only in a new window of the existing `stateguard` tmux session.

```text
recovery-update-pressure-v16-dynamic-always-commit-0001
recovery-update-pressure-v16-dynamic-candidate-0001
```

Only a validated PASS control authorizes the candidate.  Candidate failure
consumes v16 and forbids wrong/low/GT.  Only a candidate PASS authorizes
separately built v16 wrong/low capabilities; only after all six outputs freeze
may a new isolated CPU evaluator read GT.

## 5. Gate A — required committed non-CUDA proof

All items below must pass, be recorded and be committed before any v16 CUDA or
tmux release:

1. A static/transitive import and capability audit binds the exact pinned
   ReCal3R lighter span and rejects v1--v15, legacy input/archive/manifest,
   GT and prohibited operator paths.  It walks local v16 imports rather than
   relying on filename filtering alone.
2. Independent builders have poisoned-reader tests: dynamic build reads only
   raw listing plus its 30 RGBs; probe build reads only raw listing plus frame
   zero.  Duplicate-key/nonfinite/symlink/path escape/wrong hash/order/altered
   transform/mutable input tests fail closed.
3. The pressure operator proves exact add/cap, saturation, reset bootstrap,
   clear identity, CUDA floating shape/dtype/device/finite checks and no CPU
   serialization.  Fake recurrence proves unconditional clean native start,
   detector commit before injection, reset semantics, unchanged alarm-frame
   raw tensors and a strictly more conservative later native update.
4. A **fresh one-use CUDA-hidden probe** is executed with
   `CUDA_VISIBLE_DEVICES=''`.  Its actual data-access ledger, enforced by
   instrumented reader tests and output evidence, has exactly three entries:
   probe capsule, frame-zero RGB and checkpoint.  It loads no listing or
   dynamic capsule, blocks all forward/encoder/RoPE/detector/operator routes,
   never initializes CUDA, and binds `(1,768,768)`, `(1,256,1536)`,
   `(1,768,1)`.  Its `O_EXCL` `0444` evidence is never rerun.
5. An own v16 dispatcher/waiter covers normal fake child, child nonzero,
   source-dirty, early failure, pipe failure, missing pre-exec PID,
   PID-reuse-safe timeout, dispatcher exception, postflight/inventory/freeze
   and a separate `CUDA_VISIBLE_DEVICES=''` validator.  It owns O_EXCL
   journal/logs, live `pipe-pane`/`V16_PIPE_READY`, random 256-bit token,
   PID plus `/proc` start ticks, source/GPU rechecks and no unverified signal.
6. Targeted tests, the complete CUDA-hidden CPU suite, `compileall`, shell
   syntax, `git diff --check`, pin/permission/provenance and frozen-output
   checks pass.  The StateGuard worktree may retain only a documented exact
   v14/v15 untracked terminal set; all v16 pinned files must be committed HEAD
   bytes and ReCal3R must be clean.

Any failure is `V16_IMPLEMENTATION_OR_INPUT_NO_GO`, forbids all v16 CUDA and
requires a newly numbered version rather than a modified retry.

## 6. One-use tmux protocol and Gates B--E

The dispatcher obtains its exclusive owner lease, then immediately creates an
`O_EXCL` journal.  It validates committed source/capsules/pins and writes two
frozen preflights; opens one detached `stateguard` pane; attaches `pipe-pane`;
waits for `V16_PIPE_READY`; and launches a token-gated driver.  The driver
publishes PID and `/proc` start ticks before it receives exactly one random
256-bit `O_EXCL` token.  It validates the complete token before setting
`CUDA_VISIBLE_DEVICES=2` and `exec`ing the runner.  The dispatcher drains and
freezes streams after pane exit; verifies the matching PID/start pair absent
without signalling an unverified PID; rechecks source/GPU; freezes and
inventories output; then invokes a fresh CUDA-hidden validator.  There is no
rerun, repair or postflight-only CLI.

- **Gate B:** launch exactly one dynamic always-control.  It makes 30 direct
  native commits, constructs no detector/operator/pressure evidence, has no
  pending work, and must match the pre-registered protected SHA-256 constants
  for `checkpoint-load-audit.json`, `health.jsonl`,
  `predictions-summary.json`, `trajectory.json` without reading any historical
  control artifact.  Runtime must be at most `1.20 × 5.660642675124109` s.
  Any failure ends v16.
- **Gate C:** only after B PASS, launch one candidate.  It requires a real
  alarm, unchanged raw-current fingerprints for every injection,
  detector-before-injection/always-commit witnesses, no fallback/reset/
  watchdog/provenance failure, a changed later raw trajectory/prediction, and
  the same runtime limit.  Failure is `V16_AVAILABILITY_OR_RUNTIME_NO_GO` and
  forbids wrong/low/GT.
- **Gate D:** only after C PASS, independently build v16 wrong/low RGB
  capabilities, then run one control/candidate each and freeze all six outputs.
- **Gate E:** only then a fresh isolated CPU evaluator may read GT.  It passes
  only if two of three conditions jointly have positive tail ATE and
  translation RPE, both medians improve at least five percent, and median
  candidate runtime is at most `1.20x`; otherwise freeze
  `V16_FEASIBILITY_NO_GO` without tuning.
