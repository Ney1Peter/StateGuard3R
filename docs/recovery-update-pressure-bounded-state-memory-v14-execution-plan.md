# v14 bounded update-pressure: execution plan

- Date: 2026-08-13
- Status: **terminal Gate-A NO-GO —
  `V14_IMPLEMENTATION_OR_INPUT_NO_GO`.  The quarantined-archive builder reads
  the entire legacy `run.json` to calculate its SHA-256 before extracting the
  `images` projection, contradicting the required no-read-past-delimiter
  boundary.  No v14 CUDA run is authorized.**  See [the Gate-A NO-GO
  audit](audits/recovery-update-pressure-bounded-state-memory-v14-gate-a-no-go.md).
- Long-horizon objective (12--18 hours of active work): determine, without
  tuning on a response, whether a causal alarm can safely make **only future
  ReCal3R global-state writes** more conservative by injecting bounded native
  update pressure.  A version failure is a terminal v14 result, not permission
  to alter v14 and retry it.

## 1. Research hypothesis and mechanism boundary

The fixed hypothesis is that an alarm frame can increase the model's existing
write-pressure accumulator, so that later ReCal3R state updates are attenuated
while the alarm frame itself remains a completely native output.  This tests a
minimal *post-alarm write-pressure* response rather than rollback, replay,
pose correction, state/memory-row restoration, or a learned detector.

For an alarm at frame `t`, after the native update, calibration, and native
reset have completed, let the existing ReCal3R pressure tensor be
`P_native`.  The only candidate action is

```text
P_v14 = clamp(P_native + 0.25, 0.0, 1.0).
```

The addition and clamp are elementwise.  If a native reset removed the
pressure tensor, an alarm creates a CUDA floating tensor of the fixed bound
shape `(1, 768, 1)` and value `0.25`; a clear frame after reset leaves the
attribute absent.  The fixed increment (`0.25`) and cap (`1.0`) are not tuned
or changed after any v14 execution.  ReCal3R's next normal update consequently
uses its pinned equation

```text
P_(t+1) = 0.95 * P_v14 + 0.05 * alignment_gate_(t+1)
attenuation_(t+1) = exp(-P_(t+1)).
```

The v14 operator receives only the current detector boolean, an existing
`update_pressure` tensor (or its fixed shape/device/dtype metadata after
reset), and the two fixed constants.  It must not receive or inspect a raw
prediction, camera pose, RGB, state feature, memory, decoder/attention latent,
pointmap, GT/depth/label/event/source index, prior exported pose, anchor,
future frame, baseline alarm artifact, or CPU state.  It does not clone or
change `state_feat`, `mem`, `state_pos`, init/reset tensors, calibration trace,
reference state, RNG, detector state, or the current `res`.

This is deliberately distinct from v1--v13: no full discard/replay or
closure rollback; no anchor/registration/pointmap/query/decoder/encoder/patch
pose export; no selective row ranking or repair; no row or top-k policy; and
no fallback.  In particular v14 must not import or call any v1--v13 recovery
runner, operator, exporter, script, dispatcher, launcher, `QuarantineWatchdog`
or `run_recal3r_smoke` helper.  Narrow, independently audited utilities may be
used only for frozen Detector-v3, health serialization, RGB timestamp binding,
and ordinary atomic file handling.

## 2. Fixed causal order

Every frame follows this exact order:

```text
model-ready RGB overlap for current/previous frame
→ native rollout and raw current prediction
→ native ReCal3R update-pressure EMA/update mask
→ native state_feat and mem commits
→ native sequence-age and calibration record
→ native reset (including native pressure deletion when requested)
→ Detector-v3 observes raw current health and decides
→ Detector-v3 commits that raw health (never quarantines it)
→ candidate alarm only: v14 pressure injection
→ CPU transfer/export of the unchanged raw current prediction.
```

The frozen detector may continue to derive its existing raw-health fields,
including pose jump, from raw current/past predictions and causal RGB overlap.
That permission does not extend to the v14 pressure operator: raw pose/RGB can
flow only into the unchanged health/digest adapters, never into pressure,
fallback, or output replacement.  The candidate must prove that the raw
current pose, state feature, and memory GPU fingerprints are unchanged across
the injection.  A later raw prediction/trajectory difference from control is
required to prove an actual causal future effect.

## 3. GT-safe RGB-only input capsule

The legacy corruption manifests and archived `run.json` files contain
ground-truth/depth metadata and are **not runtime inputs**.  v14 runtime may
read only a new immutable RGB-only capsule, its thirty RGB files, the original
read-only `rgb.txt`, the frozen detector configuration, and the existing
checkpoint.  It may not import `input_manifest`, call its loader, or read a
legacy manifest/source-manifest/run JSON.

No standalone safe capsule currently exists.  Before any model work, a new
non-CUDA one-time capsule builder may perform a deliberately narrow archival
projection from each frozen v1 always-control `run.json`.  It must:

1. require the archive to be a non-symlink, mode-`0444` file and bind its
   SHA-256;
2. lexically locate and stream-parse only the top-level `images` JSON array;
   stop at that array's closing delimiter and never call whole-file
   `json.loads`, parse `input_frames`, or materialize any suffix;
3. validate the fixed projection digest and independently stat/hash the RGB
   files; and
4. write an atomic, new, mode-`0444` capsule with a strict whitelist and no
   archive/manifest path, GT, depth, source index, timestamp, condition,
   corruption, event, label, or future field.

This limited projection is the only explicitly allowed contact with the
quarantined historical archive.  A poisoned test must make any read beyond the
`images` delimiter fail.  If that construction cannot prove it stopped before
the suffix, Gate 0 is `V14_IMPLEMENTATION_OR_INPUT_NO_GO`; the runner does not
fall back to an old file.

For the dynamic first-release capsule, the archival source is
`outputs/recovery-policy-development-v1-dynamic-always-commit-0001/run.json`
(SHA-256 `59121c5525d3ab3a33ece85e530b36edd140d0f66c0904242b974116f8284643`),
whose top-level thirty-RGB projection SHA-256 is
`68c5684246d92116ee56488abec22c63e33fe9d0ec015d616d25ec9c3b26e285`.
The raw `rgb.txt` SHA-256 is
`d1bc510ecca08540e03be8df55af8857753b614d8a5c38526bc669ce6c802284`.
The capsule hard-codes only these five post-loader rectangles:

| frame | x | y | width | height | fill |
| ---: | ---: | ---: | ---: | ---: | --- |
| 15 | .25 | .25 | .5 | .5 | RGB 255,0,0 |
| 16 | .29 | .275 | .5 | .5 | RGB 255,0,0 |
| 17 | .33 | .30 | .5 | .5 | RGB 255,0,0 |
| 18 | .37 | .325 | .5 | .5 | RGB 255,0,0 |
| 19 | .41 | .35 | .5 | .5 | RGB 255,0,0 |

The only accepted loader is `dust3r.utils.image.load_images_for_eval` with
`size=512`, `crop=true`, `square_ok=false`; a clone receives each rectangle in
`model_input_after_resize_and_center_crop` coordinates using floor start and
ceil end rounding.  `timestamp_order_v3.capture_timestamp_records` is the
only shared timestamp interface, because it accepts only ordered RGB paths,
`rgb.txt`, and dataset root.

## 4. Immutable resources and run IDs

- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, clean tree.
- checkpoint: `src/cut3r_512_dpt_4_64.pth`, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
- frozen Detector-v3 configuration:
  `outputs/formal-v3-calibration-0001/formal-config.json`, unchanged.
- GPU: physical GPU 2, UUID
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`; two identical-UUID snapshots
  with at least 12288 MiB free and no Project2 process are required before
  each release.  No download, weight/dependency/data acquisition, ReCal3R
  modification, or raw RGB modification is allowed.

The one-use dynamic IDs are:

```text
recovery-update-pressure-v14-dynamic-always-commit-0001
recovery-update-pressure-v14-dynamic-candidate-0001
```

Only a validated control authorizes the candidate.  If that candidate passes,
the separately built wrong/low RGB-only capsules authorize exactly one control
and one candidate per condition; no GT may be read until all six outputs are
frozen.  Each candidate or control failure consumes its ID and is terminal for
v14; no cleanup, reissue, tuning, or retry is allowed.

## 5. Gate A — implementation and non-CUDA contract

Gate A must be committed before a CUDA release.  It requires all of:

1. source/import audit binding the pinned ReCal3R lighter order and the new
   v14 own-code graph; direct and transitive previous recovery imports, old
   manifest APIs, GT tokens, and prohibited operator inputs fail;
2. builder/capsule poison, whitelist, hash, path/symlink/read-only, transform,
   RGB-loader, tensor-clone, and timestamp-only tests;
3. operator tests for exact add/cap, absent-pressure reset bootstrap,
   clear identity, finite floating CUDA/shape/dtype/device contracts, and no
   CPU tensor serialization;
4. fake-recurrent tests for detector-before-injection, always raw detector
   commit, unchanged alarm-frame raw output/state/memory, reset semantics, and
   a strictly more conservative later native update;
5. a CUDA-hidden CPU interface probe that loads only the checkpoint and
   capsule frame zero to bind `(1,768,768)`, `(1,256,1536)`, and
   `(1,768,1)`, without encoder/RoPE/forward/detector/operator; and
6. dispatcher/waiter tests for normal, child nonzero, early provenance failure,
   dirty tree, pipe failure, no PID pre-exec, timeout, and dispatcher-error
   paths, plus targeted tests, full CPU suite, compileall, diff check, clean
   provenance, and permission checks.

Any failure writes `V14_IMPLEMENTATION_OR_INPUT_NO_GO` and forbids v14 CUDA.

## 6. One-use tmux protocol and gates B--E

The new v14 dispatcher first creates an `O_EXCL` dispatch journal immediately
after its owner lease, appends every stage and caught exception, then freezes
it in `finally`.  The waiter captures both dispatcher streams, unconditionally
records its exit code, and exits nonzero on dispatcher failure.  Before a pane
can run a payload, the dispatcher validates provenance/capsule hashes and
writes/freeze preflight; creates a new detached pane in existing `stateguard`,
waits for `V14_PIPE_READY`, proves driver PID plus `/proc` start ticks, then
writes a 256-bit one-use go token.  The payload validates that exact token
before CUDA.  It drains and freezes transcript/logs, verifies the PID/start
pair is absent without signalling any PID, checks source hashes and GPU
postflight, and invokes a fresh CUDA-hidden validator.

- **Gate B:** one dynamic always-control.  It must create neither detector nor
  v14 operator/pressure evidence, make 30 direct native commits, have no
  pending work, match v1 byte-for-byte in `checkpoint-load-audit.json`,
  `health.jsonl`, `predictions-summary.json`, and `trajectory.json`, and run
  at most `1.20x` v1 runtime.  Any failure ends v14.
- **Gate C:** one dynamic candidate.  It needs at least one real alarm, every
  raw-current/pressure witness, detector-before-change/always-commit proof,
  no fallback/reset/watchdog/provenance error, a changed later raw trajectory
  or prediction, and runtime at most `1.20x`.  Any failure is
  `V14_AVAILABILITY_OR_RUNTIME_NO_GO` and forbids wrong/low/GT.
- **Gate D:** only after C PASS, one control/candidate each for wrong and low,
  with the same evidence and runtime limit.  All six outputs must freeze.
- **Gate E:** only after D PASS, a new isolated CPU evaluator may read GT.
  It passes only with no causal/protocol failures, jointly positive tail ATE
  and translation-RPE in at least two of three conditions, both median
  improvements at least five percent, and median candidate runtime at most
  `1.20x`.  PASS writes only an unexecuted new-data plan; otherwise it records
  immutable `V14_FEASIBILITY_NO_GO` without tuning or a second run.
