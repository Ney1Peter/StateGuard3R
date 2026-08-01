# Formal detection pilot v1 protocol

Status: **pre-registered; no development or holdout forward from these windows
may start until the implementation, input, and provenance gates below pass and
are committed.**

This protocol replaces the scientific use of the single-run v0 formal flag.
The v0 CLI remains an exploratory compatibility interface.  The v1 suite is a
small, within-sequence pilot on one official TUM `fr1_desk` recording; it does
not establish cross-scene or cross-dataset generalization.

## Scope and fixed exclusions

- Use only the already present, read-only `rgbd_dataset_freiburg1_desk` raw
  tree.  Do not download another dataset and do not create a 50-frame run.
- RGB indices are zero-based indices among non-comment entries of `rgb.txt`.
- RGB indices 17--46 are a permanent exploratory exclusion because their clean
  and three corruption responses have already been inspected.
- Development and holdout use frame-disjoint windows from the same sequence.
  They are separated by 90 unused RGB indices: development ends at 409 and
  holdout begins at 500 (approximately 3.031849 seconds in this recording).
- Selection used only index/time and a ground-truth-pose distance proxy.
  Actual image overlap was not measured.  This limitation must accompany every
  result.

The frozen source allocation is:

| Split | Corruption | Base RGB indices | Donor RGB indices | Output positions |
| --- | --- | --- | --- | --- |
| development | dynamic occlusion | 248--277 | none | 0--29 |
| development | wrong order | 320--349 | none | 0--29 |
| development | low-overlap proxy | 380--409 | 300--304 | 0--29 |
| holdout | dynamic occlusion | 520--549 | none | 0--29 |
| holdout | wrong order | 550--579 | none | 0--29 |
| holdout | low-overlap proxy | 583--612 | 500--504 | 0--29 |

Each base has exactly 30 frames.  A low-overlap source manifest has a 35-frame
source pool: base frames at source-pool positions 0--29, donor frames at
positions 30--34, and `output_frame_count = 30`.  Donors must come only from
the tail pool and replace output positions 15--19.  The five displaced base
frames are not model inputs, but remain committed through the source-manifest
hash.

Before any forward, the builder must reject overlap among all six complete
base/donor allocations.  It must compare resolved path, device/inode, raw RGB
source index, and file SHA-256.  It must also verify that the independently
nearest depth and ground-truth associations are unique within and across the
six allocations and are at most 0.02 seconds from their RGB timestamp.

## Frozen corruption construction

Every output is exactly 30 frames and contains one enabled corruption:

- `dynamic_occlusion`: positions 15--19 inclusive; normalized rectangle
  `(x=0.25, y=0.25, width=0.5, height=0.5)`, RGB fill `[255, 0, 0]`, and
  per-frame velocity `(dx=0.04, dy=0.025)`;
- `wrong_order_segment`: positions 15--18 inclusive, mode `reverse`;
- `low_overlap_jump`: positions 15--19 inclusive, using source-pool positions
  30--34 in order.

Corruption construction seed is 0.  It is distinct from the detection random
baseline seed policy.  The same corruption severity is used on development
and holdout.  “Low overlap” remains a ground-truth-pose/index proxy, not a
measured image-overlap claim.

The generator and strict loader must agree on all of the following before an
input can be frozen: source/output frame counts, label interval, transform
type, donor source range, per-frame replacement source index, resolved source
file, and exact content hash.  A donor from the base prefix, an unchanged donor
alias, or a label/frame provenance mismatch is an error.

## Pre-forward commitments

All six source and input manifests are generated before any development
forward.  The holdout inputs may be structurally and pixel-replayed on CPU, but
no ReCal3R response from them may be computed or inspected yet.

The split registry commits, for each run:

- fixed run ID, split, and corruption type;
- source/input/corruption exact-byte SHA-256 and byte size;
- the ordered source indices and the unique set of actually consumed RGB
  SHA-256 values, including low-overlap donors;
- source-manifest SHA, raw ground-truth SHA, and the RGB/depth/GT association
  metadata already embedded in the input manifest.

A separate canonical holdout commitment contains the holdout projection of
the same registry.  The registry and holdout commitment are written with
default no-overwrite semantics, hashed, changed to mode `0444`, and placed in
a mode `0555` directory.  The six source/input manifests receive the same
read-only freeze.  CPU validation runs with CUDA hidden and records before/after
raw path, size, SHA-256, mtime, mode, and link count.  Raw files and GT must be
unchanged.

The commitment command does not accept a pre-generated validation report.  It
must launch the tracked `scripts/validate_formal_pilot_inputs.py` itself with
the pinned ReCal3R virtual-environment interpreter and
`CUDA_VISIBLE_DEVICES=''`, inside a private no-overwrite staging directory.  A
PASS report must prove exactly 385 distinct source artifacts, the exact
24-entry immutable input tree, independently reconstructed nearest-timestamp
RGB/depth/ground-truth associations for all 190 allocated RGB entries, the
official image-loader path and bytes, the interpreter and both repository
revisions, and CUDA remaining uninitialized.  The report is then copied by
exact bytes into the same atomically published commitment tree.  Public test
injection hooks or an externally supplied look-alike PASS report are not
formal evidence.

The input/protocol gate is committed before development GPU execution.  Git
does not track the manifests, ledgers, raw data, logs, checkpoint, or generated
metrics.

## Realized input freeze 0001

The manifest-only input build at
`outputs/formal-v1-inputs-0001` completed before any formal GPU forward.  It
was generated from StateGuard3R commit
`56bb55bb5fee7e66d2d9fd8c917c67262a347939`; the tracked
`scripts/prepare_formal_pilot_inputs.py` blob had SHA-256
`f895a0a47f3fb1f13b883fa646cc68a9dc92434021fd80fb803c9e07526fef23`.
The tree contains only the root, eight directories, and fifteen JSON files
(24 nodes in total): every directory is mode `0555`, every file is mode
`0444`, and there are no copied or linked images.

The realized top-level artifacts are:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `formal-pilot-manifest.json` | 491575 | `1de55c281a780002d8961fd5a05cb507fd1168804aaf1c493c2e48b459d77932` |
| `split-registry.json` | 314466 | `f07cc9661b47ff42a89e30151f72b48c3855309a5d74f6af5d47524a4891bd9d` |
| `holdout-commitment.json` | 157262 | `f146161aeec23ed5174969a806f3fc55e701a28c37c01695f33e46267f2dc8c7` |

Each input manifest is also the exact corruption JSON consumed by the strict
loader.  The six realized run artifacts are:

| Run | Source manifest (bytes, SHA-256) | Input/corruption manifest (bytes, SHA-256) |
| --- | --- | --- |
| `development-dynamic` | 76416, `61248aa0a417102b104109eb2c2e57a5841878203f40fab4be6a52358b7c4284` | 84689, `264b0c9e4f04f4da9e26a634ed008342d32c7b561d7b6ee4f9eb90d1688189c8` |
| `development-wrong` | 76468, `a61efdb96b2f8740ac5839e415ff9211e015bcdd92a509ff81d034fcde375281` | 82571, `860c45ba688da966f01df559b718e33a318352673c65d8787645b563d7e8f669` |
| `development-low` | 88676, `f446e7eb9a518847dec6dd53dcf93d55fcf3b6a2af184b427d66d3a6bd823fd7` | 82754, `72eeb1dc297992b14a6a3bd5e166832a5220ee16df82dfc9d39aefec25aec0c0` |
| `holdout-dynamic` | 76338, `575fdd90a46a529897f70728b10e616c1584a4481d20716af2aa9028e1aefae1` | 84615, `328ec2387ddffa38f606a73ed6b19e61742f9d2f17f76395bfe46dbad6d83405` |
| `holdout-wrong` | 76375, `72b82efa8e786a7aabd61746bcfe56e014ff187a8243da141afa99df25912ab6` | 82482, `d3bf2a7c0069e51e410d0b1591423058a7118c922a2f78c8d8eed3fc6ae45ba0` |
| `holdout-low` | 88675, `32454242412ed2db5caf339b8bf085b577026496112a83f9abf7ab1a7a134ef1` | 82745, `5ddcfd0bfc8682dd3b727419c70ded22b2443758c54ba2053d93ccc27704d556` |

The build binds the official archive (344011403 bytes, SHA-256
`e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`)
and raw manifest (231139 bytes, SHA-256
`5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`).
Its complete-allocation proof contains 190 unique RGB, 190 unique depth, and
190 unique ground-truth identities; all 15 run-pair intersections are zero
for every identity class.  The actual-model-input projection contains 180
unique identities in each class.  Independent Decimal timestamp replay found
190 RGB and 380 depth/ground-truth associations with no mismatch.  Depth
deltas span `[-0.017879, -0.001605]` seconds and ground-truth deltas span
`[-0.004740, 0.004153]` seconds, within the frozen 0.02-second bound.
The validator's canonical-JSON SHA-256 digests are
`7cc9051daaf2cf1387b4ffe22ae404f04f0b38ae9b87fb24af99d8e8422be2da`
for the 24-node input tree,
`850d2da4dee6ecb0a9595fb873aece5f31429056fac56baddcadc1a5cacf9348`
for the 385 distinct raw-source snapshots, and
`2661cf1bed308ad64c56b8e18453cae65edd788cbfff8bd4433b54e4bbb336d1`
for the 190 independently reconstructed association records.

The two low-overlap constructions remain pose proxies only.  For
`development-low`, the five donor/base pairs have translation separation
minimum/mean/maximum `1.5253540015353815 / 1.561770178919186 /
1.59426172882623` metres and rotation separation
`1.3860445738485307 / 1.4042530664398112 / 1.427665129672411` radians.  For
`holdout-low`, the corresponding values are
`0.68750873448997 / 0.7055181187504569 / 0.7236794179745614` metres and
`0.6730589010037727 / 0.6767409395649828 / 0.6794784641139863` radians.
These values do not measure image overlap and support no overlap-percentage
claim.

The forthcoming runtime remains pinned to ReCal3R commit
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, runner SHA-256
`091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`,
and the 3173761006-byte checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
At this freeze the scientific status remains **HOLD — evidence pending**: no
formal development or holdout response has been generated or inspected.

## Fixed execution order and resource gate

The exact run IDs and global forward order are:

```text
development-dynamic
development-wrong
development-low
holdout-dynamic
holdout-wrong
holdout-low
```

The immutable commitment records a UTC execution-not-before time.  Development
run metadata must prove that all three runs started after it and ran serially
in the listed order.  Calibration atomically freezes `search.json`,
`dev-metrics.json`, `formal-config.json`, and its manifest, then publishes a
separate immutable holdout-unlock artifact bound to that calibration manifest.
Holdout run metadata must prove that all three runs started after that unlock
and ran serially in the listed order.

Immediately before each launch, capture two GPU/process snapshots.  A selected
GPU may be shared as authorized by the user, but it must have at least 12 GiB
free in both snapshots.  Bind exactly one physical GPU with
`CUDA_VISIBLE_DEVICES`; never stop, move, or modify another user's process.
Record command, PID, GPU, log path, output directory, start/end time, runtime,
peak memory, exit status, and post-exit GPU release.  Runs are sequential and
outputs are new no-overwrite directories.

Before a run can be consumed, its directory is frozen to mode `0555` and must
contain exactly five regular, non-symlink mode-`0444` files:
`checkpoint-load-audit.json`, `health.jsonl`, `predictions-summary.json`,
`trajectory.json`, and `run.json`.  Calibration and evaluation bind the exact
bytes of all five files even though detection scores consume only the health
ledger and run metadata.

A development run may be retried only for an operational failure, with the
same frozen input and command.  No failed or successful response may be used
to change the split or corruption.  Holdout forward stays locked until the
development search result and formal configuration have been frozen.  Once
unlocked, all three holdout forwards are completed and frozen before any
holdout score or metric is inspected.

## Causal scoring and evaluation regions

All 30 frames participate in ReCal3R forward and in later causal references.
Each run is scored independently; ledgers must never be concatenated before
computing a rolling reference.  For position `t`, robust references use only
finite positions `[max(0, t-K), t)`.  Masking is applied only after every score
has been computed.

The event starts at position 15.  Its inclusive end is 19 for dynamic and
low-overlap, and 18 for wrong-order.  Let `W = 5`:

- primary positives are the exact enabled corruption interval;
- recovery boundary is `event_end + 1`;
- washout is `event_end + 1` through `event_end + 5`, inclusive;
- primary metrics exclude washout and include startup positions 0--14 as
  negatives;
- exact-interval sensitivity treats every label-external frame, including
  boundary/washout, as a negative;
- boundary alarm and all washout alarms are reported separately.

Across one three-run split, the primary pooled counts are fixed at 14 positive,
15 ignored washout, and 61 negative frames.  Exact-interval sensitivity has 14
positive and 76 negative frames.  The primary false-positive denominator is
the 61 finite evaluated negatives.  Longest consecutive false positives are
computed per run in original output position order; an ignored washout frame
breaks a primary false-positive streak.  Report the maximum across runs.

For each method report per-run, equal-corruption macro, and pooled AUROC, F1,
false-positive rate, confusion counts, event detection, and delay.  Delay is
the first threshold crossing inside the primary interval minus event start;
an alarm only at the recovery boundary is not a detection.  Also report exact-
interval sensitivity metrics, boundary/washout alarms, signal aliases actually
used, a 30-position score timeline, and run runtime/peak memory.

## Frozen development search

The four methods are seeded random, update-magnitude only, reliability only,
and combined health.  Combined scoring retains the v0 unit-weight definition.
Unavailable overlap contributes nothing and must remain recorded as missing.

`master_seed = 0`.  A random-baseline seed is independently derived from the
UTF-8 tuple `(master_seed, dataset_split, run_id)` using the first unsigned
big-endian 64 bits of SHA-256.  The derived value is recorded per run.

The ordered candidate grid is the Cartesian product below, with `window` as
the outer loop, `epsilon` as the middle loop, and `max_z` as the inner loop:

```text
window  = [5, 10, 15]
epsilon = [1e-6, 1e-5, 1e-4]
max_z   = [10, 20, null]
```

For every candidate and method, thresholds are the sorted unique finite
development primary-mask scores plus `nextafter(max_score, +inf)`.  Select a
method threshold by:

1. highest equal-corruption macro-F1;
2. lower pooled false-positive rate;
3. more detected primary events;
4. lower mean delay among detected events (`null` is infinity);
5. higher threshold.

Only the combined method selects the shared score hyperparameters.  Candidate
ties use the same first four quantities for the selected combined threshold,
then higher combined macro-AUROC, then earlier position in the pre-registered
candidate order.  After that candidate is fixed, take each method's threshold
selected within that candidate.

The search output records every candidate summary, threshold-candidate count
and hash, deterministic rules, all development input/health/run hashes, and
the selected index/config.  Deterministic recomputation from immutable
development snapshots must be byte-identical.  The search result and formal
v1 config are then atomically created with no overwrite, hashed, set to `0444`,
and placed in a `0555` calibration directory.  Only after this hash freeze may
holdout forward begin.

## Pre-registered within-sequence Go/No-Go gate

The primary scientific decision uses the untouched holdout with the frozen
development configuration.  “AUROC” below means equal-corruption macro-AUROC
on the primary mask.  All conditions are conjunctive:

1. combined AUROC is strictly greater than 0.75;
2. combined minus random AUROC is at least 0.10;
3. combined AUROC is no more than 0.02 below the better of update-only and
   reliability-only;
4. the same at least two of the three corruption types contain a primary-
   interval combined alarm in both development and holdout;
5. holdout combined pooled primary false-positive rate is at most 0.20;
6. the maximum per-run holdout combined primary false-positive streak is at
   most 3 frames;
7. all provenance, finite-score, replay, runtime, and reproducibility checks
   pass, and the combined detector uses real non-empty health signals.

Exact-interval sensitivity, recovery-boundary alarms, and washout alarms are
mandatory diagnostics but are not separately tuned and do not replace a
failed primary condition.  The gate is evaluated once; holdout must not be
used to revise thresholds, hyperparameters, masks, corruption, or splits.

Passing all conditions supports only a **within-sequence pilot Go** and unlocks
design of the already planned minimal quarantine experiment.  Any failed
condition is **No-Go for quarantine in this pilot**; record the evidence and
stop before quarantine/rollback.  Neither outcome supports a claim of broad
StateGuard3R effectiveness.
