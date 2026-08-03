# Formal Detector v3 independent blind-evaluation protocol

Status: **pre-registration draft.** This file, the v3 input builder/validator,
the v3 orchestrator and their tests must be committed and pass CPU readiness
before the project may audit or download the single candidate archive.  Until
the formal commitment is frozen, no ReCal3R response, response hash, or runner
main-log byte may be opened for the new scene.

## Objective and scope

The confirmatory objective is narrow: establish whether Detector v3 detects
dynamic occlusion, a reversed frame segment that preserves the original RGB
capture timestamps, and a low-overlap jump on one new TUM RGB-D scene while
maintaining the fixed false-positive bounds.  The sole candidate is the
official `rgbd_dataset_freiburg2_desk` archive.  It is disjoint from the
already disclosed `rgbd_dataset_freiburg1_desk` and
`rgbd_dataset_freiburg3_walking_static` evidence.

This protocol neither trains nor fine-tunes a model, modifies ReCal3R, changes
a checkpoint, evaluates a depth/pose quality target, nor makes a recovery
claim.  It uses at most this one archive and its extracted tree, together no
larger than 5 GiB.  A failed acquisition, invalid licence/source, missing
ground truth, or insufficient disk space stops before download and requires a
user decision before considering a second candidate.

All v1/v2 inputs, outputs, and formal responses are disclosed development
evidence for v3.  They may support the frozen v2 configuration and v3
development readiness, but they are not confirmatory v3 evidence.

## Fixed detector and provenance contract

The continuous component is frozen Detector v2 with its existing threshold
`2.0686323694270046`, scorer configuration, five scale floors, checkpoint
`cut3r_512_dpt_4_64.pth` SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`,
and ReCal3R commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.

Detector v3 adds exactly one causal hard channel:

```text
alarm[t] = continuous_score[t] >= 2.0686323694270046
           OR capture_timestamp[t] <= capture_timestamp[t - 1]
```

Frame zero has no timestamp predicate.  For a finite rank score only, a true
timestamp predicate receives `max(continuous_score, threshold + 1.0)`.  This
fixed margin is solely a ranking/attribution separator, not a learned weight
or calibration parameter.  Reported artifacts must distinguish continuous and
timestamp attributions.

The timestamp channel accepts only the exact first-column text of the official
raw scene `rgb.txt`, strictly parsed as finite Decimal, and binds it to the
final RGB frame path and SHA-256.  Each v3 source/input validation records:

- `rgb_capture_timestamp_text` and its strict decimal representation;
- raw `rgb.txt` physical source line and raw listing SHA-256;
- the official raw RGB path and content SHA-256; and
- the final input frame's binding to that raw file after deferred corruption.

The online sidecar receives only ordered final RGB paths plus the immutable
raw `rgb.txt` and scene root.  It has no source-index, GT, depth, label,
corruption/event, model response, or future-frame argument.  Missing,
non-finite, unmatched, duplicated, rewritten, or otherwise unproven timestamp
provenance is a formal integrity failure.  The resulting claim is restricted
to reordered packets retaining their original capture timestamps; it does not
claim universal content-reorder detection.

A timestamp predicate is intentionally **not** restricted by corruption type.
For example, a disjoint low-overlap donor inserted at positions 15--19 can
produce a timestamp descent at the insertion or return boundary; that is an
observable final-stream order anomaly and therefore legitimately activates the
same hard channel.  The validator never reads a corruption label to accept or
reject a predicate.  Instead it proves every final path's raw provenance and
requires no true predicate in the clean prefix (positions 0--14); the final
report exposes the full per-frame continuous/timestamp attribution for all
three runs.

## Input construction and labels

Before any model forward, the input builder must first prove that the raw
archive and extracted scene are regular, non-symlink, read-only artifacts,
that their hashes match the acquisition audit, and that the scene has valid
RGB, depth, and ground-truth listings.  It creates a new immutable input root
with three disjoint 30-frame base allocations and a disjoint five-frame donor
for low overlap.  RGB/depth/GT association is unique nearest absolute
timestamp within 0.02 seconds and rejects ties.  GT/depth reprojection can
rank donor candidates offline only; it must never enter the online detector.

Exactly one corruption is enabled per run, with seed 0 and start position 15:

| Run | Corruption | Event positions |
| --- | --- | --- |
| `blind-dynamic` | moving normalized RGB occlusion | 15--19 |
| `blind-wrong-order` | reverse ordered RGB packet segment | 15--18 |
| `blind-low-overlap` | explicit disjoint donor substitution | 15--19 |

Positions after an event through the next five frames are washout: forwarded
and reported but excluded from primary denominators.  Positions 0--14 form the
clean prefix.  Labels, event windows, GT, depth, source-pool role, and all
corruption metadata are kept out of online scorer/sidecar inputs.

## Commitment and one-shot execution

The commitment binds source and input manifests, their validator report, the
archive/acquisition evidence, protocol, v3 runner, v3 orchestrator, timestamp
module, hybrid module, exact commits, checkpoint, and production OpenCV
provenance.  It rejects a dirty StateGuard3R or ReCal3R worktree.  The exact
runner invocation is `scripts/run_recal3r_smoke.py --health-profile v3`,
size 512, seed 0, `beta-base 0.1`, batch one, and explicit
`CUDA_VISIBLE_DEVICES=<physical-id>`.

Six disclosed development forwards are run serially first using the existing
v1/v2 input pool, followed by deterministic fixed-config calibration.  The
calibration path has no holdout-output path or byte-reading capability.  Its
atomic frozen unlock is necessary before blind launch.

Blind forwards run exactly once and in this order:

```text
blind-dynamic -> blind-wrong-order -> blind-low-overlap
```

Each run must have two adjacent GPU/process snapshots showing at least
12288 MiB free on one explicitly selected shared GPU, the command, PID, GPU
UUID, output/log paths, start/finish times, and a post-exit snapshot.  No
other user's process is changed.  A successful v3 output contains precisely
the five standard runner artifacts plus `timestamp-order.json`; the directory
is frozen mode `0555` and every artifact mode `0444`.  A failed blind run is
a formal operational failure: no substitution, mutation, or retry on that
blind input is permitted.

The read lock is structural.  Before all three blind outputs exist and their
layouts/modes pass `lstat`/directory-listing-only preflight, no code may open,
hash, parse, or otherwise read a blind runner artifact or main log.  Only after
that all-three preflight may the unique evaluator snapshot bytes.  Its
no-overwrite output makes evaluation exactly once.

## GO / NO-GO decision

The unique blind evaluation is **DETECTOR_V3_GO** only if every condition is
true:

1. equal-corruption combined macro-AUROC is strictly greater than 0.75;
2. combined macro-AUROC minus the seeded random baseline is at least 0.10;
3. combined macro-AUROC is not more than 0.02 below the best real single
   continuous signal;
4. all three corruption types are detected in their event interval;
5. wrong-order event delay is at most one frame;
6. pooled primary FPR is at most 0.20, maximum primary FP streak at most 3;
7. clean-prefix FPR is at most 0.15 and startup FP streak at most 2; and
8. timestamp provenance, causal sidecar replay, frozen v2 configuration,
   model/input equivalence, deterministic runtime records, finite overlap, and
   no GT/source-index/label/future leakage all pass.

Every condition is conjunctive.  A failure yields the only valid blind result,
**DETECTOR_V3_NO_GO**.  No threshold, corruption, code, or second forward may
then be chosen from that holdout response.

## Conditional state-policy phase and closure

Only a detector GO permits the already validated external transactional
state-policy wrapper to run on the now-disclosed frozen input.  That result is
exploratory unless a separate independent blind scene and quality protocol are
approved.  It may conclude only
`STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM` unless a separately
pre-registered paired quality study proves the required recovery and clean
degradation thresholds.

The project ends after this conditional feasibility execution or after the
formal NO-GO report.  In both cases it must publish immutable provenance,
timelines, metrics, outcome limits, full CPU validation, repository/permission
checks, and GPU/process cleanup without altering protected v1/v2 evidence.
