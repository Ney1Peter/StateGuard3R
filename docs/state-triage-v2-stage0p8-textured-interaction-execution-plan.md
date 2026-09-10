# StateTriage3R v2 Stage 0.8 — textured local-content interaction replication

Date: 2026-09-11  
Status: **active; no Stage 0.8 forward exists**

## Objective and completion rule

Stage 0.7 produced strong 80-window typed-screen results and a genuine
local-before-coverage contrast, but only 7 rather than the pre-registered 8
activated local events.  It is closed as a formal NO-GO and may not be
relabelled, retuned, replayed, or included in this study's score.

Stage 0.8 is a new two-phase study whose goal is to establish whether the
unchanged typed decision contract can pass a fully fresh, intentionally
activated interaction replication.  Completion requires one of:

1. an immutable **`STATE_TRIAGE_V2_STAGE0P8_INTERACTION_GO`** satisfying every
   frozen final gate below;
2. a completed, auditable mechanism NO-GO; or
3. a documented design infeasibility after the predeclared calibration route
   has been exhausted without touching a final response.

No development result, calibration result, partial final run, or single
successful window is an endpoint.  This remains a development-proxy detector
study only: it cannot authorize an action policy, persistent state write,
hold/replay, recovery, rollback, upstream ReCal3R change, natural semantic
claim, generalization claim, or ATE/RPE claim.

## Why a new scene is necessary

An audited scan of the four currently installed TUM RGB sequences finds only
17 unused mutually disjoint 30-frame windows after Stage 0, 0.5, 0.6 and 0.7.
That is insufficient for an independent calibration set plus an adequately
powered final matrix.  Stage 0.8 therefore adds exactly one official scene,
not a large data collection:

```text
Dataset: TUM RGB-D, Freiburg 1 room
URL: https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
Observed HTTP Content-Length: 782,381,450 bytes
License: CC BY 4.0 (TUM RGB-D benchmark page, checked 2026-09-11)
Target: baselines/ReCal3R/data/tum/
```

Before download, the acquisition script must recheck URL, Content-Length,
available `/data` capacity, archive/tree path, and absence of an existing
identical scene.  It must write through a `.part` file, record archive and
tree hashes, validate `rgb.txt` plus RGB listing, atomically publish the new
raw tree read-only, and leave all existing data untouched.  The archive and
derived experimental outputs are never committed to Git.

## Frozen detector methods

Stage 0.8 copies the numerical decision contract exactly; no threshold or
ordering changes are permitted.

```text
T0.8 primary:
  sharpness <= .05                         -> bad observation
  clipped_fraction >= .20                  -> transient local content
  coverage_ratio < .25                     -> normal novelty
  pose_jump_z >= 3 or residual_z >= 3      -> registration/order fault
  otherwise                                 -> unresolved

C0.8 structural ablation:
  sharpness <= .05                         -> bad observation
  coverage_ratio < .25                     -> normal novelty
  clipped_fraction >= .20                  -> transient local content
  pose_jump_z >= 3 or residual_z >= 3      -> registration/order fault
  otherwise                                 -> unresolved

S0.8 scalar baseline:
  coverage_ratio < .25                     -> normal novelty
  residual_z >= 3                          -> registration/order fault
  otherwise                                 -> unresolved
```

Each method sees only the current observer row and strict observer prefix.  It
never reads the transform recipe, source index/path, event label, future row,
GT, model state, prior-stage output, or calibration result.

## New, fixed input mechanism

The Stage 0.7 failure mode was not a wrong local branch: four 75%-area constant
rectangles collapsed first-event sharpness below .05 and consequently entered
the deliberately earlier quality branch.  Stage 0.8 adds one explicit,
auditable `checkerboard_occlusion` capsule transform:

- it covers a fixed 75% rectangle, leaving 25% of the source image untouched;
- the covered rectangle receives a deterministic black/white checkerboard at
  a fixed model-ready tile size; every altered pixel remains clipped;
- it is a synthetic partial local-content proxy, not a natural object or
  semantic-change claim;
- raw source images are cloned before it is applied, and the transform is
  validated from an immutable capsule schema.

The high-frequency pattern is designed to preserve substantial spatial
variation while retaining a large clipped region.  Whether it also causes the
required low coverage is deliberately measured in a separate calibration
phase, not assumed from its recipe.

## Phase A — sealed calibration, never scored as final evidence

After code, unit tests and the new scene inventory are committed, the builder
creates exactly 12 fresh, pairwise-disjoint 30-frame **local-content-only**
capsules.  They are selected deterministically from the new scene and are
disjoint from every existing Stage 0–0.7 capsule and from the reserved final
set.  Event frames are 12–17, with the checkerboard transform on every event
frame.  Inputs, runner/observer source hashes, command, GPU, output names and
acceptance rule are frozen before any forward.

`P0.8_CALIBRATION_ACCEPTED` requires all 12 valid immutable responses and at
least 10 first event rows satisfying:

```text
sharpness > .05
clipped_fraction >= .20
coverage_ratio < .25
```

The calibration report is descriptive recipe feasibility only.  Its capsule
windows and responses are permanently excluded from final scoring.  If it
fails, record its exact failure mode and construct a new numbered, fresh
calibration route; do not modify a capsule or retry any published response.

## Phase B — new frozen final matrix

Only after Phase A accepts, construct and commit a final immutable inventory
of 31 new, pairwise-disjoint capsules from the same new scene, disjoint from
the calibration windows and all previous stages:

| Cause | Count | Event frames | Recipe |
| --- | ---: | --- | --- |
| normal novelty | 5 | 0–4 | identity |
| registration/order fault | 5 | 12–17 | audited six-frame reversal |
| bad observation | 5 | 12–17 | full-frame constant colour |
| transient local content | 16 | 12–17 | fixed 75% checkerboard |
| **Total** | **31** | | |

The final protocol must contain exact input and capsule hashes, 31 direct
output names, frozen runner/observer hashes, source selection algorithm,
methods, interaction definition and gates.  It is made read-only and committed
before any final forward.  Each successful observer-only forward is serial,
immutable and non-retryable.  Evaluation is invoked exactly once only after a
31/31 done sentinel exists.

## Final interaction definition and GO gates

An activated event is a final local-content truth whose first event row has
`sharpness > .05`, `clipped_fraction >= .20`, and `coverage_ratio < .25`.
This is computed after immutable responses exist and cannot select inputs or
change a threshold.

`STATE_TRIAGE_V2_STAGE0P8_INTERACTION_GO` requires all of:

- all 31 final responses valid, immutable, causal and provenance-checked;
- T0.8 Macro-F1 >= .85 and T0.8 minus S0.8 Macro-F1 >= .30;
- every T0.8 cause recall >= .75;
- normal-novelty false reject <= .10;
- no unsafe confusion type occurs more than once;
- at least 12 activated local-coverage events;
- within that stratum T0.8 local recall >= .75 and T0.8 exceeds C0.8 local
  recall by >= .75;
- all raw-source, input, output, source-hash, strict-prefix and read-only
  checks pass.

Any failed gate is a formal Stage 0.8 NO-GO.  It must be reported as such even
if other metrics are strong.

## Execution discipline and milestones

1. Implement the new capsule transform, Stage 0.8 builders/evaluators,
   acquisition guard and unit tests without changing legacy transform behavior.
2. Commit code/tests, inspect disk/GPU/process ownership, then acquire and
   inventory the one official scene.
3. Freeze and run Phase A in a named tmux window on one GPU with sufficient
   free memory; audit its recipe-feasibility result.
4. If accepted, freeze Phase B inputs/protocol in a separate commit, run all
   31 forwards in tmux, evaluate once, audit, test and push.
5. If calibration or final gates fail, preserve every artifact and continue
   only through a newly versioned, disjoint protocol; never adapt on a final
   response.

At each long-running step record absolute command, PID/process group, GPU,
log path, output directory, start/end time, checkpoint, random seed and Git
commit.  Do not download dependencies, alter protected ReCal3R files, use more
than one GPU, or interfere with other processes.
