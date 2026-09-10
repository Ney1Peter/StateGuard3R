# StateTriage3R v2 Stage 0.9 — non-periodic tile calibration

Date: 2026-09-11

## Goal

Stage 0.8 failed its sealed activation gate: it had 8 rather than 10 activated first-event rows because four coverage ratios were not below `.25`. Stage 0.9 will not revise, replay, score, or reuse any Stage 0–0.8 response.

Stage 0.9 tests one new fixed synthetic local-content mechanism on fresh local raw windows. The only claim it can make is calibration recipe feasibility. It cannot make a detector GO, generalization, action, recovery, state-write, ATE, or RPE claim.

The terminal results are exactly `P0.9_HASHED_TILE_CALIBRATION_ACCEPTED` or `P0.9_HASHED_TILE_CALIBRATION_REJECTED`. No response can be retried, altered, relabelled, or used in a final score.

## Capacity

The direct replay of all frozen input inventories finds 23 pairwise-disjoint fresh 30-frame windows: 0 in `rgbd_dataset_freiburg1_desk`, 2 in `rgbd_dataset_freiburg1_room`, 17 in `rgbd_dataset_freiburg2_desk`, 3 in `rgbd_dataset_freiburg3_walking_static`, and 1 in `rgbd_dataset_freiburg3_walking_xyz`.

All Stage 0.8 calibration and reserved-final capsules are occupied for this audit, even where no forward ran. A 12-window calibration plus a 31-window final matrix requires 43 fresh windows. Therefore this stage is calibration-only: existing data cannot support a formal final GO.

## Fixed new recipe

`hashed_tile_occlusion` replaces a 75%-area rectangle only on event frames with black or white 16-model-ready-pixel tiles. Tile colour is a deterministic integer hash of tile row, tile column, and seed `1909`. It is non-periodic, fully clipped, source-independent, and spatially local; it cannot read a model output, observer value, label, future frame, or prior result.

The four rectangle positions cycle as `(0,.0,1,.75)`, `(0,.25,1,.75)`, `(0,0,.75,1)`, `(0.25,0,.75,1)`.

The builder reparses every Stage 0–0.8 inventory before publishing. It creates 12 fresh, 30-frame `rgbd_dataset_freiburg2_desk` capsules at starts `110,270,730,870,1230,1430,1770,1950,2380,2630,2760,2850`; event frames are 12–17. It refuses source overlap, writable raw input, duplicate output, and changed historical inventory.

## Gate and execution

Before a forward, a read-only protocol freezes input and capsule hashes, runner and observer source hashes, `EvidenceConfig`, decision contract, output names, and this gate: all 12 responses are valid, immutable, causal and provenance-checked, and at least 10 first-event rows have `sharpness > .05`, `clipped_fraction >= .20`, and `coverage_ratio < .25`.

1. Implement and test the transform, capacity verifier, builder, protocol freezer, evaluator, and serial dispatcher. Legacy transforms remain byte-for-byte behaviourally compatible. Commit code and tests before inputs.
2. Build inputs, freeze and commit the protocol, preflight resources, then run all 12 observer-on forwards serially in a named tmux window on one explicit GPU. The evaluator runs once after a 12/12 sentinel.
3. Publish one audit. If accepted, document the candidate and the remaining minimum requirement: a separately authorized source with at least 31 fresh pairwise-disjoint 30-frame windows for a final-only matrix. If rejected, document the failed rows and close this recipe without retry.
4. Verify artifacts and tests, commit, then push only Stage 0.9 code and documents.

No new data, dependency, or upstream ReCal3R change is authorized by this plan.
