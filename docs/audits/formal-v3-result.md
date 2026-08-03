# Formal Detector v3 independent blind result

- Evaluation date: 2026-08-03
- Decision: **`DETECTOR_V3_GO`**
- Scope: one pre-registered, previously uninspected TUM scene; this is the sole v3
  confirmatory detector evaluation.

## Immutable evidence chain

| Artifact | SHA-256 |
| --- | --- |
| formal commitment | `a0a33b215797a9925565fbb48daecff10c1b8607bf8e530750bea6552a51914a` |
| CPU input validation | `01054b599d3492005b7510d2f2c3dc16ea1d43ffbdfbf114fcd763849570ce28` |
| calibration manifest | `d6a0504e9119d8fa8132c32a523e1154bd20a43cb58c151bed1c68138314d868` |
| evaluation attempt seal | `acc5d2a10a00d32f738e8269263586bbbd6942fd3e4de9717afda0d24bfedd7e` |
| evaluation manifest | `180ceaa20dd4de353d6833eddd42d97390a6e20220ef950af6e1eab671cc4e72` |
| holdout metrics | `6fbb1f117de09cbb55d673b07a82b34f093a49f321f560ce6b2af4a30a9bdd0c` |
| GO/NO-GO record | `ba7f7d412b607156e4a1109edd1c8a44654da242ab1f8c1f12f4d2131d2712e4` |

The frozen candidate was official TUM `rgbd_dataset_freiburg2_desk`. Its archive SHA-256 is
`1a0756d72510a26e26e2a02bf0b6c69c2796619fca531e176fc2e5110173807c`; archive plus raw
tree used 3,895,008,662 bytes, below the 5 GiB cap. ReCal3R remained clean at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`. The formal StateGuard3R commitment bound
commit `722858bbce69bf50658afb9b5ef4ce608e0495cc` before any blind response was read.

Six disclosed development forwards/calibration completed first. Blind forwards then ran once,
serially as `blind-dynamic -> blind-wrong-order -> blind-low-overlap`; all three directories
were structurally verified and frozen before the one evaluator wrote its irreversible attempt
seal. No blind response was used for a retune or rerun.

## Pre-registered decision checks

| Check | Actual | Result |
| --- | ---: | --- |
| combined macro-AUROC `> 0.75` | 0.9933333333 | PASS |
| combined minus seeded random `>= 0.10` | 0.5315873016 | PASS |
| within 0.02 of best real single | 0.9933333333 vs 0.9188888889 | PASS |
| detected corruption types | dynamic, wrong-order, low-overlap | PASS |
| wrong-order delay `<= 1` | 0 frames (detected at frame 15) | PASS |
| pooled primary FPR `<= 0.20` | 0.0163934426 | PASS |
| maximum primary FP streak `<= 3` | 1 | PASS |
| clean-prefix FPR / startup streak | 0.0222222222 / 1 | PASS |
| timestamp/provenance/runtime integrity | all required checks true | PASS |

The wrong-order run's frozen raw RGB capture timestamps yielded hard-channel violations at
positions 16--18; the hybrid detector also crossed at event start (position 15). The input
validator confirmed no clean-prefix timestamp violation, final-RGB-to-raw-line/hash bindings,
and that the online timestamp channel excludes GT, depth, labels, source index and future
frames.

## Interpretation limits and follow-up

This result establishes the pre-registered narrow claim: Detector v3 detects the three specified
corruptions on this one independent scene, including timestamp-preserving reordering. It does not
establish universal content reorder detection when capture timestamps are rewritten, natural attack
prevalence, pose/depth quality, ATE/RPE improvement or recovery quality.

Because and only because the detector gate passed, the separately recorded exploratory
state-policy follow-up was run on this now-disclosed input. Its outcome is documented in
[recal3r-state-policy-v3-feasibility.md](recal3r-state-policy-v3-feasibility.md) and is limited to
transactional state semantics, not recovery quality.
