# Detector v2 Phase 6 readiness audit

- Audit date: 2026-08-03
- Status: **PASS — Phase 7 single-new-scene download permitted**
- Scope: readiness on disclosed legacy development data plus the Phase 4 GPU-equivalence audit. This is neither a cross-scene result nor a formal-v2 GO decision.

## Immutable decision

The CPU-only readiness validator bound the existing development-search result to the completed instrumentation audit and published:

| Artifact | Value |
|---|---|
| Output | `outputs/detector-v2-readiness-0001/readiness.json` |
| Output SHA-256 | `5dd6a08770e5fd91d7e4b4077b5d538f4ea044421793f0da5f4a19847343bcd3` |
| Validator SHA-256 | `72fd8b78fd345a7f6ad3a27da2f8725fa6213bfbb74c5c74308b18d3b7e9d867` |
| Bound development search SHA-256 | `45287f2eeeee33c8c0eee9cf7586f6996589d9963683775fcb8c93e61250c129` |
| Bound Phase 4 audit SHA-256 | `fe636e3dc2d1e70ae47ead54af2203f8272a59f0c9f4f88dd0510c38e6b7c3dc` |

The output directory is `0555`; its JSON file is `0444`.

## Ten required checks

| Readiness requirement | Measured evidence | Result |
|---|---:|---|
| Full disclosed-development macro-AUROC >= 0.80 | 0.949365079 | pass |
| Five-fold held-out macro-AUROC >= 0.75 | 0.949365079 | pass |
| Combined minus seeded random >= 0.15 | 0.445079365 | pass |
| Pooled primary FPR <= 0.15 | 0.008196721 | pass |
| Per-run FP streak <= 2 | 1 | pass |
| All corruption types detected; mean delay <= 1 | 6/6; 0 | pass |
| Clean-prefix FP <= 6; no triplicate position | 1; none | pass |
| Reliability-only FPR < 0.50 | 0.016393443 | pass |
| Frames 1–29 use finite online overlap | all six development runs | pass |
| GPU instrumentation, determinism/provenance and cleanup | Phase 4 PASS; 30 frames, 29 updates | pass |

The preinstrumentation development artifact is intentionally not rewritten: it remains a correct historical record with its tenth check set to false. `readiness.json` is the separate, hash-bound transition that supplies the now-completed GPU evidence.

## Authorized next action and continuing restrictions

Only Phase 7 is unlocked:

1. deterministically choose one previously uninspected official TUM RGB-D+GT scene from the allowlist;
2. before downloading, record source URL, license/official origin, Content-Length, expected unpacked size and current disk space;
3. download exactly one archive using `.part`, verify it, atomically publish a read-only raw manifest; and
4. do not run ReCal3R, the online overlap hook or any health-response inspection on the new scene before the Phase 8 protocol and blind commitment are frozen.

This PASS does not authorize a second archive, informal holdout exploration, threshold adjustment from new-scene responses, quarantine, rollback, or recovery claims.
