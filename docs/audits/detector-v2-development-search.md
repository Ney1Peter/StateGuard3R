# Detector v2 disclosed-development search audit

- Audit date: 2026-08-02
- Status: **PREINSTRUMENTATION_PASS**
- Scope: all six formal-v1 runs are now disclosed development data only. This is not a blind formal-v2 result and does not permit new-data download yet.

## What was derived

The immutable output [detector-v2-development-search-0001](/data/wangzheng/Project2/StateGuard3R/outputs/detector-v2-development-search-0001) joins each frozen v1 health record with exactly one causal online-overlap value. The new ledger retains the health schema and contains no label, GT, depth, event, or future-frame field. Frame 0 has null overlap; every later frame has a finite value in [0, 1].

All 32 combinations were retained: four frozen visual candidates times the eight predeclared scoring configurations. For every combination, the search:

- derives scale floors only from clean prefixes;
- selects every method's threshold under pooled FPR <= 0.15 and per-run FP streak <= 2;
- emits full score timelines and every threshold candidate;
- evaluates three leave-one-corruption-type-out folds and two leave-one-original-window-out folds;
- derives each held-out fold's scale floors and threshold from its training runs only.

All 32 candidates produced a feasible constrained threshold. The chosen configuration is:

| Item | Value |
|---|---|
| Visual candidate | ORB-2000, ratio 0.80, RANSAC 1.0 |
| Causal score window | 10 |
| Minimum history | 5 |
| Maximum z | 10 |
| Aggregation | finite-component arithmetic mean |

It wins the fixed comparator: held-out macro-AUROC, lower held-out FPR, event count, lower delay, then stricter ratio / smaller RANSAC / candidate order.

## Development evidence

| Metric | Result | Gate |
|---|---:|---:|
| Full six-run combined macro-AUROC | 0.949365079 | >= 0.80 |
| Five-fold held-out macro-AUROC | 0.949365079 | >= 0.75 |
| Combined minus seeded random macro-AUROC | 0.445079365 | >= 0.15 |
| Combined pooled primary FPR | 0.008196721 | <= 0.15 |
| Max per-run FP streak | 1 | <= 2 |
| Detected events | 6/6 | all three types |
| Mean delay | 0 | <= 1 |
| Clean-prefix FPs | 1 | <= 6 |
| Repeated clean-prefix position in >=3 runs | none | required absent |
| Reliability-only pooled FPR | 0.016393443 | < 0.50 |
| Combined uses all five finite signals after frame 0 | yes | required |

The five folds are both corruption-type holdouts (dynamic occlusion, low-overlap jump, wrong-order segment) and the two original v1 allocation windows. The former v1 holdout allocation is explicitly treated as disclosed development data; this cross-validation is a stress test, not a confirmation claim.

## Binding and immutable outputs

- derived-ledgers.json, SHA-256 284e5edbfd76605f43cadf3fa32a86c46b35678bff4991c33132502bb041f14c
- search.json, SHA-256 45287f2eeeee33c8c0eee9cf7586f6996589d9963683775fcb8c93e61250c129
- report.md, SHA-256 a3a54967fa4e0c5c694c5bd7a11e042e92572d4dc509c9afc2084d173402908b

The output directory is mode 0555; all three files are mode 0444. It binds the frozen visual replay SHA-256 f2cbbff03a00baeb4ec93b731b003c61b72fa0d95776b6df819fd5e41dabff41, frozen development/holdout v1 metric files, and the six formal-v1 health hashes. The derivation script SHA-256 is d23043fa2d4721d140606131a1d09c7f4089c6f45a1821a34ec67502a5902cea; Detector v2 scoring module SHA-256 is 49d38af7da85bb5ca231b5b0569ba511b101e99f8687195ff8b880c7f31fa30b.

## Gate consequence

Nine of the ten development-readiness conditions are now supported by frozen CPU evidence. The remaining required condition is Phase 4: prove that enabling v2 instrumentation does not change ReCal3R model input, prediction summary, trajectory, checkpoint audit, or recurrent update count, and that its GPU process cleans up. Until that passes, development readiness is not final and no new TUM archive may be downloaded.
