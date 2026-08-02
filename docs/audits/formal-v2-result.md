# Formal Detector v2 result and close-out

## Decision

The unique cross-scene formal Detector v2 evaluation is **GO**. Its immutable
evaluation manifest is `outputs/formal-v2-evaluation-0001/evaluation-manifest.json`,
SHA-256 `8b0842fd5d12b5a109d2ba206b6cea4f71aedfe2cdc0d38ce48e63044675fa6a`.
The formal commitment is 0002, SHA-256
`865665d4548b0c349fedf8a12e4cf7159deb8551ea44795c2ded2dfb7627b4fc`; the
calibration manifest is SHA-256
`7228c58e576270dd6cd98cb32de65d11734509a2420b488649cd2e3e89ca2f59`.

The result uses the pre-registered new scene
`rgbd_dataset_freiburg3_walking_static`, not the disclosed `fr1_desk`
development pool. Three blind forwards were completed in the fixed order
dynamic, wrong-order, low-overlap. Their layouts were checked only with
metadata until all three were frozen; evaluation then ran exactly once.

| Formal v2 gate | Observed value | Result |
| --- | ---: | --- |
| combined macro-AUROC | 0.9126984127 | pass (`> 0.75`) |
| combined minus random macro-AUROC | 0.4303174603 | pass (`>= 0.10`) |
| best single real signal macro-AUROC | 0.8376984127 | pass (combined is within 0.02) |
| shared detected corruption types | dynamic, low-overlap | pass (2) |
| pooled primary FPR | 0.03278688525 | pass (`<= 0.20`) |
| maximum primary FP streak | 2 | pass (`<= 3`) |
| provenance/runtime/replay/real signals | true | pass |
| finite online overlap/combined use/no GT-label leakage | true | pass |
| clean-prefix FPR / startup streak | 0.0 / 0 | pass (`<= 0.15` / `<= 2`) |

The combined detector alarmed at event position 15 (delay zero) for dynamic
occlusion and low-overlap jump. It did not detect wrong-order within that event
interval. Passing the formal gate means only that this small cross-scene
detection pilot may proceed to the separately pre-registered quarantine
feasibility check; it does not establish broad generalization.

## Formal versioning note

`formal-v2-commit-0001`, its six disclosed development outputs, and its
calibration are retained immutable. Before any blind forward, an audit found
that its first launcher implementation would have read an individual holdout
output during postflight verification. This contradicted the protocol's
all-three-before-read lock. No holdout forward had started, so no blind
response was exposed. The launcher was corrected, independently committed,
and all six disclosed development forwards were rerun under commitment 0002.
Only 0002 supplied the formal blind decision above.

## Exploratory quarantine follow-up

The separately committed pilot is immutable at
`outputs/quarantine-v2-commit-0001`, SHA-256
`fa6f74b718fea9101e9b3a534dd6027098125bc92c2580e04c981b5a093bb5c6`.
It operates on disclosed health ledgers only; it does not alter ReCal3R state,
predictions, trajectory, or Health Ledger. The one evaluation manifest is
SHA-256 `47ca777d55f63e85fff1573ce3d8909066652538dbf6e8dda3712c634c18467b`.

The fixed policy triggers once at combined score `>= 2.0686323694270046`,
buffers at most three positions, and drops the buffer unless its final two
scores are both below threshold. It withheld the following
committed-update-magnitude proxy mass:

| Run | alarm | action | buffered positions | proxy mass withheld |
| --- | ---: | --- | --- | ---: |
| dynamic | 15 | drop | 15--17 | 2.3554569912 |
| low-overlap | 15 | drop | 15--17 | 3.5076742130 |
| wrong-order | 27 | drop | 27--29 | 1.4741402155 |

Total proxy mass is `7.3372714197`. The wrong-order action covers three
evaluation-only clean positions, so the pilot exposes a real false-positive
cost. Its immutable decision is **EXPLORATORY_PROXY_ONLY_NO_ROLLBACK**. This
is deliberately not a recovery claim: the proxy cannot prove a different
recurrent state, geometry, trajectory, or persistent-error reduction. No
rollback, safe replay, or additional recovery experiment was executed.

## Reproducibility and limitations

- Fixed visual proxy: ORB-2000, grid 8x6, ratio 0.80, RANSAC 1.0, one OpenCV
  thread, OpenCL disabled, RNG 0.
- Fixed score: window 10, min-history 5, max-z 10, finite-component mean and
  precommitted floors/thresholds. GT-depth reprojection was donor selection
  only and never entered online health/scoring.
- One official new archive was downloaded; no additional scene, model, or
  checkpoint was obtained.
- The protocol contains a small controlled 30-frame corruption setting and a
  single new scene. The wrong-order miss and exploratory quarantine clean cost
  must remain visible in any summary of this work.
