# Formal-v2 single-scene TUM acquisition audit

- Audit date: 2026-08-03
- Status: **PASS**
- Scope: the one cross-scene raw-data acquisition allowed by Detector v2 Phase 7. No ReCal3R forward, online visual overlap, health ledger, or model-response inspection was run on this scene.

## Deterministic candidate choice

`fr1_xyz` was ineligible: this project had already run ReCal3R on its RGB AVI during the earlier Gate 1 smoke. Among the documented official complete RGB-D+GT candidates with no corresponding model-output/log hit, the deterministic smallest archive is therefore TUM `fr3_walking_static`, ahead of `fr3_walking_xyz` (474,074,105 B versus 527,550,055 B).

The immutable preflight output is
`outputs/formal-v2-data-preflight-0001/preflight.json` (SHA-256
`09656dcca1a3190f719254e79b4420b21018b33a5c2b67b25d063f133e2f24df`). It records:

| Item | Value |
|---|---|
| Dataset | `rgbd_dataset_freiburg3_walking_static` |
| Canonical URL | `https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz` |
| Final official URL | `https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz` |
| Live HTTP response | 302 then 200; byte ranges supported |
| Content-Length | 474,074,105 B |
| License | CC BY 4.0 |
| Available space before download | 301,751,332,864 B |
| Required data budget | 1,572,864,000 B |
| Candidate model-response/log hits | none |

## Archive and raw-tree result

Exactly one archive was downloaded to the prescribed `.part` path, checked, then atomically renamed. The final successful output is
`outputs/formal-v2-data-0002/acquisition.json` (SHA-256
`c04e7e835edf861b234ecd8fbeae3f9f24842a3d0151d11e7b1a3660b0e0b391`).

| Item | Result |
|---|---|
| Archive | `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg3_walking_static.tgz` |
| Archive bytes | 474,074,105 |
| Archive SHA-256 | `8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1` |
| Archive mode | `0444` |
| Tar members | 1,473 = 3 directories + 1,470 regular files |
| Regular-file bytes | 512,766,395 |
| Raw root | `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg3_walking_static` |
| Raw modes | directories `0555`; files `0444` |
| RGB / depth / GT rows | 743 / 723 / 2,484 |
| RGB frames with depth within 20 ms | 717 |
| GT duplicate timestamps | 1 (nondecreasing stream; not an online signal) |
| Remaining `.part` or staging directory | none |

The raw validation streamed the gzip payload, rejected non-regular/link/path-traversal tar members, checked the expected single top-level directory and required RGB/depth/index/GT files, verified every PNG's TUM geometry and bit depth, parsed all index/GT rows, and SHA-256 hashed all 1,470 raw files.

## One failed validation attempt, preserved without redownload

The first acquisition process completed the byte download and archive safety check, then stopped before raw publication because the validator incorrectly required strictly increasing GT timestamps. The official stream has one duplicate timestamp. The archive remained read-only, no raw tree or acquisition output was published, and the task-generated staging directory was removed. The issue was corrected in commit `778ea7a` with a regression test in `eec7438`: raw validation now accepts nondecreasing timestamps while later formal input selection must still require unique nearest associations. The successful second process reused the same verified archive and performed no network download.

This is an acquisition/format correction, not a model-derived adjustment: the new scene remains blind with respect to ReCal3R response. The first process log is retained at `logs/formal-v2-data-acquisition-0001.log`; the successful reuse log is `logs/formal-v2-data-acquisition-0002.log`.

## Gate consequence

Phase 7 is complete. The next phase must freeze a v2-specific cross-scene input protocol, source/association proof, code commit, candidate configuration, thresholds and blind read lock before any ReCal3R invocation on this raw tree. The data gate does not authorize an additional archive, an exploratory forward, quarantine, rollback, or recovery experiment.
