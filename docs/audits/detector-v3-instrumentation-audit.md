# Detector v3 GPU instrumentation equivalence audit

- Audit date: 2026-08-03
- Status: **PASS**
- Scope: one 30-frame, already disclosed `formal-v1` development manifest. This is an
  observational-equivalence audit only; it is neither a new-scene evaluation nor a
  Detector v3 formal result.

## Frozen replay pair

Both profiles used the same final manifest, checkpoint, seed, image size, beta base,
and one explicit GPU binding. The only v3-only input was the raw official `rgb.txt`
listing and its read-only dataset root, from which the runner generated a sidecar
before model loading. It does not alter the manifest, image bytes, model input, or
health JSONL schema.

| Item | Value |
| --- | --- |
| Input | `outputs/formal-v1-inputs-0001/development/development-dynamic/input-manifest.json` |
| Input frames / recurrent updates | 30 / 29 |
| Checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| Runner commit / SHA-256 | `b32b1bb0bfa6aa7de26c2a5a6dd4624596eeafd3` / `89b5d86f5ed7ae7e5b61b9971ad57cb9a7ad34a578f40cc9d9983c5c76bf5024` |
| GPU binding | `CUDA_VISIBLE_DEVICES=0`, NVIDIA L20, UUID `GPU-34f2e8e8-05a4-58b3-116d-cd5cc9cec2a2` |
| v2 output | `outputs/detector-v3-instrumentation-v2-0001` |
| v3 output | `outputs/detector-v3-instrumentation-v3-0001` |
| v3 raw timestamp listing/root | `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk/rgb.txt` / its scene root |

Immediately before each serial launch, GPU 0 had 43,351 MiB free, above the 12,288 MiB
gate. `gpustat` was not installed, so no package was installed solely for the audit.
`nvidia-smi` and `ps` identified an unrelated root Web service (PID 5877, about 388 MiB)
and another user's training process (PID 735375, about 1,788 MiB); neither was changed,
stopped, or otherwise interfered with. The v2 PID 962974 and v3 PID 970796 are both
absent from their recorded postflight snapshots. GPU 0 returned to 43,351 MiB free.

## Machine-checked result

The CPU-only validator
`scripts/validate_detector_v3_instrumentation.py` wrote the immutable report
`outputs/detector-v3-instrumentation-equivalence-0001/instrumentation-equivalence.json`
(SHA-256 `0444caa05a16c185987903b2d569ea5de8d248e62d764efe193220bc43f9e6bd`).
Its own SHA-256 is `0f9bd09bb97f22f63b773110e1c6a30057389645c0ed70650ec097168d6cd958`.

All nine required checks passed:

- checkpoint-load audit, health JSONL, prediction summary, and trajectory are byte-identical;
- model-relevant `run.json` invariants and the 29 recurrent updates are identical;
- visual-correspondence provenance is identical except for separately measured wall time;
- the v3 timestamp sidecar replays its causal predicate, its run metadata hash matches,
  and every sidecar RGB hash matches the corresponding final model-input frame;
- both runner PIDs are absent in their postflight `nvidia-smi` snapshots.

The clean, ordered development input correctly produced no timestamp violation positions.
This checks the observational path; it is not evidence that timestamp-rewritten or
content-only reorders are detected.

| Artifact | v2 SHA-256 | v3 SHA-256 |
| --- | --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` | identical |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` | identical |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` | identical |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` | identical |
| v3 `timestamp-order.json` | n/a | `31904032042a63c38f38ff419b8ea9719db39516c3ec49e4088610b3a1ff797a` |

The v2/v3 runner wall times were 5.170659 s / 6.421300 s; their inference-only times
were 5.052231 s / 4.660243 s, and peak allocated model memory was 6366.722 MiB /
6365.548 MiB. Timing is descriptive only; byte-identical model and health artifacts
are the equivalence criterion.

All three output directories are mode `0555` and all contained output files are `0444`.
The six GPU snapshot/main logs are also `0444` and are retained under `logs/`.

## Gate consequence

The Detector v3 instrumentation-equivalence half of Phase 3 is complete. It proves that
adding the timestamp-order sidecar is observational with respect to the pinned ReCal3R
forward on this disclosed input. It does not yet complete Phase 3: state-aware wrapper
semantic feasibility remains a separate gate, and no new archive or formal-v3 input may
be prepared until that gate and the formal readiness checks pass.
