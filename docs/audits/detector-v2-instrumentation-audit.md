# Detector v2 GPU instrumentation equivalence audit

- Audit date: 2026-08-03
- Status: **PASS**
- Scope: one disclosed 30-frame development manifest only. This audit proves that the v2 online visual-overlap hook is observational with respect to the ReCal3R forward; it is not a formal-v2 evaluation.

## Frozen replay pair

Both runs used the same frozen manifest, checkpoint, seed, size, beta base and physical GPU binding:

| Item | Value |
|---|---|
| Input | `outputs/formal-v1-inputs-0001/development/development-dynamic/input-manifest.json` |
| Input frames | 30 |
| Checkpoint SHA-256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| ReCal3R baseline commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| StateGuard3R runner commit | `e12c5e8` |
| Device binding | `CUDA_VISIBLE_DEVICES=2` |
| v1 profile output | `outputs/detector-v2-instrumentation-v1-0001` |
| v2 profile output | `outputs/detector-v2-instrumentation-v2-0001` |

Each launch was preceded by two new `nvidia-smi` process/memory snapshots. The host did not have `gpustat` installed; this was recorded in the snapshots and no package was installed merely for this audit. GPU 2 met the plan's `free >= 12288 MiB` requirement. During the second run a separately identified same-user Movie3R process used the same device; it was neither modified nor interrupted, and sufficient free memory remained. Both ReCal3R PIDs exited. The v1/v2 postflight logs bind the absence of PIDs `801139` and `816096`, respectively.

## Machine-checked result

The CPU-only validator
`scripts/validate_detector_v2_instrumentation.py` produced the immutable output
`outputs/detector-v2-instrumentation-equivalence-0001`:

| Artifact | SHA-256 |
|---|---|
| `instrumentation-equivalence.json` | `fe636e3dc2d1e70ae47ead54af2203f8272a59f0c9f4f88dd0510c38e6b7c3dc` |
| validator source used for the audit | `1ac87de8d28860ba733d63010064f4cf4306ba43659c15f85e28bc117829ed3a` |
| v1 postflight log | `9d46af183ad8f9504796da299d439b80477000f7147c1bf95fe33a76f8f09cb9` |
| v2 postflight log | `47042e3579b6fd547be5d4251dbaa8b26c481a36f11edd8183d1f038f5ac615a` |

The validator requires and passed all of the following:

- `checkpoint-load-audit.json`, `predictions-summary.json`, and `trajectory.json` are byte-identical.
- All model-relevant `run.json` invariants match, including frozen input hashes, checkpoint, runtime configuration, device, and the 29 expected recurrent updates.
- Every v1 health field equals its v2 counterpart after removing only `overlap`.
- v1 overlap is null on all 30 frames; v2 overlap is null only at frame 0 and finite in `[0,1]` on frames 1–29.
- v2 diagnostics use only the immediately previous frame and agree exactly with the health-ledger overlap values.
- OpenCV provenance is pinned to version 4.11.0, one thread, OpenCL disabled and RNG seed 0.
- Both recorded runner PIDs are absent in their postflight snapshots.

The byte-identical model artifact hashes are:

| File | SHA-256 |
|---|---|
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` |

All three output directories are immutable: directories are mode `0555` and their files are mode `0444`.

## Runtime separation

| Measurement | v1 | v2 |
|---|---:|---:|
| Runner wall time (process start to finish) | 31.998411 s | 36.920814 s |
| ReCal3R inference-only time | 4.901038 s | 4.997153 s |
| Peak allocated GPU memory | 6366.722 MiB | 6365.548 MiB |
| v2 CPU visual-overlap time | n/a | 4.508043 s |

The visual-overlap number is explicitly separate from inference-only time. The modest inference timing difference is not interpreted as a model-output difference; the model-output artifacts and recurrent update count are the controlling equivalence evidence.

## Gate consequence

Phase 4 is complete. Combined with the frozen CPU development search, the only missing Phase 6 condition has been supplied. New data remained prohibited until this audit passed; the subsequent readiness audit is the sole authorization for the next, tightly bounded Phase 7 download.
