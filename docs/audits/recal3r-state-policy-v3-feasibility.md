# ReCal3R v3 state-policy transactional feasibility audit

- Audit date: 2026-08-03
- Status: **PASS — `TRANSACTIONAL_STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM`**
- Scope: a disclosed 30-frame `formal-v1` development input and one synthetic control
  alarm. This is not a Detector v3 formal result, an attack evaluation, ATE/RPE
  comparison, or a recovery-quality claim.

## What was tested

The pinned ReCal3R `forward_recurrent_lighter` implementation returns an empty
`all_state_args` list. Its own source also updates `update_pressure`,
`recal3r_state0`, `_recal3r_sequence_age`, and calibration trace/queue attributes;
therefore changing only a view's `update` flag is not a state transaction.

StateGuard3R adds an **external** hash-bound clone, without modifying the baseline:

- protected source: `baselines/ReCal3R/src/dust3r/model.py`, SHA-256
  `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1`;
- cloned source span: lines 1661--1833, SHA-256
  `03d3c534f5f59f6eb1fa1852ff1cbeb8e874e97b027cd863e48fb6347b100023`;
- protected baseline commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, clean;
- transaction closure: local `state_feat`, `state_pos`, `init_state_feat`, `mem`,
  `init_mem`, prior reset state, model-side pressure/reference/age/trace/queue,
  and CPU/CUDA RNG state.

For frame `t`, policy can read only alarm `t-1`. A prior alarm holds the current
candidate (maximum three pending); a later clear score restores the newest held
post-state before the next candidate forward. At end of stream, remaining held
transactions are explicitly reported as dropped rather than silently committed.

## Reproducible evidence

| Run | Policy | GPU / PID | Result |
| --- | --- | --- | --- |
| `state-policy-v3-always-commit-0003` | no alarms, always commit | GPU 0 / 1083206 | success |
| `state-policy-v3-forced-hold-0003` | synthetic alarm at frame 1 | GPU 0 / 1090089 | success |
| `state-policy-v3-feasibility-0003` | CPU validator | n/a | PASS |

Both GPU runs used the same disclosed `development-dynamic` manifest, checkpoint,
seed, size, beta base and `CUDA_VISIBLE_DEVICES=0`. GPU 0 (NVIDIA L20, UUID
`GPU-34f2e8e8-05a4-58b3-116d-cd5cc9cec2a2`) had 43,404 MiB free before each run.
Existing root Web-service and other-user training processes were identified and left
untouched; both task PIDs are absent from postflight snapshots and GPU memory returned
to its prior range. `gpustat` was unavailable; no package was installed for this audit.

The always-commit external path is byte-identical to the independent v2 reference for
all model/health artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` |

The forced policy deliberately keeps those same four artifacts byte-identical. Its
purpose is state-path verification, not to manufacture an output-quality difference.
The immutable feasibility result is
`outputs/state-policy-v3-feasibility-0003/state-policy-feasibility.json`, SHA-256
`42015712e396137ba0d19549aac573e19968958671fdbc0f08166c0fc3285127`.

## Transaction result

The synthetic alarm at frame 1 influenced only frame 2:

| Frame | Action | State evidence |
| ---: | --- | --- |
| 2 | `hold` | proposed digest `631123...ff176` differed from pre-state; committed digest restored exactly to `42654b...14a4c2` |
| 3 | `release_replay_then_commit` | pre-state digest exactly matched the held proposal `631123...ff176`; post-replay commit digest was `8d7ea8...241619` |

There were 30 transactions, no unreported pending transaction, and the always-commit
timeline contains only `commit`. The validator independently checked all these facts,
plus postflight PID absence and immutable permissions. Every successful output directory
is mode `0555` with files mode `0444`; the associated six success logs are `0444`.

Two earlier, separately retained attempts are not hidden: `always-commit-0001` failed
before baseline import because direct script execution lacked its repository bootstrap;
`always-commit-0002` reached model loading but exposed the missing `torch.no_grad()`
context through an OOM. The fixes were StateGuard3R-only commits, followed by CPU tests;
the failed logs/minimal audit artifact and postflight snapshot remain read-only. Neither
attempt changed ReCal3R or produced a policy conclusion.

## Gate consequence and limit

The Phase 3 wrapper-feasibility gate is now complete: an external policy can both leave
the normal forward byte-equivalent and demonstrably hold/restore/replay full recurrent
state. This does **not** show geometric recovery, lower drift, safer detection, or a
benefit on natural corruption. The alarm here was forced and the scene is disclosed.
Any recovery-quality experiment remains prohibited until Detector v3 formal GO and a
separate pre-registered quality protocol are in place.
