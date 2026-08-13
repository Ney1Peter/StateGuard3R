# v16 bounded update-pressure: Gate-B availability/runtime NO-GO

- Date: 2026-08-14
- Decision: **`V16_AVAILABILITY_OR_RUNTIME_NO_GO`**
- Bound one-use control: `recovery-update-pressure-v16-dynamic-always-commit-0001`
- Scope: the only Gate-B dynamic always-commit control authorized by the committed [v16 execution plan](../recovery-update-pressure-bounded-state-memory-v16-split-capability-execution-plan.md).

## Decision

The one-use control completed its real pinned-GPU native execution successfully, but it did not satisfy either independent Gate-B acceptance requirement. Its fresh CUDA-hidden validator is frozen with `status="FAIL"`. Therefore v16 is terminal: the control cannot authorize its candidate, and no v16 wrong/low condition, GT read, quality evaluation, source repair, new run ID or rerun is permitted.

Two protected artifacts differ from their pre-registered SHA-256 constants. The checkpoint-load audit and prediction summary match, while the health ledger and trajectory do not:

| protected artifact | pre-registered SHA-256 | frozen control SHA-256 | result |
| --- | --- | --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` | PASS |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` | `0164fcfc1b9d3e78da400bb77f3579b27dc9ecb8d451d0b07ef3a9ba98ed831d` | **FAIL** |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` | PASS |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` | `29c64430e9c31323e48322aa32dcecd13b8f1fb5a24db38dec25a577110cfd7a` | **FAIL** |

The synchronized runtime is also an independent hard failure: `9.320409298874438 s` versus the pre-registered reference `5.660642675124109 s`. The fixed `1.20x` limit was `6.79277121014893 s`; observed ratio is `1.646528465722329x`, exceeding the limit by `2.5276380887255074 s`. The validator stopped at the protected-hash failure first, but recomputation from the immutable `run.json` establishes the runtime failure as well.

## What the control did establish

This is not a failed launch or partial model result. The tmux child on physical GPU 2 exited `0` after loading the pinned ReCal3R checkpoint and processing the complete frozen 30-frame capability. Its output tree is immutable (`0555` root, `0444` files), and its control timeline records 30 direct native commits, all with no alarm, no pending transaction and no pressure evidence. It did not construct the Detector-v3, candidate operator or update-pressure injection path.

The one-use dispatcher completed its evidence protocol. Two preflight snapshots and one postflight snapshot selected GPU UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`, met the free-memory condition, and found no remaining Project2 process on GPU 2 after exit. The PID/start-tick pair was proved absent without an unverified signal. Frozen preflight, main log, tmux transcript, driver, result, postflight and validator are all present.

| terminal artifact | SHA-256 | mode |
| --- | --- | --- |
| [`validator`](../../logs/recovery-update-pressure-v16-dynamic-always-commit-0001-validator.json) | `5ec3a58e1e4bc81e66b8f1981292261d127ec3975aa88311e5796286d7cdd73e` | `0444` |

The source and inputs remain pinned as established at Gate A: StateGuard3R `baba317f94f3469e169cbfe0fec272ae06de4792`, clean ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, checkpoint SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`, and dynamic capability SHA-256 `c734a8af297f3d4f5ede70a1d7fba36809b87423338ea8c3f332ddca4fb12792`. No data, weight or dependency was downloaded, and ReCal3R, checkpoint, raw RGB and `rgb.txt` were not modified.

## Required stop and successor boundary

Do not alter the v16 constants, runner, detector, pressure operator, capability, dispatcher, waiter, validator or plan to make this control pass. Do not launch `recovery-update-pressure-v16-dynamic-candidate-0001`, a v16 wrong/low run or a GT evaluator. Do not overwrite, thaw or regenerate its frozen outputs or terminal logs.

A successor must use a newly numbered, separately committed pre-registration and a mechanism other than v16's alarm-triggered bounded update-pressure write. It must not import, call, read or use v16 recovery/input/operator/run artifacts and must pass a new Gate A before any GPU release.
