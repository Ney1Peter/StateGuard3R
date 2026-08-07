# Recovery early-decoder pose-token export v8: Gate B provenance NO-GO

- Audit date: 2026-08-07
- Bound plan: `docs/recovery-early-decoder-pose-export-development-v8-execution-plan.md`
- Bound recovery addendum: `docs/recovery-early-decoder-pose-export-v8-gate-b-log-recovery-addendum.md`
- Terminal decision: **`EARLY_DECODER_POSE_EXPORT_V8_AVAILABILITY_OR_RUNTIME_NO_GO`**.

## Decision

v8 does **not** authorize its dynamic candidate, wrong/low matrix, quality evaluator,
or any v8 rerun. This is a terminal Gate-B provenance/protocol failure, rather than a
negative measurement of the early-decoder-token scientific hypothesis. Its numerical
control artifacts are retained for forensic transparency but are not Gate-B PASS evidence.

No data, checkpoint, dependency, ReCal3R source, Detector-v3 configuration, raw image,
or GT was changed or downloaded for this audit. No v8 candidate output/log exists.

## Immutable chronology and the unaccepted `0001`

The committed recovery addendum is `fd4ee53222f704f24501a03e45cd343978b08665`, with
author and committer time `2026-08-07T21:45:00+08:00`. It explicitly states that
`recovery-early-decoder-pose-v8-dynamic-always-commit-0001` had no GPU forward and may
not be reused, and authorizes only the fresh `0002` control.

The immutable `0001` artifacts contradict that prerequisite:

| evidence | immutable fact |
| --- | --- |
| `outputs/recovery-early-decoder-pose-v8-dynamic-always-commit-0001/run.json` | PID `3984986`; start `2026-08-07T21:45:36.509761+08:00`; finish `2026-08-07T21:46:06.728623+08:00`; status `succeeded` |
| `logs/recovery-early-decoder-pose-v8-dynamic-always-commit-0001-main.log` | `V7_DRIVER_START`, exact CUDA command, PID, and `V7_DRIVER_EXIT ... exit_code=0` |
| `logs/recovery-early-decoder-pose-v8-dynamic-always-commit-0001-tmux-transcript.log` | live-pipe terminal copy of the same command/PID/exit sequence |
| `outputs/...0001` | root mode `0555`; each of its seven output files is `0444` |

Thus its CUDA child began 36 seconds *after* the addendum was committed. `0001` is
permanently **`UNACCEPTED_AUDIT_PROTOCOL_BREACH`**. It must not be deleted, overwritten,
or cited to authorize a candidate. It did finish before `0002` began, so the audit found
no forward overlap or result-file dependence; that independence does not repair the
violation.

## `0002`: technically normal, but incomplete canonical evidence

The separately named `0002` run did complete normally. Its `run.json` records PID
`3987169`, start `2026-08-07T21:46:16.152263+08:00`, finish
`2026-08-07T21:46:46.983376+08:00`, and policy runtime
`3.3956317994743586 s`. Against the frozen v1 dynamic baseline runtime
`3.4633694058284163 s`, the arithmetic ratio is `0.9804417033192983`, below `1.20`.

Its four protected files are byte-identical to the v1 dynamic baseline:

| file | SHA-256 |
| --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` |

The `0002` output root is `0555`; all seven files are `0444`. Its preflight records two
GPU-2 snapshots of `45586 MiB >= 12288 MiB`; preflight/main/transcript plus the frozen
driver/result are NUL-free and `0444`; its transcript has the `V7_DRIVER_START`, exact
command, child PID, and zero-exit markers. The control begins after `0001` ends.

Nevertheless, its required canonical postflight path was never produced:

```text
logs/recovery-early-decoder-pose-v8-dynamic-always-commit-0002-postflight.log
```

It is absent, and the frozen `tmp/recovery-early-decoder-pose-v8-dynamic-always-commit-0002-driver.sh`
contains no postflight-generation action. Both the v8 plan and the recovery addendum
require NUL-free, `0444` canonical preflight/main/**postflight**/transcript evidence
before candidate authorization. Creating a same-named postflight now would be a
reconstruction rather than the required original termination snapshot, and is therefore
not permitted.

## Binding stop rule and successor boundary

Neither `0001` nor `0002` may be validated as a v8 Gate-B PASS. Do not run
`recovery-early-decoder-pose-v8-dynamic-candidate-*`, wrong/low v8 controls or
candidates, a v8 GT evaluator, or a v8 recovery/retry with modified token index,
precision, decoder, detector, or terminal-evidence handling.

Any continuation must use a separately committed mechanism and a new, single-owner
launcher that atomically records its two preflights, live `pipe-pane` transcript,
dispatch marker, child result, and postflight before freezing. It cannot reuse a v8
output, log identifier, or numerical result as candidate authorization.
