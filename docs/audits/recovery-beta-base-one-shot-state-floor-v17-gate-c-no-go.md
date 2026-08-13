# v17 native one-shot beta-base floor: terminal Gate-C NO-GO

- Date: 2026-08-14
- Decision: **`V17_AVAILABILITY_OR_RUNTIME_NO_GO` — terminal.**
- Completion condition: Gate-A and one Gate-B always-control were frozen, then
  the single pre-registered Gate-C candidate was released once and terminated
  nonzero.  Per the v17 execution plan, **do not retry v17, alter its
  implementation or constants, run wrong/low conditions, or read GT/run Gate
  E.**

## Frozen Gate-B control PASS

The sole dynamic capability was built directly from the pinned raw selector and
its 30 selected RGB files, then published immutable mode `0444`:

| artifact | SHA-256 |
| --- | --- |
| `outputs/recovery-beta-floor-v17-dynamic-rgb-list-capsule-0001.json` | `61f2bc78135e6700f25bfa79b1d4370c2878b4e3dba38e7706958fdd863fec1b` |

The released `recovery-beta-floor-v17-dynamic-always-commit-0001` control had
two fresh GPU-2 preflights with UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`, `45586 MiB` free in each snapshot,
and no Project2 GPU process.  Its child exited zero and its independent
CUDA-hidden validator is mode `0444`, SHA-256
`8e7ac7d2fcc041bfcea650d42fa5976d8dd0ceab443851bd43af30d81b9c1a2e`.

The validator passed all Gate-B conditions:

| protected output | expected SHA-256 | observed |
| --- | --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` | identical |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` | identical |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` | identical |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` | identical |

It performed exactly 30 direct native commits; detector and operator were both
absent from that branch.  The measured runtime was `4.492682149633765 s`,
`0.7936699783890288x` the reference `5.660642675124109 s`, below the frozen
`6.79277121014893 s` / `1.20x` limit.  This establishes that the v17 control
serialization and the selected raw input are reproducible; it does not test
the intervention.

## Single Gate-C candidate and failure boundary

Only after the frozen control PASS, the sole candidate ID
`recovery-beta-floor-v17-dynamic-candidate-0001` was released in its own new
`stateguard` tmux pane.  It received the same two required GPU-2 snapshots
(`45586 MiB` free; no Project2 process), and the same committed source pin:

| provenance item | value |
| --- | --- |
| StateGuard3R commit | `21a96df390f0e56f1a5234b68755cb1945af1923` |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| candidate child PID/start ticks | `1577366` / `648571289` |
| child exit | `1` at `2026-08-14T02:13:38+08:00` |
| postflight GPU-2 free memory | `45068 MiB`; no Project2 process |

The child successfully loaded and audited the pinned checkpoint, then failed
at `recal3r_beta_base_floor_runner_v17.py:425`, during native
`_compute_recal3r_update_mask`.  CUDA reported an attempted `432.00 MiB`
allocation with `41.98 GiB` already allocated by PyTorch, `1.45 GiB` reserved,
and only `90.25 MiB` free in the visible GPU.  This is an availability failure
of the candidate's required native recurrence path, not a control mismatch.

The failure occurred before an observer health decision, alarm, future arm,
floor consumption, strict-mask witness or candidate output export.  Therefore
there is no valid candidate availability/effect evidence and no basis to
evaluate quality.  The candidate output was never atomically published; the
only preserved staging tree is the immutable mode-`0555`
`outputs/.recovery-beta-floor-v17-dynamic-candidate-0001.j_q_g1rn.staging`,
containing only frozen checkpoint audit and `failure.json` (SHA-256
`93c7a7fa39cb572d8efb82093c832a506222e38d1d87743fc988edc0db2ee5b5`).

Its token-gated-driver, two snapshots, closed pipe, dead PID/start pair and
nonzero child evidence are frozen.  The immutable candidate validator is
`CHILD_NONZERO_EVIDENCE_COMPLETE` (SHA-256
`17eb0ddba83975534c182983a20a332cec330f2b30eb77dd8216ed500fd624c3`);
the dispatcher consequently returned terminal exit code `70`.  It did not
silently convert this nonzero child into a candidate PASS.

## Consequence and preservation record

Gate A had already passed and been committed as `21a96df` after 52 CUDA-hidden
CPU tests passed.  All control/candidate release artifacts, the capability and
both candidate staging files are frozen; ReCal3R remained clean at its pin.
No data, weights or dependencies were downloaded, and no upstream file was
modified.  The only StateGuard3R untracked source paths remain the previously
preserved 28 v14/v15 terminal paths; none were staged or edited.

The supported conclusion is narrow but decisive: under this fixed 30-frame,
native state-floor protocol and one-release resource envelope, v17 can
reproduce the control exactly but cannot make the candidate recurrence
available.  It cannot support a claim about intervention quality.  Any future
research direction must be a separately pre-registered version with fresh
IDs, source/audit boundaries and resource proof; it may not repair or rerun
v17.
