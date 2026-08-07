# v10 aspect-ratio spatial-grid pose export: execution plan

- Status: **PRESERVED / DISABLED — PRE-GATE-A NO-GO; DO NOT EXECUTE OR
  MODIFY.**  The count-only design is rejected in
  `docs/audits/recovery-aspect-ratio-spatial-grid-pose-export-v10-preimplementation-no-go.md`.
- Objective: determine whether a **resolution-locked 32x24 pre-projection
  spatial-grid mean** (768 tokens at the frozen 512x384 input geometry), then
  the frozen `decoder_embed` and official pose head, can export an alarm pose
  without raw-pose fallback and within 1.20x runtime.

## Why this is distinct from v9

v9 froze a 24x24/576-token mean and its only control proved that assumption is
unavailable in the real pinned recurrent execution. v10 does not alter v9 or
retry its ID: it binds a different, aspect-ratio-aware source domain, derived
from the declared 512x384 input grid and fixed 16-pixel patch geometry. The
candidate-only source must be exactly `(1,768,1024)` before `decoder_embed`;
any other batch, count, width, dtype, finite or device value fails closed.

## Boundaries and gates

1. Reuse only the existing development manifests/checkpoint; download no data.
2. Always-control must not inspect, guard, pool, project, clone or export a
   v10 token. Its four protected files must equal frozen v1 byte-for-byte.
3. Candidate captures `decoder_embed(dec[0].mean(dim=1, keepdim=True))` only
   after complete rollout and before `update_mem`, only if `dec[0]` is exactly
   `(1,768,1024)`. Alarm decode and rollback/no-fallback/SO3 rules remain v9.
4. A new one-use launcher revision and fresh IDs provide original preflight,
   pipe transcript, PID/start-time, result, postflight and CPU validator.
5. Gate A requires source/AST/runtime-shape tests, actual CPU pose-head test,
   dispatcher regression and clean worktrees. Gate B runs one dynamic control;
   only PASS permits one dynamic candidate. Any failure freezes v10 no-go.
6. Gate C, only after Gate B PASS, runs wrong/low control/candidate and the
   existing blind evaluator; it must meet all control-equivalence, provenance,
   runtime and quality rules without retuning.
