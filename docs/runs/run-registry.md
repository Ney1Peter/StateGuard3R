# StateGuard3R Run Registry

This registry records setup attempts and experiments that materially affect the
initial feasibility decision. Large logs and outputs stay in the repository that
produced them and are not committed here.

## Setup record

### SETUP-0001: Pin the official ReCal3R baseline

- Time: 2026-07-31 19:00–19:28 CST
- Operator: Codex
- Target: `/data/wangzheng/Project2/baselines/ReCal3R`
- Official remote: `https://github.com/Powertony102/ReCal3R.git`
- Pinned commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Branch: `main`
- Result: success

Actions and observations:

1. A direct HTTPS clone was attempted first and failed with
   `GnuTLS recv error (-110)`; it left no target directory.
2. A clean, `wangzheng`-owned local official clone at
   `/data/wangzheng/iJCV-CODE/ReCal3R` was used only as the source for a new
   Project2 working clone.
3. The new clone's `origin` was reset to the official GitHub URL.
4. `git fetch origin main` succeeded and the working copy was fast-forwarded
   from `2b920d3f` to the pinned official commit above.
5. Final `git status --short --branch` was clean and matched `origin/main`.

No dataset, checkpoint, environment, or GPU task was created during this setup.

### SETUP-0002: Create the isolated ReCal3R environment

- Time: 2026-07-31 19:35–21:30 CST
- Repository: `/data/wangzheng/Project2/baselines/ReCal3R`
- Git commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Result: environment and CPU import checks succeeded
- Python: 3.11.14
- PyTorch: 2.4.0+cu121
- torchvision: 0.19.0+cu121
- CUDA toolkit used for RoPE build: `/usr/local/cuda-12.1`
- Environment: `/data/wangzheng/Project2/baselines/ReCal3R/.venv`
- Full freeze: `docs/runs/recal3r-environment-freeze.txt`
- GPU work: not started

The first dependency resolution installed incompatible newest versions of
transformers, gradio, viser, and websockets. The compatible pins and actual
failure evidence are recorded in `docs/audits/recal3r-environment-audit.md`.
The final `uv pip check` passed and `dust3r.model` imported from the expected
`src/dust3r/model.py` path.

### SETUP-0003: Acquire the official CUT3R checkpoint

- Time: 2026-07-31 21:30–23:00 CST
- Official file ID: `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD`
- Intended target: `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`
- Result: pending; official Google Drive endpoint timed out
- Partial file left behind: no
- Third-party mirror used: no

This network failure blocks real checkpoint loading but does not block the
independent corruption, health-ledger, and detection-only implementation work.

A second read-only connectivity check at approximately 22:40 CST used the same
official file ID and `drive.google.com` download endpoint. DNS resolution or
connection establishment again timed out within the bounded request timeout.
Checks against the official CUT3R GitHub README and GitHub Releases API found
the same Google Drive source and no GitHub release asset for this checkpoint.
No checkpoint download was started and no partial file was created.

### SETUP-0004: Recheck GPU availability before any CUDA execution

- Time: 2026-07-31 23:08 CST
- Result: no completely idle GPU; no CUDA process started

All eight NVIDIA L20 cards had an existing compute process and non-zero memory
use. GPU 2 was the least occupied at about 1.2 GiB, but it still belonged to an
existing `/data/jiale/.../python` process and therefore did not satisfy the
server rule requiring a completely idle card. No process was stopped or
modified. The next CUDA attempt must repeat this check and bind an eligible card
with an explicit `CUDA_VISIBLE_DEVICES` value.

## Experiment record template

Copy this section for every smoke test and formal run.

### RUN-ID: short description

- Status: planned | running | succeeded | failed | cancelled
- Start time:
- End time:
- Repository:
- Git commit:
- Git worktree state before run:
- Full command:
- Configuration:
- Dataset and split:
- Dataset source/hash:
- Checkpoint path/hash:
- Random seed:
- GPU and CUDA:
- Main PID or process group:
- Log path:
- Output path:
- Number of frames:
- Runtime:
- Peak GPU memory:
- Final metrics:
- Validation performed:
- Cleanup result:
- Notes and failure analysis:
