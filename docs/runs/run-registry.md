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

### SETUP-0005: Retry official checkpoint connectivity after runner freeze

- Time: 2026-08-01 01:11–01:14 CST
- Final checkpoint file ID: `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD`
- Result: blocked before download; no real checkpoint or partial file created
- `/data` free space at retry: about 515 GiB (97% used)

Read-only probes covered the official `drive.google.com`,
`drive.usercontent.google.com`, and `docs.google.com` download forms. IPv4
connection attempts returned HTTP code `000`: the first and third endpoints
timed out while connecting to port 443, and the user-content hostname timed out
during resolution. No third-party mirror was used. Same-named 31-byte files
under `StateGuard3R/tmp/pytest-*` are unit-test fixtures, not checkpoints; no
real `cut3r_512_dpt_4_64.pth` or `cut3r_224_linear_4.pth` was found.

### SETUP-0006: Recheck all GPUs after runner freeze

- Time: 2026-08-01 01:11 CST
- Result: no completely idle GPU; no CUDA context or task started

Every L20 had at least one existing compute process. Approximate memory use by
index was `7755, 30854, 1195, 18909, 24651, 26139, 25247, 41495` MiB. GPU 2
remained the least occupied but still had another user's process, so it did not
pass the empty-card rule. No process was stopped or modified.

## Development pipeline records

### DEV-SYNTH-0001: Original synthetic CLI smoke

- Status: stale development artifact retained without modification
- Output: `outputs/synthetic-smoke-0001`
- Reason: manifest declares `stateguard3r.corruption.v0` and predates the
  required dynamic-occlusion coordinate-reference contract
- Current use: historical debugging only; not reproducible with the v1 loader

Its earlier perfect synthetic detector scores must not be cited as either a
current pipeline run or a ReCal3R result.

### DEV-SYNTH-0002: Corruption v1 and detection CLI smoke

- Status: succeeded
- Time: 2026-08-01 02:03–02:07 CST
- Repository commit: `f0b2748ad276ffaf875e93f7a70937c5d6ea9075`
- Output: `outputs/synthetic-smoke-0002`
- Output size: 128 KiB
- Input files copied: none
- Frames: 60 synthetic ledger rows referencing two existing Chateau images
- Corruptions: `low_overlap_jump`, `dynamic_occlusion`, and
  `wrong_order_segment`, five labeled frames each
- Manifest schema: `stateguard3r.corruption.v1`
- Detection execution mode: exploratory, not formal

Commands, all with project-local `TMPDIR` and `UV_CACHE_DIR`, were run in this
order:

```text
export TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp
export UV_CACHE_DIR=/data/wangzheng/Project2/.cache/uv
uv run python scripts/generate_synthetic_smoke_inputs.py \
  --output-dir outputs/synthetic-smoke-0002/inputs \
  --frame-a /data/wangzheng/Project2/baselines/ReCal3R/src/croco/assets/Chateau1.png \
  --frame-b /data/wangzheng/Project2/baselines/ReCal3R/src/croco/assets/Chateau2.png
uv run python -m stateguard3r.corruption \
  outputs/synthetic-smoke-0002/inputs/source_manifest.json \
  outputs/synthetic-smoke-0002/corruption_manifest.json \
  --config outputs/synthetic-smoke-0002/inputs/corruption_specs.json \
  --seed 0 --check-paths --sequence synthetic_smoke_0002_v1
uv run python -c "from stateguard3r.input_manifest import load_input_manifest; load_input_manifest('outputs/synthetic-smoke-0002/corruption_manifest.json')"
uv run python -m stateguard3r.detection \
  outputs/synthetic-smoke-0002/inputs/health.jsonl \
  outputs/synthetic-smoke-0002/corruption_manifest.json \
  --output outputs/synthetic-smoke-0002/metrics.json \
  --window 15 --threshold 3.0 --seed 0 \
  --method-threshold random=0.5 \
  --method-threshold update_magnitude_only=3.0 \
  --method-threshold reliability_only=-0.5 \
  --method-threshold combined=3.0
uv run python -m stateguard3r.timeline \
  outputs/synthetic-smoke-0002/inputs/health.jsonl \
  outputs/synthetic-smoke-0002/metrics.json \
  --output outputs/synthetic-smoke-0002/timeline.svg \
  --title "Synthetic smoke 0002 — NOT A REAL EXPERIMENT"
```

Synthetic-only metrics:

| Method | AUROC | F1 | FPR | Mean delay (frames) |
|---|---:|---:|---:|---:|
| random | 0.5659259259 | 0.3829787234 | 0.5111111111 | 0.3333333333 |
| update magnitude only | 1.0 | 1.0 | 0.0 | 0.0 |
| reliability only | 1.0 | 1.0 | 0.0 | 0.0 |
| combined | 1.0 | 1.0 | 0.0 | 0.0 |

Artifact and input hashes:

```text
Chateau1.png             71ffb8c7d77e5ced0bb3dcd2cb0db84d0e98e6ff5ffd2d02696a7156e5284857
Chateau2.png             c3a0be9e19f6b89491d692c71e3f2317c2288a898a990561d48b7667218b47c8
health.jsonl             c725772db7761a92cc019adb6170a983bc39c01e60e5116091fb08c210b4d525
corruption_manifest.json 193026f9af2cb8fbf2626a0ec67276a75c979553d2327d4415ca0d4350811da5
metrics.json             075bad94ad25b55fd3e0ade9f2bc8db1ec34441030277c87dc174298faba7453
timeline.svg             3bf67fb6fc0e65d7a240de418fa7430bd674a2cd82f465ec3087d83ad908958d
```

The metrics' embedded health/corruption snapshot hashes match the files above;
the SVG parsed successfully. ReCal3R was not loaded or run, and none of these
numbers is evidence about real corruption detection.

### GATE0-REAL-0001: Official ReCal3R two-image smoke

- Status: blocked before launch
- Intended inputs: the two Chateau images above
- Intended checkpoint: official 512 DPT 4–64
- Blocking conditions: checkpoint unavailable and no completely idle GPU
- CUDA process/PID/output: none

The CPU-only official image loader did successfully load both source images at
shape `1x3x384x512` with value range `[-1, 1]`, without changing their hashes.
It used the ReCal3R `.venv`, project-local `TMPDIR`, and an empty
`CUDA_VISIBLE_DEVICES`; no CUDA context was requested. This validates
preprocessing only, not checkpoint loading or model forward.

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
