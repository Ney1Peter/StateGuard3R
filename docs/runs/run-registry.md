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

### SETUP-0007: Final pre-report Gate 0 resource recheck

- Time: 2026-08-01 02:17 CST
- Result: both external blockers remain; no CUDA context or download started
- `/data` free space: about 513 GiB (97% used)

A Project2-scoped filename search, excluding pytest fixtures, again found no
real `cut3r_*.pth` checkpoint and no `.part`/`.partial` file. The official 512
Google Drive URL was probed once with an HTTP HEAD request bounded by an
8-second connection timeout and 15-second total timeout; DNS resolution timed
out and returned HTTP code `000`, with zero response-body bytes. No third-party
source was contacted. The pinned ReCal3R worktree remained clean at
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`.

Every L20 still had at least one existing compute process. Approximate memory
use by index was `21038, 30854, 21175, 18909, 24651, 26139, 25247, 41495` MiB,
so no card met the empty-card rule. A task-path process sweep and listening-port
check found no long-lived StateGuard3R/ReCal3R process or service owned by this
work. No existing process was changed.

### SETUP-0008: Exhaust official checkpoint metadata and local reuse paths

- Time: 2026-08-01 02:27–02:29 CST
- Result: no confirmed alternate official checkpoint path and no local copy
- Download or partial file created: no

Read-only GitHub API checks found no ReCal3R issue mentioning the checkpoint
and no release assets in either `Powertony102/ReCal3R` or `CUT3R/CUT3R`. CUT3R
issue 4 contains an unanswered request to host models on Hugging Face. In issue
18, a repository contributor links a Hugging Face dataset repository for
processed datasets, not model weights. Bounded HEAD probes for the two expected
checkpoint filenames under that dataset repository could not connect; they do
not establish that such files exist. No unverified URL was promoted to an
official checkpoint source.

A broader read-only filename search across `/data/wangzheng`, as permitted by
the server rules for reuse checks, found neither supported checkpoint under an
alternate local path. This exhausts the currently discoverable safe sources;
future progress requires the official Google endpoint to recover or a
user-provided, source-traceable file.

### SETUP-0009: Retry Gate 0 resources after the source audit

- Time: 2026-08-01 02:32 CST
- StateGuard3R commit: `957499e457e4717bff4df901e18a0ccc2998ae6c`
- Result: checkpoint and GPU gates remained blocked; no download or CUDA task started
- `/data` free space: about 512 GiB (97% used)

No real supported checkpoint or partial download was present. A bounded probe
of the official Google Drive source timed out again, and no third-party source
was contacted. Approximate L20 memory use by index was
`395, 30854, 20595, 18909, 24651, 26139, 25249, 41495` MiB. Every card still
had an existing compute process, including the low-memory process on GPU 0, so
none met the empty-card rule. No existing process was changed.

Subsequent runner-readiness work added pinned checkpoint-loader overrides
(`8f6360c`, `e2d5f86`), froze the official runtime parameters
(`59a1eac`, `8bf6154`), and recorded the loaded cuRoPE binary provenance
(`2c44ea8`, `483ade1`). These commits and their tests strengthen the Gate 0
guards; they do not constitute checkpoint loading or a model forward.

### SETUP-0010: Final pre-launch recheck after runner hardening

- Time: 2026-08-01 02:58–03:14 CST
- StateGuard3R code commit: `483ade139400bce27a5f94bc9c9a08a8b0d40c65`
- Result: checkpoint and GPU gates remained blocked; Gate 0 was not launched
- `/data` free space at the final check: about 506 GiB (97% used)
- CPU regression check with CUDA hidden: `174 passed in 1.16s`

Bounded HEAD probes of the official `drive.google.com`,
`drive.usercontent.google.com`, and `docs.google.com` URL forms all timed out
with HTTP `000` and zero response bytes. No 512 or 224 supported checkpoint,
`.part`, or `.partial` file existed under `/data/wangzheng`, and no download was
started.

At 03:14 CST, approximate memory use across GPU indices 0–7 was
`395, 30854, 8592, 18909, 24651, 26139, 25249, 41495` MiB. Every card had an
existing compute process. In particular, GPU 0's low-memory process was an
existing root-owned Open WebUI service, so it was neither idle nor available
to this task. No process was stopped or modified, and no CUDA context was
started. The two external gates must both be rechecked before any later launch.

### SETUP-0011: User-supplied official checkpoint and renewed GPU gate

- Time: 2026-08-01 17:19–17:25 CST
- Source declared by the user: the official ReCal3R README Google Drive link,
  file ID `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD`
- Path: `/data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`
- Size: 3,173,761,006 bytes (about 2.96 GiB)
- SHA-256: `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Owner: `wangzheng:wangzheng`; regular file; one hard link; not a symlink
- Final mode: read-only `0444`
- `/data` free space after upload: about 312 GiB (98% used)
- Result: static checkpoint and GPU pre-launch gates passed; no deserialization
  or CUDA work occurred in this setup step

The uploaded file had a PyTorch-style ZIP header and 1,240 archive entries,
contained no obvious HTML/error-page marker in its leading bytes, and had no
adjacent `.part` or `.partial` file. Its size, mtime, and SHA-256 remained
unchanged across the static audit and permission hardening. The official source
does not publish a size or checksum, so this first local digest is an
intra-project integrity baseline and does not independently authenticate the
source. The runner must verify it again before and after deserialization.

The user explicitly allowed a GPU with sufficient free memory even when it is
not completely empty, provided existing processes are not stopped or
interfered with. The task still prefers an empty card. Two snapshots seven
seconds apart found GPU 4 and GPU 3 stable at 45,586 MiB free, 0% utilization,
P8, and no compute PID. GPU 4 is the first choice and GPU 3 the fallback; the
actual launch must repeat the check immediately before binding one card.

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

Post-run audit found that this smoke's declared dynamic occlusion used
`dx=dy=0`, so its rectangle was static. The v1 coordinate contract is valid,
but `DEV-SYNTH-0003` supersedes it as the current moving-occlusion pipeline
evidence.

### DEV-SYNTH-0003: Moving-occlusion synthetic CLI smoke

- Status: succeeded; current synthetic pipeline evidence
- Time: 2026-08-01 02:13–02:14 CST
- Repository commit: `08e59d0f43c7ec0af7e48d487215b29c000ec39a`
- Output: `outputs/synthetic-smoke-0003` (128 KiB; no raster copies)
- Manifest schema: `stateguard3r.corruption.v1`
- Detection execution mode: exploratory, not formal

This run repeated the 0002 command sequence in a new directory after freezing
non-zero occlusion velocity `dx=0.04`, `dy=0.025`. The five normalized rectangle
origins were `(0.25,0.25)`, `(0.29,0.275)`, `(0.33,0.30)`, `(0.37,0.325)`, and
`(0.41,0.35)`, proving that replay metadata changes on every occluded frame.
There were still 60 frames and three disjoint five-frame corruption intervals.

The synthetic health fixture did not change, so overall synthetic-only metrics
remained identical to 0002: combined, update-only, and reliability-only had
AUROC/F1 1.0, FPR 0, and mean delay 0; random had AUROC 0.5659259259, F1
0.3829787234, FPR 0.5111111111, and mean delay 0.3333333333. This equality is
expected and is not evidence that pixels were processed by ReCal3R.

```text
source_manifest.json     55bab88af20e7fd21e7b13b77c34d100e2e6fd75ad6c708dceb11b38f9a17e41
corruption_specs.json    ec0327f7424889354dea8743b0d7b2a30e0b277ddbb3410764949c7b3d2cbb10
health.jsonl             c725772db7761a92cc019adb6170a983bc39c01e60e5116091fb08c210b4d525
corruption_manifest.json 864133efe020a024674f49ca3c0eaaa5a6a84ee45422e492bc14bdaba11657f2
metrics.json             afe496ebbd285ffb95dbb2cdfb940e136d3451f941fe1d642d12efc3bd2ae051
timeline.svg             f9bc651def0d97cad33ed101b2efc6d458c6671629772ed245a1a7ee926e0bea
```

The embedded health/corruption snapshot hashes matched, the SVG parsed, and
both Chateau source hashes remained unchanged.

### DEV-SYNTH-0003-MATERIALIZATION-AUDIT: CPU pixel replay

- Status: succeeded; supplements the original 0003 CLI smoke
- Time: 2026-08-01 02:49–02:50 CST
- StateGuard3R clean commit: `483ade139400bce27a5f94bc9c9a08a8b0d40c65`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Input: the 60 final ordered paths in `synthetic-smoke-0003`
- Loader: official `load_images_for_eval`, size 512 with its normal crop
- Materialization: `stateguard3r.input_manifest.materialize_manifest_views`
- Output shape: every view was `1x3x384x512`
- CUDA: hidden with an empty `CUDA_VISIBLE_DEVICES`; PyTorch reported
  `torch.cuda.is_initialized() == False`
- Files written: none; zero raster outputs

The first inline attempt incorrectly passed `crop=True` to the generic
`load_images` function. That API does not accept a `crop` keyword, so it raised
`TypeError` before `materialize_manifest_views` ran; it produced no output file
or CUDA work. The corrected audit then used the runner's actual preprocessing
entry point, `load_images_for_eval(size=512, crop=True)`, and completed the
materialization checks reported below.

This audit actually applied the deferred transforms in memory. Every output
image used storage distinct from its corresponding loader input, all 60 input
tensor hashes were unchanged after materialization, and the two source-image
hashes were unchanged:

```text
Chateau1.png  71ffb8c7d77e5ced0bb3dcd2cb0db84d0e98e6ff5ffd2d02696a7156e5284857
Chateau2.png  c3a0be9e19f6b89491d692c71e3f2317c2288a898a990561d48b7667218b47c8
```

For every dynamic-occlusion frame, the observed changed-pixel bounding box
equalled the rectangle produced by the v1 normalized-coordinate contract. The
right and bottom endpoints below are exclusive. Every pixel inside was exactly
the normalized red fill `[1, -1, -1]`, and every pixel outside was unchanged.

| Frame | Normalized origin | Pixel bbox `(x0,y0,x1,y1)` | Changed pixels |
|---:|---|---|---:|
| 25 | `(0.25,0.25)` | `(128,96,384,288)` | 49,152 |
| 26 | `(0.29,0.275)` | `(148,105,405,298)` | 49,601 |
| 27 | `(0.33,0.30)` | `(168,115,425,308)` | 49,601 |
| 28 | `(0.37,0.325)` | `(189,124,446,317)` | 49,601 |
| 29 | `(0.41,0.35)` | `(209,134,466,327)` | 49,601 |

The other corruption routes also matched their manifest provenance:
`low_overlap_jump` selected source indices `50–54`, and
`wrong_order_segment` selected `[44,43,42,41,40]`. Because this fixture repeats
only two Chateau images, the selected path changed on only 3/5 low-overlap
frames and 4/5 wrong-order frames; the center frame of the reversal maps to
itself. Those routes validate ordering and provenance, not realistic
low-overlap or temporal-corruption strength.

This is CPU-only pixel-materialization evidence. It did not load ReCal3R
weights, run a model forward, or produce real health signals, and it does not
upgrade the synthetic detection metrics into research results.

### GATE0-REAL-0001: Official ReCal3R two-image smoke

- Status: **failed before model-to-CUDA/forward due to a runner interface
  false-negative**; retained and not overwritten
- Start/end evidence: 2026-08-01 17:28:59–17:29:25 CST
- StateGuard3R commit: `1fb53690254221daabdec97c3b32d1a1783b04cc`
- Runner SHA-256: `03b535ce9ceb18de75eb3c39795746eea4047617dd315d0f416a4724b1492622`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Main PID: `3754415`; observed process exit code: `1`
- GPU binding: `CUDA_VISIBLE_DEVICES=4`; preflight 45,586 MiB free, no compute
  PID, 0% utilization, P8
- Checkpoint: 3,173,761,006-byte official 512 DPT 4–64, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Inputs: Chateau1 SHA-256 `71ffb8c7d77e5ced0bb3dcd2cb0db84d0e98e6ff5ffd2d02696a7156e5284857`;
  Chateau2 SHA-256 `c3a0be9e19f6b89491d692c71e3f2317c2288a898a990561d48b7667218b47c8`
- Log: `logs/gate0-real-0001.log`, 1,554 bytes, SHA-256
  `6a337604a3ab6f1070eb103f74f79fd44cc719a798ab05d57ddf0d5af4491dc3`
- Output: `outputs/gate0-real-0001/checkpoint-load-audit.json`, 106 bytes,
  SHA-256 `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d`

Full model command after the separately recorded Git/checkpoint/GPU preflight:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 \
TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp \
UV_CACHE_DIR=/data/wangzheng/Project2/.cache/uv \
HF_HOME=/data/wangzheng/Project2/.cache/huggingface \
TORCH_HOME=/data/wangzheng/Project2/.cache/torch \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
  /data/wangzheng/Project2/StateGuard3R/scripts/run_recal3r_smoke.py \
  --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
  --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --image /data/wangzheng/Project2/baselines/ReCal3R/src/croco/assets/Chateau1.png \
  --image /data/wangzheng/Project2/baselines/ReCal3R/src/croco/assets/Chateau2.png \
  --output-dir /data/wangzheng/Project2/StateGuard3R/outputs/gate0-real-0001 \
  --device cuda --size 512 --seed 0 --beta-base 0.1 \
  2>&1 | tee /data/wangzheng/Project2/StateGuard3R/logs/gate0-real-0001.log
```

Static preflight and preprocessing passed. The official loader
deserialized the checkpoint, instantiated `ARCroco3DStereo` with
`head_type='dpt'` and `output_mode='pts3d+pose'`, and reported
`<All keys matched successfully>`. The load audit confirms `strict=false` with
empty missing and unexpected key lists.

The runner then incorrectly read the three outer architecture fields from
`model.config`, which the pinned CroCo parent constructor replaces with a
narrower `CrocoConfig`. It therefore observed `head_type=None` and stopped at
the interface guard. `model.to(cuda)`, cuRoPE execution, forward, the expected
single calibrated update, runtime/peak-memory measurement, health output,
trajectory, and `run.json` did not occur. This failure is not checkpoint
incompatibility and is not a research result.

Commits `5a8e609` and `ee5768c` correct the guard to validate the effective
model/downstream fields and add pinned-like conflict/absence/pose-structure
tests. GPU 4 returned to 4 MiB used, 45,586 MiB free, 0% utilization, P8, with
no compute PID. Checkpoint and input hashes remained unchanged. The next
attempt must use new run ID `GATE0-REAL-0002`; TUM remains locked until it
passes.

### GATE0-REAL-0002: Corrected official ReCal3R two-image smoke

- Status: **succeeded — Gate 0 PASS**
- Start/end: 2026-08-01 17:49:00–17:49:29 CST
- StateGuard3R commit: `5231580b15750cfaa6ce374ce7b67b29f65fa574`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Main PID: `3882890`; exit code: `0`
- GPU: physical index 4, NVIDIA L20; two preflight snapshots each reported
  45,586 MiB free, 0% utilization, P8, and no compute PID
- Checkpoint: 3,173,761,006 bytes, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Inputs: the same two Chateau files and hashes recorded for 0001
- Log: `logs/gate0-real-0002.log`, 6,961 bytes, SHA-256
  `b3678bb06e6a13be94e369100e038b8a1f684696cb441617d66ab4b35627a22c`
- Output: `outputs/gate0-real-0002`

The model command matched 0001 except for the new output/log paths and the
corrected clean runner commit. It used the ReCal3R isolated interpreter,
project-local TMP/cache paths, `CUDA_VISIBLE_DEVICES=4`, `--device cuda
--size 512 --seed 0 --beta-base 0.1`, and the frozen checkpoint SHA. The exact
argv is also embedded in `run.json`. Both repositories were tracked-clean at
launch.

Observed result:

- the checkpoint loaded with `strict=false`; missing and unexpected keys were
  empty;
- the effective interface was `head_type=dpt`, `DPTPts3dPose`,
  `output_mode=pts3d+pose`, with pose flag and decoder present;
- the actual cuRoPE extension path/size/SHA matched the frozen provenance;
- two frames caused exactly `N-1 = 1` calibrated update and trace step `[1]`;
- inference-scope runtime was 1.052124 seconds, 1.900916 FPS, and peak allocated
  memory was 6,362.670 MiB; the approximately 28.995-second wall time included
  startup and checkpoint loading and must not be confused with FPS scope;
- health, trajectory, and prediction summaries each aligned to two frames and
  all recorded numeric values were finite;
- frame 1 recorded uncertainty 0.797944, reliability 0.202056,
  `global_state_delta=1.665824`, pose jump 5.136780, and geometric residual
  0.0558443.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                3d621a94ce71870b186ff24724f32331c31a38986767d71918f32fb70e8b67e6
predictions-summary.json    e713a28cf2cd44b5eb718571140e1f879dbf5e1ed913dfe93ae3ee9e45d6f0ca
run.json                    92f365150742fce0684ed60e9733d0db607a8d7888773fd52765caaa32bc76b3
trajectory.json             2003ceefbd22dc6c85b4ae133523e424a46590802203dc32d9b9ab43012b8bc0
```

`run.json.log_path` is `null` because shell `tee` owns the external log; the
registry path and hash above bind it to this run. `update_magnitude`,
candidate/final beta, and local-memory delta remain `null`; only the recorded
`global_state_delta` may be claimed. The two Chateau images are not a real
continuous sequence, so the pose jump is a smoke signal rather than accuracy
or drift evidence. Checkpoint/input hashes remained unchanged. After exit,
GPU 4 returned to 4 MiB used, 45,586 MiB free, 0% utilization, P8, with no
compute PID. This PASS unlocks only the next 4–8-frame smoke, not detection or
quarantine.

### GATE0-STATE4-0001: Four-frame mechanical state smoke

- Status: **succeeded — four-frame state/trace gate PASS**
- Start/end: 2026-08-01 18:19:38–18:20:05 CST
- StateGuard3R commit: `4e97a06cc6e6b70f5c92aaa76516683c9cc0a642`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Main PID: `4079642`; exit code: `0`; GPU: physical index 4
- Inputs, in order: Chateau1, Chateau2, `arch.jpg`, Chateau1; files were
  referenced read-only and not copied
- `arch.jpg` SHA-256: `05fbf12896a79819a3864a800b174896bd3b6fa29b4f4f580d06725ff7c30dc7`
- Log: `logs/gate0-state4-0001.log`, 8,325 bytes, SHA-256
  `837de34cede7f7ad15e8eec29506d67d3acd84ebe51e7c49ad5de764b2f49830`
- Output: `outputs/gate0-state4-0001`

The exact argv is embedded in `run.json`. It matched the successful 0002
command and environment, added the third and fourth `--image` arguments above,
and used unique `gate0-state4-0001` output/log paths. Two immediate preflight
snapshots again showed GPU 4 at 45,586 MiB free, 0% utilization, P8, and no
compute PID.

All four frames completed with exactly `N-1=3` calibrated updates and trace
steps `[1,2,3]`. Health, trajectory, and prediction summaries each contained
four aligned frames, and every recorded numeric value was finite. The official
loader produced `512x384` model inputs for the Chateau images and `512x176`
for the wide `arch.jpg`, exercising mixed aspect ratios. Inference-scope
runtime was 1.348210 seconds, 2.966896 FPS, and peak allocated memory was
6,367.173 MiB.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                60c979e2898242b45f7e43dbdd44e2cb446b80a5868a8c0723b91e1866a00b23
predictions-summary.json    55dccdc74af7375aa2b1af2c2ce5c3f5cc99c905ef2f1d78284cba67d2f687a0
run.json                    9326a04653283473470d3fb3545c50c8fd663a0035cb1eda925af457aa0c5a5a
trajectory.json             57847393c73d90f47bed68cbccaa14002196f37bea68b6f1d9c5c5547855ce3b
```

Checkpoint and input hashes remained unchanged, and GPU 4 returned to 4 MiB
used with no compute PID. Because this sequence repeats assets and is not a
continuous capture, it proves recurrent count/trace/output alignment only; it
is not a clean-scene, accuracy, drift, or detection result. It unlocks only the
small TUM AVI Gate 1. Candidate/final beta, `update_magnitude`, and
local-memory delta remained `null`; only `global_state_delta` was measured.
`run.json.log_path` is also `null`, so the external log path and hash above are
the binding log record.

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
