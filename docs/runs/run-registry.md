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

### GATE1-DATA-0001: Official TUM `fr1_xyz` AVI and ten-frame prefix

- Status: **download and CPU prefix-decode PASS; GPU smoke not yet run**
- Time: 2026-08-01 18:34–18:38 CST
- Canonical URL: `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz-rgb.avi`
- Final official redirect: `https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz-rgb.avi`
- Download target: `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_xyz-rgb.avi`
- Size: 8,059,298 bytes; mode `0444`; no remaining `.part`
- SHA-256: `1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022`
- Derived prefix: `baselines/ReCal3R/data/tum/derived/fr1_xyz-gate1-10-v1`
- Frame manifest SHA-256: `1905a62ea7aa4285c2607a009540ada2844490c279b0daea50dd5260028e6b7e`
- Derived size: about 4.1 MiB; ten PNGs and the manifest are mode `0444`

A bounded official HEAD check returned one 302 followed by HTTP 200 with
`Content-Length: 8059298`, `Content-Type: video/x-msvideo`,
`Last-Modified: Thu, 29 Sep 2011 14:32:22 GMT`, and ETag
`"7af9a2-4ae15606ac980"`. `curl --fail --location --continue-at -` wrote only
the `.part` path; the received size matched exactly before atomic rename.
RIFF/AVI magic, `file`, and `ffprobe` passed. The stream is MPEG-4/FMP4,
640x480, 30 FPS, 26.6 seconds, and reports 798 frames.

With CUDA hidden, ReCal3R's isolated Python and OpenCV 4.11 sequentially read
indices 0 through 9. Every decoded frame was `480x640x3 uint8`; each was saved
once as a read-only PNG. The manifest records every path, byte size, SHA-256,
shape, dtype, OpenCV version, source hash, and nominal `frame_index/fps` time.
That nominal time is explicitly not an original TUM timestamp. Source AVI
SHA-256 was unchanged after decoding.

Read-only audit logs and SHA-256 values:

```text
gate1-fr1-xyz-avi-headers.txt   a3c40eea2745ecc4f6b2e0455dd8d1f4785a88ba9a1bbe5548ac15dd7f1bd9d4
gate1-fr1-xyz-avi-download.log  2bc6bff8ec828eb798c4538e4d8e4fce43ccf0393a5dfb5ea8dfdcbaf04c6360
gate1-fr1-xyz-avi-file.txt      b86a4c9ac1beac91b30799522e742c720df27371d165be2f10d463479b36b9ff
gate1-fr1-xyz-avi-ffprobe.txt   002248e507346c0fcba915ad4c884c4b1afaf353d6616c05bc4d1dac55bb533b
gate1-fr1-xyz-decode10.log      0f8e7787f2d79a36f69c2392c4184b1dc99a5af09ad4c398eaef698747757c12
```

This AVI has no original RGB/depth association files, timestamps, or
ground-truth trajectory. It is only a continuous RGB smoke input and cannot
support ATE/RPE or a formal research conclusion. The next gate is a ten-frame
GPU run with exactly nine updates; neither the 30-frame expansion nor the TUM
TGZ is unlocked yet.

### GATE1-FR1XYZ-10-0001: Ten-frame continuous RGB smoke

- Status: **succeeded — ten-frame Gate 1 sub-gate PASS**
- Start/end: 2026-08-01 18:51:24–18:51:51 CST
- StateGuard3R commit: `95033bad226df5d13fad3e2481c90aba27050dcb`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Main PID: `98422`; exit code: `0`; GPU: physical index 4
- Source AVI SHA-256: `1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022`
- External input manifest: `baselines/ReCal3R/data/tum/derived/fr1_xyz-gate1-10-v1/frame-manifest.json`,
  SHA-256 `1905a62ea7aa4285c2607a009540ada2844490c279b0daea50dd5260028e6b7e`
- Checkpoint: `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`, 3,173,761,006 bytes,
  SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Configuration: CUDA, image size 512, seed 0, base beta 0.1; the hashed `run.json`
  is the authoritative full argv including all ten explicit `--image` arguments
- Log: `logs/gate1-fr1xyz-10-0001.log`, 13,435 bytes, SHA-256
  `1ae047ce0da226902c461ec49a4ba8db21786fb38655dd011b04fdec30fd9e14`
- Output: `outputs/gate1-fr1xyz-10-0001`

The exact clean-run argv in `run.json` lists the ten read-only PNGs strictly
from `frame_000000.png` through `frame_000009.png`; `run.json.input_manifest`
is `null` because the video-frame manifest is not the corruption schema. This
registry externally binds every run input to the manifest and source AVI.
Two immediate preflight snapshots found GPU 4 at 45,586 MiB free, 0%
utilization, P8, and no compute PID.

All ten frames completed with nine expected/observed calibrated updates and
trace steps `[1,2,3,4,5,6,7,8,9]`. Health, trajectory, prediction, input, and
manifest frame IDs aligned exactly to 0–9; all recorded numeric values and 60
tensor summaries were finite. Frames 1–9 contained non-null, non-constant real
signals: geometric residual ranged 0.00716487–0.01079948, pose jump
0.00945283–0.02421211, uncertainty 0.79086339–0.80310434, reliability
0.19689566–0.20913661, and `global_state_delta` 0.88016944–1.28335999.
Reliability and pose-jump formulas independently recomputed within numerical
precision.

Inference-scope runtime was 1.988277 seconds, 5.029481 FPS, and peak allocated
memory was 6,364.891 MiB; the 26.761966-second wall time includes loading.
After exit, PID `98422` was absent and GPU 4 returned to 4 MiB with no compute
process. Checkpoint, AVI, manifest, runner, relpose source, cuRoPE, and all ten
input hashes remained unchanged.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                151435ad8312206042c2208946e9e2435f7a4f090f262142aae8d36be38fa04a
predictions-summary.json    78f6dafcc67f0f1a458a1351a0d3d83d52b06f11caa7f1134dc2ad57cfc2fb4c
run.json                    08b8e4adab654bfff0d8604c092fd780427d0475ba90779bd07f9ed01ceffceb
trajectory.json             f17f3677eae35fd83b1173ea9d531c5e00c87269f34bda309cc7359deee0b085
```

Candidate/final beta, internal `update_magnitude`, local-memory delta,
overlap, risk/decision, and timestamp remain `null`; the first-frame
update/reliability fields are also null by reference-frame design.
`run.json.log_path` is null, so the external log binding above is authoritative.
This is only the first 0.3 seconds of RGB video without depth, original TUM
timestamps, or GT. It cannot support ATE/RPE, long-term stability, detection,
or quarantine claims. It unlocks only the same AVI's 30-frame smoke.

### GATE1-DATA-0002: Thirty-frame prefix without duplicating frozen frames

- Status: **CPU preparation and independent validation PASS**
- Time: 2026-08-01 19:16–19:18 CST
- StateGuard3R base commit: `762e9d7f08ab38569909aff40036634363339052`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Source AVI SHA-256: `1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022`
- Frozen ten-frame manifest SHA-256: `1905a62ea7aa4285c2607a009540ada2844490c279b0daea50dd5260028e6b7e`
- Derived manifest: `baselines/ReCal3R/data/tum/derived/fr1_xyz-gate1-30-v1/frame-manifest.json`,
  12,524 bytes, SHA-256 `1db34e2f88e7bbb67104dc5bc5dbbfb3b4bebfc5a0b0398b3373dacc5deee041`
- Preparation script: `baselines/ReCal3R/logs/gate1-fr1-xyz-decode30.py`, SHA-256
  `7f1f9db56d72ca152b36534846c91a38917d84f716d8e3727504f9cc43f98130`
- Preparation log: `baselines/ReCal3R/logs/gate1-fr1-xyz-decode30.log`, 639 bytes,
  SHA-256 `2d04fa9c05a46258648e49b4d8f677ea347586e17a980875cdbc1f766e6c56ca`

With CUDA hidden, the pinned ReCal3R environment's OpenCV 4.11 read AVI
indices 0–29 sequentially. Frames 0–9 were compared pixel-for-pixel with the
frozen read-only prefix and were neither written nor copied. Only frames 10–29
were losslessly encoded into a same-filesystem staging directory, read back,
set to mode `0444`, validated, and atomically renamed to the final directory.
The final directory therefore contains exactly 20 PNGs plus the manifest;
the manifest references the old directory for indices 0–9. Actual new bytes,
including the manifest, are 7,968,045 bytes.

The v2 manifest records all 30 ordered frame indices, resolved storage policy,
per-frame path/size/SHA/shape/dtype, nominal time, AVI metadata, OpenCV version,
source hash, and frozen-prefix lineage. A separate post-publication audit again
decoded indices 0–29 and compared all 30 referenced PNGs pixel-for-pixel; all
paths, hashes, modes, shapes, dtypes, and index sets matched. The source AVI
and old manifest hashes remained unchanged, and no partial file remained.

This is data preparation only and used no GPU. It unlocks exactly
`GATE1-FR1XYZ-30-0001`; the TUM TGZ, ATE/RPE, formal detection,
quarantine, and rollback remain locked until that run passes its own gate.

### GATE1-FR1XYZ-30-0001: Thirty-frame continuous RGB smoke

- Status: **succeeded — thirty-frame Gate 1 PASS**
- Start/end: 2026-08-01 19:38:39–19:39:14 CST
- StateGuard3R commit: `c1fd41f13a159f1572518fb0d82dfc4d4f50a505`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Main PID: `525828`; exit code: `0`; GPU: physical index 4
- Source AVI SHA-256: `1820b52939af2a2e7afdf78f6398cfbc7816399c8809b3836a9ba8da9bef8022`
- External input manifest: `baselines/ReCal3R/data/tum/derived/fr1_xyz-gate1-30-v1/frame-manifest.json`,
  SHA-256 `1db34e2f88e7bbb67104dc5bc5dbbfb3b4bebfc5a0b0398b3373dacc5deee041`
- Checkpoint: `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`, 3,173,761,006 bytes,
  SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Configuration: CUDA, image size 512, seed 0, base beta 0.1; the hashed `run.json`
  is the authoritative full argv including all 30 explicit `--image` arguments
- Log: `logs/gate1-fr1xyz-30-0001.log`, 29,177 bytes, SHA-256
  `1658e2ec847b1b77f41d9dd4972347e3d449b787041732c2b38b941b28009ac9`
- Preflight GPU log: `logs/gate1-fr1xyz-30-0001-gpu-preflight.log`, 397 bytes,
  SHA-256 `ce52af5e972744ffd7cf612cd8bc7f14b886c977692f1e6ad90eb50862ffb177`
- Postflight GPU log: `logs/gate1-fr1xyz-30-0001-gpu-postflight.log`, 383 bytes,
  SHA-256 `77ead2adf78bcff0fc84a7504723002635761d9a9d52a2ca22ec51c2e27d2b80`
- Output: `outputs/gate1-fr1xyz-30-0001`

The launch used the frozen 0–9 paths followed by the incremental 10–29 paths,
all explicit and ordered. `run.json.input_manifest` is `null` because the v2
video-frame manifest is provenance rather than a StateGuard3R corruption
schema; this registry externally binds the run to it. Two earlier interactive
snapshots and the persisted preflight found GPU 4 at 45,586 MiB free, 0%
utilization, P8, and no compute PID. `gpustat` was unavailable, so the persisted
check used `nvidia-smi` GPU, compute-app, and `pmon` views; no process was
stopped or disturbed.

All 30 frames completed with 29 expected/observed calibrated updates and trace
steps `[1..29]`. Health, trajectory, prediction, input, and manifest frame IDs
aligned exactly to 0–29. All JSON numeric values and 180 tensor summaries were
finite. The first ten health, trajectory, and prediction entries matched the
independent ten-frame run exactly. Frames 1–29 contained non-null,
non-constant real signals: geometric residual ranged
0.00543514–0.01079948, pose jump 0.00945283–0.02580239, uncertainty
0.79086339–0.80325240, reliability 0.19674760–0.20913661, and
`global_state_delta` 0.51733727–1.28335999. Reliability, pose-jump, and
cross-file signal equality independently recomputed within numerical precision.

Inference-scope runtime was 4.440337 seconds, 6.756244 FPS, and peak allocated
memory was 6,366.722 MiB; the 34.749758-second wall time includes loading.
At postflight, PID `525828` was absent and GPU 4 was back to 4 MiB with no
compute process. Checkpoint, source AVI, both manifests, runner, all 30 input
images, and recorded code provenance hashes remained unchanged.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                1169e293c771e4d865ff8dd18e86b084da05d858708d37c571cf25993d45bd80
predictions-summary.json    5b9a942739d2470666b583a8ce6a9859ad237d4e2eb01b50fc4da4327cf4e7ab
run.json                    7ccad5540cd0625953204192e663aad9cc2d8480e67264af80228db0fe1dc0c4
trajectory.json             f83506e6ecf2498e03c332a18a571d7c8833cef8844616acdf29024e6e9f111b
```

Candidate/final beta, internal `update_magnitude`, local-memory delta,
overlap, risk/decision, and timestamp remain `null`; the first-frame
update/reliability fields are null by reference-frame design.
`run.json.log_path` is also null, so the external log binding above is
authoritative. The run covers only the first approximately 0.97 seconds of an
RGB-only video without depth, original TUM timestamps, or GT. It cannot support
ATE/RPE, long-term stability, formal detection, or quarantine claims. It
unlocks only the bounded Gate 2 `fr1_desk.tgz` acquisition and smallest smoke;
it is not a research Go decision.

### GATE2-DATA-0001: Official `fr1_desk` TGZ and read-only raw tree

- Status: **download, archive validation, extraction, and independent audit PASS**
- Time: 2026-08-01 19:59–20:23 CST
- StateGuard3R base commit: `23f91e382c953f072907b44a0313aaa607400ae6`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Canonical URL: `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz`
- Final URL: `https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz`
- Archive: `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk.tgz`,
  344,011,403 bytes, mode `0444`, SHA-256
  `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`
- Raw tree: `baselines/ReCal3R/data/tum/rgbd_dataset_freiburg1_desk`,
  directories mode `0555`, files mode `0444`
- Raw manifest: `baselines/ReCal3R/logs/gate2-fr1-desk-raw-manifest.json`,
  231,139 bytes, SHA-256
  `5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`

A `/data/wangzheng` filename search found no existing archive or partial. The
pre-download filesystem snapshot had 342,318,374,912 bytes free. A fresh HEAD
returned the expected 302/200 redirect, `Content-Length: 344011403`, byte-range
support, 2011-09-30 Last-Modified, and ETag. Only this planned TGZ was downloaded
to a same-directory `.part`; curl completed 344,011,403 bytes in 720.696681
seconds before any publication. No `fr1_xyz` TGZ, `fr1_desk` AVI, standalone
GT, fr2/fr3 sequence, or other dataset was acquired.

While still partial, exact size, `gzip -t`, gzip listing, local SHA-256, and a
Python tar-member audit passed. The tar stream is 368,220,160 bytes and has
exactly 1,215 members: three directories and 1,212 regular files totaling
367,285,514 bytes. It has one expected top-level directory, no duplicate,
absolute, parent-traversal, link, or special member, and contains 613 RGB PNGs,
595 depth PNGs, `rgb.txt`, `depth.txt`, `groundtruth.txt`, and the official
`accelerometer.txt`. The TGZ was set to `0444`, atomically renamed, then audited
again with the same hash. TUM publishes no cryptographic checksum; the SHA above
is this project's first-download identity, not an official authenticity claim.

Extraction used a unique same-filesystem staging directory. All archive members
were first checked in staging, all 1,212 files were hashed, and every RGB image
decoded as 480x640x3 `uint8`; every depth image decoded as 480x640 `uint16`.
`rgb.txt` and `depth.txt` contain 613 and 595 strictly ordered timestamp/path
rows matching the image files; `groundtruth.txt` contains 2,335 finite
eight-field poses with ordered timestamps and unit quaternions. The raw manifest
was regenerated after publication and matched its pre-publication bytes exactly.

The first directory-publication attempt was deliberately stopped when moving a
staged root already set to `0555` returned permission denied. The final target
was still absent. The same fully audited staging root alone was temporarily set
to `0755` for the parent-changing atomic rename, immediately restored to `0555`,
and fully re-audited; no re-extraction, overwrite, or half-published target
occurred. The extract log preserves both attempts. No `.part`, staging,
symlink, hardlink, or temporary manifest remains, and both Git tracked trees
are clean. This entire gate was CPU/storage-only and used no GPU.

Read-only audit hashes:

```text
gate2-fr1-desk-tgz-headers.txt             5be3bec9136a1068f0c71975f7421b73ce4e008bab4898a9c3b6693227ee2b5b
gate2-fr1-desk-tgz-download.log            7023c17c78e10a7195fffa89c67ae1a021fa42b3f3c960630b210c1bea55d7a8
gate2-fr1-desk-tgz-audit.py                1e7b35bab4e1fb348f82c9686e73f3c688c75e5def36741aeb8da96253246e91
gate2-fr1-desk-tgz-file.txt                ce9bf79350499d6042f49f24f12619dcd5899aa7caf53f6ffc185e3f6f3653cc
gate2-fr1-desk-tgz-gzip.txt                b29edcf2149514f47f8e6dcf796ea308538190c0c098d9c381ef4eacb20edc97
gate2-fr1-desk-tgz-tar-audit.json          cc332a0188e3173f0fdfca11af91f58040f5621fcb649cfde5c08975bd3e0f5b
gate2-fr1-desk-tgz-postpublish-audit.json  da2425c79c5765ea0e9777df0043334b6e593c833ee09ac7f7db3acffce20405
gate2-fr1-desk-extraction-audit.py         9d1b175af46bf3045613ad01af4c21caaa788c5526600caf8235e3e20990ad95
gate2-fr1-desk-extract.log                 7adfc412c370a5e2281249c6a259e9c1ae17635148f0a55ceb580691577bf21a
```

This PASS proves archive/raw-layout integrity and unlocks only the smallest
eight-frame I/O/forward gate. It is not an inference, pose-accuracy, health,
detection, or research result. The raw tree remains untouched; any future
continuous-window manifest must live under the writable sibling `derived/`
and reference the raw files without copying them.

### GATE2-WINDOW8-DATA-0001: Deterministic contiguous `fr1_desk` window

- Status: **reference-only input preparation and independent audit PASS**
- Time: 2026-08-01 20:50–20:56 CST
- StateGuard3R base commit: `4c13eba7093105de0c56847df33cbd4b3bcd25ff`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Window manifest: `baselines/ReCal3R/data/tum/derived/fr1_desk-gate2-8-v1/window-manifest.json`,
  11,697 bytes, mode `0444`, SHA-256
  `24548d1110af58ca5599bba5737396b8c4b96478b6a4d4f137cc832ae55e9e35`
- Preparation script: `baselines/ReCal3R/logs/gate2-fr1-desk-window8.py`,
  SHA-256 `110989ab9601632946f7664f131934e085b833456b252889b9b4e3cf289b2879`
- Preparation log: `baselines/ReCal3R/logs/gate2-fr1-desk-window8.log`,
  330 bytes, SHA-256 `c1687088896243b27aead364c92c64b67dff1c8b2c11b674d2bb20256888f93d`
- Frozen raw-manifest SHA-256: `5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`
- Frozen TGZ SHA-256: `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`

The deterministic rule scans `rgb.txt` in source order and selects the earliest
eight consecutive RGB entries for which every frame has a unique nearest depth
and ground-truth timestamp within 0.02 seconds; ties are ordered by absolute
delta, timestamp, then physical source line, and matches must remain unique
within the window. The first eligible window is 0-based non-comment RGB data
indices 17–24 (physical `rgb.txt` lines 21–28), from timestamp
1305031453.359684 through 1305031453.591640, spanning 0.231956 seconds. Its
maximum absolute RGB-depth delta is 0.014616966 seconds and maximum absolute
RGB-GT delta is 0.004184008 seconds.

The `stateguard3r.tum-window.v1` manifest records all eight original RGB,
depth, and GT associations, source lines/timestamps, poses, relative paths,
sizes, hashes, raw-manifest/archive lineage, and the original TUM timestamp
semantics. It was written in a same-filesystem staging directory, all references
were revalidated, then the one-file directory was atomically published and set
read-only. No RGB or depth image was copied, linked, rewritten, or generated;
the derived directory contains only this manifest. A separate audit rebuilt the
earliest-window selection and every association from the three original text
files and reproduced the same indices, deltas, paths, poses, and hashes. No
partial or staging file remains, and the raw tree stayed read-only and unchanged.

The official evaluator's `--max_frames 8` is intentionally not used for this
gate: on this sequence it approximately samples source indices
`[0,1,88,175,262,349,436,523]`, which is not a contiguous recurrent window.
The next run is `GATE2-FR1DESK-8-0001` with eight explicit RGB paths and an
expected seven updates/trace `[1..7]`. Depth and GT are provenance associations
only because the external smoke runner consumes RGB. Therefore this data PASS
cannot support ATE/RPE, a depth metric, long-term stability, detection, or a
research decision, and it unlocks no 30–50-frame experiment before the
eight-frame forward itself passes.

### GATE2-FR1DESK-8-0001: Eight-frame contiguous raw-TUM smoke

- Status: **succeeded — Gate 2 eight-frame I/O/forward PASS**
- Start/end: 2026-08-01 21:20:51–21:21:15 CST
- StateGuard3R commit: `4d4a30286296d630e9f6fbd9eb67be218d9a2c9a`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Main PID: `1107500`; exit code: `0`; GPU: physical index 4
- External window manifest: `baselines/ReCal3R/data/tum/derived/fr1_desk-gate2-8-v1/window-manifest.json`,
  SHA-256 `24548d1110af58ca5599bba5737396b8c4b96478b6a4d4f137cc832ae55e9e35`
- Raw-manifest SHA-256: `5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`
- TGZ SHA-256: `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`
- Checkpoint: `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`, 3,173,761,006 bytes,
  SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Configuration: CUDA, image size 512, seed 0, base beta 0.1; the hashed `run.json`
  is the authoritative full argv with eight explicit ordered `--image` paths
- Log: `logs/gate2-fr1desk-8-0001.log`, 12,154 bytes, SHA-256
  `70f95b529fda91985a6a477e1cd797aaa6520743dd05ff727c9d19d88653cd3f`
- Preflight GPU log: `logs/gate2-fr1desk-8-0001-gpu-preflight.log`, 397 bytes,
  SHA-256 `9118d614ff76c73a1517a21a0822f40aa958bc189a16cdead50010f3f7282743`
- Postflight GPU log: `logs/gate2-fr1desk-8-0001-gpu-postflight.log`, 383 bytes,
  SHA-256 `202c59505faad2b1001f4e795a2219b5105bc9276f5242eb731571f12a0433d9`
- Output: `outputs/gate2-fr1desk-8-0001`

The exact argv uses the manifest's eight raw RGB paths in source order.
`run.json.input_manifest` is `null` because this TUM window manifest is
provenance rather than the corruption schema; this registry externally binds
the window, raw tree, depth/GT associations, and run. Both tracked trees were
clean. Two interactive checks and the persisted preflight found GPU 4 at
45,586 MiB free, 0% utilization, P8, and no compute PID. `gpustat` was
unavailable, so `nvidia-smi` GPU, compute-app, and `pmon` views were recorded;
no process was stopped or disturbed.

All eight frames completed with seven expected/observed calibrated updates and
trace steps `[1..7]`. Health, trajectory, prediction, input, and window frame
IDs aligned exactly to 0–7; every JSON numeric value and all 48 tensor summaries
were finite. Frames 1–7 had non-null, non-constant signals: geometric residual
ranged 0.01338305–0.01868337, pose jump 0.01803411–0.03922328, uncertainty
0.78875065–0.81150341, reliability 0.18849659–0.21124935, and
`global_state_delta` 1.11154728–1.32416431. Reliability, pose-jump, and
cross-file signal equality were independently recomputed within numerical
precision.

Inference-scope runtime was 1.501171 seconds, 5.329174 FPS, and peak allocated
memory was 6,364.891 MiB; the recorded post-validation runner interval,
including imports and model loading, was 23.313784 seconds.
Postflight found PID `1107500` absent and GPU 4 back at 4 MiB with no compute
process. Checkpoint, TGZ, raw manifest, window manifest, all eight RGB inputs,
runner, and recorded code provenance hashes remained unchanged.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                f423d2dc46239098a69d0ad103ec6557fb9ec06d9cd2237815112315ecd87a35
predictions-summary.json    055624fa500a3e2e1ddbbdfd9e54d1953f93b7c0116c9c2f7dbe4c79fa15c0e8
run.json                    86031349efd5ed1364eae66eddfd687c0a4619aaf9bc7f33b4011110078afede
trajectory.json             c9e548537c7857046b2ce2bf9fa989bb970cb5c55958bc5395281064c79cac37
```

Candidate/final beta, internal `update_magnitude`, local-memory delta,
overlap, risk/decision, and per-frame timestamp remain `null`; first-frame
update/reliability fields are null by reference-frame design. The external
manifest retains original TUM timestamps and depth/GT associations, but this
runner consumes RGB only. Therefore this approximately 0.232-second result
cannot support ATE/RPE, a depth metric, long-term stability, formal detection,
or quarantine. It unlocks only a reference-only contiguous 30-frame clean
ledger preparation and smoke, not a research Go decision or 50-frame/formal
corruption experiment.

### GATE2-WINDOW30-DATA-0001: Deterministic contiguous 30-frame `fr1_desk` window

- Status: **succeeded — Gate 2 30-frame input preparation PASS**
- Prepared/audited: 2026-08-01 21:39–21:41 CST
- StateGuard3R commit: `e326be3acb19f1a92147747f6de4e4802b8c43dc`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Window manifest: `baselines/ReCal3R/data/tum/derived/fr1_desk-gate2-30-v1/window-manifest.json`,
  38,255 bytes, SHA-256
  `8c7ade9ee01d22ebd895e034f46130aa8c6d834a80b3e24d2b5a5ca1a2b08575`
- Parent eight-frame manifest SHA-256:
  `24548d1110af58ca5599bba5737396b8c4b96478b6a4d4f137cc832ae55e9e35`
- Raw-manifest SHA-256: `5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`
- TGZ: 344,011,403 bytes, SHA-256
  `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`
- Materialization: references only; one manifest, no copied or linked image

The same deterministic rule as the eight-frame gate was extended to the
earliest 30 consecutive entries whose independent nearest depth and ground
truth timestamps are unique within the window and no more than 20 ms away.
This selects 0-based non-comment RGB indices 17–46 (physical `rgb.txt` lines
21–50), timestamps 1305031453.359684–1305031454.327699, and a
0.968014956-second span. All 30 depth and all 30 ground-truth associations are
unique; maximum absolute RGB-depth and RGB-GT deltas are respectively
0.014616966 and 0.004184008 seconds. Every earlier possible start fails the
20-ms depth condition.

The manifest freezes all 30 RGB/depth file sizes and hashes, all source lines
and exact timestamp text, GT poses, TGZ/raw-manifest/text-file provenance, and
the parent manifest path/hash. Its first eight frame objects are structurally
identical to the already-passed eight-frame manifest. An independent audit
rebuilt the earliest-window search and every association from the raw text,
then matched every referenced raw file against the raw manifest. The derived
directory is mode `0555`, its only file is mode `0444`, and all selected raw
files remain single-link read-only files. Both tracked worktrees were clean at
publication, no partial or staging path remains, and no new dataset was
downloaded.

This input PASS unlocks only `GATE2-FR1DESK-30-0001` from a clean commit with
30 explicit ordered RGB paths. Required invariants are 29 calibrated updates,
trace `[1..29]`, 180 finite tensor summaries, exact cross-file frame alignment,
and a first-eight-frame output prefix equal to the independent eight-frame run.
Depth and GT remain external provenance and are not consumed by this RGB-only
runner, so
the prepared input cannot support ATE/RPE, depth metrics, detection, long-term
stability, or a research Go. The 50-frame, formal corruption/detection, and
quarantine gates remain locked.

### GATE2-FR1DESK-30-0001: Thirty-frame contiguous raw-TUM clean ledger

- Status: **succeeded — Gate 2 30-frame clean ledger PASS**
- Start/end: 2026-08-01 21:59:17–21:59:45 CST
- StateGuard3R commit: `bdcd73cd8197abe73604b21c6142c96ecdbfb514`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Main PID: `1315129`; exit code: `0`; GPU: physical index 4
- External window manifest: `baselines/ReCal3R/data/tum/derived/fr1_desk-gate2-30-v1/window-manifest.json`,
  SHA-256 `8c7ade9ee01d22ebd895e034f46130aa8c6d834a80b3e24d2b5a5ca1a2b08575`
- Parent eight-frame manifest SHA-256:
  `24548d1110af58ca5599bba5737396b8c4b96478b6a4d4f137cc832ae55e9e35`
- Raw-manifest SHA-256: `5908db0f357fd4a21b2c777220de38e651b48b80c7d53182123c6eb4e7163d87`
- TGZ SHA-256: `e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c`
- Checkpoint: `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth`, 3,173,761,006 bytes,
  SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Configuration: CUDA, image size 512, seed 0, base beta 0.1; the hashed
  `run.json` is the authoritative argv with 30 explicit ordered `--image` paths
- Log: `logs/gate2-fr1desk-30-0001.log`, 30,244 bytes, SHA-256
  `bdfbac7b045a073c48d1555e3b2297819b7d970d7ae95c17cdfa5879368f7e46`
- Preflight GPU log: `logs/gate2-fr1desk-30-0001-gpu-preflight.log`, 14,593 bytes,
  SHA-256 `84b64af82b450d629e54428bd77f0b630ba9968012c21d0f728eaa73db00f286`
- Postflight GPU log: `logs/gate2-fr1desk-30-0001-gpu-postflight.log`, 7,225 bytes,
  SHA-256 `640c45d601f5a30c10583458298e3c762d1065ee7663f3752965f3be676e5f90`
- Output: `outputs/gate2-fr1desk-30-0001`

The manifest's 30 raw RGB paths were passed explicitly in order. Both tracked
trees were clean, and the runner recorded that code and input hashes matched
the frozen values. Two persisted preflight snapshots found GPU 4 at 4 MiB
used, 45,586 MiB free, 0% utilization, P8, and without a compute PID;
`gpustat` was unavailable, so full `nvidia-smi`, GPU, compute-app, and `pmon`
views were retained. No process was stopped or disturbed.

All 30 frames completed with 29 expected/observed calibrated updates and trace
steps `[1..29]`. Health, trajectory, prediction, image, and input-frame IDs
aligned exactly to 0–29; all JSON numbers and all 180 tensor summaries were
finite. Across the available frames, geometric residual ranged
0.01146577–0.01876780, pose jump 0.01803411–0.05329814, uncertainty
0.77685690–0.81150341, reliability 0.18849659–0.22314310, and
`global_state_delta` 0.97143427–1.32416431. Reliability bounds, pose-jump
composition, and cross-file signal equality were independently recomputed.
The first eight health objects, trajectory frames/metadata, prediction
summaries, images, and input-frame records exactly equal the independent
eight-frame run, confirming deterministic prefix behavior.

Inference-scope runtime was 3.419778 seconds, 8.772500 FPS, and peak allocated
memory was 6,365.548 MiB. The recorded post-validation runner interval was
28.159521 seconds. Postflight found PID `1315129` absent and GPU 4 at 4 MiB
with no compute PID; a later audit found it at 45,586 MiB free, 0%, P8.

Output hashes:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                0438de6108831b6bc3d0b911d47763e73e7a4b81730577bb73479bd60b5aa6bb
predictions-summary.json    29a88e0d10d09a8a28c2eda232c134767c7b2c8b28b94129df0c98236afd0b5b
run.json                    425ec6a01d486f228895d1cef711e2c5a0f09f7e4e3dad4d064da5e7f053d9cc
trajectory.json             bfc3389e00c538e950b81670f239580ea1c2a354abadca80ea7963616c2d41e8
```

`run.json.input_manifest`, `log_path`, and per-frame timestamp remain `null`;
the registry therefore externally binds the window and logs. Runner source
indices 0–29 are local and correspond to raw TUM indices 17–46. Depth/GT did
not enter forward, and overlap, internal `update_magnitude`, candidate/final
beta, local-memory delta, and risk/decision remain unavailable. This result is
only a deterministic, approximately 0.968-second RGB recurrent smoke: it
cannot support ATE/RPE, a depth metric, long-term stability, formal detection,
quarantine, or a research Go. It unlocks only preparation and separate
exploratory smokes for the three planned corruptions on the same frozen source;
the 50-frame and formal gates remain locked.

### GATE2-CORRUPTION-INPUTS-0001: Three isolated 30-frame exploratory inputs

- Status: **succeeded — source adaptation, generation, and CPU replay PASS**
- Prepared/audited: 2026-08-01 23:08–23:13 CST
- StateGuard3R base commit: `fb26038d087cf29e03c4a77c95a836f138c51388`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Parent 30-frame window SHA-256:
  `8c7ade9ee01d22ebd895e034f46130aa8c6d834a80b3e24d2b5a5ca1a2b08575`
- Shared source adapter: `baselines/ReCal3R/data/tum/derived/fr1_desk-gate2-30-corruption-source-v1/source-manifest.json`,
  37,827 bytes, mode `0444`, SHA-256
  `ca7f2532a529f3eb8044644d4b8434f7a2f654d55003da8f9995dad4ee2ce8cc`
- Input bundle: `outputs/gate2-fr1desk-30-corruption-inputs-v1`, directory mode
  `0555`; exactly three configs and three generated manifests, all mode `0444`
- Generator SHA-256: `683132672136bdb60c940147556fc1c55ca8063c1b309bba6a4e8875975f5b5a`
- Strict replay/materializer SHA-256:
  `e42852f87e704c5b9b90e12a9fa10506ec6f38290b9cd1a01d065e5dc77106c2`
- Generation/audit log: `logs/gate2-fr1desk-30-corruption-inputs-v1.log`,
  4,946 bytes, mode `0444`, SHA-256
  `cbd53547f235d7c4c1b59926f552d3068e09223aa1fc55961a5fcb67cd02b674`

Frozen bundle hashes and sizes:

```text
dynamic-occlusion-config.json       1,333  561850ad129f9e231ad6fabb62944c2d5f82ec113656a49ca30adcc252357d56
dynamic-occlusion-manifest.json    44,989  8a9746079ab56c7b0020de2c73f0544e6bfaeee2bb59e2fef8c4150174588485
low-overlap-jump-config.json        1,064  0ef31b8c4bd1dc9adb5a381faf9c4c8dfc4e4d3886e781ad288a81da272d8b5a
low-overlap-jump-manifest.json     43,026  8da158a85d19aaec12be53ea52357cf65cf038909fe59f673c0491a6e37be830
wrong-order-segment-config.json     1,024  307f20c07a350b35211e2e819defd6f36bb2aca25b54488c20d1842be2ecf22d
wrong-order-segment-manifest.json  42,779  b1f205411da42e96707f7d158cc9caf73f5d0f69c621ee0188ded965928259dc
```

The TUM window schema stores RGB paths under `frames[i].rgb.path`, while the
corruption v1 generator requires `frames[i].path`. A deterministic one-file
adapter therefore promotes each RGB path and preserves its original TUM
timestamp text/value, raw index/line, size/hash, depth association, GT pose,
parent-window hash, and archive/raw/text lineage. It remains beside the other
ReCal3R derived data so its relative paths resolve directly to the same 30
single-link read-only RGB files. Its directory is mode `0555`; it contains no
image, link, cache, or materialized frame.

Each generated `stateguard3r.corruption.v1` manifest has seed 0, exactly one
corruption, 30 output frames, a ten-frame clean prefix, and the same absolute
source-adapter path/hash:

- `low_overlap_jump` substitutes local target positions 10–14 with source
  indices 25–29 (raw RGB indices 42–46). The target corresponds to raw indices
  27–31. This is an explicitly frozen distant-segment index/GT-pose proxy;
  actual image overlap was not measured and is not claimed.
- `dynamic_occlusion` applies a normalized red rectangle over positions 10–14.
  It starts at `(0.25,0.25)`, has width/height 0.5, and moves by
  `(dx,dy)=(0.04,0.025)` per frame without clipping. The coordinate reference
  is `model_input_after_resize_and_center_crop`.
- `wrong_order_segment` reverses positions 10–13 to source indices
  `[13,12,11,10]`; its even length ensures all four paths change.

All three strict-load checks passed. A CUDA-hidden CPU audit then used the
official `load_images_for_eval(size=512,crop=True)` path and materialized each
manifest in memory. Every result tensor was an independent clone; low-overlap
and wrong-order pixels exactly matched their final ordered source paths. The
five dynamic rectangles changed only the expected exclusive pixel boxes:

```text
(128,  96, 384, 288)  49,152 pixels
(148, 105, 405, 298)  49,601 pixels
(168, 115, 425, 308)  49,601 pixels
(189, 124, 446, 317)  49,601 pixels
(209, 134, 466, 327)  49,601 pixels
```

Every rectangle interior was exactly normalized red `[1,-1,-1]`, every pixel
outside was unchanged, loader inputs remained unchanged, and CUDA was never
initialized. All source RGB hashes, sizes, mtimes, modes, and link counts were
identical before and after. No PNG or related staging, partial, or temporary
artifact remains in the source-adapter or published bundle.

The first read-only adapter audit attempted to construct an expected path by
splitting a `../../` relative path incorrectly and raised an assertion before
generation; it wrote nothing. The corrected check resolved both paths against
their manifest directories and passed. This failure and retry are retained in
the frozen log rather than hidden.

This input gate is not model-forward, low-overlap measurement, detection, or a
research result. The ten-frame prefix is also shorter than the example formal
window of 15. It unlocks only the low-overlap exploratory GPU smoke. Dynamic
occlusion remains sequentially locked until that run is audited and recorded;
wrong-order remains locked until both preceding runs pass. No threshold may be
tuned from these smoke outputs and described as frozen development calibration;
the 50-frame, formal detection, quarantine, and rollback gates remain locked.

### GATE2-FR1DESK-30-LOW-OVERLAP-0001: Exploratory index/GT-pose proxy response

- Status: **succeeded — exploratory Health Ledger and evidence audit PASS**
- Launch/model/postflight: 2026-08-01 23:26:45–23:27:20 CST
- StateGuard3R commit: `e37a86afd56e99713acea10085cb8aed8546b16d`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Manifest SHA-256: `8da158a85d19aaec12be53ea52357cf65cf038909fe59f673c0491a6e37be830`
- Source-adapter SHA-256:
  `ca7f2532a529f3eb8044644d4b8434f7a2f654d55003da8f9995dad4ee2ce8cc`
- Checkpoint: 3,173,761,006 bytes, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Main PID: `1801419`; exit code `0`; physical GPU 4, UUID
  `GPU-0227ecd0-4186-14bc-2c6b-6bf6377fbd7f`
- Output: `outputs/gate2-fr1desk-30-low-overlap-0001`, final directory mode
  `0555`; exactly five files, all mode `0444`
- Main log: `logs/gate2-fr1desk-30-low-overlap-0001.log`, 63,231 bytes,
  mode `0444`, SHA-256
  `af4355b0ff6cb9913598e1c6317241ceef4dbabab67a8d88415ffc700126a0b9`
- Preflight log: `logs/gate2-fr1desk-30-low-overlap-0001-gpu-preflight.log`,
  17,069 bytes, mode `0444`, SHA-256
  `efa2389b0fbaa3d8cdc45a9886bfb72d88d4d3149da94611efe8cdf5c83ae099`
- Postflight log: `logs/gate2-fr1desk-30-low-overlap-0001-gpu-postflight.log`,
  7,965 bytes, mode `0444`, SHA-256
  `e613bb807288de2e1971d922adbbb9780d103d01f808b038358250f9a1aa913f`
- Audit log: `logs/gate2-fr1desk-30-low-overlap-0001-audit.log`, 2,190 bytes,
  mode `0444`, final external SHA-256
  `308c8cc0429683e60feb94edcc2a2e3dac836e2027800f5852a2ca25281aa420`

The exact command, also retained in the preflight log and `run.json`, was:

```text
CUDA_VISIBLE_DEVICES=4 \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
/data/wangzheng/Project2/StateGuard3R/scripts/run_recal3r_smoke.py \
  --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
  --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --input-manifest /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-corruption-inputs-v1/low-overlap-jump-manifest.json \
  --output-dir /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-low-overlap-0001 \
  --device cuda --size 512 --seed 0 --beta-base 0.1
```

Both immediately persisted preflight snapshots found physical GPU 4 at 4 MiB
used, 45,586 MiB free, 0% utilization, P8, and without a compute PID. The two
tracked worktrees were clean, the unique output/log targets did not exist, and
all frozen code, input, source, and checkpoint hashes matched before launch. No
existing process was stopped or changed.

The strict manifest replay and the runner independently agreed on the exact
source-index order:

```text
0..9, 25..29, 15..29
```

Thus local frames 10–14 substitute source 25–29 (raw RGB indices 42–46), while
all other frames remain identity references. All resolved paths, current image
hashes, metadata, transforms, and the `run.json.images` projection matched the
manifest exactly. The 30 source RGB files remained size/hash/mode/link-count
identical. Health timestamps follow this final order, including the expected
14-to-15 time reversal and the repeated timestamps at 10–14 and 25–29.

The model completed 30 predictions, 29 expected/observed calibrated updates,
trace `[1..29]`, and 180 finite tensor summaries. Health, trajectory, prediction,
and input IDs all align to 0–29; checkpoint missing/unexpected keys are empty.
The first ten prediction summaries and trajectory objects equal the clean
30-frame run exactly. The first ten health objects also equal clean exactly when
only the manifest-supplied `timestamp` field is omitted. This summary equality
does not claim bitwise equality of unpersisted full prediction tensors.

Observed signal ranges were:

```text
geometric_residual  0.0112456265 .. 0.0187678006
pose_jump           0.0180341104 .. 0.5918007661
uncertainty_u       0.7683731914 .. 0.8115034103
reliability         0.1884965897 .. 0.2316268086
global_state_delta  0.9001695568 .. 1.6197634888
```

Frame 10 has the maximum pose jump (`0.5918007661`) and global-state delta
(`1.6197634888`). All five substituted frames have global-state delta
`1.3776924675..1.6197634888`, above the clean run's overall maximum
`1.3241643105`. Frame 15 has a second pose-jump peak (`0.5828957277`) when the
input returns from source 29 to source 15. That frame is outside the manifest's
10–14 label but is an adjacent causal boundary/spillover, so a later formal
protocol must state how it is scored rather than silently counting it as an
ordinary clean false positive. Frames 10–29 differ from clean in all three
persisted output families, as expected once the recurrent state sees a different
frame; this alone is not a formal detection or damage-propagation conclusion.

Inference-scope runtime was 3.476418 seconds (8.629573 FPS), and peak allocated
memory was 6,366.722 MiB. PID `1801419` was absent after exit. Immediate
postflight found GPU 4 at 4 MiB with no compute PID; utilization was still 71%
for that sample, then a later check found 0% and P8. The output hashes are:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                8425d6d0977d56dac85177afcb43ebc705fd23dd751703c6d0631221da7f9f57
predictions-summary.json    73f3ca326958990ee84b5f16f4b7c9b46cc1948eb74b4e79b4784eb57825c979
run.json                    26431ae868f43358837c5c6618edf642307af3feedc27888b763849daeefe65d
trajectory.json             ce5c9d39dc53e139903057da18d16d567bd515fbbb507ee7c9a483ae0d0e465c
```

The first local read-only audit incorrectly assumed that two top-level
trajectory semantic-description strings were signal arrays and stopped on that
assertion without writing any evidence file. The corrected audit used the
per-frame schema, passed all finite/mapping/prefix/source assertions, and records
the failed assumption in the frozen audit log. A digest printed for that log
while `tee` was still writing is explicitly marked provisional; the final
external digest above is authoritative.

This is one exploratory response to a distant-index/GT-pose proxy. Actual image
overlap was not measured; there is no threshold calibration, holdout, AUROC,
F1/FPR/delay, ATE/RPE, or comparison against the four detection methods. It
therefore keeps the research decision at HOLD and unlocks only the separately
frozen dynamic-occlusion exploratory smoke. Wrong-order, 50 frames, formal
detection, quarantine, and rollback remain locked.

### GATE2-FR1DESK-30-DYNAMIC-OCCLUSION-0001: Controlled moving-red-box response

- Status: **succeeded — exploratory Health Ledger, pixel replay, and audit PASS**
- Launch/model/postflight: 2026-08-02 00:08:58–00:09:33 CST
- StateGuard3R commit: `dbc7c73c2d9911ff6841cf0e425629e056517ee6`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Manifest SHA-256: `8a9746079ab56c7b0020de2c73f0544e6bfaeee2bb59e2fef8c4150174588485`
- Source-adapter SHA-256:
  `ca7f2532a529f3eb8044644d4b8434f7a2f654d55003da8f9995dad4ee2ce8cc`
- Checkpoint: 3,173,761,006 bytes, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Main PID: `2035444`; exit code `0`; physical GPU 4, UUID
  `GPU-0227ecd0-4186-14bc-2c6b-6bf6377fbd7f`
- Output: `outputs/gate2-fr1desk-30-dynamic-occlusion-0001`, final directory mode
  `0555`; exactly five files, all mode `0444`
- Main log: `logs/gate2-fr1desk-30-dynamic-occlusion-0001.log`, 64,335 bytes,
  mode `0444`, SHA-256
  `3c4187c2073f0111b44445fceb56dff978891217b3bf570a5ec9c6e5e26b4042`
- Preflight log: `logs/gate2-fr1desk-30-dynamic-occlusion-0001-gpu-preflight.log`,
  17,075 bytes, mode `0444`, SHA-256
  `992d9a62a59e6ebd696cb43338eb8e112214d2b8cb784eaa155b9c8ca3bc4cfe`
- Postflight log: `logs/gate2-fr1desk-30-dynamic-occlusion-0001-gpu-postflight.log`,
  7,965 bytes, mode `0444`, SHA-256
  `e90c4a752d8ec1f6fc6606c3f1d32c92f84bff4c1147b4baacf7459225cd5e46`
- Audit log: `logs/gate2-fr1desk-30-dynamic-occlusion-0001-audit.log`,
  3,263 bytes, mode `0444`, SHA-256
  `26743569627c573fdb411938459d3a621d1543873f577f694d7e30b42a899d51`

The command differed from the preceding smoke only in its frozen manifest and
unique output directory:

```text
CUDA_VISIBLE_DEVICES=4 \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
/data/wangzheng/Project2/StateGuard3R/scripts/run_recal3r_smoke.py \
  --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
  --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --input-manifest /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-corruption-inputs-v1/dynamic-occlusion-manifest.json \
  --output-dir /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-dynamic-occlusion-0001 \
  --device cuda --size 512 --seed 0 --beta-base 0.1
```

Both persisted preflight snapshots found GPU 4 at 4 MiB used, 45,586 MiB free,
0% utilization, P8, and without a compute PID. Both tracked worktrees and all
frozen hashes were clean/current, and no unique target existed. No other process
was stopped or changed.

The strict manifest and runner both retain identity source order 0–29. Only
frames 10–14 have one `rectangle_occlusion` transform each; all other transforms
are empty. The red `[255,0,0]` rectangle starts at normalized `(0.25,0.25)`, has
width/height 0.5, and moves by `(0.04,0.025)` per frame in
`model_input_after_resize_and_center_crop` coordinates. A CUDA-hidden official
CPU-loader replay independently obtained 30 cloned `1x3x384x512` tensors and
these exact exclusive boxes:

```text
(128,  96, 384, 288)  49,152 changed pixels
(148, 105, 405, 298)  49,601 changed pixels
(168, 115, 425, 308)  49,601 changed pixels
(189, 124, 446, 317)  49,601 changed pixels
(209, 134, 466, 327)  49,601 changed pixels
```

Every box interior was exactly normalized red `[1,-1,-1]`; every outside pixel,
loader input, and all 30 source PNG size/hash/mtime/mode/link-count records were
unchanged. CUDA remained uninitialized throughout replay. `run.json.images`
correctly records raw PNG hashes, not transformed-tensor hashes; the pixel claim
comes from the frozen transform plus official-loader replay, not that field alone.

The CUDA run completed 30 predictions, 29 expected/observed updates, trace
`[1..29]`, and 180 finite tensor summaries. The three persisted output families
equal clean exactly only at frames 0–9 when the manifest timestamp is excluded
from health. Health timestamps equal the unchanged, strictly increasing TUM
order. Health and trajectory signals align exactly, and all 29 pose jumps were
independently recomputed from the saved matrices.

Observed full-run ranges were:

```text
geometric_residual  0.0113145745 .. 0.0187678006  (maximum frame 8)
pose_jump           0.0180341104 .. 0.0535758505  (maximum frame 26)
uncertainty_u       0.7790557146 .. 0.8115034103
reliability         0.1884965897 .. 0.2209442854
global_state_delta  1.0170857439 .. 1.3241643105
```

At frame 10, the largest-magnitude signed differences for each field within the
five-frame corruption interval versus the corresponding clean frames were pose jump
`+0.02050335`, uncertainty `+0.01832145`, reliability `-0.01832145`, global-state
delta `+0.18258630`, and geometric residual `-0.00187669`. Across all five
occluded frames, uncertainty and global-state delta were higher than their clean
counterparts while reliability was lower. At frame 15 the occlusion is gone;
pose-jump difference is only `-0.00073739`, but global-state delta remains
`+0.08418447`, and all three persisted output families remain different through
frame 29. This is an observed recurrent spillover, not proof of damage propagation.

The global pose maximum occurs at frame 26, geometric residual peaks inside the
clean prefix, and the full-run global-state maximum remains clean frame 1. These
facts preclude describing this smoke as a successful detector peak. Inference
runtime was 3.240399 seconds (9.258118 FPS), with 6,366.722 MiB peak allocated.
PID `2035444` was absent after exit. Immediate postflight found 4 MiB and no
compute PID at 91% sampled utilization; the later check returned 0% and P8.

Output hashes are:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                095fc04ac188cb3b132d3f35fb73c9730dc0e5c4abfba7a53540e432f3cfc627
predictions-summary.json    b3221b662ffd703d40e36910b40ef6730b07844393b888fae0a0f0f0cee356ec
run.json                    c359862c216d44a2a5c4aadd5cad6f64b80863b9deda251f7f7222b1c09e3e01
trajectory.json             ce82480f02100502a92c9ad2a3670ca44696c5ef6602f92345e9f5030f0b79c0
```

One read-only static helper initially accessed the nested rectangle as if it
were a flat transform key and raised `KeyError`; it wrote nothing. The corrected
nested parse matched all five rectangles, and the independent CPU pixel replay
above passed on its first attempt.

This is a controlled in-memory red rectangle, not a natural dynamic-object
experiment. It has no threshold calibration, holdout metrics, full-tensor
bitwise comparison, detection result, or research Go. It keeps the decision at
HOLD and unlocks only the separately frozen wrong-order exploratory smoke; 50
frames, formal detection, quarantine, and rollback remain locked.

### GATE2-FR1DESK-30-WRONG-ORDER-0001: Controlled four-frame reverse response

- Status: **succeeded — exploratory Health Ledger, pixel replay, and three-way audit PASS**
- Launch/model/postflight: 2026-08-02 00:19:44–00:20:17 CST
- StateGuard3R commit: `e134ae7808f3afa5166f11ee42c5036fd5dc93bc`
- ReCal3R commit: `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- Runner SHA-256: `091ac783fb3eae62ea3835e58c3e3f56fd32b2306477ba158a8ba722abc8a40e`
- Manifest SHA-256: `b1f205411da42e96707f7d158cc9caf73f5d0f69c621ee0188ded965928259dc`
- Source-adapter SHA-256:
  `ca7f2532a529f3eb8044644d4b8434f7a2f654d55003da8f9995dad4ee2ce8cc`
- Checkpoint: 3,173,761,006 bytes, SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`
- Main PID: `2096352`; exit code `0`; physical GPU 4, UUID
  `GPU-0227ecd0-4186-14bc-2c6b-6bf6377fbd7f`
- Output: `outputs/gate2-fr1desk-30-wrong-order-0001`, final directory mode
  `0555`; exactly five files, all mode `0444`
- Main log: `logs/gate2-fr1desk-30-wrong-order-0001.log`, 63,004 bytes,
  mode `0444`, SHA-256
  `553023575b9bce13cf8ac35ce26f2d51f111bbf40d69a291aa9f451ebe02154e`
- Preflight log: `logs/gate2-fr1desk-30-wrong-order-0001-gpu-preflight.log`,
  17,071 bytes, mode `0444`, SHA-256
  `9eb710273b5b150e0341d05e47249d6830a1cbbad783beb4d2dba6ff461605f0`
- Postflight log: `logs/gate2-fr1desk-30-wrong-order-0001-gpu-postflight.log`,
  7,965 bytes, mode `0444`, SHA-256
  `4d4e64795fede754918604dd7206f4a994ce539dc0e8fbcccd95fd8b0443d02a`
- Audit log: `logs/gate2-fr1desk-30-wrong-order-0001-audit.log`, 5,273 bytes,
  mode `0444`, SHA-256
  `6d707cd5f4e6a398b1bf2a5ea82f20a34b4c342df1930726d0d94fedeeb6e567`

The command differed from the preceding smoke only in its frozen manifest and
unique output directory:

```text
CUDA_VISIBLE_DEVICES=4 \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
/data/wangzheng/Project2/StateGuard3R/scripts/run_recal3r_smoke.py \
  --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
  --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --input-manifest /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-corruption-inputs-v1/wrong-order-segment-manifest.json \
  --output-dir /data/wangzheng/Project2/StateGuard3R/outputs/gate2-fr1desk-30-wrong-order-0001 \
  --device cuda --size 512 --seed 0 --beta-base 0.1
```

Both persisted preflight snapshots found GPU 4 at 4 MiB used, 45,586 MiB free,
0% utilization, P8, and without a compute PID. Both tracked worktrees and all
frozen hashes were clean/current, the output target did not exist, and no other
process was stopped or changed. Inference completed in 3.114958 seconds at
9.630949 FPS, with 6,365.548 MiB peak allocated. PID `2096352` was absent after
exit. Immediate postflight found 4 MiB and no compute PID at 7% sampled
utilization.

The final source-index order is exactly:

```text
0,1,2,3,4,5,6,7,8,9,13,12,11,10,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

It is a bijection of `0..29`. Frames 10–13 carry exactly one
`temporal_reorder` transform each: `10->13`, `11->12`, `12->11`, and `13->10`;
all other transforms are empty and retain identity paths. `run.input_frames`
matched the strict manifest path, SHA, metadata, and transforms field by field,
and `run.images` matched their path/SHA projection. Health timestamps follow
the final source paths: frames 10–13 are strictly decreasing, then frame 14
jumps forward. Local `frame_id` nevertheless remains the output position
`0..29`; the model consumes reordered RGB, not timestamp, GT, or depth.

A CUDA-hidden official-loader replay produced 30 independent cloned
`1x3x384x512` tensors. Because temporal order is already encoded by the final
paths, every materialized tensor was pixel-equal to its corresponding loader
input; loader tensors remained unchanged and CUDA was never initialized. All
30 source PNG SHA/size/mtime/mode/link-count snapshots were identical before
and after.

The CUDA run completed 30 predictions, 29 expected/observed updates, trace
`[1..29]`, and 180 finite tensor summaries. Health and trajectory signals align
exactly, reliability equals `1 - uncertainty_u`, and all 29 pose jumps were
independently recomputed from saved matrices with zero observed JSON-roundtrip
error. The three persisted output families equal clean exactly only at frames
0–9 after removing timestamp from both health records. This comparison covers
persisted summaries; it does not claim that unpersisted full tensors are
bitwise equal.

Observed full-run ranges were:

```text
geometric_residual  0.0113718032 .. 0.0187678006  (maximum frame 8)
pose_jump           0.0180341104 .. 0.1209270919  (maximum frame 14)
uncertainty_u       0.7761573792 .. 0.8115034103
reliability         0.1884965897 .. 0.2238426208
global_state_delta  0.8707307117 .. 1.3241643105  (maximum frame 1)
```

At the entry boundary frame 10, pose jump is `0.11747319`, or `+0.09392930`
versus the corresponding clean frame; global-state delta is `+0.19213671`,
uncertainty `+0.00137693`, reliability `-0.00137693`, and geometric residual
`-0.00227916`. Frame 14 restores identity path but jumps from source 10 to
source 14, outside the labelled reverse interval. Its pose jump is
`0.12092709`, or `+0.09371665` versus clean. Both pose boundaries exceed the
clean run's global pose-jump maximum `0.05329814`.

Uncertainty/reliability and global-state delta do not move in a single
consistent direction throughout frames 10–13, and the full-run global-state
maximum remains clean-prefix frame 1. Frames 10–29 differ from clean across all
three persisted output families, but recurrent spillover is not proof of
damage propagation. In particular, frame 14 is a label-external recovery
boundary and must not be silently scored as an ordinary clean false positive.

Output hashes are:

```text
checkpoint-load-audit.json  3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d
health.jsonl                af99fcbd745053740428c5ed8fac8f80698944d7c8f05957b67b371653093117
predictions-summary.json    bd0a0e94f4f9dea7ac418c035a6371c4f0c1ac98383e41e6b8a1712b940f8c6b
run.json                    85cbae1b656de107e24f762a1384b6b1906188cd5e77e26ec7acb5b8d2cbb167
trajectory.json             bc7aee5745b48aa3fed9c0cbfcfba70eec8e52eba5f274f3ec4e8c9c77ffa844
```

The main read-only audit retained two harmless local checker retries. First,
all CPU replay assertions passed but a report-only set attempted to hash a
list and raised `TypeError` before printing. Second, the JSON checker removed
timestamp only from the wrong-order side of a clean-prefix comparison and
raised `AssertionError`; the corrected comparison removes it from both sides.
Neither attempt wrote or modified a file. The corrected local audit and two
additional independent read-only audits all passed before freeze.

This is one controlled four-frame reverse response. It is not natural disorder,
packet loss, a timestamp-anomaly experiment, formal detection, damage
propagation, holdout evidence, or research Go. All three exploratory ledgers
are now complete, but they share the same 30-frame source window. The next gate
is only to prove and freeze a non-leaking development/holdout protocol,
including recovery-boundary and spillover scoring. Fifty frames, formal
detection, quarantine, and rollback remain locked.

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
