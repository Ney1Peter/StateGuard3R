# Recovery early-decoder pose-token export v8：Gate A 审计

- 审计日期：2026-08-07
- Gate A 结论：**PASS；仅授权一次新的 dynamic always-control。**
- StateGuard3R commits：计划 `c68598f`；实现 `7861755`；hardening
  `fe5ac1b`；output guard `17de293`；tests `e9fe425`。
- ReCal3R：clean commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`。
- 本 Gate 未下载数据、权重或依赖，未启动 GPU run，未读取 development GT，未运行
  wrong/low 或质量 evaluator。

## 固定 v8 行为

candidate 在 pinned lighter body 的 `_recurrent_rollout(...)` 完成且 decoder-length
guard 通过后、`update_mem(..., dec[-1][:,0:1])` 前，仅一次 clone
`dec[0][:,0:1]`。该 `(1,1,768)` layer-0 token 是 external temporary export input：
只在 candidate observer 存在时创建，既不回写 model/state/memory/detector，也不会传到
CPU。always-control 不建 observer、不 clone token，保持 direct raw `to_cpu(res)` 路径。

报警时，Detector-v3 先根据 raw current prediction 作决定；full structural rollback/witness
成功后，policy 才将已经捕获的 early token 交给 frozen
`DPTPts3dPose.pose_head`、official `postprocess_pose` 和 official camera decoder。clear
frame 原样导出 raw pose；alarm 只替换 exported `camera_pose`，没有 raw-pose numeric
input、copy/hold/motion fallback、retry、GT、future frame、anchor、ORB/RANSAC 或 pointmap
solver。

pose head 保持 model dtype；仅其已有的 `(1,7)` output 在同一 device 升至
`float64` 后进行 official postprocess/camera validation。token/head-device mismatch、shape、
finite、head output、postprocess/camera shape、reflection、non-SO(3) 或 malformed homogeneous
row 全部 fail closed。alarm token evidence 使用 GPU-resident deterministic digest；不会在
standard `to_cpu(exported)` 前 materialize a full tensor on CPU。

## Source、因果与 normal-path 合约

`audit_pinned_early_decoder_pose_sources` 固定并 fail-close 以下 ReCal3R source
SHA-256：

| source | SHA-256 |
| --- | --- |
| `dust3r/model.py` | `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1` |
| `dust3r/heads/dpt_head.py` | `c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e` |
| `dust3r/heads/postprocess.py` | `ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7` |

它要求 official recurrent topology 的 rollout、final `dec[-1]` memory update、official
DPT pose-head/postprocess path 均不漂移；独立 runner audit 只检查 executable
`run_early_decoder_pose_recurrent_lighter` body，证明 `rollout < candidate dec[0] capture
< update_mem`、candidate finalize 在 CPU transfer 前、always-control 没有 token/observer
work。layer/index/order mutation 和仅注释的 upstream source mutation都 fail closed。

decoder signature 只有 token、frozen torch/head/postprocess/mode/camera decoder；AST checks
拒绝 raw `camera_pose`、prediction、RGB、pointmap、confidence、anchor/history、detector、GT/
future，以及 decoder/export modules 中 `.cpu()`、`.numpy()`、`.tolist()`。export alarm path
不读取 prediction 的任何值，只 shallow-copy mapping 后写入 decoded pose。

## 已执行验证

| 验证 | 结果 |
| --- | --- |
| ReCal3R Torch targeted v8 tests | `32 passed in 12.88s` |
| checkpoint-loaded official DPT CPU check | pass；加载已有 `cut3r_512_dpt_4_64.pth`，strict audit 无 missing/unexpected key；pinned `DPTPts3dPose`、official postprocess/camera、float64 SO(3) contract 通过 |
| full project CPU suite | `509 passed, 28 skipped in 49.41s` |
| compileall (`src scripts tests`) | pass |
| `git diff --check` and both worktrees | pass；StateGuard3R/ReCal3R clean |

Torch-targeted command deliberately uses StateGuard3R's pytest-capable interpreter with the
existing ReCal3R environment's site-packages; the ReCal3R interpreter itself has Torch but no
pytest, so no dependency was installed:

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/tmp" \
PYTHONPATH="src:/data/wangzheng/Project2/baselines/ReCal3R/.venv/lib/python3.11/site-packages" \
.venv/bin/python -m pytest -q \
  tests/test_early_decoder_pose_v8.py \
  tests/test_recal3r_early_decoder_pose_runner_v8.py \
  tests/test_run_recal3r_early_decoder_pose_export_v8.py \
  --basetemp "$PWD/tmp/pytest-v8-gate-a-targeted-0005"
```

The full suite uses `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH=src`, and
`--basetemp "$PWD/tmp/pytest-v8-gate-a-full-0001"`. Its 28 skips are expected only in that
Torch-free project interpreter; all v8 Torch tests run in the targeted command above.

## Gate B authorization and stop rule

This audit authorizes exactly one run:
`recovery-early-decoder-pose-v8-dynamic-always-commit-0001`.

It must run only from the existing `stateguard` tmux session on GPU 2 UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`, after two persisted preflights each reporting at
least `12288 MiB` free, with a live pre-dispatch `pipe-pane` transcript. Its four protected
files must be byte-identical to the frozen v1 dynamic baseline and its runtime ratio must be
`<=1.20`. Only then is the one-use dynamic candidate authorized. Any Gate B failure freezes v8
as `EARLY_DECODER_POSE_EXPORT_V8_AVAILABILITY_OR_RUNTIME_NO_GO`; wrong/low/GT must not run and
v8 may not retune index, precision, decoder or detector.
