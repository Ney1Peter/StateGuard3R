# Recovery current-pointmap consensus export development v5 结果

- 评测日期：2026-08-07
- 唯一开发结论：**`CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_AVAILABILITY_OR_RUNTIME_NO_GO`**
- 范围：仅既有、只读的 dynamic development manifest，按预注册 Gate A/B 依次执行。没有下载数据、权重或依赖；没有修改 ReCal3R/checkpoint/raw RGB/GT/Detector-v3 config；没有读取 GT、运行 wrong/low 或 quality evaluator。

## 一句话结论

v5 证明当前帧 self/cross pointmap 的 GPU-resident 16 x 16 robust consensus 能在连续六个真实报警帧上，完整 rollback 后
导出 non-raw、proper SO(3) camera pose；但 candidate runtime 是 v1 baseline 的 **1.41137 倍**，超过固定部署上限
`1.20`。所以不能声称跨条件质量改善，也不能继续在已解盲 dynamic 上调 lattice、weight、IRLS、threshold 或运行矩阵。

## 固定候选与实现边界

唯一 candidate 为 `detector-v3-incremental-current-self-cross-consensus-export`。alarm 时先执行 frozen Detector-v3 的
raw-candidate decision 与 complete pre-state rollback；随后只读取该 current prediction 的 self/cross 3D pointmaps 与两张
confidence map，固定取 256 个边界包含的 lattice pair，以两轮 Tukey IRLS weighted Kabsch 解 `self -> cross/reference`。
没有 RGB/ORB/history/anchor/raw pose numeric input/GT/future/fallback；raw pose 只作为 pinned encoder 的 device/dtype template。
clear frame 仍输出原始 pose，任何 exported pose 都不会反馈给 model、detector、state 或后续 frame。

本轮实现绑定 StateGuard3R `22ad08021749b41ca9db8e9109c8a47637bd1ee4`、ReCal3R
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`、checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` 和 frozen Detector-v3 config SHA-256
`2c89d356cb6e33af536975def0106bfe9bd6368edaa1094e26414d1c51052748`。cross head audit 明确记录：camera pose numeric output
不直接输入 cross head，但 cross pointmap 会使用 shared pose-token latent；故科学表述不夸大为 latent-independent recovery。

## Gate A：实现与 CPU 合约

- full project-local CPU suite：`489 tests, 0 failures, 0 errors, 3 skipped, 49.848s`；两个 v5 Torch module 因 project
  `.venv` 无 Torch 而 skip，另有一个既有 skip；含 ReCal3R Torch site-packages 的 v5 targeted tests 为 `12 passed in 2.42s`。
- direct real ReCal3R camera encode/decode CPU self-check 通过，max round-trip error `2.06e-15`；outlier、rank/shape/
  nonfinite、proper SO(3)、no-raw-export、AST no-forbidden-input/no-CPU-solver、rollback order 均有 tests。
- `compileall` 与 `git diff --check` 通过；实现、tests、Gate A audit 分别提交为 `d2e4cdb`、`b68e1aa`、`22ad080`。

旧的 `...v5-dynamic-always-commit-0001` 保留但不接受：其 provenance 绑定旧 NumPy/CPU primitive hash，而不是本轮固定的
GPU-resident implementation。

## Gate B control：通过

接受的 control 是 `outputs/recovery-current-pointmap-consensus-v5-dynamic-always-commit-0002`。GPU 2（L20 UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`）两次 preflight 都有 `45,586 MiB` free；run exit 0，output 目录 `0555`、文件
`0444`，preflight/main/postflight/tmux transcript/validator 均是不同名称、NUL-free 的 `0444` immutable logs。

| Artifact | 与 v1 dynamic baseline | 结果 |
| --- | --- | --- |
| `checkpoint-load-audit.json` | byte-identical | PASS |
| `health.jsonl` | byte-identical | PASS |
| `predictions-summary.json` | byte-identical | PASS |
| `trajectory.json` | byte-identical | PASS |
| runtime | `3.4266027724 / 3.4633694058 = 0.9893841433` | PASS |

direct-child timeline path 与 `run.json` SHA binding、当前 component hashes、log modes、two preflights 与 postflight exit 都由
`logs/recovery-current-pointmap-consensus-v5-dynamic-always-commit-0002-gate-b-control-validation.log` 验证。

## Gate B candidate：几何可用，但 runtime 失败

唯一 candidate 是 `outputs/recovery-current-pointmap-consensus-v5-dynamic-candidate-0001`。它同样冻结为 `0555/0444`，30 frames、
29 calibrated updates、exit 0，并具完整 UUID/PID/command/log/timeline provenance。所有 six real alarms 在 frame `15--20` 都通过：

| Availability contract | 实测 | 结果 |
| --- | --- | --- |
| alarm / rollback frames | 6 / 6，15--20 | PASS |
| rollback/witness/pending/watchdog | `quarantine_current_rollback`、witness present、0、max 6 | PASS |
| export | `export_current_pointmap_consensus_se3`，6 / 6；raw/export pose digest 均不同 | PASS |
| fixed lattice / finite positive pairs | 256 / 256 每帧 | PASS |
| IRLS final positive weights | min 220（`>=128`） | PASS |
| source/target rank | 3 / 3 每帧 | PASS |
| normalized residual | max `0.0101512`（`<=0.05`） | PASS |
| proper SO(3) / encoder round-trip | det≈1；max orth error `2.55e-15`；max round-trip `2.24e-08` | PASS |
| runtime | `4.8881027559 / 3.4633694058 = 1.4113720436` | **FAIL** |

这项 runtime 已包含 CUDA-synchronized recurrent policy、online detector、rollback、GPU consensus 和 encode/export；共同 causal
RGB overlap 不在可比 interval 内。因而 `+41.14%` 的开销是本候选部署 gate 的独立失败，不是继续在同一 v5 mechanism 调参的授权。
完整 availability/runtime audit 在
`logs/recovery-current-pointmap-consensus-v5-dynamic-candidate-0001-gate-b-validation.log`。

## 决定与后续边界

Gate B short-circuit 已触发。不得运行 v5 wrong/low、GT evaluator、quality selection 或 blind acquisition；不得重用 output id，
也不得在同一 disclosed development dynamic 上改变 v5 lattice、confidence weight、IRLS iterations/Tukey/MAD、residual gate 或
Detector threshold 以追逐 runtime/quality。

下一版如继续，必须预注册一个机制上不同的 candidate，先单独完成 Gate A，再重走 dynamic control 与一次 candidate probe；它不应
把本次 six-frame availability 或任何未运行的 quality 结果当成可搜索调参集。
