# Recovery geometric-registration export development v4 结果

- 评测日期：2026-08-07
- 唯一开发结论：**`GEOMETRIC_REGISTRATION_EXPORT_V4_AVAILABILITY_OR_RUNTIME_NO_GO`**
- 范围：仅三个已解盲、只读的 `formal-v1-inputs-0001/development` manifests；本轮实际只运行了 dynamic 的
  fixed Gate A/B forwards。没有下载数据、权重或依赖，没有改动 ReCal3R/checkpoint/raw RGB/GT，也没有读取、重跑或
  覆盖任何 `recovery-quality-formal-*` response。

## 一句话结论

v4 证明 anchor-RGB ORB 与 3D--3D registration 能在全部六个真实连续告警帧上，从 raw candidate pose 之外因果地
生成有效的 camera pose，并保持 recurrent state 完整 rollback；但该 CPU registration 的 candidate runtime 是 baseline
的 **4.6307 倍**，不可部署。因此不运行 wrong/low 或质量 evaluator，不能以单条 dynamic 输出主张质量改善。

## 固定候选与完整性边界

唯一候选为 `detector-v3-incremental-anchor-orb-3d3d-registration-export`：clear frame 仅在 model 外保存其
model-ready RGB、`pts3d_in_other_view` 和 real-pose audit witness；alarm frame 先以 frozen Detector-v3 的 current
online decision 执行完整 pre-state rollback，再以该 last real-safe anchor 的 RGB/reference pointmap 与当前 RGB/self
pointmap 做 deterministic ORB mutual match 和 fixed 3D--3D RANSAC Kabsch。注册矩阵只通过 pinned
`camera_to_pose_encoding` 写入临时导出 `camera_pose`；它不会进入 model、detector 或后续 anchor。

绑定的实现为 StateGuard3R `27f18e88282df1f894c0ad33a82020f397f11033`、ReCal3R
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`、checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` 和 immutable Detector-v3 config SHA-256
`2c89d356cb6e33af536975def0106bfe9bd6368edaa1094e26414d1c51052748`。v4 runner 是独立 source-bound recurrent
loop；不会 monkeypatch 或调用 v3 main/fallback。

## Gate A：实现与 CPU 合约

- targeted v4 geometry/export/ORB/runner tests：`11 passed in 3.08s`；覆盖 fixed RANSAC 的 proper transform、outlier、
  rank-three refined cloud、anchor direction/indexing、pinned ReCal3R encoding round-trip、独立 runner 与 reset guard。
- 全量 CPU suite：`483 passed in 56.83s`，`--basetemp` 固定在 Project2 内；`compileall` 与 `git diff --check` 通过。
- ReCal3R worktree 固定且干净于 `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`。

## Gate B control：正常路径与 provenance

早期 `...dynamic-always-commit-0003` 的 metadata timeline path 指向已经移动的 staging path，且早于独立 runner；
它保留但不作为 Gate B evidence。接受的 control 是
`outputs/recovery-geometric-registration-v4-dynamic-always-commit-0004`，目录 `0555`、文件 `0444`，其 final direct-child
timeline path 可解析且 SHA-256 匹配 `run.json`。

| Artifact | 与 v1 dynamic baseline 的 SHA-256 | 结果 |
| --- | --- | --- |
| `checkpoint-load-audit.json` | `3e12875a701e0afc534d8f3a90b6a5fb54a58cfca59e2ce055109e6db4a4153d` | PASS |
| `health.jsonl` | `9de0c0b690c00e603fd16c81ff6258ce3c275388e48d66a0c1a64672848131fe` | PASS |
| `predictions-summary.json` | `cb11415bfe3765e7ad87fd9655873f319cdaaa553ef948f3e3f935037a8504ed` | PASS |
| `trajectory.json` | `fdba3312b792e4c67f56c8a0f8feba908846cd61190fa5a8665f0b0be81d013b` | PASS |

control runtime 为 `3.7447120929 s`，v1 baseline 为 `3.4633694058 s`，ratio `1.0812338085`，通过 `<=1.20`。
其 tmux execution transcript 和 post-hoc control validator 分别记录在
`logs/recovery-geometric-registration-v4-dynamic-always-commit-0004-tmux-execution-transcript.log` 与
`logs/recovery-geometric-registration-v4-dynamic-always-commit-0004-gate-b-control-validation.log`。

## Gate B candidate：availability 通过，runtime 失败

唯一 candidate 是
`outputs/recovery-geometric-registration-v4-dynamic-candidate-0001`；它冻结为 `0555/0444`，`status=succeeded`，30 frames、
29 calibrated updates，timeline final path 与 SHA 均有效。GPU 2（L20 UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`）在 preflight 两次均有 45,586 MiB free；完整 command、GPU snapshots 与
exit 0 在 `logs/recovery-geometric-registration-v4-dynamic-candidate-0001.log`。

| Availability contract | 实测 | 结果 |
| --- | --- | --- |
| real alarm / rollback frames | 15--20, 6 / 6 | PASS |
| rollback action / witness / pending / watchdog | `quarantine_current_rollback`, witness present, 0, `<=8` | PASS |
| export action | `export_anchor_orb_3d3d_registration`, 6 / 6 | PASS |
| continuous safe anchor | frame 14 for all six alarms | PASS |
| finite 3D pairs / inliers | min 70 / min 29 | PASS |
| normalized residual | max 0.0268972 (`<=0.05`) | PASS |
| proper SO(3) | det≈1; max orthogonality error `1.55e-15` | PASS |
| candidate / baseline runtime | `16.0376898451 / 3.4633694058 = 4.6306610603` | **FAIL** |

该 runtime 包含 CUDA synchronized recurrent policy、当前 detector、structural rollback 和 v4 registration/export；共同
causal RGB overlap 单列到 wall runtime，不能将 registration 成本移出可比区间。runtime 失败独立于 geometry availability，
也不是以 development 数值调整 ORB feature count、ratio、RANSAC hypothesis、threshold 或 anchor policy 的理由。

## 决定与后续边界

Gate B 的 fixed short-circuit 已触发，故 **没有** wrong/low v4 forward、没有 v4 quality evaluator、没有 ATE/RPE
selection、没有新的 blind acquisition/commitment。v4 的唯一科学结论是：外部 anchor geometry 可恢复报警帧 pose，但该
ORB/CPU registration 实现不满足 deployment runtime budget。

后续版本必须是机制上不同的、预注册的候选；它不得把 v4 ORB/RANSAC 常数、anchor policy 或任何已解盲 v4 quality
数值作为可搜索参数。一个可审计的下一方向是：仅当源码审计证明 self/reference pointmap 并非由 raw camera pose
循环生成时，测试当前帧 device-resident dense pointmap registration；该方向在启动 GPU 前仍须有自己的计划与 Gate A。
