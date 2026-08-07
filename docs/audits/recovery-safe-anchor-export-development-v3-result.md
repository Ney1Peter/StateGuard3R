# Recovery safe-anchor export development v3 结果

- 评测日期：2026-08-07
- 唯一开发结论：**`SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_FEASIBILITY_NO_GO`**
- 范围：三个已解盲、只读的 `formal-v1-inputs-0001/development` corruption manifests；不是新的
  recovery-quality formal result。

## 一句话结论

v3 首次把 current-frame recurrent rollback 与因果的 SE(3) safe-anchor pose export 组合起来：每个报警帧均恢复
完整 pre-state，且只替换输出 `camera_pose`，连续报警由最近的 external fallback 推进一步。它消除了 v2 的性能失败，
并在 low-overlap 条件大幅改善；但 dynamic 与 wrong 均恶化，两个质量中位数为负。因此这条固定机制不能进入新盲测。

## 固定候选、边界与实现

唯一候选为 `detector-v3-incremental-safe-anchor-se3-export`：frozen Detector-v3 的 current online decision 只读取
candidate model health、相邻 model-ready RGB overlap 与 raw capture timestamps；alarm 时 rollback candidate recurrent
closure，并以

```text
Delta = T(t-1) @ inverse(T(t-2))
T_hat(t) = Delta @ T(t-1)
```

导出 fallback pose。连续 alarm 将上一 fallback 仅作为 external export history，绝不反馈 model 或 detector；clear
frame 立即导出真实 candidate pose。没有 GT、future、source index、synthetic alarm、v1 artifact、replay 或下载。
pointmap/health 保留 raw candidate 语义；candidate trajectory 明确标记 pose 为 exported、geometric residual 为替换前
raw candidate pointmap 量。

实现绑定 ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`、checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` 与 immutable Detector-v3 config SHA-256
`2c89d356cb6e33af536975def0106bfe9bd6368edaa1094e26414d1c51052748`。接受的 v3 runner script SHA-256 是
`8767f1bb9c3341284da61d52da4797b737ac3f85ae0c14d290f86250a8b7dd3e`。

## Gate A：合约、等价与恢复证据

- CPU suite：`472 passed in 48.30s`（受控 `--basetemp` 在 Project2 内）；`compileall`、`git diff --check` 通过。
- 新增 SE(3) left/world extrapolation、连续 fallback、anchor shortage fail-closed、incremental/reference
  detector、sanitized rollback history 与 structural identity/storage/version witness 的单元覆盖。
- 每个 accepted candidate 都在 15--20 六次 rollback；六条 witness、六条 fallback、24 条 real export，无 pending、
  watchdog abort 或 restore failure。
- 对三份 frozen v3 observed ledger，bounded incremental detector 在全部 90 frame 的 score、attribution、action
  与 CPU `OnlinePrefixDetector` reference 完全一致。
- GPU current scalar 与 v2 CPU post-hoc scalar 的最大绝对差（float32 GPU vs float64 NumPy）分别为
  dynamic pose/geometry `1.34e-05`/`7.84e-05`、wrong `5.66e-06`/`6.70e-05`、low
  `2.93e-06`/`1.62e-04`；所有 detected actions 相同。该数值精度事实不被表述为 bitwise scalar equivalence。

三个 v3 always controls 和冻结 baseline 的 checkpoint audit、health、prediction summary、trajectory 四文件均逐字节
一致。runtime 用逐 frame CUDA synchronization 计量 recurrent+policy，明确排除共有 RGB-overlap；其 ratio 为
dynamic 1.0015、wrong 1.0161、low 1.1036，均通过 1.20。

一个早期 always output `...dynamic-always-commit-0002` 包含 overlap 的错误 runtime scope，因而没有作为 Gate B
证据；另一个 `...0003` 暴露 trajectory schema byte-equivalence 缺口。两者保留但均被拒绝。candidate `...0001`
在第一次 forward 的 camera-shape implementation failure 后原子清理，accepted dynamic candidate 是 `...0002`。

## Gate B/C：冻结矩阵与选择

effect 为 `(baseline - candidate) / baseline`，正值代表改善。全部 six GPU outputs 冻结之后，CPU evaluator 才读取
source-manifest logical-base GT；prefix 0--14 做 Sim(3)，tail 19--29 计分。

| Condition | ATE effect | translation-RPE effect | candidate / baseline runtime |
| --- | ---: | ---: | ---: |
| dynamic | -10.1576% | -102.8026% | 1.1051 |
| wrong | -16.2783% | -20.9873% | 1.0917 |
| low | +91.4097% | +92.3408% | 1.1306 |
| median | **-10.1576%** | **-20.9873%** | **1.1051** |

| Required gate | Result |
| --- | --- |
| all four-file baseline/always equivalences | PASS (3 / 3) |
| real rollback, witness, safe anchor and observed ledger | PASS (3 / 3) |
| at least one trajectory changed | PASS |
| both metrics positive in at least 2 / 3 | **FAIL** (1 / 3) |
| median ATE effect `>= +5%` | **FAIL** (-10.1576%) |
| median translation-RPE effect `>= +5%` | **FAIL** (-20.9873%) |
| candidate runtime median `<= 1.20` | PASS (1.1051) |

冻结 evaluator 是 `outputs/recovery-safe-anchor-v3-evaluation-0001/evaluation.json`，SHA-256：
`6379563dcbb1959f5ec63cf6c7fd7d7626fbcffaf31fbd1db45f06538b6b8e45`。

## 解释与后续边界

v3 证明了 fast causal isolation 与 export-only recovery 可以同时满足安全和运行时门禁，却没有产生跨 corruption
的稳定质量改善。low 的单独成功不能授权选择 hold、另一 delta direction、时间缩放、anchor 长度、阈值或另一个
pose fallback；这些都将把同一 development matrix 变成机制搜索集。

因此不下载新数据、不写 v3 blind acquisition/commitment，也不把 low 的大效果作正式 claim。若继续，必须先以不同于
safe-anchor pose fallback 的机制假设写独立计划，并将当前已解盲 development 结果只作为失败取证，而不是用于选取
v4 运动模型或参数。
