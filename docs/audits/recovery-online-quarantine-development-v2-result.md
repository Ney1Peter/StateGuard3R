# Recovery online-quarantine development v2 结果

- 评测日期：2026-08-07
- 唯一开发结论：**`ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO`**
- 范围：三个已解盲、只读的 `formal-v1-inputs-0001/development` corruption manifests；不是新的
  recovery-quality formal result。

## 一句话结论

v2 首次证明“当前帧、连续 episode、完整 state rollback”隔离了 15--20 帧的 recurrent 污染；但它保留了坏 RGB
导出的当前 pose，因此三个条件的 ATE/RPE 都更差，同时候选 runtime 中位数为 baseline 的 2.9076 倍。它不能进入
新盲测。

## 固定问题、边界与实现

v1 的 rising-edge policy 仅在 15 处触发一次，实际成为 `f15 commit -> f16 discard -> f17+ commit`，未测试连续
风险 episode。v2 的唯一预注册候选 `detector-v3-online-current-quarantine` 改为：frame `t` 产生 current
prediction 后，以只含 `<=t` 的模型 health、相邻 RGB overlap 和 RGB capture timestamp 计算 frozen Detector-v3
score；若报警就保留 prediction，但恢复完整 pre-frame local/model/RNG closure。连续报警逐帧 rollback，首个 clear
才 commit；没有 replay、pending、GT、future data、baseline alarm artifact、synthetic alarm 或 output fallback。

accepted forwards 绑定 StateGuard3R `a8b0f2ce92f622ec3b0255c487241d2ec7c9c3f7`、ReCal3R
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`、checkpoint SHA-256
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` 和 immutable Detector-v3 config SHA-256
`2c89d356cb6e33af536975def0106bfe9bd6368edaa1094e26414d1c51052748`。没有下载数据、权重或依赖；没有修改
ReCal3R、raw RGB、GT、corruption manifest 或任何 `recovery-quality-formal-*` response。

## Gate A：正常路径等价与性能

三个 v2 light always-commit output 与 frozen v1 baseline 的 `checkpoint-load-audit.json`、`health.jsonl`、
`predictions-summary.json`、`trajectory.json` 全部逐字节一致。

| Condition | baseline s | v2 always s | ratio | 结果 |
| --- | ---: | ---: | ---: | --- |
| dynamic | 3.4634 | 3.2236 | 0.9308 | PASS |
| wrong | 3.4450 | 3.3935 | 0.9850 | PASS |
| low | 3.2501 | 3.4126 | 1.0500 | PASS |
| median | — | — | **0.9850** | `<= 1.20` PASS |

范围是 `external_recurrent_and_current_policy_only`。因果 RGB overlap 是各方法共有的 detector instrumentation，
冻结 baseline 已在 `runtime_seconds` 外计算；v2 保持逐帧因果计算但单列 `wall_runtime_seconds`。candidate 的 online
health scoring、digest 和 rollback 都在可比 runtime 内。

## Gate B/C：current episode 状态隔离

三个 candidate 都成功完成 30 frame、29 calibrated updates、30 observed health rows，timeline 无 pending transaction
和 restore failure。每个 condition 的 online alarm 与 rollback 都是 frames 15--20：

| Condition | rollback frames | 当前 alarm | digest / ledger 结果 |
| --- | --- | --- | --- |
| dynamic | 15--20 | 6 / 6 | committed digest = pre digest；6 ledger rows |
| wrong | 15--20 | 6 / 6 | committed digest = pre digest；6 ledger rows |
| low | 15--20 | 6 / 6 | committed digest = pre digest；6 ledger rows |

所有 candidate trajectory 都与 normal baseline 不同，证明不是 no-op。wrong 的 timestamp channel 在 16--18 触发，其余
alarm attribution 来自 frozen continuous score；policy 从未读取 v1 alarm 文件。

## Gate D：开发质量与最终判定

effect 定义为 `(baseline - candidate) / baseline`，正值代表改善。后验 evaluator 只在全部 GPU output 冻结之后才读取
source manifest logical-base GT rows 0--29；Sim(3) 使用 prefix 0--14，ATE/RPE 仅在 tail 19--29 计分。

| Condition | ATE effect | translation-RPE effect | candidate / baseline runtime |
| --- | ---: | ---: | ---: |
| dynamic | -3.2650% | -2.6789% | 4.6388 |
| wrong | -2.9494% | -15.4509% | 2.9076 |
| low | -1.7153% | -2.3883% | 2.7462 |
| median | **-2.9494%** | **-2.6789%** | **2.9076** |

| Required gate | Actual | Result |
| --- | ---: | --- |
| all four-file baseline/always equivalences | 3 / 3 | PASS |
| real current-frame quarantine and complete audit ledger | 3 / 3 | PASS |
| changed post-action trajectory | 3 / 3 | PASS |
| both metrics positive in at least 2 / 3 | 0 / 3 | **FAIL** |
| median ATE effect `>= +5%` | -2.9494% | **FAIL** |
| median translation-RPE effect `>= +5%` | -2.6789% | **FAIL** |
| candidate/baseline runtime median `<= 1.20` | 2.9076 | **FAIL** |

冻结 evaluator 是 `outputs/recovery-online-quarantine-development-v2-evaluation-0001/evaluation.json`，SHA-256
`90818667e2d54f898d3343d8bb246822f8af4eb71da6f23e477a31a027ddbdf0`；其 outcome 正是 stated NO-GO。

## 解释与下一个边界

state mechanism 因果有效但不充分：rollback model state 无法修复由坏 RGB 产生并已导出的 current prediction。low-overlap
特别清楚地表明，corrupt donor 虽不能进入未来 recurrent state，仍保留在当前输出。运行时失败独立存在：per-alarm
observation/digest work 使候选过慢。

不得在 v2 改 threshold、watchdog、episode 定义、snapshot detail 或 fallback，也不得据此下载盲测数据。下一 v3 如继续，
必须以新的独立计划预注册一个固定 causal output-export mechanism，并先建立诚实的性能门禁；本 development matrix 不能作为
confirmatory evidence。
