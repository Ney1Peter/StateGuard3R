# Recovery-policy development v1：因果诊断与候选定型执行计划

- 制定日期：2026-08-07
- 状态：**执行中**
- 计划窗口：12--18 小时；以门禁完成为终点，而不是以时间耗尽为终点
- 上一阶段：[Recovery-quality v1 正式 NO-GO](audits/recovery-quality-v1-result.md)
- 受保护 baseline：ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`

## 1. 明确目标与正常结束状态

本阶段只在已解盲开发输入上回答一个工程因果问题：为何已经被验证正确的
transactional `hold/replay` 会发生完整状态恢复，却使正式 trajectory、health 和 prediction
summary 与 baseline 字节一致；以及“hold 后丢弃可疑 candidate state，而非稍后 replay 它”是否是一个
值得进入**新**盲测的开发候选。

本阶段的唯一可接受结束状态有且仅有两个：

1. **`DEVELOPMENT_RECOVERY_CANDIDATE_READY`**：根因被独立复现；新增 candidate 的因果、安全、
   等价控制和开发质量门禁全部通过。它只表示可花费新盲测数据，不表示 recovery-quality 已经成立。
2. **`DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`**：根因被独立复现，但 candidate 未能得到真实
   detector-triggered action、未能改变后续状态/输出、违反安全/运行时门禁，或未达到预注册开发
   效果门槛。

两种状态都完成本计划。不得因 NO-GO 在本阶段外继续搜索、偷看新数据或改变门槛。

## 2. 不可变边界

- 不修改、重跑、覆盖或作为新 response 读取 `recovery-quality-formal-*` 的冻结输入/输出；仅允许
  对其已公开 JSON 做只读 forensic 比较。
- 不下载数据集或权重。实际 forward 仅允许使用
  `outputs/formal-v1-inputs-0001/development/{development-dynamic,development-wrong,development-low}`
  三个已解盲、只读的 `stateguard3r.corruption.v1` manifests。
- 不改 ReCal3R source、checkpoint、Detector v3 threshold/config 或 raw RGB/GT。GT 只能由新开发
  evaluator 在 forward 后读取；alarm 与 policy 接口拒绝 GT、depth、label、event/source-index 和未来帧。
- candidate 不接受 `--recovery-quality-commitment`，并只接受上述 legacy development manifest schema；
  因而不能误用于上一轮 formal commitment。
- 单卡、batch 1、显式 `CUDA_VISIBLE_DEVICES=<id>`；每次 forward 前记录 GPU UUID、空闲显存
  （至少 12 GiB）、已有 PID、命令、日志、时间和退出码。不干预其他用户进程。

## 3. 已预先登记的根因假设

### H0：现有 replay 在数学上复原 baseline 链

对于 alarm `a[t-1]`，当前 policy 在 frame `t` 先完整计算 candidate 和 prediction，再把 post-state
hold 并恢复 pre-state。下一次 clear 时，它在处理下一 candidate 前重新安装被 hold candidate 的
post-state。若 forward 是确定性的，该 post-state 正是 baseline 在 frame `t` 已提交的 state；之后的
recurrent state 链与 baseline 相同，且 frame `t` 的输出已经在 hold 决定前保存。因此 policy 可以
有真实 hold/replay timeline，却仍产生全量 byte-identical model output。

已公开的 recovery-quality v1 八条 policy output 支持 H0：每条 trajectory、health 和 prediction
summary 均与其 baseline 字节相同；每个 `release_replay_then_commit` 的 next pre-state 等于先前 held
candidate 的 proposed post-state。本阶段必须用独立 analyzer 和自动测试复现该结论，而不只引用手工
观察。

### H1：bounded discard 是有因果杠杆的最小候选

新增 development-only `hold/discard` policy 保留与 replay 完全相同的 **t-1 causal alarm**、完整
transaction capture、最多 3 个 pending candidate 和恢复 pre-state 的动作；不同点仅在 clear：它永久
discard 最新 held post-state，不重新安装它，并从最后已知 safe pre-state 处理 clear frame。被 discard
frame 的已产生 prediction 会保留且显式标记，绝不伪造、回填或用未来图像重算。它因此可能改变下一帧
及后续 output；这种改变本身不等于改善。

## 4. 固定开发矩阵与门禁

### Gate A — 只读 forensic 与 CPU 合约

新增一个 CPU-only analyzer，读取八项公开 formal-v1 evaluation/policy timeline，逐项验证 H0 的
state-chain identity 及 model-output byte identity。新增 unit tests 覆盖 replay 的 H0、discard 的
state边界、end-of-stream discard、causality、legacy-manifest 限制、GT 不可进入 online interface，
以及开发 evaluator 的 prefix-only 对齐/GT binding/拒绝路径。

**通过条件：** H0 report 对全部 8 条公开 runs 一致；新增与已有 CPU tests 全部通过；代码、测试和
文档分别提交，ReCal3R 保持 clean。

### Gate B — development baseline 与真实 shadow alarm

对三个固定 development manifests 各跑一次当前 smoke baseline（v2 health），再以不变的 Detector v3
formal config 调用 CPU-only alarm builder。三个 alarm artifact 均冻结；builder 的 online 输入只能是
baseline health、model-ready RGB visual overlap 和原始 RGB capture timestamp。

若三条 artifact 都没有 rising-edge policy alarm，本计划仍运行一个固定的 structural forced probe
（每个 corruption 的 alarm frame 固定为 14，使 frame 15 candidate 被 hold），但直接得到
`DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`，因为没有真实 detector-triggered candidate 可进入新盲测。

### Gate C — 完整 candidate 矩阵

只要 Gate B 至少有一个真实 rising edge，三个 manifests **全部**运行下列四项，不按中间指标挑选：

| Method | Purpose |
| --- | --- |
| smoke baseline | independent reference and v2 health source |
| external `always-commit` | required byte-equivalent transactional control |
| current detector-triggered replay | reproduces H0 operational path |
| new detector-triggered discard | the sole development candidate |

另外固定的 forced replay/discard pair 只验证 H1 的机械因果性，绝不用于质量结论。每个成功目录均原子
冻结为 `0555/0444`，不能覆盖或重跑。

### Gate D — 后验开发评价与候选选择

新增独立 CPU development evaluator，从 legacy manifest 的已存在 GT metadata 读取 logical-base pose，
只以 prefix frames 0--14 拟合 Sim(3)，在 tail 19--29 计算 ATE/RPE。它不得被 alarm/policy 导入。

`DEVELOPMENT_RECOVERY_CANDIDATE_READY` 要求全部满足：

1. 三条 baseline/always control 四项文件逐字节一致；所有 runs 有 30 个有限 trajectory frames，
   无 restore failure、无静默 pending transaction、无超出 `max_hold=3` 的 hold；
2. 每个实际 detector-triggered discard action 都明确记录其 held transaction 被 discard，且 clear
   frame 的 pre-state 不等于被 discard candidate 的 post-state；至少一条 detector-driven run 在该
   clear frame 或之后与 replay 的 trajectory/prediction 不同；
3. 三个开发 corruption 的 ATE 与 translation-RPE effect 都完整报告；至少 **2/3** trial 的两项 effect
   均严格为正，二者 median 均至少 **5%**，且 candidate/baseline runtime 中位比不超过 **1.20**；
4. `git diff --check`、compileall、完整 CPU suite、权限/哈希/PID/GPU 审计均通过。

未满足任一项即 `DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`。开发 GT 数值不得被写成正式结论，也不得
用于改 alarm 阈值、hold 上限或输入 corruption。

## 5. 实施顺序与交付物

1. 新增 `analyze_recovery_policy_h0.py` 及测试，冻结一份只读 forensic report；
2. 在不替换 replay 实现的前提下，新增 discard controller/mode、development-only manifest guard 和
   exhaustive timeline evidence；新增单元/集成测试；
3. 新增 development evaluator 与测试；完成 CPU suite 后以独立 commits 固定代码、测试、计划；
4. 按 Gate B/C 串行 GPU/CPU 矩阵运行，保存每次 pre/postflight log；
5. 写 `docs/audits/recovery-policy-development-v1-result.md`、更新 run registry 与本计划状态；
   仅当 candidate ready 时再写一个**尚未执行**的新盲测 acquisition/commitment plan。

最终报告必须区分：已验证的 H0/H1 机械事实、开发数据上的探索性数值、以及尚未得到的新盲测质量结论。
