# Recovery-quality v1：冻结的正式协议

- 日期：2026-08-06
- 状态：**待 formal commitment；不得在 formal response 上修改本协议**
- 前提：Detector v3 已 GO；state transaction 已通过 feasibility 验证。

## 目标与范围

本协议仅检验 Detector-v3 shadow alarm 所驱动的 external hold/replay policy，是否在 TUM
相机轨迹 GT 上改善错误输入后的 tail quality。它不重训模型、不改变 v3 calibration、不更改
ReCal3R tracked source，也不将 shadow replay 说成单路径实时部署。

确认性场景固定为此前无 ReCal3R response 的 `rgbd_dataset_freiburg3_walking_static` 与
`rgbd_dataset_freiburg3_walking_xyz`。已解盲 `fr1_desk`/`fr2_desk` 只能用于开发；其任何数值
绝不能参与 formal decision。

## 固定输入与运行矩阵

每个场景通过 CPU-only builder 生成一个 30-frame logical-base window，event 位于 frames 15--18：

| Condition | Model input | GT target |
| --- | --- | --- |
| clean | 连续原始 RGB | logical base RGB 的关联 GT |
| dynamic | 15--18 的固定 in-memory occlusion | 同一 logical base GT |
| wrong-order | reverse `[18,17,16,15]`，保留真实 RGB capture timestamps | 未重排的 logical base GT |
| low-overlap | 15--18 的固定、远离 base 的同场景 donor | 未替换的 logical base GT |

每个 input 必须严格串行完成 `baseline -> alarm builder -> always-commit -> detector-policy`。
alarm builder 只读取 baseline `health.jsonl` 的 v2 online fields、同一 final RGB 的 visual
correspondence 与 `rgb.txt` capture timestamps；拒绝 GT/depth/label/event/source-index/future
字段。policy 对 candidate frame `t` 只读 alarm `t-1`，并使用固定 rising-edge transform 与
`max_hold=3`。每条成功 output 都在继续前冻结为 directories `0555`、files `0444`。

在任何正式 forward 之前，commitment 必须冻结全部两场景、八个 manifest、代码 revision、
checkpoint、Detector-v3 config、runner contract 与三类 forward。后续 baseline、alarm builder、
always-commit、detector-policy、evaluator 与 aggregate 都必须显式接收同一个
`--recovery-quality-commitment`；每一层拒绝未承诺的 manifest、代码/检查点/配置漂移或
scene/condition substitution，并将该绑定写入自己的只读 artifact。aggregate 的唯一 attempt seal
仅接受八个与 commitment 一一对应的 evaluation。

总计 8 个 inputs、24 个 GPU forwards。每一条 run 使用相同的 pinned checkpoint SHA-256、seed
0、size 512、`beta_base=0.1`、batch 1 与单张启动时可用显存至少 12 GiB 的 GPU；不停止或清理
他人进程。

## 评价

GT 是独立 post-hoc evaluator 的唯一输入。每条 predicted trajectory 只用 prefix frames 0--14
作 non-reflective Umeyama Sim(3) 对齐；固定该变换后，tail frames 19--29 报告：

- ATE center RMSE（m）；
- 相邻帧 translation RPE RMSE（m）；
- 相邻帧 rotation RPE RMSE（deg）；
- runtime、peak GPU memory、alarm/hold/replay/dropped transaction counts。

baseline 与 always-commit 的 checkpoint audit、health、prediction summary、trajectory 必须逐字节
一致。否则该 input 不可评价。wrong-order 的每个 scene 必须产生 rising-edge alarm 与真实
hold/replay；其余条件无 alarm 是观察结果，不能触发阈值或 policy 改动。

## 决策规则

六个 event trials 是 two scenes × dynamic/wrong-order/low-overlap。对 ATE、translation RPE，
定义 effect 为 `1 - policy / baseline`。

`RECOVERY_QUALITY_GO` 要求全部成立：

1. 所有 input/GT/commitment/provenance/permission 验证通过，24 条 run 成功，zero restore failure；
2. six-event median tail ATE effect >= 10%，median translation-RPE effect >= 10%，且至少 4/6 的
   ATE effect 为正；
3. seed `20260806` 的 10,000 次 scene-stratified paired bootstrap 对两项 median effect 的
   95% lower bound 都 > 0；
4. 两个 clean trial 的 ATE 与 translation-RPE 均不恶化超过 5%，policy median runtime 增幅不超过
   20%。

其余任何完整结果均为 `RECOVERY_QUALITY_NO_GO`；无法得到有效、可重放的 GT/policy evidence
则为 `RECOVERY_QUALITY_FEASIBILITY_NO_GO`。正式 results 不得事后删除不利 input、改 window、
指标或门槛。
