# Detector v3 后续：冻结 detector 触发的 state-policy feasibility

- 制定日期：2026-08-03
- 状态：**执行中；仅因正式 Detector v3 已 GO 而解锁**
- 上游正式证据：`outputs/formal-v3-evaluation-0001/`
- 范围：已解盲 `blind-wrong-order` 输入上的探索性事务语义验证；不是第二次 detector
  evaluation，也不是 recovery-quality 研究。

## 目标

证明外部、hash-bound 的 ReCal3R transactional wrapper 能由**冻结的 formal Detector v3
hybrid alarm**（而非手写 alarm、标签、GT 或未来帧）因果地控制 state/memory commit：报警 `t`
至多影响候选 update `t+1`，hold 后的 clear 能显式 replay，且 always-commit control 与未包装
baseline 的模型/health artifacts 逐字节一致。

该目标的成功标签固定为
`TRANSACTIONAL_STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM`。即使成功，也不得声称 ATE/RPE、
几何恢复、漂移降低、自然攻击鲁棒性或 end-to-end quality improvement。

## 冻结输入与 alarm 来源

唯一输入为已经过正式 evaluate 而解盲的：

```text
outputs/formal-v3-inputs-0001/blind-wrong-order/input-manifest.json
```

唯一允许的 detector evidence 为同一输入的冻结 formal v3 runner metadata 和 evaluator timeline：

```text
outputs/formal-v3-runs-0001/blind-wrong-order/run.json
outputs/formal-v3-evaluation-0001/timelines/blind-wrong-order.json
```

`detector-v3-prior-alarm` 必须验证 run metadata 的 input-manifest path/SHA-256/size 与当前 wrapper
input 完全一致，验证 v3 health profile，并只逐帧读取 timeline attribution 的
`hybrid_alarm` boolean。它禁止访问 `labels`、corruption/event metadata、GT、depth、source index
或 future score。连续的 true 表示同一持续风险 episode，因此 controller 使用固定、因果的
`false -> true` 上升沿过滤器；它完整记录原始 hybrid positions 与 policy positions，但只将每个
episode 的第一帧作为 bounded hold/replay trigger。controller 在 frame `t` 只能查看 policy
alarm `t-1`。

## 三组固定运行

使用同一 checkpoint、30 帧 input、size 512、seed 0、beta base 0.1、GPU 0（或另一张单卡且每次
启动前两次快照均有至少 12 GiB 空闲）、ReCal3R pinned commit 和 v2 health profile，串行运行：

1. `state-policy-v3-v2-control-0004`：未包装 pinned smoke runner；
2. `state-policy-v3-always-commit-0004`：external wrapper，零 alarm；
3. `state-policy-v3-detector-hold-0005`：external wrapper，
   `detector-v3-prior-alarm` 及上述两个冻结 evidence path。

每条 run 的 stdout/stderr、启动命令、PID、GPU UUID、两次 preflight 与 postflight `nvidia-smi`
快照都保存在本项目 logs。成功 output root 及全部 artifact 必须为 `0555/0444`。不修改
ReCal3R source、checkpoint、formal v3 inputs/runs/evaluation，且不下载、训练或启动第二个场景。

## 验证与门槛

CPU validator 输出固定为 `outputs/state-policy-v3-detector-feasibility-0005/`。先前
`state-policy-v3-detector-hold-0004` 在连续 raw alarm 下显式保留为 failed operational attempt；
它没有被覆盖、删除或用于结论。新编号使用上述预先固定的 causal rising-edge policy，并要求：

1. baseline、always-commit、detector-policy 使用相同 input/checkpoint/runtime identity；
2. baseline 与 always-commit 的四个 model/health artifacts 逐字节一致；detector policy 的同四个
   artifacts 也保持一致，因此状态语义不会被伪装成 output-quality gain；
3. detector evidence files 自身只读，SHA-256/size、v3 profile、input binding 与 hybrid alarm
   positions均重放一致；
4. 至少一个 hold 的 `committed_state_digest` 等于该 transaction pre-state，而不同于 proposal；
   至少一次随后的 clear replay 以 held proposal digest 为 pre-state；所有 hold 都由前一帧的
   frozen hybrid alarm 触发；
5. always control 无 state intervention、pending transaction 不静默丢弃，三个任务 PID 都不在
   postflight snapshot 中。

任一项失败的结论是 `STATE_POLICY_FEASIBILITY_NO_GO`，并保留失败日志和已有 immutable formal
evidence；不得用该已解盲输入调整 Detector v3。全部通过后，更新审计和 run registry，随后执行
full CPU suite、worktree/permission/GPU/PID audit 并收尾。
