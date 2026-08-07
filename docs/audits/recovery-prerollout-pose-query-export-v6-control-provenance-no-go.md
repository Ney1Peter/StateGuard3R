# Recovery pre-rollout pose-query export v6：dynamic control provenance NO-GO

- 审计日期：2026-08-07
- 绑定 run：`recovery-prerollout-pose-query-v6-dynamic-always-commit-0001`
- 结论：**`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_PROVENANCE_NO_GO`**。

## 已通过的 control 数值与冻结条件

| 检查 | 证据 | 结果 |
| --- | --- | --- |
| command/exit | `*-preflight.log`、`*-main.log`、`*-postflight.log` | GPU 2 两次 preflight 均为 45,586 MiB free；`EXIT_CODE=0` |
| protected byte equivalence | v1 dynamic baseline 与 v6 control 的 `checkpoint-load-audit.json`、`health.jsonl`、`predictions-summary.json`、`trajectory.json` | 四个 SHA-256 均一致、逐字节 `cmp` 通过 |
| runtime | control `3.949031349271536 s` / baseline `3.4633694058284163 s` | `1.1402281670057521 <= 1.20` |
| immutable output | `outputs/recovery-prerollout-pose-query-v6-dynamic-always-commit-0001` | directory `0555`、7 个文件均 `0444` |
| timeline binding | `run.json.state_policy.timeline_sha256` | 等于实际 `state-timeline.json` SHA-256 `5abbc8f9...de6afaa4`；30 transactions，pending `0` |
| source provenance | `state-timeline.json` | ReCal3R clean `466c7cdf...`，pinned lighter line span `1661--1833` 及 hashes 与 Gate A 一致 |

## 不能满足的不可变 provenance 条件

v6 计划第 4 节明确规定每次 GPU run 冻结 canonical
`preflight/main/postflight/tmux` logs 为 `0444`。实际 launcher
`tmp/launch-v6-control-0001.sh` 定义了：

```text
transcript="$repo/logs/$run_id-tmux-transcript.log"
```

并把它纳入 one-use existence guard，但没有任何写入、`pipe-pane`、`capture-pane` 或 `tee` 到该
path 的命令。实际结果是：

- 三个 existing logs 都为 `0444`；
- `logs/recovery-prerollout-pose-query-v6-dynamic-always-commit-0001-tmux-transcript.log` 不存在；
- `tmux` session 已只有 windows `0--4`，v6 control window 不存在，不能从 pane history 恢复；
- launcher shell 已退出，故不能再产生原始 terminal transcript。

主日志确实保存了 command、Python PID 与 runner JSON，且没有显示任何计算错误；但它不是计划要求的
独立 tmux transcript。事后用现有日志构造同名文件会伪造不存在的过程证据，因此不可接受。

## 约束性决定

v6 Gate B 不完整。不得启动
`recovery-prerollout-pose-query-v6-dynamic-candidate-0001`，也不得运行 wrong/low、GT evaluator 或对
v6 source/decoder/detector/threshold 做变更后重跑。该决定是 provenance NO-GO，不是对 pre-rollout
pose-query 科学假设的 quality 否定。

若继续研究，只能在新的预注册版本中保留完全相同的 v6 scientific mechanism，并在 GPU 前用一个无数据、无 GPU
的 tmux smoke test 证明 transcript 从 pane 创建开始被捕获、包括 command/exit code，且在 child run 结束后被
独立 `0444` 冻结。新版本必须以新 output/log IDs 重新走 Gate A lineage 与 dynamic control；不能把 v6 control
作为 candidate 的资格证据。
