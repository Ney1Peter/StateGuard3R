# Recovery pre-rollout pose-query export development v6：结果

- 评测日期：2026-08-07
- 唯一结论：**`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_AVAILABILITY_OR_RUNTIME_NO_GO`**。
- 范围：只使用既有、只读的 three development manifests；没有下载数据、权重或依赖，没有改动
  ReCal3R/checkpoint/raw RGB/GT/Detector-v3 config。没有读取 GT、运行 quality evaluator、wrong candidate 或 low
  control/candidate。

## 结论

v6 的 pre-rollout pose query 是安全可用且 dynamic runtime 合格的：six dynamic alarm frames 都完成 current-frame
rollback/witness，并导出 proper SO(3) 的 non-raw pose。但是 fixed each-condition runtime gate 同样适用于
always-control。wrong condition 的 accepted control 虽逐字节保持正常路径，仍为
`4.26521532703191 / 3.4450445268303156 = 1.2380726268743498`，超过不可变上限 `1.20`。
因此 v6 不能声称跨条件质量可行，也不得再运行其余 matrix 或调参。

## Gate A

实现/测试/source/real-model Gate A 通过，详见
`docs/audits/recovery-prerollout-pose-query-export-v6-gate-a.md`：checkpoint-loaded actual `DPTPts3dPose` direct
CPU check 满足 float64 proper-SO(3) `1e-8` contract，Torch targeted suite `16 passed`，full local suite
`495 passed, 10 skipped`。source、query-only AST、fail-closed、pre-rollout capture、no CPU policy input、rollback
ordering和 pinned DPT interface 均被测试。绑定 executable component hashes 和 ReCal3R commit 未在以下 GPU runs 中漂移。

## Gate B：dynamic control 与 candidate 均通过

最初的 dynamic control `...always-commit-0001` 数值正确但缺失 original tmux transcript，永久保留为
`UNACCEPTED_AUDIT_INCOMPLETE`，没有被用作 PASS。其后的已预注册 log-recovery control
`outputs/recovery-prerollout-pose-query-v6-dynamic-always-commit-0002` 是唯一接受的 control：

| 项目 | 结果 |
| --- | --- |
| 四个 protected files | 相对 `recovery-policy-development-v1-dynamic-baseline-0001` 全部 byte-identical |
| runtime | `3.536778312176466 / 3.4633694058284163 = 1.0211958060911757`，PASS |
| provenance | two GPU 2 preflights、main/postflight/transcript、output/timeline/component binding 均 PASS |

其后唯一 candidate
`outputs/recovery-prerollout-pose-query-v6-dynamic-candidate-0001` 成功退出并冻结。frames 15--20 的六个真实
Detector-v3 alarms 全部有 `quarantine_current_rollback`、full structural witness、`pending=0`、连续 watchdog 1--6，
并导出 `export_pre_rollout_pose_query`。每个 token evidence 是 `(1,1,768)`、`cuda:0`、source
`pose_retriever_inquire_pre_rollout`；raw/export pose digests 均不同；max SO(3) orthonormality error
`1.11e-16`。candidate runtime 为
`3.8006518967449665 / 3.4633694058284163 = 1.0973856529277373`，PASS。

两个 accepted validations 分别为：

- `logs/recovery-prerollout-pose-query-v6-dynamic-always-commit-0002-gate-b-control-validation-0002.log`
- `logs/recovery-prerollout-pose-query-v6-dynamic-candidate-0001-gate-b-validation.log`

## Gate C：wrong normal path 成为 runtime short-circuit

按已提交 Gate C annex，只运行了唯一 wrong always-control：
`outputs/recovery-prerollout-pose-query-v6-wrong-always-commit-0001`。它 exit 0、output `0555`/files `0444`、four
protected files 均与 `recovery-policy-development-v1-wrong-baseline-0001` byte-identical，two preflights、all four
canonical NUL-free logs、timeline binding 与 component hashes 都通过。但 runtime gate 失败：

```text
4.26521532703191 / 3.4450445268303156 = 1.2380726268743498 > 1.20
```

完整 immutable validator 是
`logs/recovery-prerollout-pose-query-v6-wrong-always-commit-0001-gate-c-control-validation.log`。这不是质量下降或
可通过参数调节修复的授权：v6 plan 明定任何 Gate C control/candidate runtime failure 都立即 short-circuit。

## 禁止的后续与下一版边界

没有运行 wrong candidate、low control、low candidate、GT evaluator 或 recovery-quality selection；没有解盲任何
v6 quality numbers。不得在 dynamic/wrong 上改变 pre-rollout token source、float64 conversion、pose head/postprocess、
Detector-v3、watchdog 或 logs 后再运行 v6。若继续研究，下一版必须预注册一个与 v1--v6 scientific recovery mechanism
不同的 route，先重做 Gate A 和 dynamic Gate B；不能仅以 tmux/provenance wrapper 重新包装 byte-identical v6 components，
因为 wrong normal-path runtime failure 与 detector/candidate 无关。
