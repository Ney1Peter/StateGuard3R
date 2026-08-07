# Recovery-policy development v1 结果

- 评测日期：2026-08-07
- 唯一开发结论：**`DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`**
- 范围：只读 forensic 加三个已经解盲的 30-frame development manifests
  (`development-dynamic`、`development-wrong`、`development-low`)；不是新的 formal
  recovery-quality 结果。
- 长时目标完成状态：**已完成并停止**。固定的 H0 forensic、Gate B/C 的三条件完整矩阵和
  Gate D 评估均已经完成。由于预注册门槛未通过，本阶段不再调参、增加 forward、下载数据/权重，
  或制定/执行新的盲测 acquisition commitment。

## 结论

这轮开发工作证明了两件不同的事。

1. 现有 detector-triggered `hold/replay` 不改变轨迹的因果解释 H0 是正确的：它在 clear frame
   前重新装回被 hold candidate 的 post-state，因此再次接回 baseline 的状态链。
2. 新的 development-only `hold/discard` 的确具有预期的因果杠杆：三条真实 detector-driven
   run 都在 frame 16 hold、frame 17 discard，并且从 frame 17 开始与 replay 的 trajectory 和
   prediction summary 都不同。

但 H1 的“会改变后续状态”并不等于“会改善质量”。在三个已解盲 development corruption 中，
discard 没有任何一条同时改善 tail ATE 和 translation-RPE；两项 relative effect 的中位数均为负，
且 discard/baseline runtime ratio 的中位数为 1.7541，超过 1.20 上限。因此它不是可以进入新盲测的
候选，正常结束状态只能是 **`DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`**。

这些数值只用于这次已解盲开发判定，不能作为新的 formal recovery-quality 声明，也不得用于改动
Detector v3 threshold、hold 上限或 corruption 输入。

## 不可变边界与来源

- 没有修改、重跑或把 `recovery-quality-formal-*` 封存输出作为新的 model response；H0 只读取已公开
  JSON。
- 没有下载数据集或权重；forward 仅使用
  `outputs/formal-v1-inputs-0001/development/` 下的三个只读
  `stateguard3r.corruption.v1` manifest。
- ReCal3R source 与 checkpoint 未改动。所有新 forward 绑定 ReCal3R
  `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` 和 checkpoint
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`。
- 所有 GPU forward 串行绑定 `CUDA_VISIBLE_DEVICES=2`，即 NVIDIA L20
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`。每次 preflight 记录了超过 12 GiB 的可用显存和
  已有 PID；没有停止、迁移或修改其他用户的进程。

## Gate A：H0 forensic 与候选实现

冻结的只读 H0 report 为
`outputs/recovery-policy-h0-analysis-0001/h0-analysis.json`，SHA-256
`d0de5e8bc5ca96615b6cfd0c3cd47a5705f47da81b0d6541e366afb289de60c2`。
它对八个公开 formal-v1 baseline/policy pairs 全部通过：四类 model-output 文件逐字节一致，
13/13 次 replay 在 clear-frame forward 前重新安装 held candidate post-state，且共有 13 次 hold。
这独立复现了 H0，而没有制造新的 formal response。

候选实现始终是 development-only 的
`detector-v3-quality-prior-alarm-discard`：它拒绝 formal commitment，并只接受受限 development
manifest。关键提交为：

| 内容 | Code | Test |
| --- | --- | --- |
| H0 analyzer | `f0ec85b` | `af34a0a` |
| discard policy 与 manifest guard | `c3201a2` | `7d7b753` |
| development evaluator | `817149b` | `01f2f19` |
| discard 后 health ledger reconstruction | `74a52a0` | `2168cf8` |

第一次 `dynamic-discard-0001` 正确恢复 pre-model state 后，runner 的旧 health trace
完整性假设拒绝了被 discard 的 frame 16，因此以 trace-step mismatch 失败。该失败目录与日志被保留，
没有覆盖。后续修复只把已观察的 held-frame summary 补回**后验 health ledger**，从不把它重新提供给
model、detector 或 policy。定向修复测试为 25 passed；随后完整 CPU suite 为 447 passed in 49.61 s，
`compileall` 通过。

## Gate B/C：冻结的完整开发矩阵

三条 shadow alarm 都是现有 baseline health、model-ready RGB overlap 和 RGB capture timestamp 的
CPU-only 产物；没有读取 GT/depth/label/event/source-index，也没有 future-frame input。它们都给出
policy rising edge 15，因此未使用 forced fallback。每个成功 output 目录为 `0555`、其文件为 `0444`。

| Condition | manifest SHA-256 | alarm SHA-256 | baseline / always / replay / discard | evaluator SHA-256 |
| --- | --- | --- | --- | --- |
| dynamic | `264b0c9e4f04f4da9e26a634ed008342d32c7b561d7b6ee4f9eb90d1688189c8` | `ffa3de6c5b97dc7a133078e0593825a93f7cf91527f2ad2833286f7d0c532b09` | `dynamic-{baseline,always-commit,replay}-0001`, `dynamic-discard-0002` | `397d2dd647e4767fb052dece8368f47562108e852438372a1c35706b90cdd662` |
| wrong | `860c45ba688da966f01df559b718e33a318352673c65d8787645b563d7e8f669` | `d1c854e1527193213343fb41fbaf9e7827ec0b62afa26bc5e64939661ef9d461` | `wrong-{baseline,always-commit,replay,discard}-0001` | `7c7fc6ba6aed553d9c470ac7140a9a71d0448be9200a68a442ad8d7f86d34eac` |
| low | `72eeb1dc297992b14a6a3bd5e166832a5220ee16df82dfc9d39aefec25aec0c0` | `1701566f366540c4cee31b5892babe7f86834e28bcd3a6a36ffbbca6f4ca896c` | `low-{baseline,always-commit,replay,discard}-0001` | `3121e2e55bc0e916b40035f99928395e078f1e25ec349a20b688c45fa4240adf` |

各 evaluator 位于相应的
`outputs/recovery-policy-development-v1-<condition>-evaluation-0001/`，只对 source manifest 的
logical-base GT rows 0--29 进行后验读取；Sim(3) 只在 prefix 0--14 拟合，并只在 tail 19--29 计分。
它们各自冻结了 run/manifest SHA、baseline/always 原始字节比较、metrics、effects、runtime ratio 和
discard state evidence。

## Gate D：因果、安全与质量门禁

### 已通过的机械与安全事实

- 三个 baseline/always control 的 `checkpoint-load-audit.json`、`health.jsonl`、
  `predictions-summary.json` 与 `trajectory.json` 均逐字节一致。
- 所有 12 个 matrix run 都是 `status=succeeded`、30 frame、30 行 health、29/29 calibrated updates，
  且 trajectory 为有限值。
- 每个 replay/discard policy timeline 都有 30 个连续 frame ID；仅在 frame 16 hold，且在 frame 17
  release。没有 dropped transaction、超出 `max_hold=3` 的 hold、未记录 pending transaction 或
  restore failure。
- 三个 discard timeline 都明确记录丢弃 held frame 16；frame 17 的 pre-state digest 不等于 frame 16
  held candidate 的 proposed post-state。
- 每个 condition 的 discard 与 replay 在 frame 17--29 的 trajectory **和** prediction summary 都
  不同（而不是仅整文件 hash 不同）。这证明 H1 的 state-path intervention 在 clear frame 起真实生效。

### 开发质量数值（探索性，不是 formal 声明）

Effect 定义为 `(baseline - discard) / baseline`，所以正值才是改善。

| Condition | baseline → discard ATE (m) | ATE effect | baseline → discard translation RPE (m) | translation-RPE effect | discard / baseline runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| dynamic | 0.05502792 → 0.05517429 | -0.2660% | 0.01161666 → 0.01164103 | -0.2098% | 1.5045 |
| wrong | 0.03875308 → 0.03838191 | +0.9578% | 0.01189775 → 0.01231558 | -3.5118% | 1.8108 |
| low | 0.42494489 → 0.42581647 | -0.2051% | 0.44507481 → 0.44659615 | -0.3418% | 1.7541 |
| median | — | **-0.2051%** | — | **-0.3418%** | **1.7541** |

| Gate D requirement | Actual | Result |
| --- | ---: | --- |
| baseline/always four-file byte equivalence | 3 / 3 | PASS |
| real detector-driven discard with non-reinstalled held state | 3 / 3 | PASS |
| clear-or-later trajectory/prediction difference from replay | 3 / 3, first frame 17 | PASS |
| trials with both ATE and translation-RPE effect strictly positive | 0 / 3, required >= 2 / 3 | **FAIL** |
| median ATE effect | -0.2051%, required >= 5% | **FAIL** |
| median translation-RPE effect | -0.3418%, required >= 5% | **FAIL** |
| discard / baseline runtime median | 1.7541, required <= 1.20 | **FAIL** |
| run/timeline/permissions/hash/PID/GPU audit | passed for all substantive outputs | PASS, with retained log note below |

The initial `dynamic-discard-0002` tmux submission accidentally expanded its postflight shell
return variable before the command reached tmux, leaving the final `exit_code=` field blank in
that one postflight log. The output itself is a complete frozen `status=succeeded` run with all
required artifacts, and its runner JSON records normal completion; the error is retained rather
than rewritten. Every later newly launched policy forward records `exit_code=0`. This audit-log
defect cannot improve the decision and is an additional reason not to advance the candidate.

## Frozen candidate disposition and follow-up boundary

The development-only discard implementation is frozen in the commits above and the three
successful discard output directories. It is **rejected as a recovery candidate**, not deleted:
the frozen matrix explains that it changes the causal state path but degrades the two decision
metrics on the disclosed development set while adding excessive runtime. The old failed
`dynamic-discard-0001` is separately retained only as implementation-failure evidence and is not
an evaluation input.

Because the only allowed result is `DEVELOPMENT_RECOVERY_FEASIBILITY_NO_GO`, no new blind-test
acquisition or commitment plan is authorized by this development plan. Any future policy search
would require a new, separately authorized development protocol; it must not reuse these
disclosed data as confirmatory evidence and must acquire/commit new unseen data before making any
renewed recovery-quality claim.

## Final reproducibility checks

The final code state has a clean StateGuard3R worktree after docs are committed; ReCal3R remains
clean at its pinned commit. The final CPU suite and `compileall` are rerun after this report is
written, with project-local `TMPDIR`; their logs, `git diff --check`, recursive permission/hash
verification, GPU/PID sweep and tmux status are the final handoff evidence.
