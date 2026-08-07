# Recovery early-decoder pose-token export v8：开发终态

- 终态日期：2026-08-07
- 唯一结论：**`EARLY_DECODER_POSE_EXPORT_V8_AVAILABILITY_OR_RUNTIME_NO_GO`**。
- 范围：仅 v8 的 Gate A 与 dynamic Gate B terminal evidence。未下载数据、权重或依赖；未读
  development GT，未运行 v8 candidate、wrong、low 或 quality evaluator。

## 结论

v8 的机制实现与 Gate A 均通过，但 Gate B 的唯一运行授权已经不可恢复地失去
预注册完整性，故不能以任何 control 授权 candidate。这个结论不是对 `dec[0]` early-pose
token 质量的负面数值判断；它是 availability/provenance hard stop。v8 不得用新 ID 补跑、
不得改 early index、precision、head、detector、runtime scope 或 launcher 后再试。

## 冻结证据

原计划仅允许一次固定 control
`recovery-early-decoder-pose-v8-dynamic-always-commit-0001`，并规定它 PASS 后才允许
一个 candidate（[v8 plan](../recovery-early-decoder-pose-export-development-v8-execution-plan.md)
第 92--100 行）。`0001` 的真实 terminal evidence 是：

| 项目 | 结果 |
| --- | --- |
| CUDA child | transcript 记录 child PID `3984986`，并以 `V7_DRIVER_EXIT ... exit_code=0` 结束；并非 preflight-only attempt |
| protected files | `checkpoint-load-audit.json`、`health.jsonl`、`predictions-summary.json`、`trajectory.json` 全部与 v1 dynamic baseline byte-identical |
| runtime | `3.567192484624684 / 3.4633694058284163 = 1.0299774776036152`，小于 `1.20` |
| freeze | output root `0555`、所有 output files `0444`；preflight/main/postflight/transcript 均存在、NUL-free、`0444` |

随后提交的 Gate-B recovery addendum `fd4ee53` 声称 `0001` “没有启动 GPU forward”，并以该
前提授权 `0002`。该前提被上述 immutable `0001` output/main/transcript/driver/result 直接否定：
addendum 在 21:45:00 提交，而 `0001` CUDA child 于 21:45:36 已实际 dispatch。因此该 addendum
不能追溯改变 `0001` 的事件，也不能让 `0002` 成为“唯一恢复 control”。

`0002` 也已产生一个 frozen output 和 exit-0 transcript，但它不是原计划授权的 control；并且在
冻结审计时缺少 addendum 明定的 canonical postflight log。即使之后生成新文件，也不能消除两个
actual controls 已发生这一不可逆事实。`0001` 与 `0002` 仅保留为完整 forensic evidence，均不得
用于 v8 candidate 选择或质量比较。

## 已完成、但不足以越过 Gate B 的工作

Gate A 由 `docs/audits/recovery-early-decoder-pose-export-v8-gate-a.md` 记录：pinned source/
runner contract、checkpoint-loaded official DPT CPU check、32 个 Torch targeted tests、以及
`509 passed, 28 skipped` 的全项目 CPU suite 均通过。它不能豁免 Gate B 的 one-use control、
canonical artifact 与 preregistration 约束。

## 禁止与下一步

1. 不得启动 v8 candidate、wrong/low controls/candidates、GT evaluator 或 blind acquisition。
2. 不得覆盖、删除、解冻或事后重造 `0001`/`0002` output、log、driver、result 或 transcript。
3. 如继续研究，必须先完成一个**科学上不同**的 v9 preregistration；其 Gate A、single-owner
   dispatch protocol 与 dynamic control 必须从零开始。建议的 v9 边界见
   [recovery-early-spatial-pooled-pose-export-development-v9-execution-plan.md](../recovery-early-spatial-pooled-pose-export-development-v9-execution-plan.md)。
