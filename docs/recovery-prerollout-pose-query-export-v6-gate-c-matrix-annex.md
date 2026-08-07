# Recovery pre-rollout pose-query export v6：Gate C matrix 执行附录

- 日期：2026-08-07
- 状态：**预注册；Gate B 已通过，尚未读取 development GT。**
- 唯一目标：用 each-once wrong/low controls 与 candidates 完成冻结的 development matrix，之后才执行一次
  CPU-only quality evaluation；不以任何中途结果改变 v6 mechanism、detector 或参数。

## 已冻结的 Gate B 结论

dynamic `...always-commit-0001` 因缺 tmux transcript 保留为 `UNACCEPTED_AUDIT_INCOMPLETE`。按已提交的
log-retention recovery addendum，唯一有效 control 是
`recovery-prerollout-pose-query-v6-dynamic-always-commit-0002`：四个 protected files 与 v1 baseline byte-identical，
runtime `3.536778312176466 / 3.4633694058284163 = 1.0211958060911757`，四份 canonical logs 都是 NUL-free `0444`。

唯一 candidate 是 `recovery-prerollout-pose-query-v6-dynamic-candidate-0001`：frames 15--20 的六个报警全部为
`quarantine_current_rollback` + full witness + `export_pre_rollout_pose_query`，每个 query 都是
`(1,1,768)`、`cuda:0`、proper SO(3)，raw/export pose digest 均不同；runtime
`3.8006518967449665 / 3.4633694058284163 = 1.0973856529277373`。两项 Gate B validator 分别冻结于
`logs/recovery-prerollout-pose-query-v6-dynamic-always-commit-0002-gate-b-control-validation-0002.log` 与
`logs/recovery-prerollout-pose-query-v6-dynamic-candidate-0001-gate-b-validation.log`。

后续 repository-level docs commit 不改变四个 v6 executable component hashes；它们继续被
`recovery-prerollout-pose-query-export-v6-gate-b-log-retention-recovery-addendum.md` 的固定 SHA-256 约束。任何 hash
漂移使本 Gate C 失败而非授权改代码。与这些 frozen validation artifacts 冲突的、未引用它们的 speculative v7 文档不构成
v6 Gate B 结论，也不授权替代或重跑 v6 candidate。

## 唯一 Gate C matrix

执行顺序与 output IDs 固定如下；每个 output/log ID 只能使用一次：

1. `recovery-prerollout-pose-query-v6-wrong-always-commit-0001`，读取
   `outputs/formal-v1-inputs-0001/development/development-wrong/input-manifest.json`，相对
   `outputs/recovery-policy-development-v1-wrong-baseline-0001`。
2. 仅 1 通过后：`recovery-prerollout-pose-query-v6-wrong-candidate-0001`，同一 wrong manifest。
3. 仅 2 通过后：`recovery-prerollout-pose-query-v6-low-always-commit-0001`，读取
   `outputs/formal-v1-inputs-0001/development/development-low/input-manifest.json`，相对
   `outputs/recovery-policy-development-v1-low-baseline-0001`。
4. 仅 3 通过后：`recovery-prerollout-pose-query-v6-low-candidate-0001`，同一 low manifest。

每个 control 必须再证明其四个 protected files byte-equivalent、runtime/baseline `<=1.20`、state timeline binding、
component hashes、output/log freeze 和 tmux provenance。每个 candidate 必须再证明所有 real alarm 的 rollback/full witness/
query evidence/no fallback，且 runtime/baseline `<=1.20`。任何一步失败即
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_AVAILABILITY_OR_RUNTIME_NO_GO`：不运行该步骤之后的 matrix、GT evaluator 或 v6 调参。

四个 run 都必须保持 v6 plan 的 checkpoint/config/seed/size/beta/health/watchdog、GPU 2 UUID 规则、两次
`>=12288 MiB` preflight，使用同一预建立 tmux pipe-pane transcript pattern。所有 outputs 与 logs 最终分别为
`0555/0444` 和 NUL-free `0444`。control 的 exported default path 必须永远是 raw pose；candidate alarm path 才可使用
pre-rollout query decode。

## 允许的唯一后续

只有上述四个 outputs 全部冻结且 validators PASS 后，才能在 CPU 上读取 frozen logical-base GT，按 v6 原计划的
prefix 0--14 Sim(3)、tail 19--29 ATE/RPE 定义计算 six-output quality result。candidate-ready 仍要求：三个 controls
byte-equivalent、zero restore/query/watchdog/leakage failure、至少 2/3 条件的 ATE 与 translation-RPE 同时正改善、两项
median `>=+5%`，以及 candidate runtime median `<=1.20`。否则发布 v6 feasibility NO-GO；无论结果都不授予对已解盲
matrix 的第二次参数搜索。
