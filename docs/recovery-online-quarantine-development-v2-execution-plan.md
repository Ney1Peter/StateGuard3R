# Recovery online-quarantine development v2：连续风险 episode 隔离执行计划

- 制定日期：2026-08-07
- 状态：**已完成 — `ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO`**
- 计划窗口：12--18 小时；以预注册门禁完成为终点，不因时间耗尽而停止
- 前置证据：[recovery-policy-development-v1-result.md](audits/recovery-policy-development-v1-result.md)
- 受保护 baseline：ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`

## 1. 目标与允许的结束状态

v1 已经严格否决了**冻结 baseline shadow alarm 的 `t-1` rising-edge 单帧 hold/discard**。它没有否决
同帧、连续 episode 的 state quarantine：v1 把连续 raw hybrid alarm 压缩为单个 position 15，导致三个
条件都是 `f15 commit -> f16 discard -> f17+ commit`；而真实 raw hybrid ranges 分别为 dynamic 15--18、
wrong 15--19、low 15--20。首个异常状态已提交，后续异常状态也再次提交，所以 v1 对完整风险 episode
没有因果保护。

v2 要检验唯一的、可证伪的假设：**在每个当前帧 output 已产生后，仅用至该帧为止的固定 Detector-v3
online observables 判断 current alarm；若报警，保留该帧已产生 output 但立即恢复完整 pre-frame recurrent
closure；连续报警逐帧重复，首个 clear 才 commit。** 这可阻断 f15 起整个已检测 episode 对未来状态的
污染，且不使用未来帧、GT 或标签。

本阶段有且仅有两个正常结束状态：

1. **`ONLINE_QUARANTINE_DEVELOPMENT_CANDIDATE_READY`**：候选的因果、安全、等价、运行时与开发
   质量门禁全部通过。仅此时写一个**尚未执行**的小规模新盲测 acquisition/commitment plan；它不构成
   recovery-quality 正式结论。
2. **`ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO`**：candidate 无实际 action、不能维持完整
   episode rollback、没有改变后续输出、违反安全/运行时门槛，或未达到预注册开发质量门槛。该结果同样
   完成本计划；不得用同一计划继续搜索阈值、系数、fallback 或其他候选。

这是一轮用户新授权的 development-only 研究，不撤销 v1 NO-GO，也不重新解释或重跑 formal v1。

## 2. 不可变边界

- 绝不修改、重跑、覆盖或读取 `recovery-quality-formal-*` 的模型 response；只允许继续引用已公开的
  v1 H0 report。
- 不下载数据集、权重或依赖。`/data` 仅约 85 GiB 可用且使用率显示 100%，新 output 必须是受控的
  JSON/日志；实际 forward 只允许使用三个只读 manifest：
  `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`。
- 不修改 ReCal3R source、checkpoint、raw RGB/GT、Detector v3 threshold/config 或 corruption manifest。
  v2 detector adapter 绑定已有冻结 Detector-v3 config，但每帧只接受当前/过去的 health、RGB overlap 与
  capture timestamp；它拒绝 GT、depth、label、event/source-index、baseline alarm artifact 和 future frame。
- 新候选及其 runner 拒绝 `--recovery-quality-commitment`、`--quality-alarm-artifact` 和 synthetic
  `--alarm-frame`；只接受 development root 内、`stateguard3r.corruption.v1`、source read-only 的 manifest，
  所以不能误用于 formal v1。
- 单卡、batch 1、显式 `CUDA_VISIBLE_DEVICES=<id>`。每条 GPU forward 前记录 UUID、至少 12 GiB
  空闲显存、已有 PID、完整命令、时间、日志和退出码；不干预其他用户的进程。

## 3. 预注册候选：online current-frame episode quarantine

候选名称为 `detector-v3-online-current-quarantine`，语义固定如下：

1. 在 frame `t` 进入 pinned recurrent computation 前，捕获完整 pre-frame local closure
   (`state_feat/state_pos/init_state_feat/mem/init_mem/reset_mask`) 与所有已登记 model mutable/config/RNG
   state。candidate forward 照常产生 `prediction[t]`。
2. 在 prediction 已产生、任何策略 state 恢复前，online adapter 从 t 的 candidate health/prediction、
   当前/前一 RGB overlap 和 timestamp 计算 fixed-config `hybrid_alarm[t]`。该决定不能改变
   `prediction[t]`，也不能读取 t 后的任何量。
3. `hybrid_alarm[t]=true` 时，立即恢复 t 的完整 pre-frame closure、model mutable/config state 和 RNG；
   `prediction[t]` 原样保留，timeline 记录 `quarantine_current_rollback`。连续 true 的每帧都重复本动作，
   因而任何被 quarantine candidate post-state 都不会成为下一帧 pre-state。
4. `hybrid_alarm[t]=false` 时，正常 commit。本 episode 的首个 clear frame 因此从最后 safe pre-state
   正常处理和提交；没有 replay、pending transaction 或延迟补算。
5. 若 episode 连续 rollback 超过固定 watchdog 8 frame，runner fail-closed 退出，绝不越界后静默 commit。
   reset boundary 也明确 fail-closed，而不是不记录地改变 reset 语义。

被 quarantine 帧的 health 由独立 append-only observed ledger 保存，用于后验审计；它永不反馈给 model、
adapter 或 policy。v2 **不**替换或回填已产生 pose，也不引入 output fallback；若 state-only candidate
因当前坏 RGB 的 exported pose 而失败，那个新机制只能在下一份独立计划中研究。

## 4. Gate A：online adapter、快照正确性与性能前提

新增独立 v2 runner/primitives/evaluator，不改旧 v3 replay/discard runner。runner 必须逐行绑定 pinned
ReCal3R lighter source，normal path 显式使用 `recal3r / beta_base=0.1 / health=v2`。

必需 CPU tests：

- online detector prefix-invariance：改变 future rows 不改变 t 的 score/action；拒绝所有禁止字段；
  commit-only prefix scores 与已冻结 Detector-v3 config 的 CPU reference 一致；
- current-frame causality：输出在 action 前产生并保留；full closure/RNG 恢复逐字节等于 pre state；
  false alarm normal commit 等价；frame 0/reset/watchdog 都 fail-closed；
- 连续 episode：raw true run 的每个 candidate 都 rollback、pending 恒为零、首 clear 正常 commit，且
  quarantine candidate post digest 永不成为后续 pre digest；
- observed health ledger 对每一个 rollback frame 有观测行，但 policy interface 从不读取它；
- development/commitment/alarm-frame guards；短/非有限/错误 provenance/权限/policy output 拒绝；
- light no-alarm `always-commit` 与 v1 frozen smoke baseline 的
  `checkpoint-load-audit.json`、`health.jsonl`、`predictions-summary.json`、`trajectory.json` 逐字节一致。

v1 的全 transaction wrapper 每帧同时 clone pre/post closure，always control 的 runtime median 已为 1.645，
不可能满足 1.20。因此 v2 必须先以 pre-snapshot-only normal path 做性能门：正常 runner 不构造 post
snapshot/digest；仅实际 quarantine action 才记录 post digest。若三条 v2 light always 的 runtime median
仍高于 corresponding baseline 的 1.20，直接 NO-GO，不启动完整 candidate matrix。

**Gate A 通过条件：** 所有专项及完整 CPU suite、`compileall`、`git diff --check` 通过；代码、测试、
计划分别提交；ReCal3R clean；light always 的三条 preflight/run-time controls 通过。

## 5. Gate B/C：固定 development 矩阵

复用 v1 已冻结的三条 smoke baseline **只读 reference**，不重跑。每个 condition 新跑且只跑一次：

| Method | Purpose |
| --- | --- |
| v2 light `always-commit` | 证明轻量 external clone 的 normal path 与 frozen smoke baseline byte-equivalent，并完成 runtime precondition |
| v2 online current-frame quarantine | 唯一候选；每个真实 current raw alarm 立即恢复 pre-state，直到 first clear |

正常 path 的 detector action 必须由同次 forward 的 prefix-only adapter 产生；不得把 v1 baseline
`alarm.json` 当作 intervention 后的 action oracle。只可将已公开 raw-alarm episode 用作 post-hoc
expected diagnostic，不得作为 candidate 的输入。每个成功目录以
`recovery-online-quarantine-development-v2-<condition>-<method>-0001` 命名，原子冻结 `0555/0444`；
不得覆盖任何 v1 output。

## 6. Gate D：后验 development 评价与选择

新增独立 CPU-only v2 evaluator。它仅在 forward 后从 legacy source manifest 的 logical-base GT rows
读取 pose；prefix 0--14 拟合 proper Sim(3)，tail 19--29 计算 ATE/RPE。它必须验证：

1. 三条 frozen v1 baseline 与对应 v2 light always 的四项文件逐字节一致；所有 runs 有 30 个有限
   trajectory frame、30 health row、29 calibrated update，正常 path 无 restore/pending state；
2. 每个 actual quarantine 有 current online alarm、pre/proposed/committed digest、observed health row；
   committed digest 等于 pre digest，连续 action 无 post-state leakage，first clear commit 被记录，且至少
   一条 condition 在 action 后 trajectory 或 prediction summary 与 normal path 不同；
3. 三条 corruption 完整报告 candidate/baseline ATE 与 translation-RPE effect；至少 2/3 条件两项严格
   为正，两项 median 都至少 +5%，candidate/baseline runtime median 不超过 1.20；
4. 所有 policy output、adapter config/input SHA、权限、PID/GPU audit 和 final CPU suite/compileall 通过。

任何一项失败即 `ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO`。不得根据 development 数值调整
threshold、watchdog、episode定义、snapshot范围或门槛。

## 7. 交付与下一数据边界

完成后交付：冻结的 per-condition evaluator output、代码/测试提交、
`docs/audits/recovery-online-quarantine-development-v2-result.md`、更新 run registry 与本计划。

仅当候选 ready，另写一个**不执行**的盲测 acquisition/commitment plan：先做磁盘/来源/大小预检，
下载量以单个可验证的小型官方序列为上限，冻结 source/association/code/config/candidate/threshold 后再
允许读取新 response。若 v2 NO-GO，报告必须明确下一研究需要新的、单独授权的机制假设，不能继续在
本计划或同一开发矩阵上调参。

## 8. 实际执行记录与结束判定

本计划已完成。final CPU suite 为 **458 passed**，`compileall` 通过；ReCal3R 保持 pinned clean。所有
substantive GPU forward 都在 `CUDA_VISIBLE_DEVICES=2` 的 L20 上串行执行，并在每次启动前记录 GPU UUID、
可用显存、其他 PID、命令与退出码。

Gate A 的 v2 always-commit 与 v1 frozen baseline 的四个模型产物在三个 condition 都逐字节一致。可比较的
`external_recurrent_and_current_policy_only` runtime ratios 是 dynamic `0.9308`、wrong `0.9850`、low
`1.0500`，中位数 `0.9850`，所以轻量 normal path 通过性能前提。RGB overlap 是所有方法共有的输入
instrumentation；冻结 baseline 原本在 `runtime_seconds` 外计算，v2 改为逐帧因果计算并单列为
`wall_runtime_seconds`，不把它混入可比 recurrent/policy 计时口径。

唯一 candidate 在三条 development corruption 都完成并冻结，且每条的 frames 15--20 都有 real current
online alarm 和 `quarantine_current_rollback`。每次 rollback 的 committed digest 等于 pre digest、pending
始终为零，observed-health ledger 完整记录被隔离 candidate。因果机制通过；但 ATE effect 为
`-3.2650%/-2.9494%/-1.7153%`，translation-RPE effect 为 `-2.6789%/-15.4509%/-2.3883%`，
candidate/baseline runtime ratios 为 `4.6388/2.9076/2.7462`（中位 `2.9076`，上限 `1.20`）。

冻结的 CPU evaluation 为
`outputs/recovery-online-quarantine-development-v2-evaluation-0001/evaluation.json`
（SHA-256 `90818667e2d54f898d3343d8bb246822f8af4eb71da6f23e477a31a027ddbdf0`），outcome 是
**`ONLINE_QUARANTINE_DEVELOPMENT_FEASIBILITY_NO_GO`**。详见
[recovery-online-quarantine-development-v2-result.md](audits/recovery-online-quarantine-development-v2-result.md)。

两个 retained pre-forward implementation failures（路径导入与 `no_grad` helper）只产生 staging cleanup 和
exit 1，不产生 model response，也不作为 matrix evidence；修复后才有统一 code commit 的 six accepted
forwards。此前 dynamic always output `...-0001` 是 runtime-scope 修正前的 equivalence preflight，保留但
不作为 Gate A 的 accepted timing run；当前实现对应的 frozen accepted output 是 `...-0002`。本计划到此停止：
不得调整 threshold、watchdog、snapshot 范围或加入 fallback。新的 output-export mechanism 必须另立 v3 假设与计划。
