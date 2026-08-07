# Recovery safe-anchor export development v3：因果输出恢复执行计划

- 制定日期：2026-08-07
- 状态：**已完成：`SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_FEASIBILITY_NO_GO`**
- 计划窗口：12--18 小时；以全部预注册门禁的正常结束状态为终点，不因时间耗尽停止
- 前置 NO-GO：[v1 state discard](audits/recovery-policy-development-v1-result.md) 与
  [v2 online quarantine](audits/recovery-online-quarantine-development-v2-result.md)
- 受保护 baseline：ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`，checkpoint SHA-256
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`

## 1. 问题与唯一候选

v2 已经证明 current-frame episode rollback 有效：15--20 的 candidate recurrent state 从未进入后续 committed
state。但 v2 故意保留了当前坏 RGB 导出的 pose，三种 corruption 的 ATE/RPE 均变差。因此 v3 只检验一个新、
可证伪的机制假设：**当相同的 current online alarm 隔离 state 时，使用仅由过去两个 safe exported poses
导出的固定 SE(3) constant-velocity pose 替换当前被隔离的 `camera_pose`，能否避免坏 RGB 的当前输出污染，同时
仍让 clear frame 的真实 prediction 正常导出。**

候选唯一名称为 `detector-v3-incremental-safe-anchor-se3-export`，固定语义如下：

1. 在 frame `t` 前捕获 v3 结构化 pre-state closure；candidate forward 正常完成。online detector 只能读取
   `<=t` 的 candidate model health、`(t-1,t)` model-ready RGB overlap 和 raw RGB capture timestamps。
2. detector 的 signal/config/floors/threshold 必须等价于 frozen Detector-v3 config；它只把整批 prefix
   recomputation 改写为有界 rolling implementation。对同一 health prefix，逐 frame score、attribution 和
   action 必须和 CPU reference bit-for-bit/数值精确一致；不能重新校准或改变 feature、threshold、window。
3. clear frame：正常 commit，并把真实 exported `camera_pose` 追加到 external safe-export history。
4. alarm frame：完整恢复 pre-state；原 candidate health 留在 append-only observed ledger，绝不反馈给 model 或
   detector。若 history 有两个 safe poses `T_{t-2},T_{t-1}`，令
   `Delta = T_{t-1} @ inverse(T_{t-2})`，导出 `T_hat_t = Delta @ T_{t-1}`；连续 alarm 时用上一个
   **exported fallback** 作为 `T_{t-1}`，同一个 `Delta` 每次只推进一步。用 pinned upstream
   `camera_to_pose_encoding(..., "absT_quaR")` 写回 **仅** `camera_pose`。pointmaps、candidate health 和 model
   state 均不被伪造或回填。
5. frame 0 必须 commit；非初始 reset 与连续 rollback 超过 8 frame 都 fail-closed。没有 replay、pending、GT、
   future frame、baseline alarm artifact、synthetic alarm、donor/source-index 或 output-method 搜索。

在现有 30-frame protocol 中 event 前已有 15 个安全 export，故 alarm 发生时一定有两个 safe anchor。没有
second anchor 的任何未来数据、复制 current candidate pose 或静默 commit 都是 error，不是 fallback。

## 2. v3 的独立性能与审计设计

v2 candidate 的 2.9076x runtime 中位数主要来自每个 alarm 都将完整 GPU state 三次内容 hash 到 CPU，以及逐帧
重建整个 health prefix/CPU pointmap residual。v3 不把这些已失败实现视为可以调参的部分，而采用一个新的
预注册实现契约：

- normal path 沿用 v2 已通过的 source-bound lightweight recurrent clone；
- current health 在 GPU 上计算为有限 scalar，only scalar transfer；完整 pointmap 仍按 baseline output contract
  导出，但不为 detector 额外做 CPU median；
- rolling detector 保存至多 frozen window 的 sanitized health rows，并逐帧对 CPU reference differential test；
- actual rollback 记录 structural witness（five local object identities/storage/version、registered mutable attribute
  identities、trace-list lengths、RNG restoration success、pre/committed identity equality），而不是在主 matrix
  中把完整 tensor 内容反复搬到 CPU。Gate A 的 CPU and one small real-GPU restore probe 必须独立证明 pinned
  source rebinds rather than mutates those references; 任一 alias/in-place witness 失败即拒绝 fast path；
- `runtime_seconds` 固定为 recurrent + current detector + structural restore + export fallback；causal RGB overlap
  与 frozen baseline 一样单列为 `wall_runtime_seconds`。两者都报告，不能在文档中混用。

这是新的 engineering hypothesis，不是降低 v2 的门槛：最终 candidate/baseline runtime 上限仍是 `1.20`。

## 3. 不可变边界

- 不修改、重跑、覆盖或作为 v3 response input 读取 `recovery-quality-formal-*`。不下载数据、权重或依赖。
- 所有 forward 只用三个只读 legacy manifests：`formal-v1-inputs-0001/development/development-{dynamic,wrong,low}`。
  这些只支持 development decision，不能生成 formal recovery claim。
- 不改 ReCal3R source/checkpoint、raw RGB、GT、corruption manifest、Detector-v3 config/floors/threshold。
- v3 runner 必须拒绝 recovery-quality commitment、v1 alarm artifact、synthetic `--alarm-frame` 和非 development
  manifest；detector interface 拒绝 GT/depth/label/event/source-index/future fields。
- 单卡 batch 1；每个 GPU forward 前记录 UUID、至少 12 GiB 空闲、existing PID、完整 command、日志和 exit code，
  在 `tmux stateguard:recovery-dev` 中运行；绝不干预他人进程。

## 4. 预注册 Gate A：实现与 CPU/GPU 合约

新增独立 v3 primitives/runner/evaluator，不修改 v2 runner。必需测试：

- SE(3) fallback：identity、translation、rotation、quaternion normalization、连续 export、clear reset、无两个
  anchors/error、candidate pose 不会作为 fallback history；
- CPU reference vs incremental detector：每个 prefix 的 score/attribution/action 相同；future mutation 不影响过去；
  rollback health 只以 sanitized overlap placeholder 存入 history；禁止字段全部拒绝；
- structural snapshot/source guard/RNG/trace witness、rollback full restore、no pending、watchdog/reset fail-closed；
- fast GPU health scalar 与 existing CPU post-hoc implementation 在同一 frozen prediction 上一致；
- normal always four-file byte equivalence；manifest/commitment/output atomic guards；
- full CPU suite、`compileall`、`git diff --check` 通过，代码/测试/文档分离提交，ReCal3R clean。

任一 source alias、fallback causality、reference-equivalence、restore 或 safety test 失败，直接
`SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_IMPLEMENTATION_NO_GO`，不启动 quality matrix。

## 5. 预注册 Gate B：性能先行

先在 development-dynamic 跑一次 v3 light always-commit，与 frozen v1 baseline 四文件逐字节比较；随后以相同
manifest 跑一次 v3 candidate performance probe（其 quality 不读取 GT、不写 evaluator）。两者都必须
`runtime_seconds / baseline <= 1.20`，candidate 不得 watchdog abort，且 probe 有真实 quarantine + causal fallback。
失败直接 `SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_RUNTIME_NO_GO`，不运行 wrong/low candidate，不下载数据。

只有 Gate B 两项通过，才各跑一次 wrong/low always 与 dynamic/wrong/low candidate，所有 output 原子冻结；任何
forward 失败也不能重试同名 output 或修改候选。

## 6. Gate C：冻结 development matrix 与选择

CPU-only evaluator 在所有 six v3 forward 冻结后才读 source-manifest logical-base GT：prefix 0--14 proper Sim(3)，
tail 19--29 ATE/RPE。它验证：

1. 三个 baseline/always 四文件 byte-equivalent、30 finite trajectory/health rows、29 updates、permissions and
   provenance 正确；
2. 每个 rollback 有 current alarm、structural restore witness、observed health、safe-anchor inputs 与 exactly-one
   export action；candidate state/output history zero leakage，首 clear 真 export 被记录；
3. 每条 corruption 质量 effect、runtime ratio 和 fallback-vs-raw output difference；
4. 至少 2/3 同时正 ATE 与 translation-RPE；两项 median `>= +5%`；candidate runtime median `<=1.20`；无
   restore failure、watchdog abort、trace gap 或 policy leakage。

全部通过才为 `SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_CANDIDATE_READY`，才写**未执行**的最小新盲测 acquisition /
commitment plan。任一失败为 `SAFE_ANCHOR_EXPORT_DEVELOPMENT_V3_FEASIBILITY_NO_GO`；不得从 v3 development
数值选择另一个 motion model、delta direction、anchor length、threshold 或 fallback。

## 7. 交付与终点

交付独立 code/test/docs commits、frozen per-condition outputs、CPU evaluation、
`docs/audits/recovery-safe-anchor-export-development-v3-result.md` 和 run registry。v3 NO-GO 也完成本计划；其后
若继续必须提出新的机制假设，不能把同一开发集变成反复调试的确认集。
