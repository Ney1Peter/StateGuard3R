# StateGuard3R Detector v3：时序顺序检测与可验证隔离执行计划

- 制定并激活日期：2026-08-03
- 状态：**执行中**
- 代码起点：StateGuard3R `2b0c28c`
- 受保护 baseline：ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- v2 正式结论：cross-scene Detector v2 **GO**，但 `wrong_order_segment` 在唯一盲测中漏检
- 长时目标：完成一个可审计的 Detector v3 正式研究周期；不以未验证的开发结果或 ledger
  proxy 作为终点。预期持续 12--24 小时，若 development 门禁未通过则继续在已解盲数据上
  诊断和版本化迭代，直到达到 formal-ready 或得到可复现的开发不可行结论。

## 1. 明确目标与完成定义

### 1.1 主目标

在不改变 ReCal3R tracked source、checkpoint、既有 v1/v2 正式证据或在线模型输入的前提下，
实现 **causal temporal-order consistency signal**，使 Detector v3 能区分“正常平滑运动”与
“短段反序/错误顺序”这种仍有大量视觉匹配、但运动方向与近期历史不一致的输入；随后用一条
从未产生 ReCal3R response 的小型独立场景执行一次冻结的盲测。

### 1.2 安全目标

在 v3 detection gate 通过后，完成一个单独预注册的 **state-aware quarantine feasibility
experiment**：它必须在 StateGuard3R wrapper 中真实决定并记录 ReCal3R 的 state/memory
commit，能证明“哪个更新被阻止、保留或重放”，并以 alternate forward 的 prediction、trajectory
和 state digest 验证政策确实改变了执行状态。它不同于 v2 的 ledger-level proxy，且绝不把
“状态改变”自动表述为“几何恢复成功”。

### 1.3 可验收的成功条件

正式 **V3_GO_AND_STATE_POLICY_VALIDATED** 需要同时满足：

1. v1/v2 所有 immutable manifests 的 byte hash、权限和内容保持不变；ReCal3R tracked
   source 仍位于原提交且 clean。
2. v3 order signal 只读取当前及此前的 model-ready RGB / 当前模型运行时状态；不读取 GT
   pose/depth、污染标签、事件区间、future frame 或既有 formal response。
3. 公开 development pool 的全量 macro-AUROC `>= 0.85`、leave-one-corruption-type-out
   macro-AUROC `>= 0.75`、pooled primary FPR `<= 0.15`、每 run FP streak `<= 2`、三类
   corruption 都在事件内检出且平均 delay `<= 1`。同时，wrong-order 留出折必须检出。
4. 新场景的唯一 formal blind evaluation 保持 combined macro-AUROC `> 0.75`、pooled FPR
   `<= 0.20`、clean-prefix FPR `<= 0.15`、FP streak `<= 3`，并且三类（dynamic、wrong-order、
   low-overlap）均在事件内检出，wrong-order delay `<= 1`。
5. state-aware quarantine 的基线和干预 forward 均成功、输入和 checkpoint 相同、干预仅由已
   冻结的 detector score 触发；state digest/commit timeline 表明至少一个报警更新被实际阻止或
   延迟提交，且未报警 clean control 不触发。输出/轨迹/health provenance 完整且可重放。
6. 若要写“恢复/持久误差下降”结论，必须另有预注册的输出质量指标、独立评价数据、统计门槛和
   与 baseline 的比较：corruption 后配对 ATE/RPE 或预定义 revisit-consistency 至少改善 10%，
   clean degradation 不超过 5%，端到端开销不超过 20%，并且 state restore failure 为零；否则
   结论固定为 **STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM**。

科学指标不能被保证。若在严格固定的 blind evaluation 出现 NO-GO，或 state wrapper 的
运行时不允许在不改 baseline source 的条件下进行可信干预，则发布完整 NO-GO/feasibility
结论是本计划的有效终点；不得在同一 blind response 上调参、改阈值或重跑。

## 2. 已知证据与核心假设

v2 的正式 combined macro-AUROC 为 `0.9126984127`，FPR 为 `0.03278688525`，但
`holdout-wrong` 的事件位置 15--18 没有 crossing：事件内分数为约
`[1.132, 0.279, 0.794, 0.222]`，正式阈值为 `2.0686`；位置 27--28 才有两个假阳性。
这与“coverage 证明可匹配性，但本身不表达运动方向”一致。故 v3 的唯一 detector 变化是
增加一个有符号、因果、图像对应关系派生的 **motion-reversal/order-consistency** 候选，而非
降低 v2 阈值、读取标签或重用 v2 holdout。

原 v2 新场景、v1 场景和所有 formal outputs 都已解盲；它们只可作 development/fixture，
不能成为 v3 confirmatory holdout。

## 3. 不可变边界

- 不修改、移动、删除、重新冻结或重新评价 `formal-v1-*`、`formal-v2-*`、
  `quarantine-v2-*` 的任何内容；新结论必须引用而非覆盖它们。
- 不修改 ReCal3R tracked source、checkpoint、官方原始 RGB-D、GT、划分或标注。
- 不训练/微调模型，不下载新权重；development 先复用现有少量输入。只有 v3 readiness PASS
  才允许下载**一个**预审计的小型官方 TUM RGB-D+GT archive（archive + 解压总预算 5 GiB）。
- 不以 GT、depth、label、corruption type、event mask、未来帧或 v2/v3 response 构造 online
  detector 特征；GT 仅可在输入构造和事后评价中使用，并以独立字段/模块隔离。
- 不以 monkey-patch 方式写回 baseline 文件；可在 StateGuard3R 中实现明确标注为 experimental
  wrapper 的独立执行函数，并逐字节记录其抽取自的 baseline source 和行范围。
- GPU 只使用启动时检查后有至少 12 GiB free 的一张卡，batch=1；不影响他人进程。

## 4. 技术方案

### 4.1 Causal order-consistency signal

对 frame `t`，复用 v2 的 ORB、mutual-ratio match、RANSAC inlier 集合。仅对已验证的 inlier
端点计算二维位移向量 `d_t = median(p_current - p_previous)`；若 inlier 数不足、鲁棒中位位移
小于固定 noise floor，结果为明确 neutral/invalid，不能虚构高风险。把 `d_t` 与严格过去的
`d_{t-1}`（或至多 window 内的 causal direction reference）比较：当两者充分反向且 coverage
仍非退化时，生成 `order_reversal` 风险。信号必须保留 support、direction cosine、magnitude、
status、reference IDs 和配置 provenance；health ledger 只存最终有限 signal 或 null，详细
视觉诊断存 run metadata。

候选限制（全部保留）：

| 参数 | 候选 |
| --- | --- |
| 方向 reference | previous-valid displacement；过去 3 个有效 displacement 的 unit-median |
| 最小 inlier support | 12；24 |
| 位移 noise floor（归一化图像对角线） | 0.002；0.005 |
| 反向 cosine 门槛 | `<= -0.50`；`<= -0.25` |
| coverage gate | `>= 0.02`（固定，避免把低 overlap 当成顺序证据） |

该候选网格最多 16 组；它和原 v2 的四个 visual/scoring 候选组合时，必须先进行合成与已解盲
development 筛选，并完整保存失败组合。combined v3 采用现有五个 v2 component 加一个
`order_consistency` high-risk component 的 finite mean；每一种方法的 scale floor 仍只从
development clean prefix 派生。若实际 valid order signal 在 development 无法达到足够 coverage，
先报告 feasibility failure，不悄悄用 GT pose 代替。

### 4.2 State-aware quarantine wrapper

v2 proxy 在已有 ledger 上事后“扣除 update magnitude”，没有 alternate model forward，所以不是
recovery 证据。v3 只在 detector formal GO 后实现以下最小验证：

1. 在 `StateGuard3R/scripts` 创建独立、行级 source-hash-bound 的 recurrent wrapper；它不能复用
   `forward_recurrent_lighter` 的返回 state（该函数不保留可恢复的 state args）。每个 transaction
   必须闭包保存 `state_feat`、`state_pos`、`init_state_feat`、`mem`、`init_mem`、
   `update_pressure`、`recal3r_state0`、`_recal3r_sequence_age`、trace/pending-queue 相关属性、
   必要 RNG/config 状态及无 alias 的 tensor digest；绝不修改 ReCal3R 文件。
2. detector 只使用当前/过去已经可获得的 signals。对依赖当前 prediction 或 trace 的 score，政策
   在下一 update 生效；不得反向声称阻止了已经发生的同帧更新。
3. 检测到风险后，按冻结 policy 将之后至多 `B=3` 个 state/memory commit 置为 hold；buffer
   release/replay 的条件只读取后续 causal scores。必须记录 proposal、commit、hold、release、
   replay、state digest 和 reason。
4. 先在 synthetic state machine 与 8-frame真实输入上测试 base/guarded 行为；必须有一个
   `update=False` 反例证明其仍会改变 pressure/sequence age，再证明 transaction rollback 会让
   完整状态闭包逐 tensor 恢复。再运行相同30-frame输入的 baseline/wrapper pair，正常（always
   commit）路径须与 pinned lighter runner byte-equivalent。比较 input hash、pre-alarm prediction、
   对应 state digest、trajectory、health、update count、runtime、peak memory；不修改输入或 baseline output。
5. 任何 quality claim 需独立 protocol，预注册改善/退化/开销/零恢复失败门槛；若只证实状态
   路径变化，结论固定为 `STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM`。

## 5. 分阶段执行与门禁

### Phase 0 — 激活、保护与基线审计（CPU）

1. 重读服务器规则、v2 result、formal/quarantine protocols；记录 StateGuard3R/ReCal3R
   HEAD、worktree、磁盘、GPU、PID、端口与受保护 manifest hashes。
2. 创建本计划并单独提交，之后才改代码。
3. 建立 `outputs/detector-v3-analysis-0001/`，只读导出 wrong-order score/component timeline，
   证明 v2 漏检的具体位置和信号缺口。

门禁：v2 evidence byte-identical、两仓库 clean（仅本计划文件除外）、不启动 CUDA。

### Phase 1 — order-signal 原型与 CPU 测试

1. 在 `visual_overlap.py` 添加可注入的 correspondence displacement/order helper，不改变 v2
   `VisualOverlapResult` 语义或正式 v2 路径；必要的新 dataclass/version 使用 v3 名称。
2. 在新 `detection_v3.py` 中实现 six-signal causal scoring、严格 schema、scale floor、warm-up、
   finite mean、候选配置和 constrained threshold selection。v1/v2 模块保持字节不变。
3. 测试 synthetic forward motion、reversal、rotation/near-zero motion、遮挡、低纹理、descriptor
   failure、coverage failure、ties、NaN、short history、future-invariance、determinism 与 no-GT API。
4. 对每个 v2 frozen input 运行两次 visual replay，写入一个新的 v3 derived-development output；
   检查 frame `t` 在改变 `t+1...` 后相同。

门禁：原完整 CPU suite 继续通过；新增测试全绿；CUDA 不初始化；候选回放 deterministic，
order signal 没有 future/GT/label provenance。

### Phase 2 — disclosed development search 与根因闭环（CPU）

1. 从 v1/v2 已解盲 ledgers/inputs 构建 v3 development pool；必须明确每个正式旧 run 已解盲。
2. 运行预先枚举的 visual/order/score search，保留每个候选、scale floor、threshold、AUROC、FPR、
   event delay 与 clean-prefix timeline。
3. 使用 leave-one-corruption-type-out，且对 wrong-order 留出折单独报告。不能只以 full-pool
   指标选择 config。
4. 只在失败时作一次有记录的单因素改动，重新生成新编号 output；不访问新 scene。

门禁：满足 §1.3 第 3 项的全部 development 条件，并能指出 v2 wrong-order miss 是被哪个
v3 component 在何时纠正；否则返回本 phase。

### Phase 3 — instrumentation equivalence 和 wrapper feasibility（最小 GPU）

1. GPU preflight 记录 `nvidia-smi`、PID owner/command、GPU UUID、空闲显存和启动命令；选择
   free >=12 GiB 的单卡并显式 `CUDA_VISIBLE_DEVICES`。
2. 相同已有 8--30 frame input 运行 v2 与 v3 detector-instrumented forward；证明模型输入 hash、
   checkpoint audit、prediction summary、trajectory、recurrent update count 与旧 health fields
   完全一致，差异只限 v3 detector metadata/field。
3. 在 8-frame input 上运行 baseline wrapper 与 policy-disabled wrapper，要求 byte-equivalent
   prediction/trajectory/state traces；再在 synthetic forced-alarm 中验证 hold/release/replay 的真实
   state digest 变化。失败时只修 StateGuard3R wrapper，绝不改 ReCal3R。
4. 每次结束核对 PID 已退出、显存释放、无监听端口。

门禁：instrumentation equivalence 与 wrapper semantic tests 全部通过；否则返回 Phase 1 或 3。

### Phase 4 — formal-v3 readiness、唯一新数据与预注册

1. 仅在 Phase 2--3 全 PASS 后，先审计 archive URL、许可、Content-Length、预期解压大小、磁盘
   余量和现有本地副本；最多下载一个新的官方 TUM RGB-D+GT scene。
2. raw archive 以 `.part`、hash、原子发布和 read-only manifest 冻结；在 protocol/commitment
   冻结前禁止运行 ReCal3R、视觉 detector 或读取新场景 response。
3. 输入 builder 可用 timestamps/GT-depth 只为构造并审计三类 corruption；online ledger 禁止这些
   字段。冻结 scene、input split、source hash、candidate order、selected v3 config/floors/threshold、
   GPU规则、all-three-before-read lock 和唯一 evaluation path。
4. 先运行六条 disclosed development forwards/calibration；完成并冻结后，再按 dynamic →
   wrong-order → low-overlap 顺序运行三条 blind forwards。三条都退出并冻结前，launcher 不得读取、
   hash、解析单条 response/main log。

门禁：CPU commitment validator 通过；commitment 后直到唯一 evaluation 结束不得修改 tracked 文件。

### Phase 5 — 唯一正式评价与条件式 state-policy evaluation

1. evaluate 恰好一次，发布只读 manifest、指标、timeline、source/config hash 和 v3 GO/NO-GO。
2. 若 detection NO-GO：发布失败项和分析，禁止对该新 holdout 调参以及禁止 state-policy quality
   experiment；完成资源收尾。
3. 若 detection GO：单独预注册 quarantine/state-policy protocol、commitment、policy、input pair
   和评价输出路径。新 blind data 已因 detector evaluation 解盲，因此此 state-policy 结果默认
   exploratory，除非另有未看过的独立场景并获 protocol 允许。
4. 用同一冻结输入进行 baseline vs guarded alternate forwards；检查 §1.3 第 5 项。仅当另行的
   quality protocol and threshold 全部通过，才使用 “recovery-quality” 语言。

### Phase 6 — 审计、报告、提交和收尾

1. 更新计划状态、formal-v3 result、state-policy result、run registry，明确 v2 的 wrong-order
   limitation是否已被独立证据消除。
2. code、tests、protocol、result docs 分开提交；formal commitment 后不提交任何改动，至 evaluate
   结束才继续文档提交。outputs/log/data/weights 不入 Git。
3. 重新运行 full CPU suite、compileall、`git diff --check`、evidence checker；复查两仓库
   worktree、GPU/port/PID、权限、磁盘。仅清理由本任务新建且已列明的临时 fixture，绝不删除正式证据。

## 6. 正式评价门槛

所有门槛为合取：

| 项目 | 门槛 |
| --- | --- |
| combined macro-AUROC | `> 0.75` |
| combined - seeded random macro-AUROC | `>= 0.10` |
| combined vs best real single signal | 不低于 `0.02` |
| detected corruption types | 3/3，含 wrong-order |
| wrong-order delay | `<= 1` frame |
| pooled primary FPR | `<= 0.20` |
| clean-prefix FPR / startup streak | `<= 0.15` / `<= 2` |
| max primary FP streak | `<= 3` |
| signal integrity | finite、causal、v3 signal used、无 GT/label/future leakage |
| runtime integrity | deterministic replay、input/model equivalence、provenance、GPU cleanup 全通过 |

任何一个失败即唯一 formal conclusion 为 `DETECTOR_V3_NO_GO`。这不是可以通过降低阈值或重跑
同一 blind scene 修正的开发失败。

## 7. 目录、资源与命名

所有新运行产物保存在 `StateGuard3R/outputs`/`logs`，按递增且从未使用的 ID 创建，例如：

```text
outputs/detector-v3-analysis-0001/
outputs/detector-v3-derived-development-0001/
outputs/detector-v3-development-search-0001/
outputs/detector-v3-instrumentation-0001/
outputs/state-policy-v3-feasibility-0001/
outputs/formal-v3-data-0001/
outputs/formal-v3-inputs-0001/
outputs/formal-v3-commit-0001/
outputs/formal-v3-runs-0001/
outputs/formal-v3-calibration-0001/
outputs/formal-v3-evaluation-0001/
outputs/state-policy-v3-commit-0001/
outputs/state-policy-v3-evaluation-0001/
logs/detector-v3-*.log
```

正式发布目录为 `0555`、文件为 `0444`；临时 development 目录不覆盖旧 ID。总预算：新 raw
archive/tree <=5 GiB，新 JSON/timeline <=150 MiB，测试 fixture <=2 GiB。当前 `/data` 空闲空间
约 248 GiB，下载前必须重新检查，且不删除任何现有数据来腾空间。

## 8. 停止条件与用户介入条件

本计划在以下任一情形才停止：完整的 V3_GO_AND_STATE_POLICY_VALIDATED；合规的唯一 formal
NO-GO；或可复现地证明 wrapper 无法在不修改 protected baseline 的条件下提供真实 state policy
并发布 feasibility NO-GO。development 指标不佳、一次脚本异常、GPU暂时被占用或网络重试均不是
停止条件。

只有以下情况需要用户决定：唯一候选 archive 因下载/许可/格式不合格而需第二个 archive；所有
可用 GPU 持续低于 12 GiB free；或要把 state-policy validity 扩大为有独立统计效力的
recovery-quality claim（需要新的评价范围）。
