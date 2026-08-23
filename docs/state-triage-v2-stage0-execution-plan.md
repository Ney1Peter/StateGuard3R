# StateTriage3R v2 Stage 0：覆盖条件化冲突分诊执行计划

- 日期：2026-08-23
- 状态：**计划已冻结，尚未启动任何 Stage 0 GPU forward。**
- 上位项目：StateGuard3R v2（`StateGuard3R: Cause-Aware Recovery and
  Versioned State Maintenance for Long-Sequence 3D Reconstruction`）
- 本阶段预计投入：10--14 个活跃小时；以最终的冻结 `GO` 或 `NO-GO`
  审计为结束条件，而不是以时间耗尽为结束条件。
- 数据策略：只使用当前本地已有的 ReCal3R TUM 开发数据和已存在权重；**不下载**
  数据、权重或依赖。

## 1. 结论与本阶段唯一目标

StateGuard3R v1 已经严谨地验证了两件不同的事：

1. Detector v3 能检测预注册的 synthetic conflict（正式独立结果的 pooled
   AUROC 为 `0.9933`）。
2. 检测后采用统一的 `hold`、`replay`、`discard` 或 recurrent-state
   quarantine，未改善最终的 ATE/RPE，且通常超过 `1.20x` 运行时预算。

因此，不能再把“检测到异常后少更新一次状态”当作主要研究假设，也不能重跑、
调阈值或修补已冻结的 v1--v20 recovery 实验来获得不同结果。

本阶段的唯一目标是回答一个更基础、也更可证伪的问题：

> 对于同样表现为 observation--state 不一致的事件，**覆盖条件化的结构化证据**
> 是否能比既有的单一 scalar health 更准确地区分：
> `registration_or_order_fault`、`bad_observation`、`transient_local_content`
> 和 `normal_novelty`？

成功不等于已经完成 StateTriage3R 系统，更不等于改善重建质量。成功仅表示：在
一个完全冻结、只读的开发协议上，原因分诊这个前提值得进入 action/branch 的下一
阶段。失败同样是有价值的结论：它会阻止我们继续投入版本图、rollback 或第二个
baseline。

## 2. 新的研究切入点：先判定“这是不是可比较的冲突”

v2 的关键不只是把一个 `health score` 变成四分类器，而是引入下面这个判定顺序：

```text
当前观测
  -> 它是否落在先前状态已经覆盖的区域？
       否：normal novelty，允许普通更新，不叫 conflict
       是：形成 observation--state conflict
             -> 空间形状 + 配准迹象 + 图像质量 + 时间支持
             -> 原因 posterior 或 unresolved
```

这称为 **coverage-conditioned conflict triage**。它解决了旧 detector 的一个根本
歧义：大残差可能是状态被破坏，也可能只是第一次看到一个原来不可见的表面。若不先
做 coverage 判定，任何保守 gate 都会把正常扩图误拒绝为异常，地图也就无法随真实
世界生长。

对于已经确认存在覆盖重叠的冲突，Stage 0 不直接预测一个动作，而是比较竞争性的
证据签名：

| 可操作标签 | 可检验证据签名 | 此阶段允许的解释边界 |
| --- | --- | --- |
| `registration_or_order_fault` | 多个已覆盖静态区域共同出现、方向一致的几何/位姿不连续；局部性低、全局性高 | 当前 ReCal3R 输入协议能直接构造的是顺序/配准代理，**不是**真实 Sim(3) 尺度故障的结论 |
| `bad_observation` | 低清晰度、裁切/低重叠或遮挡等观测质量下降；失配更弥散，后续不在同一空间区域稳定支持 | 表示图像或可见性退化，不能据此声称发现了真实物理变化 |
| `transient_local_content` | 已覆盖区域中局部、短暂的冲突；消失或恢复后缺少持续空间支持 | Stage 0 只验证可控的短暂局部内容/遮挡代理，不把它冒充为带对象 GT 的动态 3D benchmark |
| `normal_novelty` | 先前空间覆盖低，或只在新可见表面产生差异；没有“旧状态应当匹配”的前提 | 这是拒绝所有异常化的安全对照，不是 persistent change |

分类器还必须能输出 `unresolved`。当证据不足时，正确行为是延期而不是强行二选一；
本阶段把这种 abstention 计入延迟和覆盖率，而不把它悄悄算作正确分类。

`persistent_change` 不在 Stage 0 的主指标中。当前已有 TUM 开发材料没有可审计的、
跨时间重访的真实持久变化标签。只有取得新数据、许可和数据协议后，它才能在后续
Stage 1/2 作为独立类别进入；在此之前不得用颜色块、模糊或 walking 序列把它命名为
“真实持久变化”。

## 3. 与旧工作的关系和严格排除项

本计划保留 v1 的 detector/instrumentation 资产，但改变研究问题。

| 已有证据 | 对本计划的约束 |
| --- | --- |
| Formal Detector v3 是 `GO` | 可作为 scalar-health detection baseline；不得重新调参或将其已解盲 formal scene 伪称为新的 confirmatory test |
| recovery-quality v1 是 `NO-GO` | 不再把 external hold/replay 当候选，也不重新读取或修改其 sealed artifact |
| online current-frame quarantine v2 是 `NO-GO` | 不在本阶段恢复、复制或优化该 policy；当前输出和 recurrent state 不做任何修改 |
| v5 pointmap 共识 runtime `NO-GO` | 不做 pointmap consensus 写回或超过 `1.20x` 的线上维护路径 |
| v17--v20 的 OOM/观测性/协议 `NO-GO` | 不复用其 runner、capability 或 state-write 机制；Stage 0 仅作后验观测 |

因此 Stage 0 的硬排除项如下。

- 不修改 ReCal3R upstream，也不对其 model state、memory、token、mask、beta、pose 或
  输出作赋值、拦截或替换。
- 不回滚、重放、暂停写入、quarantine、建版本图或报告 ATE/RPE 改善。
- 不使用 GT、人工标签、未来帧、源序号或完成后的输出作为 **online feature**；GT 仅在
  冻结所有 output 后用于不属于本阶段的审计，不进入任何分诊分数。
- 不读取以前的 detector alarm/response artifact 作为特征或标签。
- 不下载数据、模型、代码或 Python 依赖；不把开发结果称为 blind/formal result。
- 不暂存当前工作区已有的 v14/v15 未跟踪文件，尤其不能使用 `git add .`。

## 4. 完成定义和失败定义

Stage 0 只有两种合法终态。二者都要求输出目录、输入 manifest、协议、代码 commit、
指标和日志均可复核。

### 4.1 `STATE_TRIAGE_V2_STAGE0_GO`

以下条件必须全部满足，才允许进入 Stage 1 的 action simulator：

1. 观察 logger 对 frozen ReCal3R normal-control 的主要预测摘要与无 logger 的
   normal-control 完全一致；logger 只附加写日志，不能改变模型结果。
2. 所有四类各有至少 5 个、来自至少 2 条本地序列的预注册事件；事件时间段和生成
   recipe 在任何特征/结果被读取前冻结。
3. typed、因果（只用 `<= t`）规则相对最佳 scalar baseline 的 event-level Macro-F1
   提高至少 `0.10`，并且按 `sequence_id` 分层的 10,000 次 bootstrap 95% CI 下界大于
   `0`。
4. `normal_novelty` 的 recall 不低于 scalar baseline，且 typed 方法不会把超过 `10%`
   的 normal-novelty events 错判为三种需干预冲突之一。
5. 对三种 conflict 类，typed 方法的平均 false-action proxy（把 global fault 当局部
   transient、把 bad observation 当 normal novelty、把 transient 当 global fault）低于
   scalar baseline；任何一项上升超过 `5` 个百分点即失败。
6. 推理外的观测/特征开销中位数不超过 vanilla ReCal3R 每帧 wall time 的 `0.20x`，峰值
   GPU memory 增量不超过 `512 MiB`。若提取器只能离线读取全部结果，不能宣称 online
   triage，即使离线指标很好也只能报 `NO-GO`。
7. `unresolved` rate、每类 precision/recall、混淆矩阵、NLL/ECE、event detection delay
   和所有 resource metrics 完整产出；不能只挑选有利指标。

“最佳 scalar baseline”是仅用一维数值并采用同样的 pre-registered event trigger 的
baseline，候选包括 frozen Detector-v3 score、native scalar health 或当前帧 robust residual。
在 protocol 冻结前，以开发标签选择其中**唯一**一个；冻结后不得按评测结果改选。

### 4.2 `STATE_TRIAGE_V2_STAGE0_NO_GO`

任何下列事件立即终止本阶段并写入 NO-GO 审计；不以换阈值、换特征、加数据或 GPU
重跑来挽救：

- 无法从官方 forward 的允许输出稳定取得覆盖、几何、位姿/质量和时间证据，或 logger
  破坏 normal-control 等价性；
- 不能在本地数据上构造每类 5 个、标签来源清楚且事件互不泄漏的样本；
- typed evidence 不满足上节任一效果、错误代价或资源门槛；
- 提取过程被证明使用了未来/GT/旧 response/源 index 的泄漏信息；
- `registration_or_order_fault` 与 `bad_observation` 或 `transient_local_content` 不可分，
  且 `unresolved` 不能诚实地将这种不确定性显式隔离；
- 四类只有人为图案差异而没有任何输出层 evidence 支撑，说明 benchmark 只测到了输入
  recipe 指纹。

NO-GO 的推荐 fallback 是一篇更小而诚实的诊断工作：`coverage-conditioned conflict
detection and calibrated abstention`；它不再承诺 typed action、persistent-change version
或 recoverability。

## 5. 数据、标签和切分协议

### 5.1 可用数据的边界

本阶段只清点并读取已有的、属于 ReCal3R 的 TUM 序列：

```text
rgbd_dataset_freiburg1_desk
rgbd_dataset_freiburg2_desk
rgbd_dataset_freiburg3_walking_static
rgbd_dataset_freiburg3_walking_xyz
```

它们都已经在旧研究中出现或解盲，所以 Stage 0 的性质只能是 **development**。它的
用途是淘汰错误研究主线，不是产生新的 generalization claim。以后若 Stage 0 为 GO，
才单独检查许可、磁盘、已有副本与未见数据，再起草新的 formal protocol；这一步不在
本计划内。

### 5.2 事件标签只来自冻结的生成规则

每个 event 标签必须来自 input capsule 中在 forward 前已写死的条件，而不是从模型是否
出错、GT 误差或人工观看 prediction 后反推。生成器可以读原始 RGB 的明确路径，但
不得修改原始数据。每个 capsule 要记录：

```json
{
  "protocol_version": "state-triage-v2-stage0",
  "source_sequence": "...",
  "event_id": "...",
  "label": "bad_observation",
  "start_frame": 0,
  "end_frame": 0,
  "recipe_id": "...",
  "raw_rgb_path_hashes": ["..."],
  "allowed_online_evidence_end": 0,
  "coverage_expectation": "covered | novel",
  "label_provenance": "pre-forward deterministic recipe"
}
```

具体 recipe 在 capability audit 后、首个 response 前冻结。允许的类别/recipe 对应为：

| 标签 | 合法的 pre-forward recipe | 不允许的偷换 |
| --- | --- | --- |
| `registration_or_order_fault` | 已审计的时序顺序/相邻帧配准代理；其 input timestamps、原始路径和顺序都入 manifest | 称它为“真实相机位姿 GT 错误”或用 GT pose 制作标签 |
| `bad_observation` | 局部遮挡、低重叠、模糊或裁切等单帧/短帧观测退化；精确参数固化 | 看见模型失败后移动事件位置、改变强度或挑参数 |
| `transient_local_content` | 在同一已覆盖区域内连续短窗加入确定性、局部且会撤去的内容代理；开始/结束帧固定 | 把来自未标注 walking RGB 的人直接当作有物理真值的动态对象 |
| `normal_novelty` | 原始 clean RGB 中先前 state coverage 不足的首次可见区域；候选帧只用到 `<=t` 的 coverage 判定 | 将有高 coverage 的 clean residual 直接重标签为 novelty |

若某一 recipe 只能在单一序列中获得有效事件，必须停止为 `INPUT_COVERAGE_NO_GO`，而不是
降低“跨至少两条序列”的要求。事件之间不得共享完全相同的帧窗；同一原始帧的多个
变体必须归入同一 `source_group`，避免 train/test 泄漏。

### 5.3 两层切分，避免“开发调参伪装为评测”

本计划不用训练大模型，仍需严格区分 schema 开发和冻结评测：

1. **schema dry-run：**仅用 `fr1_desk` 选择 feature 是否可导出、数值方向和缺失值规则。
   不报告该部分的分诊结果，不可用它反复调分类阈值。
2. **冻结 development evaluation：**`fr2_desk` 与两条 `fr3_walking_*` 分层组成评测集；
   至少一条完整 source sequence 必须只用于 evaluation。类别 recipe、事件窗、阈值、
   baseline 选择和 metrics script 先提交/封存，再读取该集合的 response。

若现有短序列长度导致无法形成这一切分，结论必须是 protocol 不可执行，而不是降低到
“每帧随机切分”。后者会把高度相关的同一视频帧泄漏给两边。

## 6. Stage 0 的最小技术设计

### 6.1 不侵入式 evidence ledger

新增实现只能位于 `StateGuard3R`，以一个 **post-output observer** 运行：官方 ReCal3R
forward 正常结束后，observer 从允许的 prediction/export 中读取数值并写一行 ledger。
它不得注册 hook、改变 tensor、修改 RNG、持有可写 model reference，或重用 v14--v20
的 recovery runner。

每行 ledger 至少含如下字段，且每个字段标记 `online_available_at`：

| evidence family | 计划字段 | 可用时刻与用途 |
| --- | --- | --- |
| provenance | sequence/event/frame、raw input digest、recipe digest | `t`；追溯与防混淆，不喂给分类器 |
| coverage | previous projected support count、covered-area ratio、novel-area ratio | `t`；先区分 conflict 与 novelty |
| registration | relative-pose jump、已覆盖区域的统一变换残差、residual globality | `t`；识别 global/order proxy |
| observation quality | blur/sharpness、luminance/saturation、overlap/visibility、confidence dispersion | `t`；识别坏观测 |
| spatial structure | robust residual、affected-area fraction、connected-component size、locality/globality | `t`；区分局部与整体冲突 |
| temporal support | 到当前为止的同一区域连续出现次数、连续 clear 长度、age | `t`；识别短暂性；严禁读 `t+1` 以后 |
| decision | scalar score、typed posterior、abstain reason、latency/memory | `t`；审计决策 |

所谓“统一变换残差”只是一项登记候选：capability audit 必须证实能从 frozen outputs
计算；否则该字段以 `unavailable` 明确记录，且计划降级而不是填充伪数值。

### 6.2 先规则、后学习；本阶段不训练

Stage 0 使用两个不可学习的对照和一个 pre-registered typed rule：

- **S0 / scalar health：**一维 frozen Detector-v3 score 或冻结前选定的 native scalar。
- **S1 / scalar residual：**一维 robust covered-region residual；用于排除“只是换了更好残差”的
  假改进。
- **T0 / coverage-conditioned typed rule：**先看 coverage；若 coverage 低则
  `normal_novelty`。若 coverage 高，再以 globality、registration、quality、locality 和
  historical persistence 的固定词典规则给出三个类别或 `unresolved`。

T0 的常数只能在 `fr1_desk` schema dry-run 后一次性写进 protocol。例如阈值可表示为
dry-run clean prefix 的 robust median/MAD 倍数；它们不可由评测 events、GT 或 label
accuracy 选择。所有 rule branch 都要在单元测试中覆盖，尤其是 missing/nonfinite evidence
必须返回 `unresolved`，而不是任意冲突类。

轻量 MLP、时序卷积、learned action policy 和 prompt/model 微调都明确留给 Stage 1 以后。
否则少量合成事件的高分无法说明 typed evidence 本身有价值。

### 6.3 “未来证据”只作离线诊断上界

短暂内容与持久变化在发生时可能不可辨。为诚实地呈现这一事实，Stage 0 可额外生成
`T0+K` **offline diagnostic**：它最多读取事件结束后固定的 `K` 帧来计算 persistence
support。该结果必须单列为 oracle/upper-bound，不得与 causal T0 混合，也不得用它宣称
real-time performance。`K`、内存单位和确认延迟必须在 protocol 中冻结。

此上界若也无法改善 transient 判别，是未来 versioned branch 不值得开发的早期反证；若
它改善而 T0 不改善，则结果支持“需要 provisional buffer”，但仍不等于已经实现它。

## 7. 执行顺序与门禁

下列步骤必须严格顺序执行。任何 Gate 的 `NO-GO` 都停止后续 GPU/代码扩张，并生成审计
文档；不得绕过为“再试一个版本”。

### Gate A：资源、接口和因果边界审计（CPU，约 1--2 小时）

1. 记录 `StateGuard3R`、`ReCal3R` commit、checkpoint digest、现有 TUM 路径/文件数、
   可用空间和 Python environment；不下载任何内容。
2. 阅读官方 ReCal3R inference 与当前 exporter，列出每个 candidate field 是原始输出、
   observer 派生值还是不可得。
3. 在 CUDA hidden 模式下，以 fake/frozen output unit tests 验证 observer 不需要可写 model
   或 recovery imports；`compileall`、targeted pytest、`git diff --check` 必须通过。
4. 写出 capabilities audit，包括不支持字段的明确 fallback。

通过条件：能得到至少 coverage、一个 geometry/pose proxy、一个 image-quality proxy 和
一个 temporal counter，且所有 feature 的 online boundary 有据可查。否则写
`STATE_TRIAGE_V2_STAGE0_OBSERVABILITY_NO_GO`，不启动 GPU。

### Gate B：输入 capsule 与预注册冻结（CPU，约 1--2 小时）

1. 只读生成候选 event inventory，并验证每个 source path/hash、事件窗、类间数量、
   source-group 去重和 `>=2 sequence/class` 规则。
2. 在未读取 evaluation prediction 前一次性固定 recipe、feature schema、S0/S1/T0
   定义、T0 常数、`T0+K`、统计脚本、资源预算、GO/NO-GO 门槛。
3. 建立 mode-`0444` 的 capsule/manifest 和不可变 commit；hash inventory 写进 protocol。
4. 进行 schema dry-run 的 input validation，但不在 evaluation set 上打印或保存可用于人工
   调阈值的 metrics。

通过条件：标签完全由 pre-forward recipe 推出，no-GT/no-future 检查通过，且事件数/切分
满足第 5 节。否则写 `STATE_TRIAGE_V2_STAGE0_INPUT_NO_GO`，不执行任何候选 forward。

### Gate C：vanilla 与 observer 等价性（最小 GPU，约 1 小时）

每条源序列先做一个 clean normal-control：

```text
official vanilla forward
  -> official forward + observer ledger
  -> CPU comparator
```

比较 checkpoint load audit、canonical prediction summary、pose/trajectory summary、帧数、
输入 digest 和原生 runtime。observer 运行不参与 ReCal3R forward 的关键路径；若其绝对
后处理开销无法从模型 runtime 中剥离，必须同时报告 wall time。

通过条件：所有预注册 normal-control 相等，且 observer 对应行的 source/feature timestamp
均不晚于当前帧。任一不等或漏行都是 `STATE_TRIAGE_V2_STAGE0_OBSERVER_NO_GO`。

### Gate D：冻结 development matrix（单 GPU、小批量，约 4--6 小时）

只有 Gate C 通过才按固定顺序执行。每个 capsule 只允许一次有效 forward；失败首先检查
进程/路径/权限等基础设施，只有确认 **未产生任何 response** 时可重建一个有新 ID 的
同输入 attempt，原失败日志保留。任何已产生 response 的 event 不可改变 recipe 或重跑。

每个 GPU 任务必须：

- 先运行 `nvidia-smi`，选择有足够空闲显存且没有被本项目其他任务占用的单张 GPU；
  `CUDA_VISIBLE_DEVICES=<id>` 明确绑定；不要求整卡空闲，也不影响他人进程。
- 通过 `tmux` 的命名窗口运行，记录启动命令、PID、GPU、日志绝对路径、输出目录和启动
  时间；输出只写入 `StateGuard3R/outputs/`。
- 完成后确认本任务 PID 已退出、显存已释放，并冻结输出目录。不得启动网页服务或占用端口。

矩阵结束后才允许 evaluator 读取完整 event ledger。评测按 event（而非高度相关的 frame）
汇总，并同时输出 S0、S1、T0 和 T0+K 的完整混淆矩阵。

### Gate E：单一评估、审计与决策（CPU，约 1--2 小时）

1. 用冻结 evaluator 计算第 4 节所有 metrics；bootstrap 的重采样单位是
   `source_sequence`，并保留每次 seed 和完整原始 event 表。
2. 交叉检查 feature-time boundary、manifest hash、每类样本数、no-GT imports、control
   parity、输出权限和 GPU/resource ledger。
3. 只写一个最终 `result.json`、一个 attempt seal 和一份 Markdown audit；二者 0444，目录
   0555。审计准确指出 development 限制、proxy 限制和 persistent-change 尚未验证。
4. 仅以第 4 节判据决定 `GO` 或 `NO-GO`，不读取结果后新增阈值、类别、sample 或模型。

## 8. 指标和报告格式

所有结果必须按类别与按事件报告，不能用一个总体 accuracy 隐藏风险类别。最少报告：

| 组别 | 指标 | 解释 |
| --- | --- | --- |
| conflict vs novelty | AUROC/AUPRC、event F1、false-conflict rate | 测试 coverage 条件化是否有用 |
| cause triage | Macro-F1、每类 P/R/F1、混淆矩阵、NLL、ECE | 主问题；`unresolved` 作为可见输出 |
| unsafe confusion | 三项 false-action proxy、normal novelty false reject | 避免只奖励保守分类 |
| timing | event detection delay、unresolved duration、T0+K confirmation delay | 将等待成本显式化 |
| evidence ablation | S0、S1、T0、T0 去 coverage、T0 去 temporal、T0+K | 验证每一类证据而非堆砌特征 |
| resources | observer wall overhead、GPU peak delta、CPU RAM、ledger bytes/frame | 防止诊断层本身不实用 |

`unresolved` 的处理规则也必须固定：它不计为正确原因分类；但当其替代一个错的强制
action 时，在 false-action proxy 中单独报告为 abstention，而非误称改善。coverage 判定错
误（把已覆盖区域当新区域，或反之）需有独立 confusion table，因为它是本方法的新关键
环节。

## 9. 若 Stage 0 为 GO，后续阶段的精确边界

GO 只授权下一份新计划的准备，不自动授权新下载或大规模实现。后续必须按以下顺序另行
预注册：

1. **Stage 1：离线 action simulator。**固定 `realign / reject / isolate / commit-new-version`
   的 action table，在 ledger 上评价 false commit、false reject 与 action regret；仍不改
   ReCal3R recurrent state。
2. **Stage 2：有限 provisional branch。**只有 T0+K 证明等待有价值时，做固定容量候选
   分支与 delayed commit；同时测延迟、内存和 clean no-harm。
3. **Stage 3：版本与恢复。**先 checkpoint + frame-span replay，再讨论显式 map 的
   region delta；不承诺 token-level undo。
4. **第二数据门槛。**只有先确认许可、空间、已有副本与真值协议后，获取一个真正的
   persistent-change/revisit 数据源。它必须和本阶段 development 序列隔离。
5. **第二 adapter。**只有 action 与版本机制在第一种状态表示上成立，才考虑 explicit-map
   adapter；“有第二个 repo”本身不是启动条件。

如果 Stage 0 为 NO-GO，以上四步全部停止。可以改写论文问题为 calibrated abstention/
benchmark，但不得把 NO-GO 解释为“再做一个 state intervention 就会成功”。

## 10. 预期代码与文档交付物

本计划本身不新增实验代码。实际启动后的最小提交应严格分开：

1. `docs:` Gate A interface/capability audit；
2. `feat:` read-only observer、capsule builder 与 unit tests（不含运行结果）；
3. `docs:` frozen Stage 0 protocol/input manifest；
4. `docs:` final GO/NO-GO audit。

每一步只暂存明确文件；任何已有 v14/v15 未跟踪文件均不属于这些提交。每次代码改动前后
运行与风险相称的 targeted pytest、`python -m compileall -q scripts tests` 和
`git diff --check`，并把实际命令与结果记录在 audit。

## 11. 本计划的科学主张边界

若完成且 GO，允许的最强表述是：

> 在本地、已解盲 TUM development protocol 的可控 conflict proxies 上，coverage-conditioned
> typed evidence 比单一 scalar health 更能区分特定的 order/registration proxy、观测退化、
> 短暂局部内容和正常新区域。

不允许的表述包括：

- “已解决长期 3D reconstruction recovery”；
- “已经提升 ATE/RPE”或“优于 ReCal3R”；
- “已经识别真实 persistent change”；
- “已经实现 online state repair/rollback/versioning”；
- “已在未见真实数据上泛化”。

这条边界正是该计划的价值：先以低资源、可复核的实验测试 v2 的必要前提；只有前提
成立，才值得让原因匹配 action 和 versioned spatial state 成为后续工程与论文主线。
