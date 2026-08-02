# StateGuard3R Detector v2 下一阶段执行计划

- 制定日期：2026-08-02
- 状态：**已规划，尚未执行**
- 激活条件：用户明确要求“开始执行”后，才创建新的长时目标并执行本计划
- 证据基线：StateGuard3R `adb27fefdd0f85ebbba58e3b0062e3bfcd5be1f4`
- 正式运行代码基线：StateGuard3R `e9379cc14c5a629d8c316002bc212fab046f696a`
- ReCal3R 固定基线：`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- v1 结论：within-sequence detection pilot **NO-GO**，未进入 quarantine/rollback
- 预计执行时间：18–36 小时；development 迭代不满足门槛时继续，不因单次失败提前结束

本文档只制定下一阶段的目标、技术路线、执行顺序、门禁和命令模板。创建本文档不代表
Detector v2、数据下载、GPU forward、formal holdout、quarantine 或 rollback 已经开始。

## 1. 当前证据与问题定位

### 1.1 v1 已经证明的内容

1. 固定 ReCal3R checkpoint 可以在单张 GPU 上稳定执行 30 帧序列，29 次 recurrent
   update、Health Ledger、trajectory、prediction summary、runtime 和显存记录均可复现。
2. `dynamic_occlusion`、`wrong_order_segment`、`low_overlap_jump` 三类受控污染都能产生
   非空真实 health response。
3. development calibration、holdout blind execution、唯一 evaluate、哈希绑定和串行时间线
   已形成完整、可审计的 formal pipeline。
4. v1 combined detector 在三个 holdout 事件的第一个污染帧都产生报警，事件级 delay 为
   0；combined 比 seeded random 的 macro-AUROC 高 `0.1611111111`。
5. 六条 v1 forward 总 inference-only runtime 为 `18.8225668967` 秒，最大 peak allocated
   显存为 `6366.7217 MiB`，因此 v2 仍可采用单卡、小 batch、短序列策略。

### 1.2 v1 为什么没有通过

唯一失败的正式门槛是 combined equal-corruption macro-AUROC：实际
`0.6434920635`，要求严格 `> 0.75`。这说明 detector 能碰到事件入口，但污染帧与正常帧
的分数排序仍不稳定。

具体诊断为：

| 问题 | v1 证据 | v2 必须处理的方向 |
|---|---|---|
| clean prefix 分数异常 | combined 的 10 个 primary FP 中有 9 个位于事件前；位置 2、4、7 在不同 run 重复出现 | 增加显式 warm-up/min-history 和稳定的 scale floor，不能继续只靠极小 epsilon |
| reliability 阈值饱和 | reliability-only pooled FPR=`1.0`、最大 FP streak=15 | 把 reliability 改为相对历史基线的单向下降异常，不再直接使用 `-reliability` |
| overlap 完全缺失 | 三条 holdout 的 `missing_signals` 均含 `overlap` | 增加 causal、online、非 GT 的 visual correspondence coverage |
| combined 标度不稳定 | 三个 robust-z 与两个未标准化 raw signal 直接求和 | 五个信号统一成有方向、截断后的可比 anomaly component，再取平均 |
| threshold 选择允许高 FPR 解 | reliability-only 的开发集最优 F1 导向饱和阈值 | threshold 搜索先满足 FPR/streak 约束，再比较 F1、事件数和 delay |
| v1 泛化范围窄 | development/holdout 都来自同一个 `fr1_desk` 场景 | v2 formal holdout 必须来自从未产生模型 response 的新场景 |

combined v1 的逐 run primary 错误位置为：

```text
holdout-dynamic  FP=[2,7,8,27]  FN=[17,18,19]
holdout-wrong    FP=[2,4,7]     FN=[17,18]
holdout-low      FP=[2,4,10]    FN=[18,19]
```

这些位置只用于 v2 development failure analysis。v1 holdout 已经解盲，从现在起不得再被
称为 blind 或 confirmatory holdout。

## 2. 本阶段的清晰目标

### 2.1 主目标

实现并验证一个 **Detector v2**：它必须使用 causal、online、非 GT 的 visual overlap
proxy，修复 startup/reliability 标度问题，在 development readiness gate 通过后，只在一条
全新场景上执行一次 blind formal evaluation，从而给出是否可以解锁最小 quarantine pilot
的可信决定。

### 2.2 完成定义

只有下列内容全部完成，本计划才算完成；执行过程中不得因为一次代码失败、一次 development
指标下降、临时 GPU 不可用或网络重试而提前宣布结束：

- [ ] v1 正式证据保持不可变，v1 runner/profile 可以原样重放。
- [ ] 生成 byte-bound 的 v1 failure-analysis 报告，解释 startup FP、reliability 饱和和
      missing overlap。
- [ ] 实现 deterministic visual correspondence coverage；除首帧外，每个 v2 frame 都有
      finite overlap，且计算不使用 GT、depth、标签或未来帧。
- [ ] 实现 direction-aware robust component、min-history、development-derived scale floor
      和 mean aggregation，不改变 v1 score 语义。
- [ ] 完成 unit/property/integration/adversarial tests；旧测试必须全部继续通过。
- [ ] 证明 v2 instrumentation 不改变 ReCal3R model input、prediction summary、trajectory
      或 recurrent update count。
- [ ] 在已解盲的 legacy 数据上完成 development 迭代，达到 Phase 6 readiness gate。
- [ ] readiness 通过后最多下载一条新的官方 RGB-D+GT 小场景，不批量下载数据集。
- [ ] 冻结 formal-v2 protocol、代码 commit、输入、split、commitment、开发集配置和阈值。
- [ ] 全部 development forward/calibration 完成后，才顺序执行三条 blind holdout forward。
- [ ] 三条 holdout 全部退出并冻结前，不读取、解析、哈希 response 或主日志。
- [ ] formal v2 只 evaluate 一次，发布 immutable metrics、timeline、manifest 和 Go/No-Go。
- [ ] 若 formal v2 GO，继续完成独立预注册的最小 quarantine pilot 并发布其结论；若
      NO-GO，按协议停止恢复实验，但仍须完成失败分析、文档、提交和资源收尾。
- [ ] 两个仓库 clean，无遗留 PID、GPU worker、监听端口、staging 或无用临时文件。

“不到目标完成不停止”在本计划中的含义是：在 formal commitment 之前，development gate
不通过就继续诊断和版本化迭代，不能把不理想结果包装成完成；formal holdout 一旦解盲，则
必须接受唯一评价结果。研究指标不能被保证为 GO，因此一次合规的 formal v2 NO-GO 也是
本计划的有效科学终点，但绝不允许在同一 holdout 上调参后重跑。

### 2.3 条件目标

formal v2 全部门槛通过时，解锁一个独立版本的最小 quarantine pilot。GO 路径不能只写
设计文档就结束，必须继续完成该 pilot、发布结果并收尾。quarantine 不是 Detector v2
development 的默认工作，也不得为了展示恢复结果而绕过 detection gate。

## 3. 本阶段明确不做什么

- 不修改、覆盖或重新命名 `formal-v1-*` 输入、运行、校准和评估产物。
- 不把 v1 holdout 再包装成 blind 数据。
- 不使用 GT pose、GT depth、污染标签或未来帧作为 online detector 输入。
- 不下载多个 TUM/Replica/ScanNet/TartanAir 场景，不部署新的大模型或 feature checkpoint。
- 不训练或微调 ReCal3R，不修改 ReCal3R tracked source。
- 不用删除其他用户数据、停止其他用户进程或抢占 GPU 的方式解阻。
- detection gate 未通过时，不执行 high-risk skip、reduced-rate、quarantine、rollback 或
  safe replay。
- 不把 visual correspondence coverage 称为真实物理 overlap 百分比。
- 不用 ATE/RPE、depth accuracy 或 recovery gain 描述本 detection-only 阶段。

## 4. Detector v2 技术方案

### 4.1 Online visual correspondence coverage

v2 的 `overlap` 是模型输入图像之间的 **verified visual correspondence coverage**，不是
GT overlap。它只比较当前 frame 与前一 frame 的 model-ready RGB：

1. 将官方 loader 产生的同一 `[-1,1]`、center-cropped tensor 确定性还原为 `uint8` RGB；
2. 使用 ORB 提取局部特征；
3. 使用 mutual/ratio-filtered Hamming match；
4. 用 RANSAC fundamental matrix 去除不满足主相机运动的 outlier；
5. 把两幅图划分为固定网格，分别统计含 inlier endpoint 的 cell coverage；
6. score 取两幅图 coverage 的较小值，并结合 inlier ratio，固定到 `[0,1]`；
7. 描述子不足、match 不足或几何验证失败时返回 `0`，不得返回 NaN 或偷偷丢帧；
8. frame 0 没有历史 reference，明确记录为 `null`，risk component 为 neutral 0。

当前环境审计已知：StateGuard3R `.venv` 没有 OpenCV；ReCal3R `.venv` 已有
`opencv-python==4.11.0.86`。执行时优先复用后者，不立即在 StateGuard3R 环境重复安装。
OpenCV 必须设置单线程、固定 RNG、关闭 OpenCL，并记录 build/version/config provenance。
StateGuard3R core tests 通过注入 synthetic keypoint/match 结果测试纯 NumPy 逻辑；真实
OpenCV integration checker 使用 ReCal3R `.venv` 运行，默认 import v2 模块时不得强制
StateGuard3R `.venv` 安装 `cv2`。

候选参数最多允许四组，全部结果必须保留，不得只报告最好的一组：

```text
ORB nfeatures       = 2000（固定）
grid                = 8 x 6（固定）
minimum matches     = 12（固定）
ratio threshold     = [0.70, 0.80]
RANSAC pixel limit  = [1.0, 2.0]
```

选择顺序固定为：先满足 deterministic replay、finite coverage、clean-prefix 稳定和三类污染
方向正确；再比较 development macro-AUROC；仍相同则选更严格 ratio、更小 RANSAC limit、
候选顺序更早者。正式 holdout 不参与选择。

### 4.2 Offline GT-depth reprojection overlap

为了让 `low_overlap_jump` 不再只是 pose-distance proxy，输入构造阶段另行实现
`gt_depth_reprojection_overlap`：用 TUM intrinsics、GT pose 和 depth 做双向 reprojection，
统计 in-bounds 且 depth-consistent 的采样点比例，取双向较小值。

该值只用于：

- development/holdout donor 的确定性选择；
- 证明 corruption 确实降低几何视野重合；
- 记录 corruption severity。

它绝不写入 online Health Ledger，也绝不进入 detector score。输入 builder 必须分别标记
`label_construction_gt_overlap` 和 `online_visual_correspondence_coverage`，防止两者混淆。

### 4.3 Direction-aware causal anomaly component

v1 对 geometric residual、pose jump 和 update magnitude 使用 absolute robust-z，却对
overlap/reliability 使用未标准化 raw 值。v2 对五个信号统一使用有方向的 causal robust
component。

对 frame `t`，历史只允许使用严格早于 `t` 的最多 `window` 个 finite 值：

```text
center_t = median(history)
scale_t  = max(1.4826 * MAD(history), development_scale_floor[signal])

high-risk signal: max(0, (x_t - center_t) / scale_t)
low-risk signal:  max(0, (center_t - x_t) / scale_t)

component_t = min(component_t, max_z)
```

方向固定为：

| Signal | 风险方向 |
|---|---|
| geometric residual | high |
| pose jump | high |
| update magnitude/global state delta | high |
| visual correspondence coverage | low |
| reliability | low |

`development_scale_floor` 只从 development clean-prefix 的正 absolute first-difference 计算，
规则固定为该信号第 10 percentile 与 `1e-6` 的较大值；不存在正 difference 时明确回退
到 `1e-6`。holdout 不得重新估计 center、floor 或任何全局统计量。

### 4.4 Warm-up 与 combined aggregation

- `min_history` 未满足时，component 输出 finite neutral 0，并记录 `warmup=true`；不得从
  metric denominator 删除这些 clean frame。
- corruption event 必须满足 `start >= max(window, min_history) + 5`，防止把 warm-up 设计成
  针对标签位置的特殊 mask。
- combined 使用所有 finite component 的 arithmetic mean，不再直接求和。
- formal v2 在 frame 1 之后若 overlap 缺失，runner/evaluator 必须失败；不得当成 0 后继续。
- formal combined 至少必须使用 overlap、reliability 和两个 geometry/state signal；实际
  缺失的信号及原因必须写入 manifest。

Detector v2 的 development candidate grid 限制为八组：

```text
window       = [5, 10]
min_history  = [3, 5]
max_z        = [5, 10]
epsilon      = 1e-6（仅作最终数值防护，不承担 scale floor）
aggregation  = finite_component_mean（固定）
```

### 4.5 Threshold 与配置选择

每个 candidate/method 的 threshold 只能来自 development finite score。选择 threshold 时先
筛除不满足以下 operational constraint 的候选：

```text
pooled primary FPR <= 0.15
max per-run FP streak <= 2
```

在可行 threshold 中依次选择：

1. 更高 equal-corruption macro-F1；
2. 更多 detected events；
3. 更低 mean detection delay；
4. 更低 pooled FPR；
5. 更高 threshold。

没有任何 threshold 满足 operational constraint 时，该 candidate 直接判为 development
不可用，不允许退回高 FPR threshold。

shared score config 使用 leave-one-corruption-type-out/leave-one-window-out development replay
选择，优先 held-out macro-AUROC，再依次比较 FPR、事件数、delay、candidate 顺序。完整
search space、每个候选结果、选择理由和哈希全部发布；不允许手工删除失败候选。

## 5. 代码与文件改动规划

以下是预期改动，不在制定本计划时创建：

| 文件 | 计划改动 |
|---|---|
| `src/stateguard3r/visual_overlap.py` | OpenCV lazy import、ORB/RANSAC/grid coverage、deterministic provenance |
| `src/stateguard3r/detection_v2.py` | direction-aware component、warm-up、scale floor、mean aggregation、v2 schema |
| `src/stateguard3r/health.py` | 仅做向后兼容的 v2 overlap/provenance 校验，不改变 v1 JSON 语义 |
| `scripts/run_recal3r_smoke.py` | 新增显式 `--health-profile v2`；默认继续为 v1；v2 计算 overlap 但不改模型状态 |
| `scripts/analyze_detection_v1_failure.py` | 只读复算 v1 FP/FN、score distribution、startup 和 signal ablation |
| `scripts/validate_visual_overlap_v2.py` | 使用 ReCal3R OpenCV 环境做真实 loader/integration deterministic replay |
| `scripts/derive_detector_v2_development.py` | 从冻结 v1 run/input 生成带 overlap 的 development ledger，不改原文件 |
| `scripts/prepare_formal_pilot_v2_inputs.py` | 跨场景输入、GT-depth overlap donor、disjoint proof、holdout commitment |
| `scripts/run_formal_detection_pilot_v2.py` | v2 commit/calibrate/evaluate 原子编排与唯一评价门禁 |
| `tests/test_visual_overlap.py` | synthetic geometry、低纹理、遮挡、无 descriptor、determinism |
| `tests/test_detection_v2.py` | direction、future-invariance、warm-up、scale floor、missing overlap、threshold constraints |
| `tests/test_formal_pilot_v2.py` | commitment、cross-scene split、blind lock、single evaluate、atomic failure |
| `docs/audits/detector-v2-failure-analysis.md` | v1 失败的机器可复算结论 |
| `docs/audits/visual-overlap-v2-audit.md` | online/GT overlap 的语义、依赖与限制 |
| `docs/protocols/detection-formal-v2.md` | holdout 前冻结的 v2 正式协议 |

v1 public API、schema、默认 runner profile 和 formal-v1 artifacts 必须保持不变。若共享代码
重构不可避免，必须先增加 v1 golden-byte/metric regression，证明输出完全一致。

## 6. 产物目录规划

所有运行产物继续留在 `StateGuard3R/outputs` 或 `logs`，不提交 Git：

```text
outputs/detector-v2-analysis-0001/
outputs/detector-v2-derived-development-0001/
outputs/detector-v2-dev-runs-NNNN/
outputs/formal-v2-inputs-0001/
outputs/formal-v2-commit-0001/
outputs/formal-v2-runs-0001/
outputs/formal-v2-calibration-0001/
outputs/formal-v2-calibration-0001-holdout-unlock/
outputs/formal-v2-evaluation-0001/
logs/detector-v2-*.log
logs/formal-v2-*.log
```

每个 ID 只创建一次，禁止覆盖。正式目录原子发布后目录为 `0555`、文件为 `0444`；manifest
记录路径、字节数、SHA-256、schema、source commit 和上游绑定。

## 7. 分阶段执行顺序

### Phase 0：激活、资源审计与证据保护

预计：0.5–1 小时。

执行：

1. 用户明确要求开始后，创建预计 18–36 小时的长时目标。
2. 重读 `SERVER_USAGE_RULES.md`、本计划、formal-v1 报告和协议。
3. 检查两个仓库 HEAD/worktree、v1 artifact hashes、磁盘、GPU、PID、端口和 tmp。
4. 记录 v2 开始 commit；不得修改 `formal-v1-*`。
5. 只提交本计划文档，再开始代码改动。

通过条件：v1 evaluation manifest 仍为
`ce17d98ac000d9b1e5df8852553708d03990ba61f42fba743e9626d850792c15`，两个仓库除本计划
提交外 clean，空间足够，且没有需要干扰他人资源的冲突。

### Phase 1：v1 failure analysis（CPU-only）

预计：1–2 小时。

执行：

1. 从 immutable v1 evaluation/run artifacts 复算每个 method/run 的 score、FP/FN、startup、
   event、boundary 和 washout 位置。
2. 对五类 canonical signal 做 leave-one-signal-out 和 single-signal rank analysis；overlap
   明确保持 missing，不伪造。
3. 将报告输入哈希、程序哈希和全部结果写入新目录并冻结。
4. 更新 `docs/audits/detector-v2-failure-analysis.md`。

通过条件：机器结果与 v1 `go-no-go.json` byte/metric 一致，并能解释 10 个 combined FP、
7 个 FN 和 reliability saturation；不启动 CUDA。

### Phase 2：visual overlap 原型与语义审计（CPU-only）

预计：3–5 小时。

执行：

1. 实现 online ORB/RANSAC/grid coverage 和 offline GT-depth reprojection overlap。
2. 在 synthetic translation、rotation、crop、low texture、moving rectangle、blank frame、
   descriptor failure 上验证边界行为。
3. 在已解盲 v1 inputs 上运行最多四组 online candidate，保留所有结果。
4. 确认任何 frame `t` 的 score 不受 `t+1...` 修改影响。
5. 确认 online score 不读取 depth、GT、labels、corruption type 或 manifest event interval。

通过条件：相同输入跨两次执行 byte-identical；frame 1..29 全 finite；clean contiguous pair
总体高于 low-overlap donor pair；不存在 GT/label leakage；审计文档完成。

### Phase 3：Detector v2 scoring 与测试（CPU-only）

预计：3–5 小时。

执行：

1. 实现 direction-aware score、scale floor、warm-up、mean aggregation 和 constrained
   threshold selection。
2. 保留 v1 functions/schema，v2 使用新模块和新 schema version。
3. 增加 future-invariance、constant series、zero MAD、missing value、short sequence、
   nonfinite rejection、tie-break、atomic output 和 adversarial tests。
4. 以冻结 v1 ledger 为 golden fixture，证明 v1 metrics 不变。
5. 执行完整 CPU suite、compileall 和 diff check。

通过条件：旧 314 tests 全部继续通过，新增测试全部通过；CUDA 未初始化；v1 golden result
不变；没有 tracked output/log/data。

### Phase 4：v2 instrumentation 等价性 smoke（单 GPU）

预计：1–2 小时。

执行：

1. 在一个已有 8–30 帧 development input 上分别执行 v1 profile 与 v2 profile。
2. GPU 启动前保存两份新快照；显式绑定一张 `free >= 12288 MiB` 的 GPU。GPU 不要求
   完全空闲，但不得使用会造成 OOM 或干扰他人的设备。
3. 对比 checkpoint audit、input snapshots、prediction summary、trajectory、update count
   和除新增 overlap/provenance 外的 health fields。
4. 记录 overlap CPU runtime、runner wall time、inference-only time 和 GPU peak memory，
   不把它们混为一项。

通过条件：模型相关输出与 v1 完全一致；v2 只增加 detector instrumentation；frame 1..N-1
overlap finite；退出后 PID 消失、显存释放。

### Phase 5：Development 迭代循环

预计：4–10 小时；未过门槛就继续版本化迭代。

development pool 使用所有已经解盲的 v1 六条运行及其冻结 inputs。它们可以用于 v2
开发，但不再是 holdout。优先先用 CPU derived ledger 搜索，只有 runner/instrumentation
变化需要真实模型输出时才重新使用 GPU。

每次迭代必须：

1. 创建新的 config/output ID，不覆盖旧结果；
2. 一次只改变一类因素：overlap candidate、score transform、warm-up 或 threshold policy；
3. 保留全部候选和失败结果；
4. 执行 leave-one-corruption-type-out 和 leave-one-window-out；
5. 生成 score timeline、startup FP、per-event coverage、FPR/streak、runtime；
6. 若 gate 失败，写出失败原因和下一次唯一改动，不接触新 formal holdout。

禁止通过移动 event、删除 startup negatives、缩短 washout、挑选 favorable run 或修改 v1
标签来获得更高指标。

### Phase 6：Development readiness gate

预计：0.5–1 小时。

只有同时满足以下条件，才允许下载/准备新 formal holdout：

1. full-development combined equal-corruption macro-AUROC `>= 0.80`；
2. cross-validated held-out macro-AUROC `>= 0.75`；
3. combined minus seeded-random macro-AUROC `>= 0.15`；
4. pooled primary FPR `<= 0.15`；
5. max per-run FP streak `<= 2`；
6. 三类 corruption 都检测到，mean delay `<= 1`；
7. 六条 run 合计 clean-prefix FP `<= 6`，且同一 startup position 不在三个以上 run 重复报警；
8. reliability-only pooled FPR `< 0.50`，证明不再饱和；
9. combined 在所有 frame 1..29 使用 finite online overlap；
10. instrumentation equivalence、determinism、finite、provenance 和 CPU/GPU cleanup 全通过。

readiness 未通过时返回 Phase 5，不允许“先看新 holdout 再决定怎么改”。

### Phase 7：获取一条新的小型 formal holdout 场景

预计：1–4 小时，取决于网络；只允许一个 archive。

执行：

1. 从未进行过 ReCal3R response inspection 的官方 TUM RGB-D+GT 场景 allowlist 中，按
   官方 metadata 可用、具备 RGB/depth/GT、预估 archive 最小的确定性顺序选择一条。
2. 下载前记录 URL、Content-Length、预估解压大小、目标路径和磁盘余量。
3. 只下载这一条；使用 `.part`、精确字节数、gzip/tar safety、SHA-256 和原子发布。
4. raw tree 建 manifest，验证格式/关联后设只读；不得运行 detector 或 online visual
   overlap 来预览 holdout response。
5. input builder 只能用 timestamps、GT/depth reprojection overlap 和固定 corruption
   construction 选择 base/donor；不能读取 model health。

若唯一 archive 不合格，先形成正式 data-suitability failure report；不得静默连下多个场景。
是否启用第二候选必须作为新的显式、可审计范围变更。

### Phase 8：Formal v2 预注册与 commitment

预计：1–2 小时。

formal-v2 development 使用已解盲 `fr1_desk` 六条输入作为两窗口/三 corruption 的开发池；
formal-v2 holdout 使用新场景的三条输入。协议必须在任何新场景 model response 之前冻结：

1. `docs/protocols/detection-formal-v2.md`；
2. StateGuard3R/ReCal3R commit、runner、OpenCV build、checkpoint、数据 archive/raw manifest；
3. nine-run registry、跨场景/disjoint proof、online-vs-GT overlap separation；
4. event/mask/washout/startup 语义；
5. v2 candidate grid、scale floors、选中 config、四方法 threshold；
6. GPU 最低 free memory、固定执行顺序、not-before；
7. holdout commitment 与 blind read lock；
8. 唯一 evaluate 输出路径和拒绝覆盖规则。

CPU validator 必须在 CUDA 隐藏/未初始化条件下验证全部输入、source hashes、loader replay、
GT association、overlap provenance 和 v1 evidence immutability。commitment 发布后直到 evaluate
完成，不得修改任何 tracked 文件。

### Phase 9：Formal development、calibration 与 blind holdout

预计：2–4 小时。

固定顺序：

```text
development window A: dynamic -> wrong -> low
development window B: dynamic -> wrong -> low
CPU calibration + immutable config + holdout unlock
new-scene holdout: dynamic -> wrong -> low
CPU evaluate exactly once
```

每个 GPU run：

- 两份紧邻 launch 的 GPU/process/Git 快照；
- `CUDA_VISIBLE_DEVICES` 显式绑定物理 GPU；
- batch size 1、30 帧、唯一输出目录；
- stdout/stderr 直接重定向到主日志；
- 记录 PID、UUID、完整命令、开始/结束、runtime 和 peak memory；
- exit 0 后只做 postflight、结构检查和 `0555/0444` 冻结；
- 三条 holdout 完成前不读、不哈希 response 和主日志。

任一 development run 失败可在未污染 holdout 的前提下修复并创建全新 formal version；任一
holdout run 失败则停止解盲，保留失败证据，不覆盖或换目录挑选结果。

### Phase 10：Formal v2 唯一评价门槛

预计：0.5–1 小时。

保留 v1 七项门槛以便直接比较，并增加两个 v2 必要条件。九项全部为合取关系：

1. combined holdout macro-AUROC 严格 `> 0.75`；
2. combined minus random macro-AUROC `>= 0.10`；
3. combined 不低于最佳单信号减 `0.02`；
4. development/holdout 共同检测至少两类 corruption；
5. combined pooled primary FPR `<= 0.20`；
6. max per-run primary FP streak `<= 3`；
7. provenance、runtime、finite、replay、reproducibility 和 real-signal checks 全通过；
8. frame 1..29 的 online overlap 全 finite，combined 确实使用它，且不存在 GT/label leakage；
9. holdout clean-prefix pooled FPR `<= 0.15`、startup max streak `<= 2`。

evaluate 只运行一次：

- **GO：**只支持“跨场景短序列 detection pilot 可以进入最小 quarantine 设计”；随后创建
  quarantine 专用 protocol/commitment，不直接宣称 recovery 有效。
- **NO-GO：**停止在 quarantine/rollback 前，发布失败项和 v2 limitations；当前 holdout
  立即转为 disclosed，不得调参后重跑同一 gate。

### Phase 11：条件式最小 quarantine pilot

只有 Phase 10 GO 才执行。目标不是做完整 rollback，而是回答：在 detector 首次报警时
延迟提交高风险更新，是否比原始 ReCal3R 减少污染后的持续偏差，且不会因误报造成更大损害。

必须另写预注册协议，至少包含：

- baseline ReCal3R vs. detector-v2-triggered quarantine；
- 固定短 buffer、固定 release/drop 规则；
- 不使用 holdout 标签作运行时 oracle；
- output-level/ledger-level damage proxy、clean cost、latency、memory；
- 同样的新 blind 数据或明确降级为 exploratory；
- GO/NO-GO 与 rollback 是否解锁的独立门槛。

本计划不预先承诺 quarantine 会改善结果，也不在 detection NO-GO 时执行它。

### Phase 12：报告、提交与资源收尾

预计：1–2 小时。

1. 更新本计划 checklist、feasibility report 和 run registry。
2. 写 formal-v2 result section，完整报告所有候选、失败和限制。
3. code/tests/protocol/results docs 分开提交；不提交 outputs、logs、数据、权重或 cache。
4. 重新运行 CPU suite、compileall、`git diff --check` 和机器一致性 checker。
5. 核验两个仓库 clean、所有 PID/GPU worker/端口消失、GPU 释放、目录权限和磁盘。
6. 只清理本任务明确生成的临时 fixture；优先移到回收站，不删除正式证据。
7. 完成后关闭长时目标，汇报正式 decision、commit、artifact hashes、测试和资源状态。

## 8. Development readiness 与正式结果的关系

readiness gate 是“是否值得消耗一个新 blind holdout”的门槛，不是科研结论。达到
development `0.80` 不保证 formal GO；它只提供足够 margin，避免在明显未准备好时浪费
新场景。

formal v2 的唯一结果具有优先级：

```text
development readiness PASS
  -> freeze everything
  -> fresh cross-scene blind holdout
  -> unique formal decision
       -> GO: unlock a separately preregistered quarantine pilot
       -> NO-GO: publish and stop before recovery
```

任何一条路径都不得用当前或新 holdout 反复调参。

## 9. 执行命令模板

以下命令是实施完成后的预期接口；脚本尚未创建时不得照抄运行。

### 9.1 CPU tests

```bash
cd /data/wangzheng/Project2/StateGuard3R
CUDA_VISIBLE_DEVICES='' \
TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp \
.venv/bin/python -m pytest -q

CUDA_VISIBLE_DEVICES='' \
TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp \
.venv/bin/python -m compileall -q src scripts tests

git diff --check
```

### 9.2 v1 failure analysis

```bash
CUDA_VISIBLE_DEVICES='' \
TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp \
.venv/bin/python scripts/analyze_detection_v1_failure.py \
  outputs/formal-v1-inputs-0001 \
  outputs/formal-v1-runs-0001 \
  outputs/formal-v1-evaluation-0001 \
  outputs/detector-v2-analysis-0001
```

### 9.3 v2 GPU runner

```bash
CUDA_VISIBLE_DEVICES=GPU_ID \
TMPDIR=/data/wangzheng/Project2/StateGuard3R/tmp \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
scripts/run_recal3r_smoke.py \
  --baseline-root /data/wangzheng/Project2/baselines/ReCal3R \
  --checkpoint /data/wangzheng/Project2/baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth \
  --checkpoint-sha256 45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103 \
  --input-manifest INPUT_MANIFEST \
  --output-dir UNIQUE_OUTPUT_DIR \
  --device cuda --size 512 --seed 0 --beta-base 0.1 \
  --health-profile v2
```

### 9.4 Formal v2 orchestration

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/run_formal_detection_pilot_v2.py \
  commit outputs/formal-v2-inputs-0001 outputs/formal-v2-commit-0001 \
  --protocol docs/protocols/detection-formal-v2.md

CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/run_formal_detection_pilot_v2.py \
  calibrate outputs/formal-v2-inputs-0001 outputs/formal-v2-commit-0001 \
  outputs/formal-v2-runs-0001 outputs/formal-v2-calibration-0001

CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/run_formal_detection_pilot_v2.py \
  evaluate outputs/formal-v2-inputs-0001 outputs/formal-v2-commit-0001 \
  outputs/formal-v2-calibration-0001 outputs/formal-v2-runs-0001 \
  outputs/formal-v2-evaluation-0001
```

所有正式命令在真正预注册时必须替换为代码实际支持的接口，并把完整展开后的命令写入
commitment/run registry；本节模板本身不是正式 protocol。

## 10. GPU、数据与存储预算

### GPU

- 单卡、batch size 1；不使用多卡训练。
- v1 实测 peak allocated 约 `6.37 GiB`；每次 launch 要求当前 free memory 至少
  `12 GiB`，但 GPU 不要求完全空闲。
- 每次重新选择当前合适 GPU，保存 UUID 和两份快照；不得假定 GPU 2 永久可用。
- 不停止、迁移或降低任何其他用户任务优先级。

### 数据

- development 首先完全复用现有只读数据和已解盲运行，不下载。
- 只有 readiness PASS 后，最多下载一个新的官方 TUM RGB-D+GT archive。
- 不复制大量 RGB/depth；输入用只读 source manifest 引用。
- 新场景未完成 formal evaluate 前，不产生 detector response 可见性泄漏。

### 存储

粗略上限：

| 内容 | 预算 |
|---|---:|
| 新 archive + raw tree | 不超过 5 GiB，下载前复核 |
| development/正式 JSON 产物和 timeline | 小于 100 MiB |
| 临时 loader/test fixture | 小于 2 GiB，结束后回收 |
| 新模型权重 | 0；复用现有 checkpoint |

## 11. 提交计划

建议保持小提交、可独立验证：

1. `docs: plan detector v2 execution`
2. `test: specify visual overlap and detector v2 semantics`
3. `feat: add deterministic visual overlap signal`
4. `feat: add causal detector v2 scoring`
5. `feat: instrument ReCal3R health profile v2`
6. `feat: add formal v2 input and evidence gates`
7. `docs: preregister formal detection v2`
8. `docs: record detector v2 formal decision`

formal commitment 后至唯一 evaluate 完成前，不允许任何 tracked commit 或修改。

## 12. 风险与处理策略

| 风险 | 处理 |
|---|---|
| ORB 在低纹理帧无 descriptor | score 明确为 0，记录 keypoint/match count；development 检查是否导致普遍饱和 |
| OpenCV 多线程/RANSAC 不确定 | 单线程、固定 RNG、关闭 OpenCL、两次 byte replay、冻结 build info |
| warm-up 人为抬高指标 | warm-up frame 仍进入 negative denominator；增加 early-event development stress diagnostic |
| GT overlap 泄漏到 detector | 模块/字段/schema 分离；formal checker 拒绝 online ledger 出现 GT 字段 |
| 当前 v1 数据过拟合 | leave-one-window/type-out；formal holdout 使用全新场景且只评估一次 |
| 新场景 distribution shift | development readiness 留出 AUROC/FPR margin；不在 holdout 上补调 |
| download 失败 | `.part`、断点/有限重试；不自动扩大到多个数据集 |
| GPU 被其他用户部分占用 | 只看可用显存和进程归属，换本任务 GPU；不终止他人进程 |
| formal run 中途失败 | 不解盲、不覆盖；保留失败证据并审计是否需要新的 formal version |
| formal v2 再次 NO-GO | 接受结论并停止 recovery；下一版必须使用新的 protocol 和新的 blind holdout |

## 13. 执行时需要用户做什么

当前不需要用户下载或提供任何东西。正式执行后，正常情况下也由任务自动完成现有数据分析、
代码、测试、GPU development 和一个小型官方数据下载。

只有出现以下情况才需要用户介入：

1. 新官方场景的下载入口要求人工浏览器确认或凭据；
2. 所有 GPU 长时间都低于 12 GiB free，且等待/换卡无法解阻；
3. 唯一候选 archive 不适合构造三类 formal input，需要授权下载第二个小场景；
4. 需要扩大到新数据集、训练模型或改变本计划的 scientific scope。

用户明确说“开始执行本计划”之前，不创建 Detector v2 代码、不下载数据、不运行 GPU，
也不创建新的长时目标。

## 14. Execution outcome (2026-08-03)

- [x] Phase 0--6: v1 evidence was preserved; the causal visual proxy, v2
  scoring, regression tests, instrumentation equivalence, and readiness gate
  passed.
- [x] Phase 7: exactly one new official TUM RGB-D+GT archive was acquired and
  frozen: `rgbd_dataset_freiburg3_walking_static`.
- [x] Phase 8--10: protocol, CPU validation, commitment, six disclosed
  development forwards, calibration/unlock, three serial blind forwards, and
  one evaluation were completed. The formal v2 decision is **GO**; see
  `docs/audits/formal-v2-result.md` and the immutable 0002 artifacts.
- [x] A pre-blind audit superseded commitment 0001 before any blind forward
  because its launcher could have violated the all-three-before-read lock.
  The corrected, independently committed 0002 version reran development and
  supplied the only formal decision.
- [x] Phase 11: the separately pre-registered exploratory ledger-level
  quarantine proxy completed with decision
  `EXPLORATORY_PROXY_ONLY_NO_ROLLBACK`. It does not support recovery or
  rollback claims.
- [ ] Phase 12: final full-suite verification, documentation commit, artifact
  permission/process cleanup, and goal close-out remain.
