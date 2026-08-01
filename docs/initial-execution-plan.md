# StateGuard3R 初步可行性执行计划

日期：2026-07-31

状态：执行中；Gate 0 已通过，Phase 5 科研结论仍为 HOLD / evidence pending

预计持续时间：12–24 小时（包含环境安装、最小数据获取、GPU 推理和评测；若上游依赖编译或网络较慢，实际墙钟时间可能更长）

## 1. 执行目标

在不训练大模型、不批量下载数据集的前提下，完成一个可复现的最小证据闭环：

```text
ReCal3R clean baseline
  -> 三类已知 corruption
  -> per-frame health signals
  -> detection-only 量化评测
  -> Go/No-Go 结论
  -> 仅在通过门槛后进行最小 quarantine 验证
```

第一轮的目标不是完成整套 StateGuard3R，而是回答以下问题：

1. ReCal3R 是否能在当前服务器上用少量输入稳定运行？
2. 是否能访问或低成本导出 reliability、update magnitude、pose、pointmap/residual 等信号？
3. low-overlap jump、dynamic occlusion、wrong-order segment 是否会产生可重复的状态污染或错误传播？
4. training-free health score 是否能定位已知污染区间？
5. 结果是否足以支持继续实现 quarantine 和 rollback？

## 2. 完成定义

下列内容全部完成，才算完成本计划的必做部分：

- [ ] 记录官方 ReCal3R 来源、许可证、固定 commit、依赖、权重来源和实际运行命令。
- [ ] 使用独立环境完成单样本 smoke test 和至少一条短序列 clean baseline。
- [ ] 保存可获得的 trajectory、depth/pointmap、health log、runtime 和 peak GPU memory。
- [ ] 生成三类 corruption 的可复现配置与标签，不修改原始数据和 GT。
- [ ] 完成 random、update-magnitude-only、reliability-only、combined health score 四组 detection 对照。
- [ ] 汇报 AUROC、F1、detection delay、false-positive rate 和时间曲线。
- [ ] 给出有证据的 Go/No-Go 结论、失败分析和下一步建议。
- [ ] 所有运行进程、GPU 显存、端口和临时文件完成收尾核查。

以下内容是条件任务，不作为 detection 未通过时的强制完成项：

- [ ] high-risk skip 对照。
- [ ] reduced-rate 对照。
- [ ] 最小 quarantine buffer 验证。
- [ ] snapshot rollback 和 safe replay。

## 3. 严格边界

### 3.1 第一轮纳入范围

- 主 baseline：ReCal3R。
- 数据：先使用一个官方小样本或一条 TUM RGB-D 短序列；通过 smoke test 后最多扩展到三条短序列。
- corruption：low-overlap jump、dynamic occlusion、wrong-order segment。
- 方法：只做日志、training-free robust score，以及通过门槛后的最小 quarantine。

### 3.2 第一轮不纳入范围

- 不批量下载 TUM、Replica、ScanNet、TartanAir。
- 不同时部署 CUT3R、MASt3R-SLAM、VidMap、VGGT-CD。
- 不训练或微调 foundation model。
- 不进行 region/token-level rollback。
- 不进行大规模 benchmark、论文级消融或多卡训练。

## 4. 工作区与存储布局

严格保持 Git 仓库、环境和输出隔离：

```text
/data/wangzheng/Project2/
├── StateGuard3R/                     # 主方法、评测、汇总和本文档
│   ├── docs/
│   ├── configs/
│   ├── scripts/
│   ├── src/
│   └── outputs/                      # StateGuard3R 汇总结果；不提交大文件
├── baselines/
│   └── ReCal3R/                      # 唯一官方副本
│       ├── .venv/
│       ├── data/                     # 小样本或指向只读数据的软链接
│       ├── checkpoints/
│       ├── outputs/                  # 原版 baseline 输出
│       ├── logs/
│       └── tmp/
└── .cache/
    ├── uv/
    ├── huggingface/
    └── torch/
```

执行约束：

- 新内容只写入 `/data/wangzheng/Project2`。
- 先搜索 `/data/wangzheng` 下属于 `wangzheng` 的已有数据、权重和兼容环境；可用时只读复用。
- 不使用系统 `/tmp`、`/dev/shm`、`pip install --user` 或系统 Python 安装依赖。
- 原始数据只读。corruption 优先保存为 frame manifest/配置，避免复制完整序列。
- ReCal3R 原版先不修改。health logging 优先由 StateGuard3R wrapper/hook 实现；确需修改时，在 ReCal3R 独立实验分支上做最小 instrumentation，并保留原版运行结果。
- checkpoint、数据、日志、视频、点云和大规模输出不加入 Git。

## 5. 阶段计划与硬门槛

### Phase 0：前置审计

预计：0.5–1 小时。

操作：

1. 检查工作区路径、Git 状态和磁盘空间。
2. 搜索已有 ReCal3R/CUT3R、checkpoint、TUM 小序列和可兼容环境。
3. 只读核验 ReCal3R 官方仓库、LICENSE、默认分支、依赖、权重和最小运行入口。
4. 检查 GPU、进程所有者和可用显存，不停止任何现有进程。
5. 估算首个样本、权重、环境和输出的空间需求。

产物：`docs/audits/initial-resource-audit.md`。

通过条件：

- 官方来源和许可证明确；
- 存储预算可控；
- 有合法可用的 GPU 或明确的 CPU/稍后 GPU smoke-test 路径；
- 没有需要修改系统环境的硬依赖。

失败处理：停止下载和安装，记录具体阻塞，不用 `sudo` 绕过。

### Phase 1：固定官方 baseline 与独立环境

预计：2–5 小时。

操作：

1. 若本地没有官方副本，仅克隆 ReCal3R 一个仓库到 `baselines/ReCal3R`。
2. 固定并记录 commit，检查工作区干净状态。
3. 检查现有 `.venv`、`pyproject.toml`、lock 文件和官方环境说明。
4. 使用 uv 创建 ReCal3R 独立 `.venv`，将 cache/TMP/HF/Torch 路径指向 Project2。
5. 先安装最小推理依赖；不部署其他 baseline。
6. 搜索本地权重；仅在没有可用副本时下载 ReCal3R 必需权重。

产物：

- `docs/audits/recal3r-upstream-audit.md`
- `docs/runs/run-registry.md`
- 可运行的 ReCal3R 独立环境

通过条件：官方最小入口可导入，模型能够初始化，且依赖没有污染系统或其他仓库环境。

降级：若完整依赖无法建立，先验证代码导入、数据加载和无 GPU 的静态/单元测试，并记录剩余阻塞。

### Phase 2：最小数据与 clean smoke test

预计：1–4 小时。

下载限制：

1. 第一次只获取一个官方 demo 输入或一条 TUM 短序列。
2. 下载前记录 URL、文件大小、预估解压大小、目标绝对路径和剩余空间。
3. 解压后校验文件数量、目录结构、时间戳/GT 等关键文件。
4. 未经用户另行授权不删除压缩包。
5. 只有首条序列跑通后，才允许最多扩展到三条短序列。

运行顺序：

```text
单图/最小输入
  -> 8–20 帧短片段
  -> 一条完整短序列
  -> 最多三条短序列
```

每次 GPU 运行必须记录：完整命令、commit、配置、数据、checkpoint、随机种子、GPU/CUDA、PID、日志、输出目录、开始/结束时间、runtime、peak memory。

通过条件：

- 至少一条短序列完整结束；
- 输出非空、无 NaN/Inf；
- trajectory 或 pointmap/depth 至少有一类可用于后续评测；
- 进程退出后 GPU 显存释放。

### Phase 3：Health Ledger v0

预计：2–4 小时。

优先提取以下信号：

```text
frame_id
timestamp
overlap_score
pose_jump
geometric_or_pointmap_residual
update_magnitude
reliability_mean
candidate_learning_rate
calibrated_learning_rate
```

统一输出为逐帧 JSONL。不可获得的字段保留为 `null` 并记录原因，不伪造代理值。

最小验收：至少获得 `frame_id`、一个几何/位姿信号和一个 ReCal3R 内部可靠性/更新信号；若内部信号完全不可访问，则启用 output-level 路线，并把这一点列为方法局限。

### Phase 4：Corruption Protocol v0

预计：1–3 小时。

依次实现：

1. `low_overlap_jump`：以 frame manifest 拼接低重叠片段。
2. `dynamic_occlusion`：使用现有动态片段或只生成短区间遮挡派生帧，不复制整套数据。
3. `wrong_order_segment`：仅修改 manifest 中一小段帧顺序。

每个样本必须保存：

```json
{
  "sequence": "...",
  "source_sequence": "...",
  "source_is_read_only": true,
  "corruptions": [
    {
      "type": "...",
      "start_frame": 0,
      "end_frame": 0,
      "parameters": {},
      "expected_effect": "..."
    }
  ]
}
```

通过条件：三类配置均可由同一 clean source 重放；标签区间与实际输入一致；clean source 和 GT 哈希/状态不变。

### Phase 5：Detection-only

预计：2–5 小时。

在 development 序列上确定窗口和阈值，在独立 holdout corruption 上报告结果；不得利用 frozen evaluation 反复调参。

训练无关分数：

```text
risk_t =
    robust_z(geometric_residual_t)
  + robust_z(pose_jump_t)
  + robust_z(update_magnitude_t)
  + (1 - overlap_score_t)
  - reliability_t
```

缺失字段不进入分数，并在配置中记录实际使用项。`robust_z` 使用 trailing rolling median/MAD，不读取未来帧。

对照：

1. random score（固定随机种子）；
2. update magnitude only；
3. ReCal3R reliability only；
4. combined health score。

指标：AUROC、F1、detection delay、false-positive rate，并按 corruption type 分组；同时绘制 corruption 标记、risk、reliability、update magnitude、pose/geometric error 的 timeline。

Go 条件：

- combined AUROC 目标大于 0.75；
- combined 明显优于 random，并至少不劣于最佳单信号；
- 三类 corruption 中至少两类出现可重复的 risk peak；
- holdout 上误报没有使报警长期饱和；
- 所有数字来自真实日志且可由固定命令重算。

No-Go 条件：

- AUROC 接近随机；
- risk 只反映正常相机运动而不能定位 corruption；
- corruption 不产生可测的持续误差；
- ReCal3R 和 output-level 信号均无法支持检测。

No-Go 时停止实现 rollback，转为修订 corruption、信号定义，或降级为 offline diagnostic benchmark。

### Phase 6：条件式 Quarantine Pilot

仅在 Phase 5 通过后执行，预计 2–5 小时。

最小对照：

1. 原版 ReCal3R；
2. skip high-risk；
3. reduced-rate high-risk；
4. quarantine + next-K-frame confirmation（仅在接口允许时）。

通过目标：corrupted sequence 上 ATE/RPE 或 revisit consistency 改善约 10% 以上，clean sequence 下降小于 5%，runtime overhead 小于 20%。未达到时不进入 snapshot rollback。

### Phase 7：报告与收尾

预计：1–2 小时。

必须完成：

- 汇总实测值、文档值、估计值和未完成项，四者不得混写。
- 保存成功和失败实验的必要日志与结论。
- 检查所有相关 Git 仓库状态，不混合提交。
- 检查并关闭已经无用途且由本任务启动的进程。
- 检查 GPU 显存和端口释放。
- 检查输出增长、根分区、临时文件和重复权重。
- 写入 `docs/initial-feasibility-report.md`。

## 6. 提交计划

每个提交只包含一个可验证的逻辑内容：

1. `docs: add initial feasibility execution plan`
2. `chore: add experiment registry and repository scaffolding`
3. `feat: add corruption protocol manifests`
4. `feat: add health log normalization`
5. `feat: add detection-only evaluation`
6. `test: add corruption and metric tests`
7. `docs: report initial feasibility results`

如果必须修改 ReCal3R：先保留官方原版结果；instrumentation 使用独立分支和最小提交，不与 StateGuard3R 提交混合。

## 7. 当前执行状态

- [x] 已阅读服务器规则和研究计划。
- [x] 已确认 StateGuard3R 初始仓库工作区干净。
- [x] 已确认 Project2 顶层当前不是 Git 仓库。
- [x] 已确认 `/data` 最新剩余约 324 GiB、使用率 98%，必须限制下载和重复输出。
- [x] 已完成本地资源、ReCal3R 上游和最小数据方案的并行只读核查。
- [x] 已在 `baselines/ReCal3R` 固定官方 commit `466c7cdf`，工作区干净。
- [x] 已建立 ReCal3R 独立 `.venv`；CPU import、CUDA RoPE 编译和官方两图
      preprocessing 已通过。
- [x] 已实现并测试 corruption manifest v1、三类 corruption、只读 replay、
      Health Ledger、四方法 detection、formal freeze gate 和 SVG timeline。
- [x] 已实现外部 ReCal3R smoke runner；会冻结两仓库 provenance、校验 checkpoint
      SHA/大小/名称/尺寸/head/weight keys，并导出 pose、pointmap residual、trace 与
      trajectory。当前全套测试为 179 passed。
- [x] 已完成 Gate 0 runner readiness 加固：允许并审计 pinned model 的
      `load_state_dict` 委托 override（`8f6360c` / `e2d5f86`）；按来源文件及 SHA
      冻结官方 relpose 赋值语义，并为本 smoke 显式选择 `recal3r` / `beta_base=0.1`
      （`59a1eac` / `8bf6154`）；冻结实际 cuRoPE `.so` 的绝对
      路径、9,523,120-byte 大小和 SHA-256（`2c44ea8` / `483ade1`）。CPU pinned
      interface/provenance 检查已通过，但没有初始化 CUDA，也没有执行真实 kernel、
      checkpoint load 或模型 forward。
- [x] 已完成 v1 synthetic 闭环；`synthetic-smoke-0003` 在干净 tracked commit
      `08e59d0` 上使用非零遮挡速度，取代 0002 作为当前管线证据。它不是
      ReCal3R 实验结果。
- [x] 已在干净提交 `483ade1` 上完成 0003 的 CPU-only 像素 materialization
      审计：官方 loader 后的 60 帧均为独立 `1x3x384x512` tensor，五帧移动遮挡逐像素
      符合协议，输入 tensor 和 Chateau 源文件未变，且 CUDA 未初始化。两图 fixture
      只能证明路由与像素执行，不能证明真实 corruption 强度、模型响应或科研指标。
- [x] 已限制数据范围：截至 Gate 0 通过时没有下载数据集；GPU 仅运行两图 smoke，
      没有启动训练或多卡任务。
- [x] 用户已从官方 README 的 Google Drive 入口取得 512 DPT checkpoint；静态审计
      记录了 3,173,761,006-byte 大小和首次 SHA-256 `45f7e98a…f8103`，文件已设为只读，
      且没有 partial。官方未发布 checksum，因此 runner 仍须在反序列化前后复核哈希。
- [x] Gate 0 GPU 门禁已恢复：GPU 4/3 连续快照均约有 45.6 GiB 空闲、无 compute PID；
      用户另行允许使用显存充足的非完全空闲卡，但不得停止或干扰既有进程。实际启动前
      仍须紧邻复查并显式绑定单卡。
- [x] 官方 512 checkpoint 已在 0001 中成功反序列化，`strict=false` 的 missing / unexpected
      keys 均为空；这只通过加载子门槛，不代表模型 forward。
- [x] `GATE0-REAL-0001` 已保留为 pre-forward 失败证据：checkpoint 反序列化与全部
      state-dict keys 匹配，但 runner 错从被基类替换的 `model.config` 读取 head 契约，
      在 `model.to(cuda)` 前误拒绝。`5a8e609` / `ee5768c` 已按 pinned 真实 model 与
      downstream 字段修复并补齐回归测试；重试必须使用全新 0002 路径。
- [x] `GATE0-REAL-0002` 已在干净提交 `5231580` 上通过：官方 512 DPT checkpoint、
      `DPTPts3dPose`、cuRoPE、两图 CUDA forward 和恰好一次 calibrated update 均通过
      守卫；所有轻量输出 finite。推理段 1.052124 秒、1.900916 FPS，峰值 allocated
      显存 6,362.670 MiB，退出后 GPU 4 恢复为空闲状态。
- [x] `GATE0-STATE4-0001` 已用现有官方资产的四个只读引用通过机械 state smoke：
      `N-1=3` updates、trace `[1,2,3]`、四帧有限输出和混合宽高比均通过；峰值 allocated
      显存 6,367.173 MiB。该序列不是连续真实场景，只解除 7.69 MiB TUM AVI Gate 1。
- [x] Gate 1 官方 `fr1_xyz` RGB AVI 已按 `.part`→精确字节/RIFF/ffprobe/SHA 校验→
      原子改名取得：8,059,298 bytes，SHA-256 `1820b529…f8022`，只读且无 partial。
      CPU/OpenCV 已顺序解码连续索引 `0..9` 为十张只读 PNG，源 AVI 哈希保持不变。
- [x] `GATE1-FR1XYZ-10-0001` 已完成十帧真实连续 RGB forward：恰好 9 次 update、
      trace `[1..9]`、十帧有限且对齐；真实 uncertainty/reliability、pose jump、
      geometric residual 和 `global_state_delta` 非空且非恒定。推理段 1.988277 秒，
      峰值 allocated 显存 6,364.891 MiB，退出后 GPU 释放。
- [x] 30 帧输入已用 CPU/OpenCV 从同一 AVI 顺序复核：0–9 逐像素匹配冻结前缀且只被
      引用，目录内仅新增 10–29 共 20 张只读 PNG；30 帧发布后复核全部逐像素和 SHA，
      AVI 与旧 manifest 哈希不变，无 partial 或重复前缀。
- [x] `GATE1-FR1XYZ-30-0001` 已完成 30 帧真实连续 RGB forward：恰好 29 次 update、
      trace `[1..29]`、30 帧有限且对齐，前十帧的 health/trajectory/prediction 与冻结的
      十帧 run 完全一致；推理段 4.440337 秒，峰值 allocated 显存 6,366.722 MiB，
      退出后 PID 消失且 GPU 释放。
- [x] Gate 2 唯一完整包 `fr1_desk.tgz` 已按 `.part` 下载并经精确字节、gzip、tar 安全、
      SHA 和原子发布校验；raw tree 在同文件系统 staging 解压，1212 个文件逐一哈希，
      全部 RGB/depth/GT 文本格式复核后设为只读。未下载任何其他 TUM 资产。
- [x] Gate 2 连续八帧输入已冻结为只引用 manifest：按 `rgb.txt` 原始顺序选择最早满足
      depth/GT 唯一最近邻均不超过 20 ms 的 0-based 非注释数据行索引 17–24（物理行
      21–28），跨度 0.231956 秒；未复制图片，raw tree、TGZ 和 raw manifest 哈希不变。
- [x] `GATE2-FR1DESK-8-0001` 已完成连续八帧真实 CUDA forward：7 次 update、trace
      `[1..7]`、48 个 tensor summary 和全部记录数值 finite；推理段 1.501171 秒，峰值
      allocated 显存 6,364.891 MiB，退出后 PID/GPU 均释放。
- [x] Gate 2 连续 30 帧输入已冻结为只引用 manifest：最早合格 RGB 数据索引 17–46
      （物理行 21–50），跨度 0.968015 秒；30 个 depth/GT 最近邻均唯一且不超过 20 ms，
      前八帧与已通过窗口逐字段相同，未复制、链接或改写 raw 图片。
- [x] `GATE2-FR1DESK-30-0001` 已完成连续 30 帧真实 CUDA clean ledger：29 次 update、
      trace `[1..29]`、180 个 tensor summary 全 finite，前八帧四类输出与独立八帧 run
      精确相同；推理段 3.419778 秒，峰值 allocated 显存 6,365.548 MiB，PID/GPU 已释放。
- [ ] 更长 clean baseline、真实 corruption Health Ledger、formal holdout 指标和科研
      Go/No-Go 尚无证据。下一步只允许在同一冻结 30 帧来源上准备并分别运行三类最小
      exploratory corruption smoke；50 帧、formal detection、quarantine 和 rollback
      继续禁入。

状态更新规则：每完成一个阶段，立即更新本节、运行登记和验证结果；只有满足上一阶段门槛才推进下一阶段。
