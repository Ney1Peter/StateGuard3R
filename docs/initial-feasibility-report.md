# StateGuard3R 初步可行性报告

- 日期：2026-08-01
- 证据截止时间：2026-08-01 02:17 CST
- StateGuard3R 证据快照：`08e59d0f43c7ec0af7e48d487215b29c000ec39a`
- ReCal3R 固定上游：`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`
- 当前结论：**HOLD — evidence pending**

本报告中的仓库内路径均相对于 StateGuard3R 根目录；ReCal3R baseline 使用
`../baselines/ReCal3R`。本报告只承认三层互不替代的证据：已实测基础设施与单元证据、
以 `synthetic-smoke-0003` 为当前版本的 synthetic 管线证据、真实 ReCal3R 证据。
当前既不能给出科研意义的 Go，也不能把“尚未运行”解释为方法 No-Go。

## 1. 已实测基础设施与单元证据

### 1.1 已确认事实

- 官方 ReCal3R 工作副本固定在
  `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`，当前分支与 `origin/main`
  对齐且 tracked worktree 干净。来源、许可证层级、checkpoint 入口和调用路径见
  `docs/audits/recal3r-upstream-audit.md`。
- 独立环境位于 `../baselines/ReCal3R/.venv`。已实测 Python 3.11.14、
  PyTorch 2.4.0+cu121、torchvision 0.19.0+cu121；`dust3r.model` 从固定 baseline
  的 `src/dust3r/model.py` 导入。环境依赖检查已通过，完整版本冻结见
  `docs/runs/recal3r-environment-freeze.txt`。
- CroCo RoPE CUDA 扩展已用 CUDA 12.1 编译成功。这证明构建链可用，**不证明**扩展
  已在 GPU 上执行。
- 外部 runner `scripts/run_recal3r_smoke.py` 的 CLI `--help` 已在 ReCal3R 独立
  解释器中通过。runner 已静态覆盖 checkpoint 文件名—输入尺寸—head 类型绑定、
  反序列化前 SHA-256 校验、模型模块来源、baseline/StateGuard3R 工作树状态、输入快照
  与运行后哈希复核、两图 `N-1` 次 calibrated update、输出有限性和轻量 provenance。
  这些仍是接口与守卫证据，不是模型 forward 证据。
- 官方 CPU image loader 已加载 Chateau 两图，得到 `1x3x384x512`、值域
  `[-1, 1]` 的输入；加载前后源文件哈希不变。该结果只通过预处理子门槛。
- 2026-08-01 使用项目内 `TMPDIR` 和 `UV_CACHE_DIR` 实跑完整测试套件，结果为
  `170 passed in 0.81s`。测试覆盖 corruption manifest、只读 replay、Health Ledger、
  detection-only、timeline、synthetic CLI、正式 detection freeze/provenance gate 和
  ReCal3R runner 的 mock/静态契约；测试没有加载真实 checkpoint，也没有启动 CUDA。

### 1.2 已实现但尚未获得模型证据的能力

- 三类 corruption：`low_overlap_jump`、`dynamic_occlusion`、
  `wrong_order_segment`；v1 manifest 固定坐标语义并支持不复制源图的 deferred replay。
- Health Ledger：逐帧 JSONL、`reliability = 1 - uncertainty_u`、缺失值保留 `null`、
  首帧/reset 对齐、batch size 大于 1 时拒绝、按 `frame_idx` 回填 global-state delta。
- Detection-only v0：严格因果 trailing median/MAD、四方法独立阈值、按 corruption type
  分组、AUROC/F1/FPR/delay 以及 dependency-free SVG timeline。
- 正式 holdout CLI gate：要求开发集校准文件、四个显式方法阈值和所有冻结字段完全匹配；
  该门禁已实现并有单元测试，但还没有真实开发集或 holdout 运行。

### 1.3 关键提交

| 提交 | 已固定的内容 |
|---|---|
| `2023fab`、`717d359` | 初始执行计划、仓库与运行登记脚手架 |
| `9b81a23`、`09e9749`、`8271569` | 上游/资源审计、独立环境、checkpoint 与 GPU 重试记录 |
| `f77b483`、`5f76097`、`c81d37c` | corruption、Health Ledger、detection-only 核心实现 |
| `f09c93c`、`77aa446` | 核心测试与 detection-only v0 协议冻结 |
| `7e25d80`、`1d6381a`、`b528a3d` | 只读 manifest replay、SVG timeline、synthetic CLI harness |
| `d06af1c`、`09d70f9`、`64b6f3b` | corruption v1 坐标/兼容性契约及测试、文档 |
| `fde32cc`、`2648af4`、`f0b2748` | instrumented ReCal3R runner、runner 门禁测试与冻结 smoke 契约 |
| `1e411fe`、`48236ec`、`7328c5d` | 正式 detection freeze gate、provenance 测试与执行契约 |
| `233f1ac`、`0eec9d6` | 非零速度 synthetic 遮挡与回归测试 |
| `08e59d0`、`2258d2c` | 0002 运行登记、在 `08e59d0` 上执行的 0003 运行登记与最新 Gate 0 资源快照 |

以上提交证明代码和审计链已建立；它们不能替代 checkpoint forward、真实时序数据或
holdout 结果。

## 2. Synthetic 管线证据（`synthetic-smoke-0003` 为当前版本）

### 2.1 运行范围与产物

`outputs/synthetic-smoke-0003` 是 2026-08-01 02:13–02:14 CST 在提交
`08e59d0` 上完成的 **exploratory development smoke**：

- 60 条人工生成的 health ledger 记录，重复引用仓库内 Chateau 两图；没有模型推理；
- 三个已知标签区间各 5 帧：`low_overlap_jump` 为 10–14，
  `dynamic_occlusion` 为 25–29，`wrong_order_segment` 为 40–44；
- manifest 为 `stateguard3r.corruption.v1`，使用
  `deferred_transforms_no_image_copy`；动态遮挡坐标参考为
  `model_input_after_resize_and_center_crop`；
- `dynamic_occlusion` 使用非零归一化速度 `dx = 0.04`、`dy = 0.025`，五帧矩形左上角
  依次为 `(0.25, 0.25)`、`(0.29, 0.275)`、`(0.33, 0.30)`、
  `(0.37, 0.325)`、`(0.41, 0.35)`，从而覆盖移动遮挡 manifest 生成、strict-loader
  校验和标签路径；该命令未执行像素级 materialization/replay；
- 生成了 `corruption_manifest.json`、`metrics.json` 和 `timeline.svg`，六个相关文件
  原始字节数合计 112,401 B、磁盘显示 128 KiB，未复制输入图片；
- `metrics.json` 明确记录 `execution_mode = exploratory`、`formal = false`、
  `frozen_from_development = false`、`development_calibration_provenance = null`；因此它
  不是正式 development/holdout 实验。

输入与主要产物的 SHA-256 为：

| 对象 | SHA-256 |
|---|---|
| Chateau1.png | `71ffb8c7d77e5ced0bb3dcd2cb0db84d0e98e6ff5ffd2d02696a7156e5284857` |
| Chateau2.png | `c3a0be9e19f6b89491d692c71e3f2317c2288a898a990561d48b7667218b47c8` |
| `inputs/source_manifest.json` | `55bab88af20e7fd21e7b13b77c34d100e2e6fd75ad6c708dceb11b38f9a17e41` |
| `inputs/corruption_specs.json` | `ec0327f7424889354dea8743b0d7b2a30e0b277ddbb3410764949c7b3d2cbb10` |
| `inputs/health.jsonl` | `c725772db7761a92cc019adb6170a983bc39c01e60e5116091fb08c210b4d525` |
| `corruption_manifest.json` | `864133efe020a024674f49ca3c0eaaa5a6a84ee45422e492bc14bdaba11657f2` |
| `metrics.json` | `afe496ebbd285ffb95dbb2cdfb940e136d3451f941fe1d642d12efc3bd2ae051` |
| `timeline.svg` | `f9bc651def0d97cad33ed101b2efc6d458c6671629772ed245a1a7ee926e0bea` |

检测输出中嵌入的 health/manifest 单次内存快照哈希与文件一致，SVG 可解析，说明
manifest v1/schema → 移动遮挡 metadata/strict-loader/labels → Health Ledger → 四方法检测
→ metrics → timeline 的接口闭环能够按固定输入重放。像素级 deferred transform replay
由单元测试覆盖，但本次 0003 命令没有 materialize 图像；源图哈希保持不变。

### 2.2 仅用于管线自检的数字

| 方法 | AUROC | F1 | FPR | 平均 delay（帧） |
|---|---:|---:|---:|---:|
| seeded random | 0.5659259259 | 0.3829787234 | 0.5111111111 | 0.3333333333 |
| update magnitude only | 1.0 | 1.0 | 0.0 | 0.0 |
| reliability only | 1.0 | 1.0 | 0.0 | 0.0 |
| combined | 1.0 | 1.0 | 0.0 | 0.0 |

这些数值来自 harness 直接构造的、与标签有意一致的 synthetic signals，只能检测
实现是否接线正确。0003 沿用了 0002 的 synthetic health ledger，因此两次 overall
指标完全相同；这正好说明这些指标没有测量 ReCal3R 对移动像素遮挡的响应。尤其是
**synthetic AUROC/F1 不得作为科研结果、ReCal3R 性能、StateGuard3R 可行性 Go 证据或
论文表格数字**。按 corruption type 的分组数字同样只是 one-type-vs-all 管线输出，
不能改变这一证据边界。

fixture 的 clean 历史为常数，导致 MAD 为 0；在协议规定的 `MAD + 1e-6` 且不裁剪
z-score 时，标注帧出现约百万量级的 synthetic robust-z。这验证了公式实现，却也说明
risk 绝对幅值完全不具备现实校准意义。

`outputs/synthetic-smoke-0002` 保留为成功的 v1/schema 历史 smoke，但独立审计发现其
`dynamic_occlusion` 设置 `dx = dy = 0`，五帧矩形完全相同；它只覆盖静止矩形遮挡，
已由 0003 取代，不能引用为移动遮挡证据。`outputs/synthetic-smoke-0001` 使用旧 v0
坐标契约，仅保留作 stale debugging artifact，也不应引用为当前管线证据。

## 3. 尚未获得的真实 ReCal3R 证据

### 3.1 当前阻塞与缺失证据

官方 final checkpoint `cut3r_512_dpt_4_64.pth` 仍不存在。官方 README 指向的
Google Drive file ID `1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD` 在多次有限时连接中于
DNS/HTTPS 建连阶段超时；02:17 CST 的最新一次限时 HEAD 检查仍为 DNS timeout、HTTP
`000` 和零响应体。本地没有真实文件、`.part` 或 `.partial`，没有使用第三方镜像。
官方也未发布可信 checksum，所以未来首次取得文件后计算的 SHA-256 只能证明本项目内
的一致性，不能单独认证来源。224 Linear fallback 也不存在；即使取得，它也只能做
`--size 224` 接口 smoke，不能替代 512 final baseline 或正式指标。

最近一次已登记 GPU 检查（2026-08-01 02:17 CST）显示 8 张 L20 均有既有 compute
process，显存占用约为 `21038, 30854, 21175, 18909, 24651, 26139, 25247, 41495 MiB`，
没有一张卡满足完全空闲规则。未停止或修改任何他人进程，也没有启动 CUDA。GPU 状态会
变化；在新的启动前检查明确找到完全空闲卡之前，GPU gate 继续视为阻塞。同期 `/data`
剩余约 513 GiB（使用率 97%），仍不允许无门禁扩展下载或重复输出。

因此下列真实证据全部尚未获得：

- 官方 checkpoint 的安全加载、state-dict/head 兼容性和真实模型初始化；
- RoPE kernel 的 GPU 执行、两张 Chateau 图片的 ReCal3R recurrent forward，以及
  预期恰好一次 calibrated update；
- 4–8 帧 clean smoke、输出 finite 检查、runtime、peak GPU memory 和退出后显存释放；
- 来自真实 forward 的 uncertainty/reliability、attention entropy、global-state delta、
  pose jump、cross-head pointmap residual、trajectory 或 pointmap/depth 证据；
- 任意 TUM 输入、clean baseline、真实三类 corruption 响应和状态污染/传播现象；
- 独立 development/holdout、正式冻结阈值，以及真实 AUROC、F1、FPR、detection delay；
- 支持 Phase 5 Go/No-Go 或 quarantine/rollback 的真实证据。

### 3.2 Gate 状态

| Gate | 状态 | 证据与约束 |
|---|---|---|
| 官方来源、固定 commit、许可证与环境隔离 | **PASS** | 上游固定、CPU import、环境 freeze 与依赖检查已完成 |
| Corruption/Health/Detection 实现与单元测试 | **PASS（软件层）** | 170 tests passed；不等于模型或科研验证 |
| Synthetic v1/moving-occlusion metadata/detection CLI 闭环 | **PASS（开发层，有限）** | `synthetic-smoke-0003` 可重复生成并通过 strict loader，遮挡坐标逐帧移动；本次未 materialize 像素，且非 formal、非模型/真实实验 |
| 官方 512 checkpoint | **BLOCKED** | 官方入口不可达，本地无真实文件，未使用第三方来源 |
| 完全空闲单卡 | **BLOCKED** | 最新登记快照无合规空闲卡；每次启动前必须重查并显式绑定单卡 |
| Gate 0：Chateau 两图真实 ReCal3R smoke | **BLOCKED / NOT RUN** | CPU preprocessing 通过；checkpoint load、CUDA forward、一次 update 未发生 |
| 4–8 帧真实 clean smoke | **LOCKED** | 只有 Gate 0 通过后才能执行 |
| Gate 1：7.69 MiB TUM `fr1_xyz` AVI | **LOCKED；当前禁止下载/运行** | 只有前述真实 smoke 通过后才允许；AVI 不能用于 ATE/RPE |
| Gate 2：328.07 MiB `fr1_desk.tgz` | **LOCKED；当前禁止下载** | 只有 Gate 0、4–8 帧和 Gate 1 均通过后才允许 |
| 真实 Health Ledger 与三类 corruption | **LOCKED** | 依赖真实连续 forward；不得用 synthetic ledger 顶替 |
| 正式 development/holdout detection | **LOCKED** | 尚无真实日志与冻结开发集配置，不能检验 AUROC > 0.75 等 Go 条件 |
| Phase 5 科研决策 | **HOLD — evidence pending** | 证据不足，既非 Go 也非方法 No-Go |
| Quarantine/rollback | **LOCKED；当前禁止进入** | 仅在真实 formal detection 满足预设 Go gate 后才可开始 |

当前必须同时解除官方 checkpoint 和完全空闲 GPU 两个启动阻塞。任一项仍未通过时，
不得把 CPU import、单元测试或 synthetic 指标升级为 Gate 0 通过。

### 3.3 严格解阻顺序

1. 仅从官方入口取得 512 DPT 4–64 checkpoint，或由用户提供可追溯到官方来源的既有
   只读副本；记录来源、实际字节数和 SHA-256。来源无法确认时不反序列化。
2. 紧邻运行前重新检查全部 GPU。只选择一张无 compute process、无非零任务占用的卡，
   显式设置单个 `CUDA_VISIBLE_DEVICES`；不得停止、迁移或干扰既有进程。
3. 在固定且可审计的 StateGuard3R/ReCal3R commit 上运行两张 Chateau 图片的 512
   smoke。必须验证 checkpoint/head 契约、全部关键输出 finite、`N-1 = 1` 次 calibrated
   update、输入/权重运行前后 hash、runtime/peak memory、进程退出和显存释放。
4. 两图完全通过后才扩到 4–8 帧，验证连续状态、trace 对齐、pose/residual 定义及输出
   数量；失败即维持 HOLD，不进入下一层。
5. 以上通过后，才允许唯一的 Gate 1 数据下载：7.69 MiB `fr1_xyz` RGB AVI，并先限制
   为约 10 帧、再至约 30 帧。它仅用于连续推理 smoke，不报告 ATE/RPE。
6. Gate 1 通过后，才允许唯一完整包 `fr1_desk.tgz`（328.07 MiB）；先做 8 帧 I/O，
   再做 30–50 个连续帧，最后才考虑完整 clean baseline。原始数据保持只读，不运行会
   大量复制图片的 TUM long preprocessing 脚本。
7. 在真实序列上划分 development/holdout，先冻结 window、epsilon、max-z、seed 和四方法
   阈值，再运行三类 corruption 的 formal detection，报告真实分组与总体 AUROC、F1、
   FPR、delay、timeline、runtime 和 memory；不得用 holdout 反复调参。
8. 只有真实 combined detector 满足预设 Go gate（包括 AUROC > 0.75、优于 random、
   不劣于最佳单信号、至少两类出现可重复 risk peak、误报不长期饱和）后，才重新评估
   是否进入最小 quarantine pilot。否则保持 HOLD 或给出有证据的 No-Go，并停止
   quarantine/rollback。

综上，当前最积极且严谨的结论是：软件与可复现管线已具备继续实验的基础，但真实
ReCal3R 核心证据为零。**决定维持 HOLD — evidence pending；立即禁止 TUM 数据推进和
quarantine，等待官方 checkpoint 与合规空闲 GPU 解阻。**
