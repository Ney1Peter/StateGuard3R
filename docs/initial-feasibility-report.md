# StateGuard3R 初步可行性报告

- 日期：2026-08-01
- 证据截止时间：2026-08-01 23:40 CST
- StateGuard3R 证据快照：`e37a86afd56e99713acea10085cb8aed8546b16d`
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
- CroCo RoPE CUDA 扩展已用 CUDA 12.1 编译成功。runner 现会拒绝 baseline 目录外的
  扩展，并冻结实际加载 `.so` 的 9,523,120 B 大小和 SHA-256
  `3bd89991bcebb9501da085f4722c55fd5ef48aa38936608ef986072313d9aede`。这证明构建链与
  CPU import 可用。17:49 的 Gate 0 随后在 GPU 上完成真实 forward，并在运行记录中固定
  同一扩展 provenance。
- 外部 runner `scripts/run_recal3r_smoke.py` 的 CLI `--help` 已在 ReCal3R 独立
  解释器中通过。runner 已静态覆盖 checkpoint 文件名—输入尺寸—head 类型绑定、
  反序列化前 SHA-256 校验、模型模块来源、baseline/StateGuard3R 工作树状态、输入快照
  与运行后哈希复核、两图 `N-1` 次 calibrated update、输出有限性和轻量 provenance。
  02:38 的 pinned-API 审计发现并修复了对上游合法 `load_state_dict` override 的必现误拒绝；
  随后用真实固定类证明委托审计可捕获 `strict=False` 结果并完整恢复方法。runner 为本
  ReCal3R smoke 显式选择 `model_update_type=recal3r` 与 `beta_base=0.1`（上游 parser
  默认仍为 `cut3r`），并按官方 relpose launcher 的赋值语义同时冻结 model/config 两层的
  `entropy_eps=2e-14`、`entropy_head_reduce=mean`、
  `uncertainty_clamp_max=1.0` 和 `decay=0.95`。0001 暴露并修复了 pinned model 的真实
  head 字段读取问题；0002 随后让这些接口与守卫接受真实模型并完成 forward。
- 官方 CPU image loader 已加载 Chateau 两图，得到 `1x3x384x512`、值域
  `[-1, 1]` 的输入；加载前后源文件哈希不变。该结果只通过预处理子门槛。
- 2026-08-01 使用项目内 `TMPDIR` 和 `UV_CACHE_DIR` 实跑完整测试套件，最新明确隐藏
  CUDA 的复核结果为 `179 passed in 0.79s`。测试覆盖 corruption manifest、只读 replay、Health Ledger、
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
| `8f6360c`、`e2d5f86` | 修复 pinned model `load_state_dict` override 误拒绝并增加委托/旁路测试 |
| `59a1eac`、`8bf6154` | 冻结官方 ReCal3R runtime 参数、来源 provenance 与回归测试 |
| `2c44ea8`、`483ade1` | 冻结实际 cuRoPE binary provenance 与路径拒绝测试 |
| `1fb5369`、`5a8e609`、`ee5768c`、`5231580` | checkpoint/GPU 解阻、真实 pinned head 接口修复与测试、0001 失败登记 |

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
由单元测试覆盖，但本次 0003 原始命令没有 materialize 图像；源图哈希保持不变。下节的
独立 CPU 审计随后补齐真实 loader 后的像素 materialization 证据。

### 2.2 CPU-only 像素 materialization 审计

2026-08-01 02:49–02:50 CST 在干净提交 `483ade1` 上，以 ReCal3R 独立解释器、空
`CUDA_VISIBLE_DEVICES` 和官方 `load_images_for_eval(size=512, crop=True)` 对 0003 的
60 帧最终有序路径执行了实际 materialization。所有输入均为 `1x3x384x512 float32`、
值域 `[-1,1]`；60 个输出均使用独立 tensor clone，原 loader tensors 逐字节未变，两个
Chateau 源文件 SHA-256 前后不变，没有写 raster，且 `torch.cuda.is_initialized()` 为
false。

五个 normalized red occlusion 的实际空间差分与 floor/ceil 契约逐像素一致，`x1/y1` 为
exclusive：

| 帧 | 原点 | 实际 `(x0,y0,x1,y1)` | 改变的空间像素数 |
|---:|---|---|---:|
| 25 | `(0.25,0.25)` | `(128,96,384,288)` | 49,152 |
| 26 | `(0.29,0.275)` | `(148,105,405,298)` | 49,601 |
| 27 | `(0.33,0.30)` | `(168,115,425,308)` | 49,601 |
| 28 | `(0.37,0.325)` | `(189,124,446,317)` | 49,601 |
| 29 | `(0.41,0.35)` | `(209,134,466,327)` | 49,601 |

每帧矩形内均精确为 `[1,-1,-1]`，矩形外逐像素不变；左上角连续向右下移动。manifest、
materializer 源码与官方 loader 源码 SHA-256 分别为
`864133efe020a024674f49ca3c0eaaa5a6a84ee45422e492bc14bdaba11657f2`、
`e42852f87e704c5b9b90e12a9fa10506ec6f38290b9cd1a01d065e5dc77106c2` 和
`a2738085cdf0f713289a3a42dbd49a1f39325eb3ad590af65970fe4609209d96`。

路由断言也通过：low-overlap source indices 为 `50–54`，wrong-order 为
`[44,43,42,41,40]`。但 fixture 只重复两张图片，所以前者实际路径仅 3/5 改变，后者
4/5 改变；它证明路由和像素执行，不证明真实 low-overlap/时序破坏强度，也不是模型或
科研结果。

### 2.3 仅用于管线自检的数字

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

## 3. 真实 ReCal3R Gate 0 与剩余证据缺口

### 3.1 Gate 0 结果

用户从官方 README 的 Google Drive 入口取得 final checkpoint
`cut3r_512_dpt_4_64.pth`。文件为 3,173,761,006 bytes，首次项目内 SHA-256 为
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`，已设为只读；官方
没有公布 checksum，因此该哈希只能固定本地一致性，来源认证仍依赖用户声明的官方入口。

`GATE0-REAL-0001` 成功反序列化 checkpoint 且 missing/unexpected keys 为空，但 runner
错误地从 pinned 父类替换后的 `model.config` 读取外层 head 字段，在 model-to-CUDA 前
误拒绝。该失败已完整保留，不是 checkpoint 不兼容。`5a8e609` / `ee5768c` 修复并测试了
真实 `model.head_type`、`model.output_mode`、`model.pose_head_flag` 和 downstream pose
结构契约。

`GATE0-REAL-0002` 随后在干净提交 `5231580`、固定 ReCal3R commit、GPU 4 和同一输入/
checkpoint 哈希上成功完成。实测证据包括：

- 官方 512 DPT / `DPTPts3dPose` 接口与全部 state-dict keys 匹配；
- 实际 cuRoPE binary provenance 匹配，真实 CUDA forward 完成；
- 两帧恰好执行 `N-1=1` 次 calibrated update，trace steps 为 `[1]`；
- health、trajectory、prediction summary 均为两帧且全部已记录数值 finite；
- 推理段 1.052124 秒、1.900916 FPS，峰值 allocated 显存 6,362.670 MiB；约 28.995 秒
  端到端墙钟包含启动与权重加载，不能与 FPS scope 混写；
- 帧 1 的 uncertainty/reliability 为 0.797944/0.202056，`global_state_delta=1.665824`、
  pose jump 5.136780、geometric residual 0.0558443；
- checkpoint 和输入哈希运行后不变，进程退出后 GPU 4 回到 4 MiB、无 compute PID。

`update_magnitude`、candidate/final beta、local-memory delta 仍为 `null`，不得声称已获得。
两张 Chateau 图片也不是真实连续序列，其 pose jump 不能解释为精度、漂移或科研性能。
单靠这个两图结果尚不能证明多帧 clean state 行为、TUM clean baseline、真实三类
corruption 响应、独立 development/holdout、正式 AUROC/F1/FPR/delay，或支持
quarantine/rollback 的证据。

随后 `GATE0-STATE4-0001` 用 Chateau1、Chateau2、宽幅 `arch.jpg`、Chateau1 四个只读
引用完成机械 state smoke：恰好 3 次 update、trace `[1,2,3]`、四帧对齐且全部数值
finite；推理段 1.348210 秒，峰值 allocated 显存 6,367.173 MiB。它证明 recurrent
计数、trace、混合宽高比和输出对齐，但重复资产不构成真实连续 clean 序列证据。

### 3.2 Gate 状态

| Gate | 状态 | 证据与约束 |
|---|---|---|
| 官方来源、固定 commit、许可证与环境隔离 | **PASS** | 上游固定、CPU import、环境 freeze 与依赖检查已完成 |
| Corruption/Health/Detection 实现与单元测试 | **PASS（软件层）** | 179 tests passed；不等于模型或科研验证 |
| Synthetic v1 metadata/detection + CPU pixel materialization | **PASS（开发层，有限）** | 0003 可重复生成并通过 strict loader；独立审计证明五帧遮挡像素逐帧移动、源图只读；非 formal、非模型/真实实验 |
| 官方 512 checkpoint | **PASS（来源边界有限）** | 用户声明取自官方入口；大小/首次 SHA 已冻结并只读，官方无 checksum |
| Gate 0 单卡资源 | **PASS（当次快照）** | GPU 4 连续两次空闲并显式绑定；运行峰值 6,362.670 MiB，退出后释放 |
| Gate 0：Chateau 两图真实 ReCal3R smoke | **PASS** | 0002 完成真实 CUDA forward、一次 update、finite 轻量输出与完整 provenance |
| 4 帧机械 state smoke | **PASS（接口层）** | 3 次 update/trace 和四帧有限输出通过；重复资产，非真实连续场景 |
| Gate 1 AVI 下载与十帧 CPU 解码 | **PASS** | 官方 8,059,298-byte AVI 与连续索引 0–9 已校验；无 GT，不能用于 ATE/RPE |
| Gate 1 十帧 GPU smoke | **PASS** | 9 次 update/trace、十帧 finite 与真实 health signals 通过；仅约 0.3 秒 RGB |
| Gate 1 三十帧输入准备 | **PASS** | 0–9 原位引用且逐像素复核，只新增只读 10–29；30 帧发布后哈希复核通过 |
| Gate 1 三十帧 GPU smoke | **PASS** | 29 次 update/trace、30 帧 finite、真实 health、哈希和 GPU 释放均通过 |
| Gate 2 `fr1_desk` TGZ 与 raw tree | **PASS** | 唯一完整包已校验；1212 个 raw 文件逐哈希/格式/只读复核，无 partial 或额外下载 |
| Gate 2 连续八帧输入准备 | **PASS** | 最早有效 0-based RGB 数据索引 17–24（物理行 21–28）；depth/GT ≤20 ms，只引用 raw 文件 |
| Gate 2 连续八帧 I/O/forward | **PASS** | 7 次 update/trace、48 个 finite summaries、真实 health、哈希和 GPU 释放通过 |
| Gate 2 连续三十帧输入准备 | **PASS** | 索引 17–46；depth/GT 唯一关联，前八帧为精确前缀，只新增只读引用 manifest |
| Gate 2 连续三十帧 clean ledger | **PASS** | 29 次 update/trace、180 summaries finite；前八帧输出精确复现，GPU/PID 释放 |
| 三类真实输入 corruption 准备 | **PASS（输入层）** | 三个独立 v1 manifest、只引用 raw；strict replay 与 moving-occlusion CPU 像素审计通过 |
| Low-overlap proxy exploratory ledger | **PASS（响应层，有限）** | 30/29/180 finite；前十帧持久化 prediction summaries/trajectory 精确复现，health 仅 timestamp 不同；未测真实 overlap、非 detection |
| Dynamic/wrong-order exploratory ledger | **NEXT / SEQUENTIAL** | 只解锁 dynamic occlusion；通过后才运行 wrong-order，不并行跑门禁 |
| 正式 development/holdout detection | **LOCKED** | 尚无真实日志与冻结开发集配置，不能检验 AUROC > 0.75 等 Go 条件 |
| Phase 5 科研决策 | **HOLD — evidence pending** | 证据不足，既非 Go 也非方法 No-Go |
| Quarantine/rollback | **LOCKED；当前禁止进入** | 仅在真实 formal detection 满足预设 Go gate 后才可开始 |

checkpoint、GPU、两图 Gate 0、四帧机械 gate、十帧和 30 帧 GPU smoke 均已通过。
唯一的 `fr1_desk.tgz`、raw tree、八帧 forward、连续 30 帧 clean ledger、三类独立
corruption 输入和 low-overlap proxy exploratory ledger 也已通过各自门禁。当前只能
运行 dynamic-occlusion exploratory smoke；不得把不足一秒的 RGB 响应或单次 proxy
smoke 升级为 ATE/RPE、formal detection 或科研 Go。

### 3.3 严格解阻顺序

1. **已完成：**用户从官方入口取得 512 DPT 4–64 checkpoint；项目记录了来源声明、
   实际字节数和首次 SHA-256，并将文件设为只读。
2. **已完成：**紧邻运行前连续检查 GPU 并显式绑定单卡；用户允许在显存充足时共享，
   但本次实际使用的是无 compute PID 的 GPU 4，且未停止、迁移或干扰既有进程。
3. **已完成：**在固定且可审计的 StateGuard3R/ReCal3R commit 上完成两张 Chateau 图片
   的 512 smoke，验证了 checkpoint/head、finite 输出、`N-1=1` update、前后哈希、
   runtime/peak memory、进程退出和显存释放。
4. **已完成：**扩到 4 帧无下载 state smoke，验证了 update/trace、混合宽高比及输出
   数量；它是机械接口检查，不是真实连续场景证据。
5. **已完成：**Gate 1 下载、十帧 GPU smoke、30 帧增量输入与 30 帧 GPU smoke 均通过；
   30 帧 run 有 29 次 update、180 个 finite tensor summaries，并已验证进程/GPU 释放。
   AVI 仅用于连续推理，不报告 ATE/RPE。
6. **已完成：**唯一完整包 `fr1_desk.tgz`（328.07 MiB）的下载、raw 解压、连续八帧
   引用 manifest/forward、连续 30 帧引用 manifest 和 clean ledger 均已通过。下一步
   只使用相同 30 帧来源，不自动扩到 50 帧。原始数据保持只读，不运行会大量复制图片
   的 TUM long preprocessing 脚本。
7. **进行中：**共享 clean-source adapter 与三份单污染 v1 manifest 已冻结；low-overlap
   index/GT-pose proxy 已完成真实 ledger、双重独立审计和只读冻结。下一步只运行 dynamic
   occlusion，完整审计并提交后才解锁 wrong-order。三次均只是 exploratory Health
   Ledger，不在其上调 formal 阈值。
8. 在真实序列上划分 development/holdout，先冻结 window、epsilon、max-z、seed 和四方法
   阈值，再运行三类 corruption 的 formal detection，报告真实分组与总体 AUROC、F1、
   FPR、delay、timeline、runtime 和 memory；不得用 holdout 反复调参。
9. 只有真实 combined detector 满足预设 Go gate（包括 AUROC > 0.75、优于 random、
   不劣于最佳单信号、至少两类出现可重复 risk peak、误报不长期饱和）后，才重新评估
   是否进入最小 quarantine pilot。否则保持 HOLD 或给出有证据的 No-Go，并停止
   quarantine/rollback。

综上，当前最积极且严谨的结论是：软件与可复现管线已具备继续实验的基础，真实
ReCal3R 两图 Gate 0、四帧机械 gate、Gate 1 十帧/30 帧以及 Gate 2 八帧/30 帧 clean
smoke 已通过，三类 corruption 输入已冻结，low-overlap proxy 也获得一次真实模型响应，
但另外两类响应和 holdout 核心证据仍不足。**决定维持 HOLD — evidence pending；下一步
只运行 dynamic-occlusion exploratory smoke，50 帧、formal detection 与
quarantine/rollback 继续禁止。**
