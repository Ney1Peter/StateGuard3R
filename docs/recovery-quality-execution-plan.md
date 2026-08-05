# StateGuard3R Detector v3：独立恢复质量验证执行计划

- 制定并激活日期：2026-08-06
- 状态：**执行中**
- 长时目标窗口：预计 12--18 小时；不以时间耗尽为停止条件
- 前置结论：[Detector v3 正式 GO](audits/formal-v3-result.md)，以及
  [state-policy transaction 已验证](audits/recal3r-state-policy-v3-feasibility.md)
- 受保护 baseline：`baselines/ReCal3R` commit
  `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`

## 1. 明确目标、科学问题与唯一结束条件

### 1.1 目标

在**独立于 Detector v3 唯一盲测**的 TUM RGB-D+GT 序列上，完成一个预注册的、可重放的
恢复质量试验。固定既有 Detector v3 连续阈值、capture-timestamp hard channel、rising-edge
policy 与最大 hold 长度，比较：

1. 原始 ReCal3R baseline；
2. 外部 transactional wrapper 的 `always-commit` 等价 control；
3. 同一 wrapper 的 Detector-v3-alarm 驱动 hold/replay policy。

问题是：在错误输入后的正常帧上，policy 是否相对 baseline 降低相机轨迹误差，同时不在
clean control 上造成明显退化。它不是重训、阈值搜索或对既有 Detector v3 盲测反复调参。

### 1.2 完成状态

本计划只允许以下三种完成状态；任一种均为目标完成，不能因结果不理想而继续在 formal 数据上
调 policy。

1. **`RECOVERY_QUALITY_GO`**：所有完整性门禁通过，且预注册质量门槛全部通过；
2. **`RECOVERY_QUALITY_NO_GO`**：完整性门禁通过，但任一质量、clean 退化、运行时或 alarm/action
   门槛不通过；
3. **`RECOVERY_QUALITY_FEASIBILITY_NO_GO`**：在不改变 protected ReCal3R source 的条件下，无法
   得到有效 GT 对齐、可重放 alarm-policy run，或在独立 CPU/GPU 验证中证明这一点。

**硬停止规则：** formal input manifest、formal commitment 与首条 formal 模型 response 之间不得改
tracked code、阈值、policy、窗口、指标或数据清单。正式 run 出错可以仅以相同 hash 的输入与代码
恢复；已成功并冻结的 response 绝不覆盖、重跑或用于改参。

## 2. 边界、资源与数据拆分

### 2.1 不可变边界

- 不改 Detector v3、其 calibration/threshold，或已有 v1/v2/v3 formal evidence；不读取它们作为
  新的 quality test response。
- 不改 ReCal3R tracked source、checkpoint、官方 raw RGB/depth/GT，且不把 GT、depth、label、
  event 区间、source index 或未来帧传给 online detector/policy。
- GT 仅由独立的**后验** evaluator 使用，且 evaluator 与 alarm builder 的输入接口、测试和
  artifact 分离。
- 不下载新权重，不开启服务/端口，不停止其他用户进程。每次 GPU run 明确绑定一张启动前
  至少有 12 GiB 空闲显存的卡，batch=1。

### 2.2 数据角色与下载上限

| 角色 | 数据 | 用途 | 是否可用于本轮确认性结论 |
| --- | --- | --- | --- |
| development fixtures | 已解盲 `fr1_desk`、`fr2_desk` 与既有 manifests | CPU 指标、input/alarm pipeline、短 GPU trial | 否 |
| formal scene A | 已只读冻结且从未产生模型 response 的 `fr3_walking_static` | 正式质量评测 | 是 |
| formal scene B | 官方 `fr3_walking_xyz` | 正式质量评测 | 是，须先预检/下载/冻结 |

只允许新下载 scene B 的一个官方 archive（预期 527,550,055 B），且 archive+raw tree 总预算为
5 GiB。下载前必须检查许可证、HTTP metadata、已有文件/response、可用磁盘和 tar 安全性；
下载后 SHA-256、文件清单、RGB/depth/GT 时间关联和 `0444/0555` 权限都必须通过。若预检失败或
空间不足，本计划只可得 feasibility NO-GO，不能以 scene A 单场景冒充 multi-scene result。

## 3. 冻结的实验设计

### 3.1 每场景的四个输入条件

对每个 formal scene 以同一、预先确定的 30-frame RGB window 生成下列独立只读 manifests：

| ID | 输入条件 | 目的 |
| --- | --- | --- |
| `clean` | 原始连续 RGB | false-hold / clean-degradation control |
| `dynamic` | 固定的 in-memory foreground occlusion，frames 15--18 | 检验连续 channel 对短动态污染后的恢复 |
| `wrong-order` | 固定 reverse permutation `[18,17,16,15]`，保留每张 RGB 的真实 capture timestamp | 检验 v3 hard timestamp channel 与后续恢复 |
| `low-overlap` | 固定、同场景、与 base window 不重叠的 donor frames 15--18 | 检验 visual-continuity channel 与后续恢复 |

窗口选择、donor 选择、关联容忍度（RGB--GT/depth 均不超过 20 ms）、transform 参数与每帧
**评估 GT binding** 都由 deterministic CPU builder 写入 manifest；不存在人工挑选“效果好”的
窗口。所有 corruption 的 post-event evaluation tail 固定为 frames 19--29。模型看到的只是
manifest 的图片/像素变换和 RGB capture-timestamp sidecar；评价 GT 始终是该 logical base frame 的
GT，即使 low-overlap 的 model image 是 donor。

### 3.2 Detector-policy 的可审计 shadow replay

每个 input 首先运行原始 baseline，产出 health/timestamp sidecar。独立 alarm builder 从该 run 的
**逐帧、因果** health fields、同一 manifest 的 RGB capture timestamps及已有冻结 v3 calibration
生成 `hybrid_alarm`；它拒绝 GT、depth、labels、event metadata、source index 和未来字段。
随后 policy run 只读取该 immutable alarm artifact 的 frame `t-1` decision，并在 frame `t` 作
hold/replay。这个两次 forward 的设计是离线的 causal shadow replay，不是单路径实时部署证明；
最终报告必须明确该限制。

为每个输入执行恰好三条 GPU forward：baseline、always-commit、detector-policy。always-commit
必须与 baseline 在 checkpoint audit、health、prediction summary、trajectory 上字节一致；否则该
input 的结果无效，不计算 quality effect。detector policy 必须记录每个 alarm、hold、replay、
state digest 和 dropped transaction；wrong-order formal input 必须至少产生一个 rising-edge alarm
及实际 hold/replay。其它条件的无 alarm 是正式观察结果，不会触发调参。

### 3.3 后验 ATE/RPE 指标

每条 trajectory 的 predicted camera centers 与 manifest 的 GT centers 在 clean prefix frames
0--14 上估计一个 deterministic Umeyama Sim(3)（不反射、固定退化拒绝规则）。该 alignment
固定后，计算 tail frames 19--29：

- `ATE_RMSE_m`：对齐后的中心 Euclidean RMSE；
- `RPE_translation_RMSE_m`：相隔一帧的相对平移 RMSE；
- `RPE_rotation_RMSE_deg`：相隔一帧的相对旋转角 RMSE；
- runtime、peak allocated GPU memory、alarm/hold/replay/restore-failure counts。

alignment 只看 prefix，绝不使用 tail 来重新拟合从而掩盖污染后的漂移。evaluator 还报告全序列
描述性数值，但 gate 只使用上述 tail 指标。synthetic unit tests 必须证明 Sim(3) invariance、
frame/GT 绑定、prefix-only 行为、错误四元数/退化/缺帧拒绝和 no-GT-in-alarm-interface。

### 3.4 预注册 GO/NO-GO 门槛

质量主分析包含两个场景的三个 corruption（共六个配对 event trials），每 trial 的 effect 定义为
`1 - policy_metric / baseline_metric`；零 baseline 指标显式拒绝而非除零。

`RECOVERY_QUALITY_GO` 必须同时满足：

1. 所有 provenance、permissions、baseline/always-commit byte-equivalence、state transaction、
   GT binding 和 CPU evaluator checks PASS；所有 six event trials 成功且无 restore failure；
2. 六 trial 的 median tail ATE effect **>= 10%**，median tail translation-RPE effect **>= 10%**，
   并且至少 4/6 trial 的 ATE effect 为正；
3. 以固定 seed `20260806`、10,000 次 scene-stratified paired bootstrap 得到的两项 median effect
   95% percentile lower bound 都大于 0；
4. 两个 clean trial 的 ATE 与 translation-RPE 均不恶化超过 5%，且 policy median runtime 增幅不
   超过 20%。

任何一项不满足即 `RECOVERY_QUALITY_NO_GO`。这会是有价值、可审计的结论；禁止事后缩小指标、
只报告有利 corruption，或将单一场景/单指标写成质量改善。

## 4. 门禁式执行步骤

### Gate A — 设计、实现与 CPU 证明

1. 审计现有 runner、trajectory convention、GT timestamp association、现有 TUM raw assets和资源；
   记录 f3 static 仍为 no-response data。
2. 新增 deterministic quality input builder/validator、shadow alarm builder、GT evaluator、
   formal orchestrator 和针对每个拒绝路径的 CPU tests。
3. 只在已解盲 f1/f2 fixtures 上运行 input->alarm->metric CPU round trip；不运行 scene A/B GPU
   forward。执行相关 tests、完整 test suite、`compileall` 与 `git diff --check`，并把代码、测试、
   docs 分开提交。

**Gate A PASS：** 所有新增单元测试与既有 suite PASS，正式 protocol/plan 已提交，且 StateGuard3R
与 ReCal3R tracked worktrees clean。

### Gate B — scene B acquisition 与 formal input freeze

1. 仅在 Gate A 后预检并原子下载/验证 scene B；不下载第二个 archive。
2. 在不启动 CUDA、不读取/产生 scene A/B model response 的条件下，为两 scene 构造并独立验证
   全部 8 个 input manifests；原子发布且冻结。
3. 将 protocol、code commit、raw/archive hashes、input manifests、calibration manifest、
   checkpoint hash、GPU rule、metric gate 和精确 run inventory 绑定成 quality commitment。

**Gate B PASS：** 两 scene 原始数据与所有 manifests/validators PASS、只读；commitment 已冻结。

### Gate C — 单次正式运行与评测

1. 对每个 frozen input，串行运行 baseline -> alarm builder -> always-commit -> detector-policy；
   每次 GPU launch 记录 GPU UUID、free memory、PID、命令、日志、开始/结束时间和退出码，成功
   artifact 立即只读冻结。
2. 每个 input 的三条 forward 完成后才允许 CPU evaluator 读取相应 trajectories；不因中间指标
   选择性跳过正式 inventory。
3. 全部 24 forward 与 8 evaluators 成功/冻结后，仅运行一次 aggregation，它会写 attempt seal 并
   生成 `RECOVERY_QUALITY_GO` 或 `RECOVERY_QUALITY_NO_GO`。

### Gate D — 最终审计与总结

1. 复核 raw/final artifact SHA-256、权限、attempt seal、worktrees、GPU/PID/端口和磁盘；保留
   失败 output/log，不覆盖也不删除。
2. 运行 full CPU suite、`compileall`、`git diff --check`；更新 run registry。
3. 写入 `docs/audits/recovery-quality-v1-result.md`，明确有效结论、所有数值、失败项、范围和
   shadow-replay 限制；只在所有门禁达成后结束本长时目标。

## 5. 当前执行指针

当前处于 **Gate A / step 1**：只读审计已确认 project disk 仍有约 146 GiB 可用，GPU 0/2/3 各有
超过 12 GiB 空闲；`fr1_desk`/`fr2_desk` 为已解盲 fixtures，`fr3_walking_static` 已冻结且其
acquisition audit 明确记录没有 ReCal3R response。尚未读取或产生本计划 scene A/B 的模型 response。
