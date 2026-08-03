# StateGuard3R Detector v3：持续执行计划

- 制定日期：2026-08-03
- 状态：**已完成：`V3_GO_AND_STATE_POLICY_VALIDATED`（不含 recovery-quality claim）**
- 上位计划：[detector-v3-execution-plan.md](detector-v3-execution-plan.md)
- 正式盲测协议：[detection-formal-v3.md](protocols/detection-formal-v3.md)

## 1. 明确目标与唯一完成条件

本轮工作的明确目标是完成 Detector v3 的**一次、独立、可审计的正式盲测**，并只在该盲测
结论为 `DETECTOR_V3_GO` 时完成预注册的真实 state-policy feasibility 验证。

最终只允许以下三种受证据支持的结束状态：

1. `V3_GO_AND_STATE_POLICY_VALIDATED`：检测门槛和 state-policy 的真实 state/memory commit
   证据均通过；
2. `DETECTOR_V3_NO_GO`：唯一盲测的任一预注册检测门槛不通过；
3. `STATE_POLICY_FEASIBILITY_NO_GO`：Detector v3 已 GO，但不改受保护 ReCal3R source 的前提下
   无法取得可信的 state-policy 干预证据。

开发阶段脚本异常、暂时没有足够 GPU、下载恢复、或单项 CPU 测试失败不是停止条件；应记录原因、
修复或等待资源后从相应门禁继续。不得以已解盲 development 结果替代正式结论，不得在唯一盲测
response 上调参、改阈值或重跑。

## 2. 当前冻结起点

- v1/v2/formal/quarantine 证据与 ReCal3R tracked source 都是受保护对象，绝不修改、移动、覆盖
  或重评。
- v3 CPU detector、输入构造/验证器、正式编排器和正式协议已完成并经相关 CPU 测试验证。
- 唯一允许的新场景已预检并下载：官方 TUM `rgbd_dataset_freiburg2_desk`。archive 为只读的
  `1,893,351,095` bytes；不下载第二个 archive、权重或数据集。
- 下载在发布前退出，留下一个完整、只读的 extraction staging tree。恢复必须重新验证该 tree 后
  原子发布，不能重下、不能删除 staging、不能读取任何 ReCal3R response。
- 只有当单张 GPU 有至少 12 GiB 空闲显存时才运行 GPU stage；共享 GPU 可以使用，但必须
  `CUDA_VISIBLE_DEVICES=<id>`、batch=1，且不干预其他用户的进程。

## 3. 按门禁执行的工作流

### A. 恢复并冻结唯一原始场景（CPU，当前步骤）

1. 为 acquisition 脚本的可恢复 staging 发布路径补充最小单元测试：
   - 一个完整 staging + 完整 archive 时，重新计算 tree manifest、检查 5 GiB 总预算、只读冻结、
     原子移动内层 raw tree，并写入 `staging_recovery: true` acquisition manifest；
   - 出现两个 staging root 时明确拒绝，绝不猜测或删除任一个。
2. 运行 acquisition 相关测试、字节码编译和 diff whitespace 检查；以单独修复提交保存。
3. CPU-only 执行：

   ```bash
   CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/tmp" .venv/bin/python \
     scripts/acquire_tum_v3_holdout.py acquire \
     outputs/formal-v3-data-preflight-0001
   ```

4. 只读核验 final raw root、archive SHA-256、manifest、`0444/0555` 权限、`.part` 与 staging 的
   缺失状态，且确认仍未产生模型 response。

**门禁 A：** `formal-v3-data-0001/acquisition.json` 状态为 PASS；archive + raw tree 不超过 5 GiB；
无 `.part`/staging；所有 raw artifact 只读。

### B. 构造并验证正式盲测输入（CPU）

1. 在不启动 CUDA、不开 ReCal3R forward 的条件下，构造三类受控 corruption 的正式输入：

   ```bash
   CUDA_VISIBLE_DEVICES='' .venv/bin/python \
     scripts/prepare_formal_detection_v3_inputs.py \
     outputs/formal-v3-inputs-0001
   ```

2. 独立 validator 重放 `rgb.txt` 的 capture timestamp provenance 和 timestamp sidecar：

   ```bash
   CUDA_VISIBLE_DEVICES='' .venv/bin/python \
     scripts/validate_formal_detection_v3_inputs.py \
     outputs/formal-v3-inputs-0001 \
     outputs/formal-v3-input-validation-0001
   ```

3. 核验 validator 没有把 GT、depth、source index、corruption label、event mask 或 future frame
   引入 online channel。clean prefix（0--14）不得有 timestamp alarm；low-overlap 在替换边界可能
   有真实 capture-timestamp 下降，必须如实 attribution，不能误当作 builder 失败。

**门禁 B：** 输入和 validation outputs 均原子发布并冻结；所有 CPU validation checks PASS；没有
新数据下载或任何 ReCal3R response。

### C. 正式承诺、GPU preflight 与已解盲开发运行

1. 将 protocol、输入 manifest 和 validator output 绑定为唯一 commitment：

   ```bash
   .venv/bin/python scripts/run_formal_detection_pilot_v3.py commit \
     outputs/formal-v3-inputs-0001 outputs/formal-v3-commit-0001 \
     --protocol docs/protocols/detection-formal-v3.md \
     --validation-dir outputs/formal-v3-input-validation-0001
   ```

2. 在每次 GPU forward 前记录 GPU UUID、空闲显存、PID owner/command、启动命令、日志绝对路径、
   输出目录与时间；选一张空闲 >=12 GiB 的 GPU，不停止或清理任何非本任务进程。
3. 运行 protocol 已披露的六条 development forwards 和固定 calibration。每条结束后确认本任务
   PID 退出、显存释放、输出只读冻结。只允许 protocol 指定的 development result 参与 calibration。

**门禁 C：** commitment validator 通过；开发/校准 artifact 完整；模型输入、checkpoint 和 v2
continuous channel 等价性仍成立。commitment 后至 formal evaluation 前不得改 tracked code、阈值、
protocol 或输入。

### D. 单次盲测与一次 evaluation

1. 严格按 `blind-dynamic -> blind-wrong-order -> blind-low-overlap` 运行三条 blind forwards。
2. 三条都成功退出并由 launcher 冻结前，禁止读取、hash、parse 任何单条 blind response、main log
   或派生指标；只有运行状态/PID/退出码可以观察。
3. 三条均冻结后，运行唯一的 `evaluate`。attempt seal 必须阻止后续评测；输出包含 metrics、
   attribution、timeline、manifest hash、resource record 与 `DETECTOR_V3_GO`/`DETECTOR_V3_NO_GO`。

**门禁 D：** 全部 formal thresholds 同时满足才为 GO：macro-AUROC >0.75、pooled FPR <=0.20、
clean-prefix FPR <=0.15、FP streak <=3、三类均 event 内检出且 wrong-order delay <=1，并且 provenance
与运行时完整性检查全过。任一失败即正式 NO-GO，禁止以该盲测数据继续调参。

### E. 条件式真实 state-policy 验证或 NO-GO 收尾

- 若 D 为 NO-GO：发布固定的失败门槛、证据路径和不可重跑说明，进入 F。
- 若 D 为 GO：按已预注册 wrapper protocol，用相同冻结输入进行 baseline 与 guarded alternate
  forwards；证明至少一个报警 state/memory update 被 hold/delay，未报警 clean control 不触发，且
  transaction 的 state digest、commit timeline、输入/checkpoint hash、prediction/trajectory 和
  health provenance 都可重放。无独立质量 protocol 时，结论只能是
  `STATE_POLICY_VALIDATED_NO_RECOVERY_QUALITY_CLAIM`，不得声称几何恢复质量。

**门禁 E：** 输出能证实真实 commit policy，或给出 source-hash-bound、可复现的 feasibility NO-GO。

### F. 最终审计与资源收尾

1. 运行相关/full CPU tests、`compileall`、`git diff --check` 和 evidence checks。
2. 更新 formal result、state-policy result、run registry 与本计划状态；代码、测试、文档各自独立
   提交，outputs/data/logs 不进 Git。
3. 复核 StateGuard3R/ReCal3R worktree、正式 manifest 权限/hash、GPU 显存、本任务 PID 和监听
   端口。只清理经明确列出的、本任务可再生临时文件；不删除数据或证据。

## 4. 执行记录格式

每项实际运行都应在其 output/log manifest 中记录：准确命令、UTC/本地开始结束时间、cwd、git HEAD、
输入/配置/checkpoint hash、GPU UUID/显存/PID（若使用 GPU）、退出码、产物绝对路径和只读权限。
任何恢复或重试必须说明它没有读取 blind response，也没有改变 commitment。

## 5. 当前执行指针

全部门禁已经完成：A/B 的唯一场景与 CPU provenance validation 均 PASS；C 的 commitment、六条
development forward 和 calibration 已冻结；D 的一次 blind evaluation 为 `DETECTOR_V3_GO`；E 的
frozen-detector-triggered transaction validator 为 PASS。最终审计、提交和资源核对记录于
`docs/audits/formal-v3-result.md` 与 `docs/audits/recal3r-state-policy-v3-feasibility.md`。结论不包含
ATE/RPE、几何恢复或 recovery-quality claim。
