# Recovery pre-rollout pose-query export v6：Gate A 审计

- 审计日期：2026-08-07
- Gate A 结论：**PASS；允许按预注册顺序创建一次新的 dynamic always-control。**
- 绑定 StateGuard3R：实现 `47419f8`、head/source hardening `77069d0`；测试
  `8fe2d86`、`783a9d3`、`66d404d`；精度说明 `ef9900f`。
- 绑定 ReCal3R：clean commit
  `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`；没有下载数据、权重或依赖，
  也没有启动新的 GPU run、读取 development GT、运行 wrong/low 或质量 evaluator。

## 固定 v6 行为

每个 current frame 在 ReCal3R `forward_recurrent_lighter` 的同构外部 loop 中，先取得
`global_img_feat_i`，再得到 `pose_retriever.inquire(global_img_feat_i, mem)`（frame 0 为
`pose_token`）。runner 立即 `detach().clone()` 这个 `(1,1,768)` token，之后才进入
`_recurrent_rollout` 和 `update_mem`。检测器始终消费 raw post-rollout prediction；报警时完整
rollback/witness 成功后，policy 才将这个已捕获 token 输入同一冻结 `DPTPts3dPose.pose_head` 和官方
`postprocess_pose`，并只替换 exported `camera_pose`。

decoder 的唯一数值精度绑定是：pose head 仍以原 model dtype 执行，已生成的 7 个 output 在**同一设备**
提升为 `float64` 后走原始官方 postprocess/camera decoder。此澄清记录于 `ef9900f`：binary32 quaternion
matrix 对有效旋转也可有约 `1.19e-7` 正交误差，无法满足已冻结 `1e-8` proper-SO(3) gate；它不改变 token、
pose-head 权重、pose source 或引入第二个估计器。alarm export 不读取 raw `camera_pose` numeric value，且无
copy/hold/motion fallback 或 retry。

## Source、因果与 fail-closed 合约

- `audit_pinned_pose_query_sources` 绑定 `model.py`、`dpt_head.py`、`postprocess.py` 的完整 SHA-256：
  `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1`、
  `c4f4b08c844b5fea47b67cedbcf4888719e8664f4b3b7e7ab6218e33cf63a66e`、
  `ee93ae7fa16897ed9163a21efe225f46193054b19bd7d316402545ed766133d7`。
- audit 只检查实际被 wrapper 复现的 `forward_recurrent_lighter` method span，要求 current global feature、
  pre-rollout `inquire`、rollout 和 `update_mem` 依序存在；篡改 query source 会 fail closed。
- 同时绑定 DPT 的 `DPTPts3dPose`、`PoseDecoder(hidden_size=in_dim)`、官方 raw pose token/head/postprocess/output
  path。runtime 要求 loaded `downstream_head` 精确为该 pinned type，`pose_head` callable 且 `pose_mode` 是三元组；
  任一 API/source 漂移在 GPU 前停止。
- decoder signature 只有 query、injected frozen torch/head/postprocess/mode/camera decoder。AST tests 拒绝
  raw pose、prediction、RGB、pointmap、confidence、anchor/history、detector、GT/future 和 `.cpu()`；alarm policy 的
  AST 证明它不从 `prediction` load 任何值。query 仍依赖 current global image feature 与 committed pre-state memory，
  因此不声称 current-image-independent。
- runner tests 证明 clone 早于 rollout/memory update，`observer.finalize` 早于 `to_cpu(exported)`，也不委托
  v3/v4/v5 recovery runner。shape/nonfinite query、nonfinite/malformed head output、reflection/non-SO(3) camera、
  缺失 DPT head/interface 全部 fail closed，且失败绝不会返回或修改 raw pose。

## 已执行验证

| 验证 | 结果 |
| --- | --- |
| ReCal3R Torch 环境 v6 targeted tests | `16 passed in 3.02s` |
| 真实 checkpoint-loaded ReCal3R `ARCroco3DStereo.downstream_head` + official postprocess/camera direct CPU check | pass；head type 是 pinned `DPTPts3dPose`，strict load 无 missing/unexpected key，float64 exported pose/camera，det error `0`，orthonormality error `2.22e-16`，均 `<=1e-8`；immutable log `logs/recovery-prerollout-pose-query-v6-gate-a-actual-model-cpu-0001.log` |
| project-local full CPU suite（`--basetemp` 位于 repository `tmp/`） | `495 passed, 10 skipped in 50.28s`；skip 全为 project `.venv` 未安装 Torch 的既有 v4/v5 与 v6 Torch modules，不是失败 |
| compileall（`src/stateguard3r` 与 scripts） | pass |
| `git diff --check` | pass |
| ReCal3R worktree/commit | clean；`466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |

## Gate B 授权与边界

Gate A 只授权一次新 ID
`recovery-prerollout-pose-query-v6-dynamic-always-commit-0001`。它必须在 tmux、GPU 2、两次各至少
12 GiB preflight 下运行，并通过相对 frozen v1 dynamic baseline 的四个 protected files byte-equivalence 与
runtime ratio `<=1.20`。只有 control PASS 后，才允许一次
`recovery-prerollout-pose-query-v6-dynamic-candidate-0001`。control/candidate 任一失败即 v6 NO-GO：不运行
wrong/low/GT evaluator，且不在同一 v6 上改 source、precision、decoder、detector 或阈值后重跑。
