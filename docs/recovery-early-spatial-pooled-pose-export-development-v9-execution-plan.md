# Recovery early-spatial-pooled pose export development v9：预注册计划

- 制定日期：2026-08-07
- 状态：**预注册，尚未实现或运行。**
- 长时目标：在 12--18 小时内，以正常终态完成所有适用 Gate；不因单次漂亮数值、v8 control 或基础设施错误提前宣布可行。
- 唯一目标：检验一个与 v8 科学上不同的 early spatial-latent export，能否在不下载数据、仅使用已有三个 development manifests 的条件下，同时满足因果/rollback 安全、normal-path byte equivalence、runtime `<=1.20` 和冻结跨条件质量门。任何硬门失败均诚实发布 v9 NO-GO。

## 1. 前置终态与唯一新假设

v1--v8 全部冻结。特别是 v8 的 special early **pose** token
`dec[0][:,0:1]` 路线因 Gate-B terminal-evidence/provenance hard stop 结束，不能换一个 run ID、
index、precision 或 launcher 后重试。

v9 只测试以下不同假设：**同一完整 current recurrent rollout 的 special pose token 可能已带有
脆弱的 pose/state 通路影响，但早层 spatial patch tokens 的无参数平均
`mean(dec[0][:,1:], dim=1, keepdim=True)` 保留当前帧几何语境；同一个冻结 pose head 对该 pooled
spatial latent 的读出可能在 detector alarm 后提供一个不同的 export。**

这不是 v8 的 layer-index 调整：v8 输入的是一个 special pose token `[:,0:1]`，v9 输入的是排除
该 pose token 后的整个 spatial token set 的 deterministic aggregate。它仍依赖 current image、current
rollout 与 committed pre-state，因此不得声称 latent-independent recovery。

它不读取 `prediction["camera_pose"]` numeric value、RGB tensor、pointmap、confidence、anchor、GT、
future frame、ORB/RANSAC、v1--v8 fallback 或 previous export。

## 2. 不可变 v9 mechanism

1. 维持 frozen Detector-v3、raw post-rollout health、current-frame complete rollback/structural witness 与 watchdog `8`。Detector 永远消费 raw prediction；export 不得反馈 model、detector、state、memory 或 later frame。
2. 每 frame 调用 pinned recurrent lighter body。**仅** candidate observer 存在时，`_recurrent_rollout(...)` 返回 `dec`、decoder length 已验证后、`update_mem(..., dec[-1][:,0:1])` 前，计算并 clone：

   ```python
   early_spatial_pose_token = dec[0][:, 1:].mean(dim=1, keepdim=True).detach().clone()
   ```

   该值必须是 floating、finite、shape `(1,1,768)` 且在 head device。always-control 不建 token/copy/observer，保持 legacy direct `to_cpu(res)`。
3. clear frame 原样导出 raw pose。alarm 先按 raw Detector-v3 decision 完成 rollback/witness；随后只将 already-captured v9 token 送入 pinned `DPTPts3dPose.pose_head`、official `postprocess_pose` 和 official camera decoder，并只替换 exported `camera_pose`。不得 retry/re-rollout、cache、copy/hold/motion fallback。
4. head 在 native model dtype 执行，已得到的 finite `(1,7)` output 在同一 device 升至 float64 后进入 official postprocess/camera path。export pose/camera 必须 finite，camera `(1,4,4)`，bottom row homogeneous，`|det(R)-1|<=1e-8` 与 `max|R^T R-I|<=1e-8`。任何异常为 `EARLY_SPATIAL_POOLED_POSE_UNAVAILABLE_FAIL_CLOSED`，不得回退 raw pose。
5. 每条 alarm 写 token class `early_spatial_patch_mean`、layer `0`、excluded index `0`、capture relation `after_rollout_before_update_mem`、spatial token count、shape/dtype/device/GPU digest、source hashes、postprocess mode、properness、raw/exported pose digest 与 rollback witness。连续 alarm 只用自身 current token。

## 3. 不可变数据与执行边界

- 仅读取 `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`。在六个 v9 output 都冻结前，禁止下载 dataset/weight/dependency、读取 GT/logical base/event labels 或运行 quality evaluator。
- 不修改 ReCal3R/checkpoint/raw RGB/GT/corruption manifests/Detector-v3 config/formal quality artifacts 或 v1--v8 frozen output/log。
- 新增独立 v9 primitive/exporter/runner/script；不得 import/call v1--v8 recovery runner/exporter、anchor/motion/ORB/pointmap solver。可复用 frozen low-level detector、structural witness、device utility。
- GPU 仅通过 existing `stateguard` tmux session 的 tracked, test-covered v9 dispatcher 启动；优先 GPU 2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`，每 run 两次 `>=12288 MiB` preflight。无需 GPU 完全空闲。

## 4. 单一 dispatch 协议

每个 authorized run ID 只能有一个 dispatcher owner。该同一 dispatcher 必须按下列不可分的顺序执行：

1. 在任何 GPU child 前验证 StateGuard3R/ReCal3R clean、committed components、new output/log/driver/result paths、fixed manifest/checkpoint/config，且记录 two GPU snapshots。
2. 创建 fresh tmux pane，建立 live `pipe-pane` 到 empty new transcript；随后在 transcript 写 `V9_FORWARD_DISPATCHED`，包含 run ID、quoted command hash 与 child PID，**再**启动 child。
3. 同一 dispatcher 收集 child stdout/stderr 到 main log，写 immediate postflight（exit code + GPU snapshot），验证 preflight/main/postflight/transcript/driver/result NUL-free 后才冻结全部为 `0444`；成功 output 必须 `0555/0444`。
4. 一旦写出 `V9_FORWARD_DISPATCHED`，该 ID 和 scientific configuration 永不重复。任何 pre-dispatch failure 也冻结为 failed authorization；v9 不设置事后 recovery-ID branch。candidate validator 必须检查恰有一个 dispatch marker、one output 和全套 canonical artifacts。

这样取消 shell/preflight 与 tmux dispatch 分离造成的模糊状态；不得人工操作 pane 或用事后 addendum 重新解释是否 forward。

## 5. Gate A：实现与因果合约

GPU 前必须全部通过：

- token-only synthetic identity/translation/rotation tests；empty spatial set、wrong shape、nonfinite spatial token、nonfinite/malformed head output、head-device mismatch、reflection/non-SO(3)/bad bottom row 全 fail closed；alarm failure 不返回/改变 raw pose；
- AST/signature tests 拒绝 raw camera-pose/RGB/pointmap/confidence/anchor/history/GT/future/detector input 与 decoder/export math 中 `.cpu()`；audit-only evidence transfer 如存在，必须在 full validation 后且明确区分；
- source mutations 把 `dec[0]` 改成 `dec[-1]`、`[:,1:]` 改成 `[:,0:1]`、mean 移至 `update_mem` 后、或移除 `keepdim=True` 均 fail；runner tests 证明 control 没有 clone/observer/finalize，candidate captures before update and finalizes after rollback before CPU transfer；
- actual checkpoint-loaded `DPTPts3dPose` CPU check、ReCal3R Torch targeted tests、full project CPU suite（`--basetemp` 在 repository `tmp/`）、compileall、diff check；code/tests/docs 分离提交，两个 worktree clean；
- dispatcher unit tests 覆盖 exact one-use lease、two snapshots、pipe-before-marker-before-child、no-forward preflight failure、nonzero child、missing/NUL/mismatched artifact 与 no manually reconstructed transcript。

任一失败是 `EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_IMPLEMENTATION_NO_GO`，不得启动 GPU。

## 6. Gate B：dynamic short circuit

1. 只运行一次 `recovery-early-spatial-pooled-pose-v9-dynamic-always-commit-0001`。四个 protected files 必须相对 frozen v1 dynamic baseline byte-identical，runtime/baseline `<=1.20`，并通过 v9 dispatcher validator。
2. 仅在第 1 步 PASS 后，运行一次 `recovery-early-spatial-pooled-pose-v9-dynamic-candidate-0001`。每个 real alarm 必须 rollback/full witness、`export_early_spatial_pooled_pose_token`、complete evidence、zero fallback，且 runtime/baseline `<=1.20`。
3. 任一失败即 `EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_AVAILABILITY_OR_RUNTIME_NO_GO`；不得重试、跑 wrong/low/GT evaluator，或调整 pool/index/precision/head/detector。

## 7. Gate C：冻结 matrix 与质量选择

仅 Gate B PASS 授权各一次：wrong control/candidate，随后 low control/candidate。六个 v9 outputs 完全冻结后，CPU evaluator 才可读取 logical-base GT，并用 frozen prefix `0--14` Sim(3)、tail `19--29` ATE/RPE 与 `(baseline-candidate)/baseline` 计算效果。

candidate-ready 需要：所有 control protected files byte-identical；所有 alarm complete；零 restore/token/watchdog/leakage/dispatcher failure；至少 2/3 条件同时 ATE 与 translation-RPE 为正；两项 median 均 `>=+5%`；candidate runtime median `<=1.20`；全部 provenance/freeze permissions 完整。PASS 只写未执行的 blind acquisition plan `EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_CANDIDATE_READY`；否则为 `EARLY_SPATIAL_POOLED_POSE_EXPORT_V9_FEASIBILITY_NO_GO` 并结束 v9。
