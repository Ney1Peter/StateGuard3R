# Recovery early-decoder pose-token export development v8：冻结早层 latent 计划

- 制定日期：2026-08-07
- 状态：**预注册，尚未实现/运行**
- 长时目标：在 12--18 小时内完成所有适用 Gate 的正常终态；不因漂亮的单次数值、历史 v6 结果或运行耗时提前宣布可行。
- 唯一目标：检验 `detector-v3-incremental-early-decoder-pose-export` 能否在不下载任何数据、只用既有三个
  development manifests 的条件下，同时满足 rollback/因果安全、normal-path byte equivalence、runtime `<=1.20`
  和冻结跨条件质量门；任一硬门失败即诚实发布 v8 NO-GO。

## 1. 前置终态与唯一新假设

v1--v6 均已冻结。特别是 v6 的 pre-rollout pose-retriever query 在 dynamic alarm availability 和 runtime 上通过，
但 wrong always-control runtime `1.2380726268743498x` 超过 `1.20`，因此 v6 不得再调 source/precision/detector 或
重跑 matrix。未执行的 v7 tmux-only wrapper 没有提出新的 recovery signal，且相同 v6 scientific components 已被
v6 wrong normal-path gate 排除；它不构成本计划的候选。

v8 只测试一个不同的 latent hypothesis：**current recurrent rollout 的 final decoder pose token 可能受 corrupted
late-layer/state-update path 影响，但 complete rollout 的第 0 个 decoder pose token `dec[0][:,0:1]` 是较早的
current-frame latent；在 detector alarm 后 rollback complete pre-state，冻结 pose head 将此 captured early token
解码，可能提供不同于 raw final pose 和 v6 pre-rollout query 的 export。**

这不是 current-image-independent：early token 仍由 current image feature、current rollout 和 committed pre-state
state/memory 形成。它不读取 `prediction["camera_pose"]` numeric value、RGB tensor、pointmap、confidence、anchor、GT、
future frame、ORB/RANSAC 或 v1--v6 fallback。它固定只用 one `(1,1,768)` early decoder token、one frozen 768-to-7
pose head、official postprocess 和 camera validity check。

## 2. 不可变 v8 mechanism

1. 保留 frozen Detector-v3、raw post-rollout health、current-frame complete rollback/structural witness、watchdog `8`。
   detector 永远消费 raw prediction；export 永远不反馈 model/detector/state/memory/later frame。
2. 每个 frame 仍调用 official pose retriever 和 pinned recurrent lighter body。仅在 candidate observer 存在时，
   `_recurrent_rollout(...)` 返回 `dec` 后、`update_mem(..., dec[-1][:,0:1])` 前立即 clone
   `dec[0][:,0:1]` 为 temporary external early token。always-control 不建立任何 v8 token/copy/observer，使用
   legacy raw `to_cpu(res)` transfer path。
3. clear frame commit/export raw `camera_pose`。alarm frame 先依 frozen raw Detector-v3 decision 做 full rollback；
   随后只将 already-captured early token 输入 pinned `model.downstream_head.pose_head`、official
   `postprocess_pose` 和 `pose_encoding_to_camera`，并只替换 exported prediction 的 `camera_pose`。无 raw-pose
   numeric input、copy/hold/motion fallback、candidate retry 或 token cache。
4. token shape `(1,1,768)`、floating、finite，且在 head device。head raw output 必须 `(1,7)`、finite；head 在其
   model dtype 执行，随后已得 7D output 在**同一 GPU device**提升为 float64 并进入 official postprocess/camera
   decoder，使 frozen proper-SO(3) check 可达 `1e-8`。这不是第二个 pose estimator。exported pose/camera 必须
   finite；camera shape `(1,4,4)`、homogeneous bottom row、`|det(R)-1|<=1e-8`、
   `max|R^T R-I|<=1e-8`。任一异常为 `EARLY_DECODER_POSE_UNAVAILABLE_FAIL_CLOSED`，不得导出 raw pose。
5. 每条 alarm 写 early-layer index `0`、capture relation (`after_rollout_before_update_mem`)、shape/dtype/device/digest、
   source hashes、postprocess mode、properness、raw/exported pose digest 与 rollback witness。连续 alarm 每帧仅使用
   own current early token，不缓存或传播 export。

## 3. source、因果和 normal-path 审计

Gate A 必须绑定 pinned ReCal3R source SHA-256，证明：

- `forward_recurrent_lighter` 以 `_recurrent_rollout(...)` 得到 `dec`，在 `out_pose_feat_i = dec[-1][:,0:1]` 和
  `update_mem` 前存在 `dec[0]`；v8 capture 恰在这两个 source events 之间，不得换成 `dec[-1]`、raw camera pose 或
  post-state memory。
- DPT official raw path 是 `pose_token=x[-1][:,0] -> pose_head -> postprocess_pose -> camera_pose`；v8 仅替换
  其 token source 为 fixed early decoder token。DPT type/head/mode/interface/source hash 漂移 fail closed。
- runner control AST/path 明确没有 v8 token clone/observer finalize；candidate clone occurs after rollout before
  update, export before `to_cpu`，并且 no v3/v4/v5/v6 recovery runner/exporter delegation。
- early decoder token 与 raw pose numeric value 不是同一 input，但由 same model computation graph 得到；所有文档
  保留这个 causal caveat，不声称 latent-independent recovery。

## 4. 不可变边界和执行证据

- 仅读取 `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`；禁止下载
  dataset/weight/dependency、blind input 或读取 GT，直到 six v8 outputs 均冻结。
- 不改 ReCal3R/checkpoint/raw RGB/GT/corruption manifests/Detector-v3 config/formal quality artifacts；不改任何
  v1--v6 frozen output/log。
- 新增 independent v8 primitive/export policy/recurrent runner/script；不得 import/call any previous recovery runner,
  motion/anchor/ORB/pointmap solver or v6 pre-rollout exporter. 可复用 frozen low-level detector、structural witness 与
  non-policy device utilities。
- GPU 仅在 existing `stateguard` tmux session 发起，优先 GPU 2 UUID
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`；每 run exact two preflights `>=12 GiB`。每个 output/log/pane ID
  one-use；preflight/main/postflight/tmux transcript all NUL-free `0444`、output root `0555`/files `0444`。tmux
  transcript 必须在 command dispatch 前由 live `pipe-pane` 建立，不能事后 reconstruction。

## 5. Gate A：实现、source 与因果合约

GPU 前必须通过：

- synthetic early token identity/translation/rotation and pinned head/postprocess/camera tests；shape/nonfinite token、
  nonfinite/malformed head output、malformed/reflection/non-SO3 camera fail closed；clear only raw export、alarm only early
  token export、failure never returns/mutates raw pose；
- AST/signature test拒绝 RGB/pointmap/confidence/anchor/history/GT/future/detector/raw camera-pose input and `.cpu()` in
  decoder/export；source mutation tests confirm index 0 / after-rollout-before-update order；
- runner tests prove always-control bypasses capture/observer, candidate clone relation, rollback before export, finalize
  before CPU transfer, detector raw-health and watchdog/reset/no-leakage; actual checkpoint-loaded DPT head CPU check；
- ReCal3R Torch targeted tests、project full CPU suite (`--basetemp` inside repository `tmp/`)、compileall、diff check；
  code/test/docs separate commits；StateGuard3R/ReCal3R clean。

任一 Gate A failure 是 `EARLY_DECODER_POSE_EXPORT_V8_IMPLEMENTATION_NO_GO`，不得启动 GPU。

## 6. Gate B：dynamic short circuit

1. Run once `recovery-early-decoder-pose-v8-dynamic-always-commit-0001` and compare four protected files to frozen v1
   dynamic baseline; runtime/baseline must be `<=1.20`.
2. Only after 1 PASS, run once
   `recovery-early-decoder-pose-v8-dynamic-candidate-0001`. Every real alarm must rollback/full witness/export
   `export_early_decoder_pose_token`/complete no-fallback evidence, and runtime/baseline `<=1.20`.
3. Any failure is `EARLY_DECODER_POSE_EXPORT_V8_AVAILABILITY_OR_RUNTIME_NO_GO`; do not run wrong/low/GT evaluator or change
   early index/precision/decoder/detector then retry this v8.

## 7. Gate C：冻结 matrix 和质量选择

Only Gate B PASS authorizes each once, in this order: wrong always-control then candidate, low always-control then candidate.
All six v8 outputs must freeze before CPU evaluator reads logical-base GT. It uses frozen prefix 0--14 Sim(3), tail 19--29
ATE/RPE and `(baseline-candidate)/baseline`. Candidate-ready needs all controls byte-identical, all alarms complete,
zero restore/query/watchdog/leakage failure, at least 2/3 conditions where ATE and translation-RPE both positive, both medians
`>=+5%`, candidate runtime median `<=1.20`, and complete provenance/freeze permissions. PASS writes only an unexecuted blind
acquisition plan (`EARLY_DECODER_POSE_EXPORT_V8_CANDIDATE_READY`); otherwise
`EARLY_DECODER_POSE_EXPORT_V8_FEASIBILITY_NO_GO` and v8 stops without matrix retuning.
