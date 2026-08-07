# Recovery pre-rollout pose-query export development v6：冻结 latent pose prior 计划

- 制定日期：2026-08-07
- 状态：**已完成，`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_PROVENANCE_NO_GO`；不得继续 v6 candidate、wrong/low 或 GT quality 路径**
- 长时目标：在 12--18 小时内，以所有预注册 Gate 的正常终态为止；不因单次好看数值、时间或日志不便提前宣布可行。
- 唯一目标：检验 `detector-v3-incremental-prerollout-pose-query-export` 能否在不下载数据、只使用既有三个 development manifests 的条件下，同时满足 rollback/因果安全、正常路径逐字节等价、`<=1.20` runtime 和冻结跨条件质量门；任一硬门失败即诚实发布 v6 NO-GO。

## 1. 前置结论与唯一新假设

v3 safe-anchor SE(3) motion export 满足 runtime 但在 dynamic/wrong 质量失败；v4 anchor RGB--3D registration 在 six
alarms 几何可用但 runtime `4.6307x`；v5 GPU dense self/cross pointmap consensus 同样在 six alarms 可用但 runtime
`1.41137x`。这些版本已冻结；不得调整其 fallback、anchor、ORB、lattice、weight、IRLS、threshold 或 output 后重跑。

v6 只检验一个机制不同的新假设：**ReCal3R 在 current recurrent rollout 之前，以 current global image feature 与已提交
pre-state memory 形成 `pose_retriever.inquire(...)` 的 pose-query token；post-rollout decoder token 生成的 raw pose 可能受
corrupted update 路径损害，而这个 pre-rollout query token 经同一冻结 pose head 解码，仍可提供更稳定的 current-frame pose
prior。** alarm 时导出这一 token 的 pose，而不是外部 motion、anchor registration 或 pointmap fitting。

它不宣称 current image independent：query 明确包含 current `global_img_feat_i`，而且它是 same model latent prior；科学表述
仅为“pre-rollout latent pose query vs post-rollout pose token”。它不读取 `prediction["camera_pose"]` numeric value，也不访问
RGB tensor、pointmap、confidence、history anchor、GT、future frame、ORB、RANSAC 或 v5 3D solver。其额外数值工作固定为一次
`768 -> 7` pose-head MLP、pinned postprocess 和 finite/proper-camera check，故与 v5 dense IRLS mechanism 实质不同。

## 2. 不可变 v6 mechanism

1. 保留 frozen Detector-v3、current-frame complete pre-state rollback、structural witness、observed raw-health ledger 和
   watchdog `8`。detector 永远消费 raw post-rollout prediction；export 绝不反馈 model、detector、state、memory 或后续帧。
2. 在每个 frame 的 existing official lighter body 中，保持 source order：得到 `global_img_feat_i` 后、调用
   `_recurrent_rollout(...)` 前，生成 `pose_query_t = pose_retriever.inquire(global_img_feat_i, mem)`（frame 0 固定
   `pose_token`）。runner 立即 clone 这个 token 作为 temporary external value；它不能写回 `mem`、`state_feat` 或 model。
3. clear frame 正常 commit 与导出 raw `camera_pose`。alarm frame 先按 frozen Detector-v3 的 raw decision 完成 full rollback，
   然后只对该 frame captured `pose_query_t[:,0]` 调用 pinned
   `model.downstream_head.pose_head(...)`，再调用 pinned `dust3r.heads.postprocess.postprocess_pose(..., pose_mode)`，写回
   exported prediction 的唯一字段 `camera_pose`。没有 raw pose numerical input、copy/hold/motion fallback 或 candidate retry。
4. token 必须 shape `(1,1,768)`（以 pinned model audit 的 decoder dimension `768` 绑定）、finite、与 pose head 同 device；
   pose-head output/postprocess result 必须 shape `(1,7)` 且 finite。冻结 pose head 仍在其原 model dtype 上执行；仅将其已得的
   7 个输出在同一 device 确定性提升为 `float64` 后，调用同一 pinned postprocess/camera decoder。这是为使 quaternion 到 matrix
   的数值 proper-SO(3) 检查达到下述已冻结 `1e-8` 精度，不改变 token、head 权重或姿态估计机制，也不引入第二个估计器。对 exported encoding 调用 pinned
   `pose_encoding_to_camera`，结果必须 finite homogeneous 4 x 4 proper SO(3)（`|det(R)-1|<=1e-8`、
   `max|R^T R-I|<=1e-8`）。任一异常为 `PRE_ROLLOUT_POSE_QUERY_UNAVAILABLE_FAIL_CLOSED`，不得导出 raw pose。
5. 每条 alarm transaction 写 query source (`pose_retriever.inquire` 或 frame-0 `pose_token`)、token shape/dtype/device digest、
   pinned source hashes、postprocess mode、camera properness、raw/exported pose digests 与 rollback witness。连续 alarm 每帧从
   current query independently decode；不缓存或传播 prior output/token。

## 3. Pinned source 与因果审计

Gate A 必须绑定 ReCal3R source hashes 并验证下列拓扑：

- `model.py` 的 `pose_retriever.inquire(global_img_feat_i, mem)` 发生在 `_recurrent_rollout(...)` 之前；`mem` 的
  `update_mem(..., out_pose_feat_i)` 在 rollout/decoder output 之后；
- `dpt_head.py` 的 official raw pose path 是 `pose_token=x[-1][:,0] -> pose_head(pose_token) -> postprocess_pose`，且
  `final_output["camera_pose"]` 是独立 output；
- v6 只将 captured pre-rollout query 输入同一 frozen `pose_head`/postprocess，绝不读取/替换 raw camera pose numeric input；
  no export feeds subsequent model state/detector; source audit 要明确 query仍会依赖 current image feature 的 caveat。

任何 source hash/topology 漂移、alias/in-place mutation、token 在 rollout 后才取得、或 raw pose numeric dependency 都是
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_IMPLEMENTATION_NO_GO`，不启动 GPU。

## 4. 不可变边界

- 仅读取 `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`；禁止下载
  dataset/weight/dependency 或 blind input。
- 不修改 ReCal3R/checkpoint/raw RGB/GT/corruption manifests/Detector-v3 config/recovery-quality formal artifacts；直到所有
  six v6 GPU outputs 冻结后，CPU evaluator 才可读取 logical-base GT。
- 新增独立 v6 primitive/export policy/recurrent runner/script；不得 import/call v3/v4/v5 runner main、safe-anchor exporter、
  ORB/registration/v5 pointmap consensus。可复用已经测试的 low-level structural witness 与 frozen detector。
- GPU 仅在 tmux 发起，优先 GPU 2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`；每个 run two preflights 都需
  `>=12 GiB`，使用永不复用 output/log id，并冻结 canonical preflight/main/postflight/tmux logs 为 `0444`、output 为 `0555/0444`。

## 5. Gate A：实现、source 与因果合约

开始 GPU 前必须通过：

- synthetic pose-head query identity/translation/rotation/pinned postprocess/camera validation；shape/nonfinite/malformed
  query、nonfinite pose-head output、malformed/non-SO3 decode fail-closed；clear only raw export、alarm only query export、failure
  never exports raw pose；
- AST/signature audit：query decoder 仅接收 token、injected frozen pose head/postprocess/camera decoder/torch；拒绝 RGB,
  pointmap, confidence, anchor, history, GT, future, detector 与 raw `camera_pose` numeric input；source audit 必须证明 token
  capture pre-rollout、no post-rollout `dec[-1]` substitution；
- incremental detector reference equivalence、rollback structural identity/storage/version witness、watchdog/reset/no leakage、
  frame-0 commit；实际 ReCal3R CPU pose-head/postprocess/decode self-check；
- project-local `--basetemp` full CPU suite、compileall、diff check；code/test/doc 分开 commit；StateGuard3R 与 ReCal3R clean。

任一 Gate A 失败即 `PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_IMPLEMENTATION_NO_GO`，不启动 GPU。

## 6. Gate B：dynamic equivalence、availability 与 performance short-circuit

1. 新建一次 dynamic always-control `recovery-prerollout-pose-query-v6-dynamic-always-commit-0001`：四个 protected files
   (`checkpoint-load-audit.json`、`health.jsonl`、`predictions-summary.json`、`trajectory.json`) 必须与 frozen v1 dynamic
   baseline byte-equivalent，且 synchronized runtime/baseline `<=1.20`。
2. 仅 control PASS 后新建一次 candidate probe `recovery-prerollout-pose-query-v6-dynamic-candidate-0001`。每个 real alarm
   必须 rollback/full witness/`export_pre_rollout_pose_query`/complete query evidence/no fallback，且 candidate/baseline runtime
   `<=1.20`。
3. control 或 candidate 任一失败即 `PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_AVAILABILITY_OR_RUNTIME_NO_GO`，不运行 wrong/low、不
   改 token source、decode/postprocess/camera gate/Detector，且不复用 output id。

## 7. Gate C：冻结矩阵与质量选择

仅 Gate B PASS 才各一次运行 wrong/low always+candidate，得到 six frozen v6 outputs。CPU evaluator 才读取 GT，按照
prefix 0--14 Sim(3)、tail 19--29 ATE/RPE 计算 `(baseline-candidate)/baseline`。candidate-ready 的全部条件：

- 三个 controls 四文件 byte-equivalent；所有 alarm query/rollback witness 完整；zero restore/query/watchdog/leakage failure；
- 至少 2/3 条件 ATE 与 translation-RPE 同时为正，两项 median `>=+5%`，candidate runtime median `<=1.20`；
- canonical logs、final timeline path/hash、code/source provenance、freeze permissions 都完整。

全部满足才写未执行的 blind acquisition plan，结论为 `PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_CANDIDATE_READY`；否则为
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_FEASIBILITY_NO_GO`。两种均完成本计划，且 NO-GO 不授权在已解盲 matrix 继续调参。

## 8. 执行终态

Gate A 已通过。dynamic always-control
`recovery-prerollout-pose-query-v6-dynamic-always-commit-0001` 的四个 protected files 与 frozen v1
baseline 逐字节一致，runtime ratio 为 `1.140228...`，也通过 `<=1.20`。但本计划第 4 节要求冻结
canonical `preflight/main/postflight/tmux` 四类日志；实际 launcher 仅生成前三者，虽定义了
`*-tmux-transcript.log` 变量却没有写入该文件，随后对应 tmux window 已消失，不能诚实地补造 transcript。

因此 control 没有完整达到本计划的 provenance/freeze 边界，v6 在 Gate B 以前终止为
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_PROVENANCE_NO_GO`。不得基于该 control 启动 v6 candidate，也不得运行
wrong/low、读取 development GT、运行 evaluator、修改同一 v6 mechanism 后重跑。完整审计见
`docs/audits/recovery-prerollout-pose-query-export-v6-control-provenance-no-go.md`。
