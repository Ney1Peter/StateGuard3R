# Recovery current-pointmap consensus export development v5：固定格点双点图 SE(3) 计划

- 制定日期：2026-08-07
- 状态：**已完成 — `CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_AVAILABILITY_OR_RUNTIME_NO_GO`**
- 长时目标：在 12--18 小时内，以全部预注册 Gate 的正常终态为止；不得因时间、单一好看数值或日志不便而提前宣布
  candidate-ready。
- 唯一目标：验证 `detector-v3-incremental-current-self-cross-consensus-export` 能否在不下载数据、只使用既有三个
  development manifest 的条件下，同时满足 rollback/因果安全、正常路径逐字节等价、`<=1.20` runtime 和跨条件质量门；
  若任一硬门失败，诚实发布 v5 NO-GO，并只允许再提出不同机制。

## 1. 前置结论与唯一新假设

v4 已证明 anchor ORB--3D--3D registration 在 dynamic 的六个真实 alarm 上可用，但 runtime ratio 为 `4.6307`，故不得
调整其 ORB/RANSAC/anchor/fallback 或运行 wrong/low。v3 的 motion safe-anchor export 则快速但不能跨 corruption 改善。

v5 只检验一个不同假设：**raw pose head 可能受 corruption 影响，而当前帧独立输出的 self pointmap 与 cross/reference
pointmap 仍可能在同像素位置具有足够的刚体一致性。** 因此 alarm 时直接从当前 candidate 的
`pts3d_in_self_view -> pts3d_in_other_view` 与 `conf_self/conf` 估计 self-to-reference SE(3)，而不匹配 RGB、不读取历史
anchor/pose、也不读取 `camera_pose` 数值。

Pinned ReCal3R source audit 是 Gate A 前置条件：`DPTPts3dPose.forward` 分别以 `dpt_self(x)` 和
`dpt_cross(x_cross)` 产生两张 pointmap；`camera_pose` 是对 `pose_head(pose_token)` 的独立 postprocess output，未被作为
数值输入传回求解器。`x_cross` 会使用同一 latent `pose_token` 调制，故 v5 的科学表述仅为“不同 pointmap heads 的
current-frame consensus”，绝不声称 cross pointmap 与 pose latent 完全独立。若 source hash/trace 显示
`camera_pose` 数值参与 cross pointmap 或 v5 solver，v5 直接 implementation NO-GO。

## 2. 不可变机制与数值合约

1. 保留冻结 Detector-v3、current-frame complete pre-state rollback、structural witness、observed-health ledger 和
   watchdog `8`。detector 永远在原始 prediction 上做决定；registered output 永不反馈 model、detector、state、anchor 或
   后续帧。
2. clear frame 正常 commit 并导出原始 `camera_pose`；alarm frame 先 rollback，随后只对该帧的临时输出执行 current
   pointmap consensus。没有 prior anchor、RGB、ORB、feature descriptor、RANSAC、motion extrapolation、hold、CV/delta 或
   raw-pose fallback。
3. 在同 shape 的 `(B=1,H,W,3)` self/cross pointmap 和 `(B=1,H,W)` confidence 上，固定选择 `16 x 16` evenly-spaced
   nearest-integer lattice（含四条边，共 256 对）。只保留两点有限且 `conf_self > 0`、`conf > 0` 的 pair；少于 `192` 对
   即 `CURRENT_POINTMAP_CONSENSUS_UNAVAILABLE_FAIL_CLOSED`。
4. 初始 weight 固定为 `sqrt(min(conf_self, conf))`。固定执行两轮 Tukey-bisquare IRLS weighted Kabsch，Tukey constant
   `4.685`，robust scale 为 `max(1.4826 * MAD(residual), 1e-6 * max(weighted_median_l2(cross_3d), 1e-3))`；这是为 exact
   identity/noiseless synthetic case 明确固定的量纲一致 scale floor，而不是 development 调参。任一 nonfinite MAD/scale、无正
   robust weight、source/target weighted centered rank < 3、或 final positive robust weight < `128` 都 fail-closed。没有
   data-dependent hyperparameter selection。
5. 最终 transform 必须 finite homogeneous 4x4、proper SO(3)（`|det(R)-1|<=1e-8`、`max|R^T R-I|<=1e-8`），并满足
   `weighted_median_l2(residual) / max(weighted_median_l2(cross_3d), 1e-3) <= 0.05`。只通过 pinned
   `camera_to_pose_encoding(..., "absT_quaR")` 写回；raw pose 只能提供 device/dtype template，且 encoder/decode
   round-trip 必须在 `1e-5` 内。
6. 每条 alarm transaction 写 lattice、finite/positive/IRLS-positive pair counts、two MAD scales、source/target rank、
   normalized residual、transform digest/矩阵、det/orthonormality、encoder round-trip、raw/exported pose digests、rollback
   witness。任何异常不得留下 partial export/pending state，也不得替换为 raw pose。

## 3. 不可变边界

- 只能读取 `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`；不下载
  dataset、weight、dependency 或 blind input。
- 不修改 ReCal3R、checkpoint、raw RGB/GT、corruption manifests、Detector-v3 config、recovery-quality formal artifacts。
- 禁止 GT/depth/label/event/source-index/future frame/baseline alarm artifact/synthetic alarm；仅在所有 six v5 GPU output
  冻结之后 evaluator 才可读取 logical-base GT。
- 新建独立 v5 primitive、independent recurrent runner 和 script；不得 import/call v3/v4 runner main、safe-anchor exporter、
  ORB/pointmap-registration v4。可复用已测试的 low-level structural witness 与 frozen detector，但 v5 provenance 必须列出
  所有 v5 components 和 source-dependency audit。
- GPU 只在 tmux 中发起，固定优先 GPU 2 UUID `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`，两次 preflight 都需 >=12 GiB；
  每次 run 使用永不复用的 output/log id，并冻结 canonical preflight/main/postflight logs（0444）。v4 的 overwritten/
  NUL log 不能作为 v5 先例。

## 4. Gate A：实现、source 与因果合约

新增 source-bound v5 dense-consensus primitive、external export policy、independent runner/script 与 CPU source audit。开始
GPU 前必须全部通过：

- synthetic identity/translation/rotation/noise/outlier weighted-Kabsch；确定性；proper rotation；nonfinite、shape、rank、
  confidence、MAD、inlier shortage fail-closed；lattice bounds/edge inclusion；direction self-to-cross；只替换 camera pose；
  pinned encoder tensor round-trip；
- static solver signature/AST test 无 RGB/ORB/anchor/GT/future/raw `camera_pose` numerical input；pointmap/cross-head source
  dependency hash audit；
- clear commit、alarm stable no-history export、failure no raw export；v3 rollback structural identity/storage/version witness、
  incremental detector reference equivalence、watchdog/reset/no leakage 持续通过；
- project-local `--basetemp` full CPU suite、compileall、diff check；code/test/doc 分开 commit；StateGuard3R 与 ReCal3R clean。

任一 Gate A 失败即 `CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_IMPLEMENTATION_NO_GO`，不启动 GPU。

## 5. Gate B：dynamic 的等价、可用性与性能短路

1. 新建一次 dynamic v5 always-control：四个 protected files
   (`checkpoint-load-audit.json`、`health.jsonl`、`predictions-summary.json`、`trajectory.json`) 必须与 frozen v1 dynamic
   baseline byte-equivalent，且 synchronized comparable runtime/baseline `<=1.20`。
2. 仅在 control PASS 后新建一次 dynamic candidate probe。每个真实 alarm 都须 rollback、full witness、
   `export_current_pointmap_consensus_se3`、完整 consensus evidence 和无 fallback；candidate/baseline runtime `<=1.20`。
3. control/candidate 任一失败即 `CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_AVAILABILITY_OR_RUNTIME_NO_GO`，不跑 wrong/low、
   不改 lattice/weight/IRLS/threshold，不重用 output id。

## 6. Gate C：冻结矩阵与质量选择

仅 Gate B PASS 才各一次运行 wrong/low always+candidate，得到六个冻结 v5 outputs；CPU evaluator 才读取 logical-base GT，
按照 prefix 0--14 Sim(3)、tail 19--29 ATE/RPE 计算 `(baseline-candidate)/baseline`。候选 ready 的全部条件是：

- 三个 controls 四文件 byte-equivalent；所有 alarm consensus/rollback witness 完整；零 restore/consensus/watchdog/leakage
  failure；
- 至少 2/3 条件的 ATE 与 translation-RPE 同时为正；两项中位数均 `>= +5%`；candidate runtime median `<=1.20`；
- 所有 canonical logs、final timeline path/hash、code/source provenance 和 freeze permissions 完整。

全部满足才写未执行的 blind acquisition plan，结论为 `CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_CANDIDATE_READY`。否则为
`CURRENT_POINTMAP_CONSENSUS_EXPORT_V5_FEASIBILITY_NO_GO`；NO-GO 是完成，不授权在已解盲 development matrix 上调参。

## 执行结果（2026-08-07）

Gate A 与新的 dynamic always-control `...0002` 均通过；control 四个 protected files 相对 v1 dynamic baseline
逐字节一致，runtime ratio 为 `0.9893841433`。唯一 dynamic candidate `...0001` 在 six real alarms 的 rollback、fixed-grid
consensus、proper SO(3) export 与 witness 上均通过，但 runtime ratio 是 `1.4113720436 > 1.20`。因此 Gate B short-circuit
已触发：不运行 wrong/low、GT evaluator 或 quality matrix，也不调整 v5 lattice/weight/IRLS/threshold。完整不可变证据见
[v5 result audit](audits/recovery-current-pointmap-consensus-export-development-v5-result.md)。
