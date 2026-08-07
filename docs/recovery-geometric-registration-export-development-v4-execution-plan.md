# Recovery geometric-registration export development v4：因果点云定位计划

- 制定日期：2026-08-07
- 状态：**执行中（独立于已完成 v1/v2/v3）**
- 计划窗口：12--18 小时；以全部预注册门禁的正常结束状态为终点
- 前置结论：v3 的 state isolation、runtime 和 output fallback 均可行，但固定 SE(3) motion extrapolation 只改善
  low-overlap，不能跨 corruption 改善；因此 v4 不得复用或调整任何 pose hold/CV/delta/anchor-length fallback。

## 1. 唯一候选与机制假设

候选固定为 `detector-v3-incremental-anchor-orb-3d3d-registration-export`。它不从过去 pose 运动学外推，而是在
当前 online alarm 时，使用**最近 real-safe anchor 的 RGB 与其 past model pointmap**、当前 RGB 与当前 candidate
`pts3d_in_self_view`，以 deterministic ORB mutual matches + deterministic robust 3D--3D rigid registration 直接估计
`T(current-self -> anchor/reference)`，并仅导出该 registration pose。

这是一条不同机制假设：若 corruption 只污染部分 RGB/state 或改变 packet order，但仍保留可匹配静态场景区域，来自
past pointmap 的几何定位可以恢复当前 pose，而不猜测运动速度。它不得读取 GT/depth/label/event/source-index/future
frame，不能使用 candidate `camera_pose` 作为 solver initialisation，也不能在 registration 失败时静默使用 raw pose、
v3 fallback、hold 或任何 donor pose。

固定语义：

1. normal clear frame 正常 commit，并保存该帧 model-ready RGB、`pts3d_in_other_view` reference pointmap 和真实
   `camera_pose` 为 external real-safe anchor；这些对象绝不回填 model/detector。
2. alarm frame 保留 v3 的 complete pre-state rollback、observed-health ledger、bounded incremental detector 与
   eight-frame watchdog；从**最近一个 real-safe anchor**到当前 raw RGB 做 ORB correspondence。
3. 每个 match 采样 anchor `pts3d_in_other_view` 与 current `pts3d_in_self_view` 的同像素 3D pair；丢弃 nonfinite、
   out-of-bounds、degenerate pairs。固定 seeded RANSAC Kabsch 后，以 inlier transform 导出 `camera_pose`，只通过
   pinned `camera_to_pose_encoding(..., "absT_quaR")` 写回。
4. 连续 alarm 保持同一 last real-safe anchor；不把 registered pose/current pointmap/candidate health 放入下一帧
   anchor、model state 或 detector history。clear 后才替换 real-safe anchor。
5. 不足确定性门槛（足够 descriptor/mutual/finite 3D pairs、non-collinear sample、inlier count、normalized residual、
   proper SO(3)）直接 `REGISTRATION_UNAVAILABLE_FAIL_CLOSED`，不输出混合结果；这是 deployment infeasibility，
   不是允许 retune 的信号。

固定 registration config（在实现与任何 v4 forward 前锁定）：复用 frozen v2 ORB `nfeatures=2000`、ratio
`0.70`、single-thread seed `0`；3D RANSAC seed `0`、256 hypotheses、每项 3 pairs、至少 24 个 finite pairs、至少
16 inliers；inlier cutoff 为 `0.05 * max(median_l2(anchor_3d), 1e-3)`，并要求 sampled/inlier centered rank 3、
proper rotation determinant `+1`。像素 correspondence 以 nearest integer indexing 读取同分辨率 pointmap。以上皆非
由 v3 quality 数值选择，v4 内不得改动。

## 2. 不可变边界

- 仅使用三个只读 development manifests；不下载数据、权重或依赖，不读取/重跑/覆盖
  `recovery-quality-formal-*` response。
- 不修改 ReCal3R、checkpoint、raw RGB、GT、corruption manifest、frozen Detector-v3 config。
- 固定 batch 1、single GPU；每个 GPU forward 在 `tmux stateguard:0.0` 前记录 UUID、>=12 GiB free、existing PID、
  command、log、exit code，绝不影响其他任务。
- 禁止以 v3 的 ATE/RPE 数值选择 ORB 参数、RANSAC threshold、anchor policy 或 registration fallback；所有 constants
  在实现前写成 source-bound config，并只可因单位测试/安全错误修正。

## 3. Gate A：纯几何与因果安全合约

新增独立 v4 matcher/3D registration/runner；不修改 v3。CPU tests 必须覆盖：

- identity、translation、rotation、noise/outlier Kabsch recovery；det=+1、orthonormality、degenerate/collinear/fewer
  pairs fail-closed；deterministic seed；camera encoding round-trip；只替换 camera pose；
- ORB keypoint coordinate/pointmap indexing bounds、mutual matching、no RGB mutation、anchor/current direction、no
  candidate camera-pose initialization、no future/GT/source fields；
- continuous alarm uses stable real anchor，clear reset anchor，registration failure creates no raw export/pending state；
- v3 rollback structural witness/incremental CPU-reference equivalence/no leakage/watchdog/reset guards仍通过；
- full CPU suite, compileall, diff check, code/test/doc separate commits, ReCal3R clean。

任一几何/causality/restore test 失败即 `GEOMETRIC_REGISTRATION_EXPORT_V4_IMPLEMENTATION_NO_GO`，不启动 GPU。

## 4. Gate B：真实 registration availability 与性能先行

先只跑 development-dynamic 的 v4 always-control，四 protected files 必须 byte-equivalent 且 comparable runtime/baseline
`<=1.20`。然后仅跑一次 dynamic candidate availability probe：它必须在每个 real alarm 成功完成 registration，均有
inlier/residual/proper-rotation witness、rollback 与 export action，candidate/baseline runtime `<=1.20`。任一失败即
`GEOMETRIC_REGISTRATION_EXPORT_V4_AVAILABILITY_OR_RUNTIME_NO_GO`，不得跑 wrong/low 或试另一个 feature/threshold。

只有通过才跑 wrong/low always+candidate 的一次性冻结矩阵；所有 forward 均原子 freeze，失败不复用同名 output。

## 5. Gate C：冻结选择

所有 six v4 outputs 冻结后才 CPU evaluator 读取 logical-base GT；metric 为 prefix 0--14 Sim(3)、tail 19--29 ATE/RPE。
要求：三组 controls byte-equivalent、所有 alarm registration/rollback witness 完整、2/3 条件同时正 ATE 与 translation
RPE、两项中位数 `>=+5%`、candidate runtime median `<=1.20`、零 restore/registration/watchdog/leakage failure。

全部通过才得到 `GEOMETRIC_REGISTRATION_EXPORT_V4_CANDIDATE_READY` 并只写未执行的新的 blind acquisition plan；否则
为 `GEOMETRIC_REGISTRATION_EXPORT_V4_FEASIBILITY_NO_GO`。NO-GO 也完成 v4；后续必须再提出不同机制，不能将开发集
变成 registration 参数搜索集。
