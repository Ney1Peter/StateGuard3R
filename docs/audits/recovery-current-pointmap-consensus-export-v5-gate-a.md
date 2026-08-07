# Recovery current-pointmap consensus export v5：Gate A 审计

- 审计日期：2026-08-07
- Gate A 结论：**PASS；允许执行一次新的 dynamic always-control。**
- 绑定实现：StateGuard3R `d2e4cdb`（GPU-resident mechanism）与 `b68e1aa`（Gate A tests）。
- 范围：仅实现、CPU 合约、pinned source 与 provenance 审计；未读取 development GT，未运行任何新的 GPU candidate，未下载数据、权重或依赖。

## 实现固定点

v5 alarm 仅从当前 prediction 的 `pts3d_in_self_view`、`pts3d_in_other_view`、`conf_self`、`conf` 在原 tensor device 上取固定
16 x 16 lattice，并以两轮预注册 Tukey IRLS weighted Kabsch 解 `self -> cross/reference`。完整 pointmap 在
`observer.finalize(..., prediction=res)` 后才被 `to_cpu`；故没有 hidden CPU registration。只有 4 x 4 transform、lattice
evidence 与最终 camera pose 会为审计导出到 CPU。

在本轮审计中修复了一个 Gate A 阻断错误：Tukey 已拒绝的零权重离群点此前仍进入 final weighted median，导致本应通过的
四点离群 synthetic case fail-closed。现在 final residual 与 cross-scale statistic 均严格只使用 final positive robust
weights，和计划中的 final-positive contract 一致；固定 lattice、weight、IRLS rounds、scale floor 与阈值没有改动。

`PinnedConsensusPoseEncoder` 还在 GPU 上再次检查 finite homogeneous matrix、proper SO(3)、bottom row 与 pinned
camera encode/decode round-trip；raw pose 只提供 device/dtype template。alarm 共识失败会抛出
`CurrentPointmapConsensusExportError`，不会返回或修改 raw pose。

## 因果、source 与安全验证

- solver signature 只有 `(self_points, cross_points, conf_self, conf_cross, torch)`；AST test 拒绝 raw pose/RGB/anchor/GT/
  future/timestamp/detector/history/pose names，且 solver body 无 `.cpu()`。
- pinned `dpt_head.py` audit 绑定 self/cross heads，确认 `camera_pose` numeric output 不直接输入 cross head；同时保留
  cross head 使用 shared pose-token latent 的科学 caveat。
- runner 在 alarm rollback/witness 完成后才执行 temporary export；export 不反馈 model、detector、state 或后续 frame。
- fixed-grid identity/rotation/translation、four lattice outliers、rank/shape/nonfinite/insufficient-pair、non-tensor/
  non-floating、encoder malformed/reflection/homogeneous rejection、alarm no-raw-export、real ReCal3R camera API
  round-trip、runner no-CPU-finalize contract 均覆盖。

## 实际验证记录

| 验证 | 结果 |
| --- | --- |
| ReCal3R `.venv` direct CPU self-check（synthetic transform + real pinned camera encode/decode） | pass；max encode/decode error `2.06e-15` |
| 组合 Torch environment v5 targeted tests | `12 passed in 2.42s` |
| StateGuard3R project-local full CPU suite，`--basetemp` 位于 repository `tmp/` | `489 tests, 0 failures, 0 errors, 3 skipped, 49.848s`；三个 skip 是 project `.venv` 无 Torch 的两个 v5 module 与既有 skip，不是失败 |
| `compileall`（`src/stateguard3r` 与 v5 script） | pass |
| `git diff --check` | pass |
| ReCal3R source | 未修改；仍绑定 `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |

## Gate B 前置与不接受的旧结果

旧的 `outputs/recovery-current-pointmap-consensus-v5-dynamic-always-commit-0001` 被保留且不可改写，但它绑定
`56222a8` 与旧 NumPy/CPU primitive hash `ac965e...`，不代表这里固定的 GPU-resident mechanism。因此它不得作为本轮
Gate B evidence。

下一步只允许使用新 output/log id `...dynamic-always-commit-0002` 跑一次 dynamic control；它必须满足四个 protected
files 相对 v1 dynamic baseline 的逐字节等价，以及 runtime ratio `<= 1.20`。只有 control pass 才可执行一次新的 dynamic
candidate probe。
