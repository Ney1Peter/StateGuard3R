# StateTriage3R v2 Stage 0 Gate A：接口与因果边界审计

- 审计日期：2026-08-23
- 决定：**`STATE_TRIAGE_V2_STAGE0_GATE_A_PASS`**
- 范围：只读的 CPU 源码、资源、既有 observational output 与环境审计。没有创建
  Stage 0 input capsule、没有运行 Stage 0 model forward、没有读取 GT 来形成特征或标签。
- 后续授权：可以实现和测试一个独立的、只读 observer；尚不授权 Stage 0 GPU matrix。

## 1. 固定资源与环境

| 项目 | 观测值 |
| --- | --- |
| StateGuard3R commit | `17ddb0e181db344a47c7d8e972eb8a71c169e118` |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| checkpoint | `baselines/ReCal3R/src/cut3r_512_dpt_4_64.pth` |
| checkpoint SHA-256 / size / mode | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` / `3,173,761,006` bytes / `0444` |
| isolated interpreter | `baselines/ReCal3R/.venv/bin/python`，Python `3.11.14`、PyTorch `2.4.0+cu121`、NumPy `1.26.4` |
| local TUM RGB counts | `fr1_desk=613`、`fr2_desk=2965`、`fr3_walking_static=743`、`fr3_walking_xyz=859` |
| available data filesystem | `/data` has `2.3 TB` free; no download is planned |

检查过的上游文本 SHA-256：

```text
710f1ba6d2e43076b6e46355518665540bc9ae0ef46c9f77079d3566a38069d3  src/dust3r/inference.py
32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1  src/dust3r/model.py
8cf678ffef14f5753aab99914516badbf03776b555cc54776c51a12896f1b589  StateGuard3R/scripts/run_recal3r_smoke.py
712edb6499b22da31644607e226bcb7b2aa46b856905a530b36f46e14fe3b6cc  StateGuard3R/src/stateguard3r/health.py
```

当前 StateGuard3R 有用户原有的 v14/v15 未跟踪文件。它们不属于 Stage 0，审计、实现、
暂存和提交均不会涉及这些文件。

## 2. 官方 forward 的观察边界

`baselines/ReCal3R/src/dust3r/inference.py:291` 的
`inference_recurrent_lighter` 调用官方
`model.forward_recurrent_lighter(..., ret_state=True)`；后者位于
`model.py:1661`。在每帧中，官方路径先计算 output head，随后调用原生
`_compute_recal3r_update_mask`（`model.py:1199`）、提交 global state/memory，最后通过
`_maybe_record_u_calibration_step`（`model.py:949`）记录校准 trace。

该路径把每个 `res` 经过 `to_cpu` 后放入 prediction list。故 Stage 0 observer 可以只接收
已经脱离模型的 CPU prediction 和当前只读 input view；它不需要、也不得接收 model、state、
memory、mask 或可写 tensor reference。它在 forward 全部结束后按 frame 顺序处理 prediction，
但单次 API 只接收当前帧与过去的 observer history。这是 **post-output observational
replay**，不是在线修改 ReCal3R state 的机制。

现有的 v3 observational run（只用来核对 schema，不作为 Stage 0 feature input）证明每帧
prediction 至少包含：

```text
camera_pose                 (1, 7)
pts3d_in_self_view          (1, H, W, 3)
pts3d_in_other_view         (1, H, W, 3)
conf / conf_self            (1, H, W)
rgb                         (1, H, W, 3)
```

`run_recal3r_smoke.py:1014` 已使用其中的 pose 和两张 pointmap 得到 pose jump 与 independent
cross-head geometric residual；`health.py` 已从 native trace 暴露 uncertainty/reliability 和
global-state delta。新的 observer 不改变这些既有字段，也不读取任何旧 run 的数值。

## 3. 可用证据与定义

| evidence | Stage 0 的实际、可观察 proxy | availability | 限制 |
| --- | --- | --- | --- |
| coverage | 当前 `pts3d_in_other_view` 的稀疏 reference-frame anchors 到先前 anchors 的最近邻支持率 | `<=t` | 是 output-space spatial-support proxy，不是可逆 state-token map |
| registration | pose jump、当前两张独立 pointmap 的几何一致性、residual globality | `t` | `registration_or_order_fault` 是输入顺序/配准代理，不是 Sim(3) 真值 |
| observation quality | 当前 model-ready RGB 的 sharpness/luminance/saturation，以及 `conf`/`conf_self` 摘要 | `t` | 不调用 GT/depth/label；quality 不直接等价物理变化 |
| spatial structure | 稀疏 pointmap residual 的 high-residual fraction、grid component/globality | `t` | 网格是图像/输出层 proxy，不能声称对象级 segmentation |
| temporal support | 仅将当前 high-residual reference anchors 与过往 high-residual anchors 匹配 | `<=t` | 只提供 causal support count；`T0+K` 以后才允许使用未来证据作离线上界 |
| scalar comparison | native uncertainty/reliability、global-state delta、pose/geometry residual的预注册单值组合 | `t` | 只能在 Gate B freeze 前选出一个 scalar baseline |

观察 API 必须以纯 NumPy/标准库实现，显式拒绝非有限数、错误 shape 与未来 frame。所有输入
prediction 要复制/只读转换；history 只保留有上限的下采样 anchors，不能保存完整模型 state。

## 4. 因果与隔离检查

Stage 0 实现必须满足：

1. 不 import `recal3r_*recovery*`、`online_quarantine*`、v14--v20 runner/capability 模块，
   也不 monkey-patch ReCal3R。
2. Observer 的 `observe` 参数不包含 GT、depth、label、event、source index、timestamp 文本、
   whole prediction list 或 model；frame identity 仅作 ledger 排序，不作为 typed rule feature。
3. Prefix-invariance unit test 用同一 prefix 与不同 suffix 分别喂入 observer，要求 prefix
   ledger bytes 完全一致；这排除无意 future read。
4. normal-control 必须比较 vanilla 与 observer run 的 checkpoint-load audit、prediction
   summary、trajectory、health 和 canonical input binding。observer ledger 不是等价对象，
   但不得改变这些原生 artifact。
5. observer 的图像质量/anchor extract 运行在官方 forward 后，不触碰 model input；input
   bytes 在 loader 前后必须重新 hash。

## 5. Gate A 结论

覆盖、几何/pose、质量和时间四种因果 evidence 都有合法且可实现的 output/input proxy；
没有必要下载数据、权重或引入第二个 baseline。与此同时，接口不能支持 region-level
token undo，也没有真实 persistent-change 标签。该限制符合 Stage 0 的目标：只测试
cause separability，不做 state intervention。

Gate A 因此通过。下一项是一个最小、充分单元测试的只读 observer 和专属 Stage 0 runner；
实现完成后需要先做 CUDA-hidden CPU tests，再进入 Gate B 的输入协议冻结。
