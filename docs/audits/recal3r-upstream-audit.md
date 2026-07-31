# ReCal3R 官方上游审计

审计日期：2026-07-31

审计对象：<https://github.com/Powertony102/ReCal3R>

固定版本：<https://github.com/Powertony102/ReCal3R/commit/466c7cdf3acd2f589f1d82e5f6391966f19db9ff>

本地工作副本：/data/wangzheng/Project2/baselines/ReCal3R

## 1. 审计范围与结论

本审计只采用 ReCal3R 官方 GitHub 仓库、作者论文和仓库内自带文档作为事实来源，覆盖：

- 上游分支、提交和发布状态；
- 根许可证与 vendored 第三方代码的许可证层级；
- 安装依赖、缺失依赖、checkpoint 来源和加载风险；
- 官方数据格式、最小运行命令和 GPU 信息；
- 实际 ReCal3R 调用路径；
- state、reliability、candidate/final learning rate 和 update signals 的可见性；
- StateGuard3R 的最佳 health hook、snapshot 和 quarantine 边界；
- 当前上游代码中会影响首轮实验的已知风险。

截至审计日，可以进入小样本 smoke test 和 detection-only instrumentation，但不应直接开始 video-depth 全流程或 rollback。首轮应固定 commit、只下载一个 checkpoint 和一条小型 TUM 序列，先验证 ReCal3R 主路径以及 health trace。

## 2. 上游身份与固定版本

官方仓库的默认且唯一分支为 main，当前 HEAD 为：

~~~text
466c7cdf3acd2f589f1d82e5f6391966f19db9ff
~~~

该提交时间为 2026-07-17 05:20:30 UTC。仓库目前没有 tag 和 GitHub release，因此实验必须固定完整 commit SHA，不能只记录 main。

官方核验入口：

- Repository API：<https://api.github.com/repos/Powertony102/ReCal3R>
- main ref API：<https://api.github.com/repos/Powertony102/ReCal3R/git/ref/heads/main>
- 固定提交：<https://github.com/Powertony102/ReCal3R/commit/466c7cdf3acd2f589f1d82e5f6391966f19db9ff>

本地副本可用以下只读命令复核：

~~~bash
git -C /data/wangzheng/Project2/baselines/ReCal3R rev-parse HEAD
git -C /data/wangzheng/Project2/baselines/ReCal3R status --short --branch
~~~

预期第一条命令输出上述 SHA，执行实验时还应保证工作区修改被单独记录。

## 3. 许可证层级

### 3.1 ReCal3R 根许可证

仓库根 LICENSE 是 MIT License，版权人为 Xinze Li：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/LICENSE#L1-L20>

这允许使用和修改 ReCal3R 自有代码，但不能据此推断仓库中的全部第三方代码、数据或 checkpoint 都是 MIT。

### 3.2 Vendored 第三方代码

仓库内的 CroCo 文档明确写明 CC BY-NC-SA 4.0：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/croco/README.MD#L25-L29>

CroCo NOTICE 还声明部分子组件具有独立的 CC BY-NC-SA 4.0 或 Apache 2.0 条款：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/croco/NOTICE#L1-L20>

数据预处理代码中也有明确的非商业许可证声明：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/datasets_preprocess/path_to_root.py#L1-L3>

因此本项目应保留上游许可证和 NOTICE，不应把 vendored CroCo/DUSt3R 代码重新标注为 MIT。当前研究用途可以继续，但若将来发布、再分发 checkpoint 或进入商业使用，必须单独完成许可证复核。

### 3.3 Checkpoint 许可证

README 提供的是 CUT3R checkpoint，没有给出 checkpoint 独立许可证。该问题当前标记为 unresolved，不应仅凭 ReCal3R 根 MIT 许可证作推断。

## 4. 安装与依赖审计

官方 README 指定：

- Python 3.11；
- CMake 3.14.0；
- PyTorch 和 torchvision；
- 推荐 pytorch-cuda 12.1，但要求按服务器 CUDA 情况调整；
- 编译 CroCo RoPE CUDA kernel；
- 评测额外安装 evo 和 open3d。

原始说明：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/README.md#L36-L64>

requirements.txt：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/requirements.txt>

当前依赖风险如下。

| 风险 | 证据 | 执行要求 |
|---|---|---|
| gdown 未列入 requirements | README 用 gdown 下载权重 | 环境中显式补充 gdown |
| scikit-image 未列入 requirements | demo.py 导入 skimage.filters | smoke demo 前显式补充 scikit-image |
| imageio 未列入 requirements | demo.py 导入 imageio.v2 | smoke demo 前显式补充 imageio |
| evo、open3d 不在 requirements | README 只在文字中列为 evaluation 依赖 | 只在需要对应评测时安装 |
| torch 等大多数包未锁版本 | requirements 仅固定少量包 | 环境跑通后保存完整 freeze |
| gradio 重复列出 | requirements 第 5、16 行 | 无功能影响，但说明依赖文件未经严格整理 |

不应直接把上游 conda 命令混入其他 baseline 环境。ReCal3R 应维持自己的独立环境，并记录 Python、PyTorch、CUDA、编译器、依赖 freeze 和 RoPE kernel 编译结果。

## 5. 权重来源与加载安全

官方仅提供：

~~~text
src/cut3r_512_dpt_4_64.pth
~~~

下载链接是 Google Drive：

<https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link>

官方说明：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/README.md#L66-L75>

这是 CUT3R 的 4–64 views 权重。ReCal3R 是 training-free 状态更新规则，官方没有发布单独的 ReCal3R checkpoint。

上游未公布文件大小和 checksum。首次下载后必须：

1. 只从 README 中的官方链接下载；
2. 保存实际文件大小；
3. 计算并记录 SHA-256；
4. 后续任务复用同一只读文件，不重复下载或复制；
5. 在实验 manifest 中同时记录权重 hash 和 ReCal3R commit。

模型加载函数使用：

~~~python
torch.load(model_path, map_location="cpu", weights_only=False)
~~~

证据：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L80-L101>

这意味着 checkpoint 不是纯张量安全加载。不得加载来源不明或被其他用户替换的文件。官方文件下载后应校验 hash，再进入隔离环境加载。

## 6. GPU 与资源信息

官方没有公布最低显存、最低 GPU 型号或最低 compute capability。

论文中的 runtime 和 peak GPU memory 是在单张 NVIDIA RTX PRO 6000、96 GB 显存上测得，这只是论文评测平台，不是最低硬件要求：

<https://arxiv.org/html/2607.05356#S4.SS0.SSS0.Px2>

README 推荐 CUDA 12.1 并要求编译 RoPE CUDA kernel。Demo 在 CUDA 不可用时会退回 CPU：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L432-L443>

Vendored CroCo 文档称 CUDA kernel 不可用时存在较慢的 PyTorch fallback：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/croco/README.MD#L46-L58>

Relpose evaluator 已实现 CUDA elapsed time 和 peak allocated memory 统计：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/relpose/launch.py#L33-L55>

因此首轮必须从单进程、单卡、少量帧开始，并把实测峰值显存作为本服务器的资源基线。

## 7. 数据格式与最小 smoke test

### 7.1 无评测集图片 smoke

Demo 可以读取一个图片目录或视频。图片目录支持：

~~~text
.jpg .jpeg .png .bmp .tiff .tif .webp
~~~

证据：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L382-L429>

README 示例没有显式设置 model_update_type，而 CLI 默认是 cut3r：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L46-L121>

因此真正的 ReCal3R smoke 命令必须显式包含 recal3r：

~~~bash
python demo.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --seq_path /absolute/path/to/tiny-image-sequence \
  --device cuda \
  --size 512 \
  --model_update_type recal3r \
  --output_dir /absolute/path/to/new-isolated-output
~~~

建议第一轮只放 4–8 张图片。Demo 会启动 Viser viewer，并会删除所给 output_dir 中已有的 depth、conf、color、camera 子目录后重建：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L349-L377>

因此必须使用全新、隔离的输出目录。

### 7.2 单条 TUM-Dynamics smoke

Relpose 和 video-depth evaluator 能直接读取原始 TUM-RGBD 布局：

~~~text
ROOT/
  rgbd_dataset_*/
    rgb/
    depth/
    rgb.txt
    depth.txt
    groundtruth.txt
~~~

官方说明：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/eval.md#L12-L19>

首个带 GT 的小样本应走 relpose，而不是当前有已知错误的 video-depth：

~~~bash
python eval/relpose/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir /absolute/path/to/new-relpose-output \
  --eval_dataset tum \
  --dataset_path /absolute/path/to/tiny-tum-root \
  --seq_list rgbd_dataset_freiburg3_walking_xyz \
  --pose_eval_stride 1 \
  --max_frames 16 \
  --size 512 \
  --model_update_type recal3r
~~~

官方命令框架：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/eval.md#L50-L67>

注意 max_frames 不表示连续前 N 帧。实现会保留首帧，然后在剩余完整序列中近似均匀抽样：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/relpose/launch.py#L196-L205>

调用位置：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/relpose/launch.py#L367-L376>

需要分析连续状态污染时，应由 StateGuard3R wrapper 明确截取连续 prefix，不能把 max_frames 当作 prefix。

## 8. 正确调用路径

仓库根部和 src/dust3r 下各有一个 model.py。官方 demo 和 evaluator 实际使用的是：

~~~text
/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py
~~~

不要在仓库根部的 model.py 上实现 health hook。

Demo 的实际导入：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L445-L450>

ReCal3R 主推理调用链为：

~~~text
demo.py / eval/*/launch.py
  -> dust3r.inference.inference_recurrent_lighter
  -> ARCroco3DStereo.forward_recurrent_lighter
  -> ARCroco3DStereo._recurrent_rollout
  -> ARCroco3DStereo._compute_recal3r_update_mask
  -> state_feat / mem update
~~~

inference_recurrent_lighter：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/inference.py#L290-L315>

forward_recurrent_lighter 的 ReCal3R 分支：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1661-L1833>

常规 forward_recurrent 没有执行 ReCal3R calibrated update，不能拿它替换 lighter 路径：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1521-L1659>

## 9. 输出与内部状态

### 9.1 常规预测输出

每帧预测包含：

- pts3d_in_self_view；
- pts3d_in_other_view；
- conf_self；
- conf；
- camera_pose。

读取位置：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py#L281-L327>

conf 和 conf_self 是 pointmap confidence，不是 ReCal3R state-token reliability。它们可作为 output-level health signal，但必须使用不同字段名。

### 9.2 持久状态

ReCal3R 执行中至少存在以下状态：

| 状态 | 作用 |
|---|---|
| state_feat | 全局 recurrent scene state |
| state_pos | state token 位置编码 |
| mem | local pose memory |
| init_state_feat | reset 使用的初始全局状态 |
| init_mem | reset 使用的初始 pose memory |
| recal3r_state0 | state deviation 的固定参考状态 |
| update_pressure | 每 token 最近写入压力 EMA |
| _recal3r_sequence_age | 序列年龄及首帧逻辑 |

recal3r_state0 和 sequence age 的初始化/重置：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L899-L947>

## 10. 已有 trace API

上游已经提供未写入 README 的 trace API：

~~~python
model.enable_u_calibration_trace(final_state=None, oracle_window=None)
model.get_u_calibration_trace()
model.get_u_calibration_last_state()
model.disable_u_calibration_trace()
~~~

定义：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L852-L1012>

首轮 detection-only 可在 inference 前调用：

~~~python
model.enable_u_calibration_trace(oracle_window=1)
~~~

oracle_window=1 会让现有逻辑记录每步 token-level delta_norm，同时仍保留 frame-level uncertainty 和 entropy summaries。

Trace 字段包括：

| Trace 字段 | 实际语义 |
|---|---|
| u | calibrated unreliability/uncertainty U |
| frame_u_mean/min/max | 当前帧 token uncertainty 汇总 |
| frame_h_mean/min/max | attention entropy rank/quantile 汇总 |
| delta_norm | state_feat 每 token 更新范数 |
| err | 相对 oracle-window future state 或指定 final state 的误差 |
| frame_idx / frame_step | 帧索引 |

两个命名陷阱必须避免：

1. u 不是 reliability。代码中的 U 与论文可靠度满足 reliability = 1 - U。
2. Trace 中的 h 是 attention entropy 统计；论文候选学习率公式中的 h 表示 update pressure，两者不是同一个量。

因此 ledger 应保存 uncertainty_u_mean，并在需要时显式派生 reliability_mean = 1 - uncertainty_u_mean。不得把 output conf 或 trace u 直接命名为 reliability。

现有 trace 不会公开：

- alignment gate；
- state drift；
- residual score；
- update pressure；
- intermediate posterior gamma；
- candidate beta；
- final beta；
- local pose memory delta。

这些字段需要一个最小只读 instrumentation hook。

## 11. 推荐 health hook

最佳 hook 点是：

~~~text
ARCroco3DStereo._compute_recal3r_update_mask
~~~

完整实现：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1199-L1251>

该方法内部已经计算：

| 代码量 | Health 语义 | 是否当前对外公开 |
|---|---|---|
| alignment_gate / beta_t before calibration | token-frame alignment gate | 否 |
| drift_m | token 相对初始状态的 L2 deviation | 否 |
| pi_0 | 从 state deviation 构造的稳定性 prior | 否 |
| h_m | attention entropy 的 empirical quantile | trace 仅公开汇总 |
| gamma_m | cue-agreement posterior-like reliability | 否 |
| R | 实际是 U = 1 - calibrated reliability | trace 公开 |
| residual_score_scalar | 当前帧未被状态解释的程度 | 否 |
| update_pressure | 最近 token 写入压力 EMA | 模型属性，不在 trace |
| attenuation | exp(-update_pressure) | 否 |
| beta_trust | uncalibrated/candidate learning rate | 否 |
| beta_final / beta_t | final learning rate | 否 |

相关辅助函数：

- Alignment gate：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1024-L1028>
- Entropy：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1058-L1107>
- Empirical quantile：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1109-L1129>
- State drift/prior：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1148-L1184>
- Residual：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1186-L1197>

建议在返回前只保存 detached scalar summaries，不保留原始 attention 或 state GPU tensor。首版每帧 ledger 建议字段：

~~~text
frame_id
update_requested
reset_requested
alignment_mean
alignment_min
alignment_max
state_drift_mean
state_drift_max
attention_entropy_mean
posterior_gamma_mean
uncertainty_u_mean
reliability_mean
residual_score
relative_residual
update_pressure_mean
candidate_beta_mean
candidate_beta_max
final_beta_mean
final_beta_max
global_state_delta_mean
global_state_delta_max
local_mem_delta_mean
local_mem_delta_max
point_conf_mean
pose_jump_translation
pose_jump_rotation
~~~

实现时应验证：

- logger 关闭后 prediction 逐项不变；
- logger 不保存计算图；
- logger 不把 GPU tensor跨帧持有；
- 每帧只写标量或小型直方图；
- 首帧和 reset 帧明确标注，因为它们走 full update 分支。

## 12. State、mem 与 quarantine 副作用

### 12.1 ReCal3R 只校准全局 state

在主路径中：

~~~python
state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)
mem = new_mem * update_mask2 + mem * (1 - update_mask2)
~~~

update_mask1 是 ReCal3R calibrated rate，update_mask2 只是原始二值 update mask：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1790-L1813>

因此 ReCal3R 保护 global state_feat，但不会按 reliability 缩放 local pose memory。StateGuard3R 必须分别监控 global_state_delta 和 local_mem_delta，rollback 也必须同时恢复两者。

### 12.2 update=False 不是完全无副作用

即使 view["update"] 为 False，_compute_recal3r_update_mask 仍根据 alignment 更新 update_pressure，随后 _advance_recal3r_sequence_age 继续增加序列年龄。二值 mask 只阻止 state_feat 和 mem 的实际写入。

因此未来的 skip/quarantine 若要求真正冻结系统行为，不能只设置 update=False，还必须冻结或回滚：

~~~text
update_pressure
recal3r_state0
_recal3r_sequence_age
trace pending state, if enabled
~~~

否则被跳过帧仍会改变后续帧的 final beta。

### 12.3 Snapshot 最小闭包

精确恢复至少需要：

~~~text
state_feat
state_pos
init_state_feat
mem
init_mem
update_pressure
recal3r_state0
_recal3r_sequence_age
~~~

如果 trace 开启，还要明确 trace 是否属于可恢复执行状态。首轮 detection-only 不应逐帧保存完整 GPU snapshot；后续只按低频 checkpoint 将 detached clone 移到 CPU，并单独测量复制开销。

## 13. 当前上游风险清单

| 风险 | 影响 | 首轮处理 |
|---|---|---|
| forward_recurrent_lighter 返回空 state_args | ret_state=True 也无法直接取得逐帧 state | detection 先用 trace；snapshot 阶段再补接口 |
| video-depth 调用未定义函数 | 推理结束后可能 NameError | 首轮禁用 video-depth |
| Bonn relpose CLI 与 README 矛盾 | README/sample_cmd 的 Bonn 命令会被 choices 拒绝 | 首轮只用 TUM relpose |
| 根 model.py 与 src/dust3r/model.py 并存 | 容易 hook 错文件 | 只修改 src/dust3r/model.py |
| Demo 默认 model_update_type=cut3r | 未显式参数时并未运行 ReCal3R | 所有命令显式写 recal3r |
| requirements 缺少 gdown、imageio、scikit-image | 新环境下载或 import 失败 | 环境显式补齐并 freeze |
| 权重无 checksum | 复现和供应链风险 | 下载后记录 SHA-256 |
| 无最低 GPU/显存说明 | 资源预算未知 | 单卡、4–8 帧起测 |
| max_frames 是全序列均匀抽样 | 会改变连续污染实验语义 | wrapper 明确连续 prefix |
| Demo 重建输出子目录 | 可能覆盖旧结果 | 始终使用全新隔离输出 |
| 论文与代码 normalization 描述不完全一致 | 指标语义可能误读 | 固定代码 commit，以代码行为为实验事实 |

forward_recurrent_lighter 的空 state_args 证据：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1661-L1670>

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1831-L1833>

Video-depth 未定义日志函数调用：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/video_depth/launch.py#L299-L310>

Relpose 允许数据集过滤：

<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/relpose/launch.py#L19-L24>

代码使用 rank/empirical-quantile normalization，而论文正文描述 deviation min-max normalization：

- 代码：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py#L1109-L1129>
- 论文：<https://arxiv.org/html/2607.05356#S3.SS3.SSS0.Px1>

## 14. 推荐首轮执行门槛

按以下顺序执行：

1. 验证本地副本仍固定在 commit 466c7cdf3acd2f589f1d82e5f6391966f19db9ff。
2. 建立独立环境，补齐缺失依赖并保存 freeze。
3. 只下载官方单个 CUT3R checkpoint，记录大小和 SHA-256。
4. 用 4–8 张图片验证显式 recal3r demo 主路径。
5. 用单条 TUM-Dynamics、16 帧以内运行 relpose，记录 FPS 和峰值显存。
6. 启用 enable_u_calibration_trace(oracle_window=1)，确认 uncertainty、entropy、delta 与 frame id 对齐。
7. 在 _compute_recal3r_update_mask 返回前加入 detached scalar hook。
8. 验证 logger 开关前后输出一致，再开始 clean/corruption detection-only 实验。

在完成以上门槛之前，不进入：

- video-depth evaluator；
- 全量数据集下载；
- 每帧完整 state snapshot；
- quarantine/rollback；
- 多卡或多进程批量评测。

## 15. 主要官方资料

- README：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/README.md>
- Evaluation guide：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/eval/eval.md>
- 主模型实现：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/model.py>
- 推理封装：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/src/dust3r/inference.py>
- Demo：<https://github.com/Powertony102/ReCal3R/blob/466c7cdf3acd2f589f1d82e5f6391966f19db9ff/demo.py>
- 论文 HTML：<https://arxiv.org/html/2607.05356>
- 本地固定副本：/data/wangzheng/Project2/baselines/ReCal3R
