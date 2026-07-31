# Phase 0 初始资源与 ReCal3R 运行路径审计

## 审计元数据

- 审计日期：2026-07-31（Asia/Shanghai，CST，UTC+08:00）
- 资源快照时间：2026-07-31 18:24–18:32 CST
- 运行路径复核完成时间：2026-07-31 19:27 CST
- 工作区：`/data/wangzheng/Project2`
- 主方法仓库：`/data/wangzheng/Project2/StateGuard3R`
- 审计性质：只读资源发现与静态代码审计

本次审计完整遵守 `/data/wangzheng/Project2/SERVER_USAGE_RULES.md`。资源搜索仅限
`/data/wangzheng`，没有遍历其他用户的数据目录，没有跟随未知软链接，没有下载代码、
权重或数据，没有创建或修改外部仓库文件，也没有启动、终止或干扰任何 GPU 进程。
除本审计文档及其父目录外，本次记录工作未写入其他文件。

## 审计范围与方法

只读检查包括：

1. `/data/wangzheng` 下 ReCal3R、CUT3R 同名目录与 Git remote；
2. ReCal3R/CUT3R 相关 checkpoint、TUM RGB-D 序列及压缩包；
3. 属于 `wangzheng` 的 Python/uv 环境、CUDA 工具链和已有编译产物；
4. `/data` 与根分区容量；
5. `nvidia-smi` GPU 快照及计算进程所有者；
6. ReCal3R relpose 入口到实际状态更新函数的静态调用链；
7. 可在不修改 baseline 的前提下导出状态健康信号的 hook 点。

远端仓库没有执行 `fetch`，因此本文中的“与 `origin/main` 一致”仅表示与本地已有的
remote-tracking ref 一致，不代表审计时再次向 GitHub 验证了远端最新状态。

## 工作区与磁盘

审计开始时 `/data/wangzheng/Project2` 顶层不是 Git 仓库；`StateGuard3R` 是独立 Git
仓库。写入本文档前，`StateGuard3R` 状态为 `main...origin/main [领先 1]`，没有未提交
文件变化。既有提交和变化均视为用户或其他任务的工作，本次未触碰。

磁盘快照：

| 挂载点 | 总量 | 已用 | 可用 | 使用率 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `/data` | 14 T | 13 T | 552 G | 96% | 可做最小实验，但必须严格控制下载和输出增长 |
| `/` | 219 G | 158 G | 50 G | 77% | 禁止保存项目环境、权重、数据和大型临时文件 |

所有新增环境、权重、数据、缓存、日志和实验输出都必须位于
`/data/wangzheng/Project2` 的正确子仓库中。首轮只允许一个必要 checkpoint、一条小型
TUM 测试序列和有限帧输出。

## 本地源码资源

### ReCal3R

发现一个现有官方来源副本：

| 项目 | 审计结果 |
| --- | --- |
| 绝对路径 | `/data/wangzheng/iJCV-CODE/ReCal3R` |
| 磁盘大小 | 17 M |
| 所有者 | `wangzheng:wangzheng` |
| Git remote | `https://github.com/Powertony102/ReCal3R.git` |
| 分支 | `main`，本地 tracking ref 为 `origin/main` |
| HEAD | `2b920d3f6eeac4cf3a95b9a83b12d82a74412314` |
| HEAD 摘要 | `Normalize dataset paths for local evaluation defaults` |
| 工作区 | 干净；index 和 worktree 均无 diff |
| 许可证 | MIT，见仓库 `LICENSE` |
| checkpoint | 无 |
| Python 环境 | 无 `.venv` |
| CUDA 扩展 | 无已编译 `.so` |
| 数据 | 无 TUM/ScanNet/Bonn 等评测数据 |

该目录位于 Project2 外，只能作为只读代码参照或 Project2 内独立 baseline clone 的本地
来源，不应直接修改、安装环境或写入输出。建议在
`/data/wangzheng/Project2/baselines/ReCal3R` 建立 Project2 内唯一工作副本，并保留官方
remote 信息。

### CUT3R 与派生内容

在 `/data/wangzheng` 内没有发现 remote 指向 CUT3R 官方仓库的 Git checkout。发现的
`cut3r` 名称仅出现在 Movie3R 的派生代码、文档或实验输出中，例如：

`/data/wangzheng/iJCV-CODE/Movie3R/output/v14/fine_alignment_research/cut3r_virtual_person_depth`

`/data/wangzheng/iJCV-CODE/Movie3R` 的 remote 是 Movie3R，而不是 CUT3R；审计时该仓库
存在大量用户未提交变化，严禁修改、回退或把它冒充为官方 CUT3R baseline。

## 权重与数据资源

### ReCal3R/CUT3R checkpoint

全路径定向搜索没有发现 ReCal3R README 要求的：

`cut3r_512_dpt_4_64.pth`

也没有发现 `*cut3r*.pth`、`*recal3r*.pth` 或相关 Hugging Face cache 目录。以下大权重
存在，但来源和模型均不同，不能替代官方 CUT3R checkpoint：

| 路径 | 字节数 | 身份 | 可否替代 |
| --- | ---: | --- | --- |
| `/data/wangzheng/iJCV-CODE/Movie3R/src/human3r_896L.pth` | 4,670,554,642 | Human3R/Movie3R 基础权重 | 否 |
| `/data/wangzheng/iJCV-CODE/Movie3R/checkpoints/v9_mixed_60h_pose_human_lora_bs10/checkpoint-final.pth` | 4,831,184,406 | Movie3R 训练产物 | 否 |

Phase 0 后续只能下载一个必要的官方 checkpoint。下载前需再次确认来源、文件大小、
许可证、目标磁盘余量和已有缓存；不得复制上述不兼容大权重。

### TUM RGB-D

没有发现以下任一资源：

- `rgbd_dataset_freiburg*` 原始序列目录；
- `fr1/desk`、`fr2/desk`、`fr3/walking_xyz` 对应目录；
- TUM/Freiburg `.tgz`、`.tar.gz` 或 `.zip` 压缩包；
- 同时具有 `rgb.txt`、`groundtruth.txt` 的可识别 TUM 原始布局。

ReCal3R relpose 可直接读取原始 TUM 布局，因此首轮不需要下载完整数据集。只下载一个
计划内序列，并首先用 2–8 帧做 smoke test。

## Python 与 CUDA 环境

### 可用工具

| 资源 | 路径或版本 | 结论 |
| --- | --- | --- |
| uv | `/home/wangzheng/.local/bin/uv`, `uv 0.9.18` | 可用于创建 Project2 内独立环境 |
| uv CPython | `/home/wangzheng/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu`，约 100 M | 与 ReCal3R README 的 Python 3.11 匹配 |
| 系统 Python | `/usr/bin/python3`, Python 3.8.10 | 不用于项目依赖 |
| conda | PATH 中不存在 | 不作为首选方案 |
| gpustat | 未安装 | 不为此安装系统包；使用 `nvidia-smi` |
| gcc/g++ | 11.2.0 | 可用 |
| CMake | 3.22.1 | 满足 README 的最低要求 |
| Ninja | 1.10.0 | 可用 |

应在 Project2 内的 ReCal3R 工作副本创建独立 `.venv`，并设置 Project2 内的
`UV_CACHE_DIR`、`HF_HOME`、`TORCH_HOME` 和仓库内 `TMPDIR`。不得修改其他仓库已有
环境。

### 只读参考环境

| 环境 | 大小 | Python / Torch | 兼容性结论 |
| --- | ---: | --- | --- |
| `/data/wangzheng/iJCV-CODE/Movie3R/.venv` | 6.8 G | Python 3.10.19, torch 2.4.0+cu124, torchvision 0.19.0+cu124, numpy 1.26.4 | 多数依赖存在，但 Python/CUDA 不同且缺 open3d/gdown；不可修改或直接复用 |
| `/data/wangzheng/iJCV-CODE/Movie3R-dataset/Depth-Anything-3/.venv` | 7.4 G | Python 3.10.19, torch 2.3.1+cu121, torchvision 0.18.1+cu121 | 缺少大量 ReCal3R 依赖；不可直接复用 |
| `/data/wangzheng/iJCV-CODE/Movie3R-dataset/.venv_data` | 2.7 G | Python 3.10.19, torch 2.2.1+cu121, torchvision 0.17.1+cu121 | 数据处理环境，不适合作为 baseline 环境 |

这些环境只用于版本和已有缓存参考。Project2 规则要求每个 baseline 保留独立环境。

### CUDA 工具链

- NVIDIA driver：`550.127.08`；`nvidia-smi` 显示最高支持 CUDA 12.4。
- 系统存在 CUDA 11.6、11.8、12.1 和 12.4。
- `/usr/local/cuda` 解析到 `/usr/local/cuda-12.1`。
- 审计时 shell PATH 中首先命中的 `nvcc` 是 `/usr/local/cuda-11.8/bin/nvcc`。

ReCal3R 官方安装建议 PyTorch CUDA 12.1。编译 CroCo RoPE CUDA 扩展时必须显式设置
`CUDA_HOME=/usr/local/cuda-12.1` 并把对应 `bin` 放在 PATH 前部，避免 PyTorch cu121
与 nvcc 11.8 不一致。

Movie3R 中存在：

`/data/wangzheng/iJCV-CODE/Movie3R/src/croco/models/curope/curope.cpython-310-x86_64-linux-gnu.so`

该文件约 9.1 M，绑定 CPython 3.10 和 Movie3R 的 Torch/CUDA ABI，不能复制给新的
Python 3.11 ReCal3R 环境使用。

## GPU 快照与进程归属

服务器包含 8 张 NVIDIA L20，每张总显存约 46,068 MiB。以下为 2026-07-31 18:32
CST 的只读快照；GPU 状态会变化，任何任务启动前必须重新检查。

| GPU | 已用 MiB | 空闲 MiB | 利用率 | 已识别进程所有者 | 审计结论 |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 395 | 45,195 | 0% | `root`, PID 5877，Open WebUI/uvicorn | 系统服务；不要使用或终止 |
| 1 | 30,854 | 14,736 | 0% | `xiaohan`, PIDs 3175314, 3440790, 3521045, 3759637 | 他人训练进程；不可干扰 |
| 2 | 4 | 45,586 | 0% | 无 compute process | 快照时空闲候选卡 |
| 3 | 18,507 | 27,083 | 100% | `yilin`, PID 147097 | 他人活跃进程；不可干扰 |
| 4 | 4 | 45,586 | 0% | 无 compute process | 快照时空闲候选卡 |
| 5 | 26,139 | 19,451 | 100% | `yilin`, PID 2352378 | 他人活跃进程；不可干扰 |
| 6 | 25,245 | 20,345 | 100% | `yilin`, PID 150397 | 他人活跃进程；不可干扰 |
| 7 | 41,495 | 4,095 | 0% | `junqi`, PID 3342614，显存保持进程 | 他人进程；不可干扰 |

快照时没有属于 `wangzheng` 的 GPU compute process。GPU 2 和 4 只是当时的候选；正式
smoke test 前需重新运行 `nvidia-smi`，确认无新增进程后使用
`CUDA_VISIBLE_DEVICES=<id>` 绑定一张卡。不得终止任何上述进程。

## ReCal3R 实际运行路径

### 模块解析

relpose 入口使用 checkpoint 所在目录决定 `dust3r` 的导入位置：

1. `/data/wangzheng/iJCV-CODE/ReCal3R/eval/relpose/launch.py:534` 调用
   `add_path_to_dust3r(args.weights)`；
2. `/data/wangzheng/iJCV-CODE/ReCal3R/add_ckpt_path.py:6-9` 将 checkpoint 目录插入
   `sys.path[0]`；
3. `launch.py:537` 执行 `from dust3r.model import ARCroco3DStereo`。

当权重参数为官方约定的 `.../ReCal3R/src/cut3r_512_dpt_4_64.pth` 时，静态解析验证的
实际模块是：

`/data/wangzheng/iJCV-CODE/ReCal3R/src/dust3r/model.py`

因此外部只读 checkpoint 如需复用，应通过 Project2 baseline 的 `src/` 内软链接引用，
并在 smoke runner 中断言 `dust3r.model.__file__` 以 `/src/dust3r/model.py` 结尾。

### relpose 调用链

实际调用链为：

1. `eval/relpose/launch.py:391-395` 选择 `inference_recurrent_lighter`；
2. `src/dust3r/inference.py:290-315` 中该 helper 在 `:311-313` 调用
   `model.forward_recurrent_lighter(..., ret_state=True)`；
3. `src/dust3r/model.py:1661-1833` 执行逐帧轻量 recurrent 路径。

relpose 不经过 `src/dust3r/model.py:1449-1455` 的 `forward()`，也不经过
`:1338-1447` 的 `_forward_impl()`。

### 实际 ReCal3R 分支

实际 `src/dust3r/model.py` 中没有“ReCal3R 分支未给 `update_mask1` 赋值”的问题：

- `:1662-1665` 先 canonicalize `model_update_type`；
- `:1793-1794` 首帧或 reset 后直接令 `update_mask1 = update_mask`；
- `:1796-1805` 后续帧分别处理 CUT3R、TTT3R 和 ReCal3R；
- ReCal3R 在 `:1803-1805` 确实调用 `_compute_recal3r_update_mask(...)`；
- `:1808-1815` 应用最终学习率、更新状态并记录状态变化。

非 relpose 路径的 `_forward_impl()` 也在 `:1418-1420` 调用了同一个 ReCal3R helper。

ReCal3R 计算本体位于 `src/dust3r/model.py:1199-1251`：

- `:1206-1207` 取得 token alignment gate；
- `:1215-1220` 更新 recent pressure，并计算 attenuation 与 residual；
- `:1225-1235` 从 state drift prior 和 attention entropy 形成 reliability 中间量；
- `:1242-1245` 形成最终 token learning rate；
- `:1251` 返回实际应用的 `update_mask * beta_t`。

`src/dust3r/blocks.py:245-257` 明确返回 pre-softmax attention logits，所以当前实现不是
对已经 softmax 的概率再次错误取 sigmoid。

### 根目录陈旧重复实现

仓库根目录还存在另一个：

`/data/wangzheng/iJCV-CODE/ReCal3R/model.py`

它不是 relpose 正常路径所导入的文件，但确有以下问题：

- 根 `model.py:1007-1032` 的 `_forward_impl` ReCal3R 分支只计算 alignment/residual，
  没有赋值 `update_mask1`，之后 `:1037` 使用会触发 `UnboundLocalError`；
- 根 `model.py:1401-1408` 的 `forward_recurrent_lighter` 只支持 CUT3R/TTT3R，
  ReCal3R 会触发 `ValueError`；
- 根 `model.py:363-406` 定义的老 `_compute_recal3r_update_mask` 没有被上述路径调用。

任何修复或 instrumentation 都必须先确认目标是 `src/dust3r/model.py`，不能修改错根目录
重复文件。

## 可用的无侵入 health-signal hook

首轮可在 StateGuard3R 外部 runner 中对模型实例做临时 method wrapper，记录 detached
聚合值并原样返回，无需修改官方 baseline：

| 信号 | Hook 位置 | 解释 |
| --- | --- | --- |
| reliability / entropy / token delta | `src/dust3r/model.py:852-897` 的内置 calibration trace，以及 `:949-1012` recorder | 推理前调用 `enable_u_calibration_trace(oracle_window=1)`，推理后读取 `get_u_calibration_trace()` |
| alignment gate | wrap `_compute_state_update_mask`，`:1024-1028` | 返回 tuple 第 2 项 `beta_t` 即 alignment gate |
| 最终 update coefficient | wrap `_compute_recal3r_update_mask`，`:1199-1251` | 原函数返回值是实际 token 学习率 mask |
| 未 gate 的 candidate update norm | wrap `_recurrent_rollout`，`:774-792` | 比较返回的 `new_state_feat` 与输入 `state_feat` |
| 实际 state delta | wrap `_maybe_record_u_calibration_step`，`:949-1012` | 参数已经包含最终更新前后的 `state_prev/state_post` |
| 原始 attention 汇总 | `model.dec_blocks_state[*].cross_attn` 的标准 forward hook | `blocks.py:245-257` 的 output[1] 是 pre-softmax logits |

不能保存每层完整 attention tensor；应在 GPU 上立即计算 mean/min/max/quantile 或 token
聚合，然后只把小型 detached 结果移到 CPU。

内置 trace 的字段名需要特别解释：`_compute_recal3r_u()` 在
`src/dust3r/model.py:1054-1056` 返回的是论文 reliability 的补量
`U = 1 - gamma * (2*gamma - 1)^2`。因此 trace 中的 `u` 是不可靠度/回退权重，health
ledger 中的 reliability 应记录为 `1 - u`，不能直接把 `u` 标成 reliability。

## 已发现风险

### 中风险：lighter 返回空 `state_args`

`src/dust3r/model.py:1667` 初始化 `all_state_args = []`，直到 `:1831-1832` 返回前没有
任何 append。relpose 丢弃该返回值，因此当前 baseline 轨迹输出不直接受影响，但不能靠
返回的 `state_args` 做 ledger、snapshot 或 rollback。首轮应使用上述 runtime hook；若
后续修复源码，最小改动是仿照 `forward_recurrent` 在初始化和每帧结束时 append 状态。

### 中风险：relpose 未显式调用 `model.eval()`

`eval/relpose/launch.py:728-744` 加载并配置模型后直接进入评测，没有像 demo 和
mv_recon 一样调用 `model.eval()`。当前主要 dropout 配置可能为 0，但这仍会影响确定性
和未来 checkpoint。最小修复是在配置完成后、评测开始前调用 `model.eval()`。

### 中风险：导入路径依赖 checkpoint 目录

若直接把 `--weights` 指向不含 `dust3r/` 的外部目录，可能导入失败或误用环境中的其他
实现。最小防护是把 checkpoint 软链接放到 baseline `src/`，并对模块绝对路径做断言。

### 中风险：默认 TUM shell 脚本启动两个进程

`eval/relpose/run_tum.sh:15` 默认 `EVAL_NUM_PROCESSES=2`，`:48` 使用 accelerate 启动。
单卡 smoke 不应使用默认值；应直接运行单进程 Python，或显式设
`EVAL_NUM_PROCESSES=1`。

### 低风险：配置项没有实际消费

`uncertainty_clamp_max` 只在 `src/dust3r/model.py:823-831` 定义 helper，当前没有调用
点。不能在报告中把它描述为已经生效的阈值或 clamp。

### 评测范围风险

- 首帧在 `forward_recurrent_lighter:1793-1794` 无条件更新；至少 2 帧才会调用一次
  ReCal3R calibration。1 帧 smoke 不能证明 ReCal3R 分支生效。
- 2 帧只适合运行路径 smoke；Sim(3)、ATE/RPE 可能因样本过少退化。完整 evaluator
  smoke 建议 4–8 帧。
- `run_tum.sh` 默认规模是 1,000 帧，不适合作为首次测试。
- relpose parser 当前只允许 `scannet*` 和 `tum*` 前缀，而文档仍给出 Bonn relpose
  命令；Bonn 后续接入前需单独复核 parser choices。

## 最小运行验证协议

当独立环境、唯一 checkpoint 和一条 TUM 序列就绪后，先运行 2 帧单卡 smoke：

```bash
cd /data/wangzheng/Project2/baselines/ReCal3R
CUDA_VISIBLE_DEVICES=<重新确认的空闲卡> \
  .venv/bin/python eval/relpose/launch.py \
  --weights "$PWD/src/cut3r_512_dpt_4_64.pth" \
  --output_dir "$PWD/outputs/smoke_tum_recal3r_2f" \
  --eval_dataset tum \
  --dataset_path "$PWD/data/tum" \
  --seq_list rgbd_dataset_freiburg3_walking_xyz \
  --pose_eval_stride 1 \
  --max_frames 2 \
  --size 224 \
  --model_update_type recal3r \
  --device cuda
```

TUM root 应包含：

```text
rgbd_dataset_freiburg3_walking_xyz/
├── rgb/
├── rgb.txt
└── groundtruth.txt
```

无 reset 时，2 帧运行中 `_compute_recal3r_update_mask` 应调用 1 次；3 帧应调用 2 次。
外部 runner 应对此计数做断言，从而证明不是“参数写了 recal3r，但实际没有进入分支”。
2 帧路径通过后，再增至 4–8 帧检查轨迹与指标，然后才进入更长短序列。

## Phase 0 结论

| 检查项 | 状态 | 结论 |
| --- | --- | --- |
| 官方 ReCal3R 来源与许可证 | 通过 | 本地有干净 MIT 源码，可作为 Project2 内 clone 来源 |
| CUT3R 官方独立仓库 | 未就绪 | 首轮 ReCal3R smoke 不要求先复现完整 CUT3R 仓库 |
| 必要 checkpoint | 阻塞 | 本地不存在，只应下载一个官方权重 |
| 最小 TUM 数据 | 阻塞 | 本地不存在，只应下载一条序列 |
| Python 3.11 / uv | 通过 | runtime 和 uv 已有，仍需创建 baseline 独立 `.venv` |
| CUDA 编译条件 | 有条件通过 | CUDA 12.1 可用，但必须覆盖 PATH 中优先命中的 nvcc 11.8 |
| GPU | 有条件通过 | 快照时 GPU 2/4 空闲；运行前必须重查 |
| 磁盘 | 有条件通过 | 552 G 可用但 `/data` 已达 96%，禁止扩大下载范围 |
| relpose ReCal3R 实际分支 | 静态通过 | 正确调用 src lighter 和 ReCal3R update helper |
| health signal 可观测性 | 通过 | 可用内置 trace 与 runtime wrapper，无需先改 baseline |
| 真实 2 帧 smoke | 未执行 | 等待环境、权重与单序列数据就绪 |

总体判断为 **有条件 Go**：源码来源、许可证、工具链和无侵入可观测点已经明确，且实际
relpose 路径不存在所担心的 `update_mask1` 未赋值问题。当前真实执行的三个前置阻塞是
独立环境、唯一官方 checkpoint 和一条最小 TUM 序列。解除后应严格按“2 帧路径 →
4–8 帧端到端 → 短序列 health ledger”的顺序推进，不应提前下载完整数据集或启动大规模
评测。
