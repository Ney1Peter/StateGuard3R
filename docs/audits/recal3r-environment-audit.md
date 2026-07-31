# ReCal3R 独立环境执行记录

日期：2026-07-31

目标仓库：`/data/wangzheng/Project2/baselines/ReCal3R`

固定上游：`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`

## 1. 结果

ReCal3R 独立 Python 环境和 CUDA RoPE 扩展已经建立，静态模块导入通过。环境位于
baseline 自己的 `.venv`，下载缓存、临时目录、Hugging Face 和 Torch cache 全部位于
`/data/wangzheng/Project2`。

已验证：

- Python 3.11.14；
- PyTorch 2.4.0+cu121；
- torchvision 0.19.0+cu121；
- NumPy 1.26.4；
- Pillow 10.3.0；
- `uv pip check` 最终通过；
- 实际导入文件为 `src/dust3r/model.py`；
- `ARCroco3DStereo.enable_u_calibration_trace` 存在；
- CUDA 12.1 RoPE 扩展编译成功。

尚未验证：

- checkpoint 加载；
- GPU 上的 RoPE kernel 执行；
- 两帧 ReCal3R forward；
- TUM clean baseline。

原因分别是官方 Google Drive checkpoint 当前无法从本机连接，以及最近一次 GPU 复查时
没有确认完全空闲的 GPU。

## 2. 环境创建

环境创建命令：

```bash
cd /data/wangzheng/Project2/baselines/ReCal3R
UV_CACHE_DIR=/data/wangzheng/Project2/.cache/uv \
TMPDIR=/data/wangzheng/Project2/baselines/ReCal3R/tmp \
uv venv --python 3.11 .venv
```

PyTorch 使用明确的 CUDA 12.1 wheel，而不是让未锁定的 `requirements.txt` 自动选择最新
版本：

```bash
UV_CACHE_DIR=/data/wangzheng/Project2/.cache/uv \
TMPDIR=/data/wangzheng/Project2/baselines/ReCal3R/tmp \
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.0 torchvision==0.19.0
```

随后安装官方 requirements 和缺失的首轮依赖：

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python gdown scikit-image evo
```

本轮未安装 `open3d`，因为 relpose 和 detection-only 暂不需要它；避免在首轮扩大环境。

## 3. 实测依赖冲突与固定版本

官方 requirements 未锁大多数依赖，直接安装产生了运行时不兼容：

1. resolver 安装 `transformers==5.14.1`，但该版本从
   `torch.distributed.tensor` 导入当前 PyTorch 2.4 没有导出的 `DTensor`，导致
   `dust3r.model` 无法导入；固定为 `transformers==4.44.2` 后恢复。
2. 最新 `gradio==6.22.0` 要求新 Hugging Face Hub，与 transformers 4.44.2 的
   `<1.0` 约束冲突；固定为 `gradio==4.44.1`。
3. 最新 viser 与 gradio 对 websockets 的主版本要求冲突；最终固定为
   `viser==0.2.10` 和 `websockets==12.0`。
4. `hf-gradio` 是初次解析留下的孤立包，已从本独立环境卸载。

最终关键 pins：

```text
torch==2.4.0+cu121
torchvision==0.19.0+cu121
transformers==4.44.2
huggingface-hub==0.36.2
tokenizers==0.19.1
gradio==4.44.1
gradio-client==1.3.0
viser==0.2.10
websockets==12.0
numpy==1.26.4
pillow==10.3.0
```

完整 freeze 保存在 `docs/runs/recal3r-environment-freeze.txt`。

## 4. RoPE CUDA 扩展

编译使用显式 CUDA 12.1，避免 PATH 中较早的 CUDA 11.8：

```bash
cd src/croco/models/curope
CUDA_HOME=/usr/local/cuda-12.1 \
PATH=/usr/local/cuda-12.1/bin:/usr/local/bin:/usr/bin:/bin \
TORCH_CUDA_ARCH_LIST=8.9 \
MAX_JOBS=4 \
TMPDIR=/data/wangzheng/Project2/baselines/ReCal3R/tmp \
/data/wangzheng/Project2/baselines/ReCal3R/.venv/bin/python \
  setup.py build_ext --inplace
```

编译成功并生成 CPython 3.11 扩展。上游 `setup.py` 自行硬编码多个架构，因此实际 nvcc
命令包含 `sm_50` 到 `sm_90`，没有采用只编译 `sm_89` 的优化。GPU 执行仍需在确认空闲
的 L20 上做一次小张量测试。

编译目录、`.so`、`.venv` 和运行输出均通过 ReCal3R 本地 exclude 或上游 `.gitignore`
隔离，固定官方 worktree 的 tracked 状态保持干净。

## 5. Checkpoint 获取阻塞

本地审计没有找到 `cut3r_512_dpt_4_64.pth`。本轮只尝试 README 提供的唯一官方 Google
Drive 文件 ID：

```text
1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD
```

执行情况：

1. README 的 `gdown --fuzzy` 与当前 `gdown==6.1.0` 不兼容，只打印参数错误且未创建文件。
2. 改用官方 file ID 和 `--no-cookies --continue` 后，连接
   `drive.google.com:443` 超时。
3. 对官方 `drive.google.com` 和 `drive.usercontent.google.com` 入口的只读 HEAD 请求也在
   DNS/连接阶段超时。
4. 失败后没有遗留 `.part` 或伪 checkpoint 文件。

没有转用第三方镜像，也没有把 Human3R/Movie3R 权重冒充官方 CUT3R checkpoint。
checkpoint 可用前，真实 ReCal3R forward 和 baseline 维持未完成状态。

## 6. GPU 复查

18:32 的首次快照中 GPU 2/4 空闲。21:40 前后的再次快照显示：

- GPU 2 已出现 `jiale` 的两个进程；
- GPU 4 已出现约 24.6 GiB 的其他进程并满负载；
- GPU 0 有系统服务和其他用户进程；
- 其余 GPU 也有其他用户任务或显存占用。

因此本轮没有启动任何 GPU 程序，也没有终止或干扰任何进程。后续每次测试前必须再次
运行 `nvidia-smi` 并只使用当时确认完全空闲的卡。

## 7. 当前门槛判断

环境门槛：通过。

checkpoint 门槛：网络阻塞，未通过。

两帧模型 smoke：依赖 checkpoint，未执行。

可并行继续的内容：corruption manifest、health ledger、detection-only 评测和合成测试。
这些模块不得被写成“真实 ReCal3R 实验已完成”。
