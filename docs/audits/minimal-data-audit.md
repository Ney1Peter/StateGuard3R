# 最小数据获取审计

> 审计日期：2026-07-31
>
> 适用仓库：`/data/wangzheng/Project2/baselines/ReCal3R`
>
> 状态：本审计仅使用官方网页、GitHub 内容和 HTTP HEAD 元数据；编写时未下载任何数据包。

## 1. 结论

首轮数据获取必须严格按以下顺序执行，并在每一层通过验收后才进入下一层：

1. **零数据集下载：仓库内两张图片。** 使用 `src/croco/assets/Chateau1.png` 和 `src/croco/assets/Chateau2.png` 验证图片加载、checkpoint 加载、两帧 recurrent forward 和 ReCal3R 更新分支。它们只是 CroCo 随仓库提供的资产，不是 ReCal3R benchmark。
2. **单个约 8 MB 官方视频。** 只下载 TUM `fr1_xyz` 的 RGB-only AVI，做 10--30 帧视频解码和连续推理 smoke test。该视频没有可用于严谨评测的原始时间戳、depth 和逐帧 GT，不能报告 ATE/RPE。
3. **单个完整评测序列。** 前两层都成功后，只下载 `fr1_desk.tgz`。它是当前候选中最小的完整计划内序列，可直接用于 ReCal3R 的 TUM pose evaluator。
4. **停止自动扩张。** 在 `fr1_desk` 上完成 clean baseline、连续窗口 health logging 和最小 corruption smoke test 前，不下载 `fr2_desk`、`fr3_walking_static`、`fr3_walking_xyz` 或其他数据集。

严禁运行 `datasets_preprocess/long_prepare_tum.py`。当前脚本会为 50、100、150、200、300、400、500、600、700、800、900、1000 等多个目标长度反复复制 RGB 图片，造成不必要的数据重复。当前 ReCal3R 已能直接读取 TUM 原始目录，无需该预处理副本。

## 2. 官方来源与精确压缩体积

下表字节数来自 2026-07-31 对 TUM 官方 canonical URL 的 HEAD 请求。链接会重定向到 TUM 的 `webshare.cvg.cit.tum.de`，最终响应均为 HTTP 200，并支持 byte ranges。

| 序列/文件 | 官方 URL | Content-Length | 二进制体积 | 决策 |
|---|---|---:|---:|---|
| `fr1_xyz` RGB AVI | [rgbd_dataset_freiburg1_xyz-rgb.avi](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz-rgb.avi) | 8,059,298 B | 7.69 MiB | 第二层唯一下载 |
| `fr1_xyz` TGZ | [rgbd_dataset_freiburg1_xyz.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz) | 448,204,271 B | 427.44 MiB | 暂不下载；AVI 已承担调试用途 |
| `fr1_desk` RGB AVI | [rgbd_dataset_freiburg1_desk-rgb.avi](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk-rgb.avi) | 6,141,142 B | 5.86 MiB | 不下载，避免第二个同类视频 |
| `fr1_desk` TGZ | [rgbd_dataset_freiburg1_desk.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz) | 344,011,403 B | 328.07 MiB | 第三层唯一完整包 |
| `fr1_desk` standalone GT | [rgbd_dataset_freiburg1_desk-groundtruth.txt](https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk-groundtruth.txt) | 158,143 B | 154.44 KiB | TGZ 正常时不单独下载 |
| `fr3_walking_static` TGZ | [rgbd_dataset_freiburg3_walking_static.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_static.tgz) | 474,074,105 B | 452.11 MiB | 后续动态场景候选，默认不下载 |
| `fr3_walking_xyz` TGZ | [rgbd_dataset_freiburg3_walking_xyz.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz) | 527,550,055 B | 503.11 MiB | 后续动态加相机运动候选，默认不下载 |
| `fr2_desk` TGZ | [rgbd_dataset_freiburg2_desk.tgz](https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz) | 1,893,351,095 B | 1.763 GiB | 明确延后 |

TUM 官方没有公布上述 TGZ 的准确解压体积。本轮受“不下载数据”约束，不能可靠实测，因而不填入伪精确数字。下载完成后、解压前应使用 `gzip -l` 和 `tar -tzf` 记录实际信息。数据包、解压目录同时存在时，建议仅就数据本身预留至少压缩包体积的 3 倍；模型输出另设配额：

- `fr1_desk`：至少预留 1 GiB 数据空间；
- `fr1_xyz`：至少预留 1.3 GiB；
- 单个 `fr3_walking_*`：至少预留 1.5 GiB；
- `fr2_desk`：至少预留 5.3 GiB。

## 3. 格式、GT 与关联规则

TUM 官方格式说明如下：

- RGB 为 `640x480`、8-bit RGB PNG。
- Depth 为 `640x480`、16-bit 单通道 PNG。
- RGB 与 depth 已由 OpenNI 预配准，像素一一对应。
- PNG depth scale 为 `5000`：像素值 5000 表示 1 米，0 表示缺失值。
- 原始序列约为 30 Hz，但 RGB 与 depth 的时间戳不完全同步。
- `rgb.txt` 和 `depth.txt` 保存时间戳到图片相对路径的映射。
- 官方 `associate.py` 默认使用 `0.02 s` 最大时间差匹配异步流。
- GT 文本中的每一行格式为：

  ```text
  timestamp tx ty tz qx qy qz qw
  ```

- `timestamp` 是 Unix epoch 秒；平移单位为米；四元数顺序必须保持为 `qx qy qz qw`；以 `#` 开头的行是注释。

完整 TGZ 解压后应形成类似以下原始布局：

```text
data/tum/
└── rgbd_dataset_freiburg1_desk/
    ├── rgb/
    ├── depth/
    ├── rgb.txt
    ├── depth.txt
    └── groundtruth.txt
```

ReCal3R `main` 在审计时对应 commit `466c7cdf3acd2f589f1d82e5f6391966f19db9ff`。该版本的 relpose evaluator 能直接读取上述 raw layout，并可通过 `--eval_dataset tum`、`--dataset_path` 和 `--seq_list rgbd_dataset_freiburg1_desk` 只运行一个序列。

## 4. 严格下载门禁

### Gate 0：零数据集下载

输入仅限：

```text
src/croco/assets/Chateau1.png
src/croco/assets/Chateau2.png
```

验收条件：

- 两张图片可被官方 loader 读取；
- checkpoint 能加载；
- 显式选择 `model_update_type=recal3r` 后能完成 recurrent forward；
- 能观察到预期输出结构和最小 reliability/update 信号；
- 没有 CUDA kernel、dtype、显存或路径错误。

这一步只能证明程序路径可运行，不能说明长时状态稳定性，也不能形成任何数据集指标。

### Gate 1：单个 8 MB AVI

只允许下载：

```text
https://cvg.cit.tum.de/rgbd/dataset/freiburg1/
rgbd_dataset_freiburg1_xyz-rgb.avi
```

TUM 官方把 `fr1_xyz` 描述为适合 debugging 的简单序列。ReCal3R `demo.py` 能直接读取视频；初次使用高 `frame_interval` 将输入限制在约 10 帧，通过后再增加到约 30 帧。

执行注意：

- `demo.py` 默认选择 `cut3r`，必须显式传入 `--model_update_type recal3r`。
- 视频解帧使用 `tempfile.mkdtemp()`，必须把 `TMPDIR` 指向 ReCal3R 仓库内部的专用临时目录，不能写入系统 `/tmp`。
- 每次使用全新的专用 `--output_dir`；demo 会删除该目录中既有的 `depth`、`conf`、`color` 和 `camera` 子目录。
- viewer 会占用端口并阻塞运行；必须使用确认空闲的端口，自动化 smoke 应优先调用无 viewer 的推理入口。
- AVI 丢失了 TUM 原始图片文件名和精确采集时间戳，不能将其与 standalone GT 拼接后宣称正式 ATE/RPE。

验收条件：视频可解码、10--30 帧连续推理完成、状态没有立即发散、输出数量与处理帧数一致。未通过时停止，不下载 TGZ。

### Gate 2：唯一完整包 `fr1_desk`

Gate 0 和 Gate 1 均通过后，才允许下载 `rgbd_dataset_freiburg1_desk.tgz`。先运行极小评测，再扩大：

1. 8 帧 I/O、推理与结果落盘 smoke；
2. 30--50 个**连续帧**的 health ledger 和 corruption smoke；
3. 同一序列上的完整 clean ATE/RPE 与 runtime/memory baseline。

注意：当前 `eval/relpose/launch.py` 的 `--max_frames N` 会跨整个序列近似均匀取 N 帧，并不等同于“前 N 个连续帧”。均匀子采样可用于功能检查，但会改变相邻帧 overlap，不能用于状态污染传播结论。连续窗口必须通过明确的 start/end 或 manifest 选择实现；如需构造文件视图，应优先使用软链接，不能复制 RGB 数据。

### Gate 3：默认停止

完成 `fr1_desk` 前不再获取数据。后续确有证据需要真实动态场景时，只选择一个：

- `fr3_walking_static` 更小，且相机近静止，适合隔离快速动态人物造成的污染；
- `fr3_walking_xyz` 同时包含动态人物和相机运动，问题更复杂。

不得同时下载两者；`fr2_desk` 继续延后。

## 5. 下载、完整性与解压校验

每次只处理一个明确 URL，并完成以下检查：

1. 下载前搜索 `/data/wangzheng` 下是否已有同名文件或已解压目录，并检查目标文件系统剩余空间。
2. 用 HEAD 请求记录 canonical URL、最终重定向 URL、HTTP 状态、`Content-Length`、`Last-Modified` 和 `ETag`。
3. 下载到 ReCal3R 数据目录中的 `.part` 临时文件，使用断点续传；字节数完整后再改成最终文件名。
4. 用 `stat -c '%s'` 将本地文件精确字节数与本审计表对比。
5. 对 TGZ 运行 `gzip -t`，失败时不得解压。
6. 用 `gzip -l` 记录压缩和未压缩 tar 大小；对于可能超过 4 GiB 的未压缩流，注意 gzip ISIZE 的 32-bit 取模限制，并用完整 tar 列表补充核查。
7. 用 `tar -tzf` 只读检查成员，确认只有预期的单一序列顶层目录，不含绝对路径或 `..` 路径穿越，并包含 `rgb/`、`depth/`、`rgb.txt`、`depth.txt` 和 `groundtruth.txt`。
8. 计算并记录本地 `sha256sum`，后续所有实验复用同一份文件。
9. 解压到 `baselines/ReCal3R/data/tum/`，直接读取 raw layout；不生成 `long_tum_s1` 的多份图片副本。

TUM 下载页没有提供这些包的官方 MD5/SHA 清单。HTTP `ETag` 在当前服务器上是 Apache 文件标识，不能当作密码学内容哈希。首次下载产生的本地 SHA-256 是项目内部复现标识，而不是官方发布的 checksum。

## 6. 许可证与引用要求

TUM 官方声明：除非另行说明，TUM RGB-D benchmark 数据采用 [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)；随附源代码采用 BSD-2-Clause。项目必须：

- 在数据 manifest 和实验报告中记录 TUM 数据来源、序列名、canonical URL 和 CC BY 4.0；
- 保留署名，不对数据施加额外限制；
- 论文或报告引用官方 benchmark 论文：

  ```text
  J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers,
  "A Benchmark for the Evaluation of RGB-D SLAM Systems,"
  Proc. IROS, 2012.
  ```

ReCal3R 官方仓库代码声明为 MIT License。模型权重由 README 指向 CUT3R checkpoint；权重的再分发条件应单独核对，不能仅依据 ReCal3R 的 MIT License 推断。

## 7. 官方参考

- [TUM RGB-D benchmark 首页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [TUM 官方下载目录](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
- [TUM 文件格式](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
- [TUM 时间戳关联与 ATE/RPE 工具](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools)
- [TUM benchmark 论文 PDF](https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf)
- [ReCal3R 官方仓库](https://github.com/Powertony102/ReCal3R)
- [ReCal3R evaluation 文档](https://github.com/Powertony102/ReCal3R/blob/main/eval/eval.md)
- [ReCal3R TUM 预处理脚本](https://github.com/Powertony102/ReCal3R/blob/main/datasets_preprocess/long_prepare_tum.py)
- [ReCal3R demo](https://github.com/Powertony102/ReCal3R/blob/main/demo.py)
