# Detector v2 的 formal-v1 失败分析

> 分析日期：2026-08-02
>
> 状态：**PASS（CPU-only、仅作已解盲 development 诊断）**
>
> 不改变 formal-v1 的 `NO-GO`，也不构成新的 formal evaluation。

## 1. 可复算证据

分析器 [`scripts/analyze_detection_v1_failure.py`](../../scripts/analyze_detection_v1_failure.py)
对 formal-v1 的六条已冻结 ledger 和 evaluation manifest 完成了只读回放。运行时
`CUDA_VISIBLE_DEVICES=''`，未导入 `torch`，未启动 GPU forward，且没有改动任何
`formal-v1-*` 产物。

| 项目 | 值 |
|---|---|
| formal-v1 evaluation manifest | `ce17d98ac000d9b1e5df8852553708d03990ba61f42fba743e9626d850792c15` |
| 被 manifest 逐项验证的 artifact 数 | 50 |
| 分析脚本执行时 SHA-256 | `bc26308439ff0f537581e9cf4cc3ff9f6e433b6e8c28589d9116b6de91c394d1` |
| 分析输出 `analysis.json` | `9e10ead4d13d3d873b1bb6dca5b9139ce8f7659a92cce0905fbac36178a53b05` |
| 分析输出 `report.md` | `b2cb57b49daf229d15e17163a8f49719d41be19d698ae3ae7660ac59253a5318` |
| 输出目录 | `outputs/detector-v2-analysis-0001/`（目录 `0555`，文件 `0444`） |

回放重建了 development 和已公开 holdout 的全部 `6 × 4` method/run score、mask、
FP/FN、最长连续 FP、event delay、recovery boundary 与 washout 诊断。重建的
`go-no-go.json` 与 frozen 文件逐字节一致，仍为 `NO-GO`；唯一失败的合取门槛仍是
combined equal-corruption macro-AUROC 严格大于 `0.75`。

## 2. 直接失败证据

冻结 holdout 的 combined macro-AUROC 为 `0.6434920634920634`。primary mask 的
combined 混淆计数仍是 TP=7、FP=10、TN=51、FN=7；10 个 FP 中 9 个出现在污染开始前的
clean prefix。逐 run 的 combined primary errors 为：

| Run | Primary FP | FN | Washout alarms |
|---|---|---|---|
| `holdout-dynamic` | 2, 7, 8, 27 | 17, 18, 19 | 无 |
| `holdout-wrong` | 2, 4, 7 | 17, 18 | 19 |
| `holdout-low` | 2, 4, 10 | 18, 19 | 20, 21 |

这说明 v1 虽然在三个 event 的第一个污染帧均报警（delay 0），但 clean prefix 的排序
不稳定，且 event 后半段仍有漏检。Reliability-only 的 pooled FPR 为 `1.0`，最长 FP
streak 为 `15`，证实直接使用 `-reliability` 已饱和。三条 holdout 的 combined 都把
`overlap` 记录为 missing；本分析没有用 GT、标签或零值虚构该信号。

## 3. 只用于设计的 rank ablation

下表是使用冻结 v1 score transform 的 threshold-free、equal-corruption holdout macro-AUROC。
它们只用于提出 Detector v2 的实现假设，不能被当作新的 gate，也不能用于把同一 holdout
重新称为 blind。

| Signal | 单信号 | 从 v1 combined 移除该信号 |
|---|---:|---:|
| geometric residual | 0.662540 | 0.622698 |
| pose jump | 0.672857 | 0.663333 |
| update magnitude | 0.528333 | 0.682857 |
| reliability | 0.521905 | 0.643492 |
| overlap | 不可用（0 个 finite 值） | 0.643492（v1 本就没有它） |

因此 v2 的改动方向是由数据和协议共同限定的：用 causal visual correspondence coverage
补齐缺失的 online overlap；对 reliability 做相对历史的单向下降异常而非 raw negation；对
所有信号使用同尺度、direction-aware 的 robust component；并在不删除 clean negative 的
前提下加显式 warm-up/min-history 与 development-derived scale floor。

所有完整 score timeline、字段来源、位置分类，以及 development/holdout 的 single 和
leave-one-out 结果均保存在冻结的
`outputs/detector-v2-analysis-0001/analysis.json`。formal-v1 holdout 从此仅能作为 v2
development/diagnostic 数据；后续 confirmatory evaluation 必须按 v2 计划使用新场景。
