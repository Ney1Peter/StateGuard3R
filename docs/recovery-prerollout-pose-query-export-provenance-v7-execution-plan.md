# Recovery pre-rollout pose-query export provenance v7：可恢复 tmux 证据计划

- 制定日期：2026-08-07
- 状态：**预注册，尚未实现/运行**
- 长时目标：以 v7 Gate A、Gate B 与（仅在 B 通过后）Gate C 的正常终态为止；任何 hard gate 失败均停止该版本，保留全部不可变证据并进入新的、明确不同的版本。
- 唯一目标：在**不改变** v6 的 pre-rollout pose-query scientific mechanism、Detector-v3、输入 manifests、checkpoint、阈值或质量门的前提下，修复 v6 的 terminal provenance 缺口，验证该机制能否通过完整的可复现 dynamic probe 与冻结 quality matrix。

## 1. v6 的前置终态与 v7 的唯一变化

v6 Gate A 已通过，dynamic always-control 的 protected files byte-equivalence 与 runtime
`1.140228x` 都通过；但 v6 launcher 定义而未写入 `*-tmux-transcript.log`，对应 pane 已销毁，故 v6 已冻结为
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V6_PROVENANCE_NO_GO`。详细证据见
`docs/audits/recovery-prerollout-pose-query-export-v6-control-provenance-no-go.md`。

v7 的唯一新假设不是 pose 估计假设，而是**可复现性假设**：在 tmux pane 创建后、任何 command 输出前以
`tmux pipe-pane` 建立 transcript，把 exact command、child stdout/stderr、PID、exit marker 捕获到一个 one-use
file；orchestrator 只在 transcript 已验证包含起止 marker 后写 postflight，并把 preflight、main、postflight、transcript
和 driver manifest 全部冻结。它不改变任何模型输入/输出与 policy 数值。

以下 v6 scientific components 必须保持 byte-identical：

| component | v6 control SHA-256 |
| --- | --- |
| `scripts/run_recal3r_prerollout_pose_query_export_v6.py` | `af4d3d684f08b1dab16c14b9408041e70a9c35dd7375c43f1406387d9085c081` |
| `src/stateguard3r/recal3r_prerollout_pose_query_runner_v6.py` | `a5e21e231bea2fe452a2082274a13725b760ed45cf687b31f43dd08e1ebd64f3` |
| `src/stateguard3r/prerollout_pose_query_export_v6.py` | `ac7ff67981dbe9e083f81416a1f7d87fea2b7485134b9c6b5436422ca8d605bf` |
| `src/stateguard3r/prerollout_pose_query_v6.py` | `3becaa5b4d7f793e869a10200493f2e18c4b079327c3958a0d3f52973c0785e1` |

任何上述 hash 或 frozen Detector-v3/checkpoint/manifest/source hash 漂移均为
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V7_IMPLEMENTATION_NO_GO`，不启动 GPU。

## 2. v7 的不可变 tmux capture protocol

新增一个 tracked, test-covered orchestrator。其真实 GPU command 必须由这个 orchestrator 在 existing
`stateguard` session 的 fresh named pane 中启动，并遵守以下顺序：

1. caller 在非 GPU phase 检查 StateGuard3R/ReCal3R clean、new direct-child output/log IDs、four v6 component
   hashes、GPU 2 两次 `>=12 GiB` preflight；所有 command、preflight snapshot、component/source provenance 写入
   immutable manifest。
2. orchestrator 创建 empty tmux pane，**先**对该 pane 配置 `pipe-pane` 到 new transcript，再 `send-keys` 运行
   generated driver；不得让 child command 先于 `pipe-pane` 开始。
3. driver 的 terminal output 必须包括 run ID、fully quoted command、child PID、child stdout/stderr、unambiguous
   `V7_DRIVER_EXIT=<integer>` marker。main log 同时接收同一 child output；driver 用 a separate result manifest 把 exit
   code 交给 orchestrator。
4. orchestrator 验证 result/marker/run-id/hash 和 exit code 一致，写 postflight。只有成功时才将 output `0555`、其
   files `0444`，并把 preflight/main/postflight/transcript/driver/result files 一律 `0444`。缺失、空 transcript、marker
   mismatch、pane creation/capture failure 或 nonzero exit 都 fail closed，绝不写“成功”postflight。
5. run completion 之后 pane 可以自然退出；其 transcript 已是独立持久 artifact。不得从 `capture-pane` 事后生成或
   拼接 transcript，且 never reuse any output/log/pane name.

transcript 只记录 v7 command 的 terminal evidence；不得携带 secrets、GT、labels/future frames 或任何 dataset 下载。

## 3. 不可变科学与数据边界

- 只读 `outputs/formal-v1-inputs-0001/development/development-{dynamic,wrong,low}/input-manifest.json`；不下载数据、权重或依赖。
- 原封不动调用 v6 algorithm runner with `always-commit` 或
  `detector-v3-incremental-prerollout-pose-query-export`；不改 source token、float64 decoder clarification、watchdog `8`、raw-health ledger、rollback、Detector-v3 config、RNG seed、size 或 `beta-base`。
- 禁止读取 development GT、logical base、event labels 或运行 quality evaluator，直到 six v7 outputs 均冻结。
- GPU 仅使用 `CUDA_VISIBLE_DEVICES=2`，但每 run 必须以实际 UUID
  `GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250` 和 two preflights binding；只要可用空间满足 `>=12 GiB` 即可使用，不要求完全空闲。

## 4. Gate A：protocol 与 lineage

GPU 前必须全部通过：

- v7 orchestrator unit tests：one-use refusal、empty pane then `pipe-pane` then dispatch order、quoted command preservation、
  no child starts before capture, nonzero child/empty/missing/mismatched transcript fail closed, result/marker agreement, no GPU API;
- a real no-data/no-GPU tmux smoke: `stdout` plus a deliberately nonzero child must leave a complete immutable transcript with
  exact command/PID/exit marker and a frozen failure artifact; then a successful child must satisfy the corresponding success
  protocol in a separate ID;
- re-run the v6 Gate A source/AST/CPU checks, actual checkpoint-loaded model interface/float32-to-float64 camera check,
  project-local full CPU suite and compileall; verify all four scientific component SHA-256 values above;
- code, tests and docs in separate commits; StateGuard3R and ReCal3R clean; no v6 output/log altered.

Any Gate A failure is `PRE_ROLLOUT_POSE_QUERY_EXPORT_V7_IMPLEMENTATION_NO_GO`, and v7 GPU must not start.

## 5. Gate B：dynamic short circuit

1. Create exactly one v7 dynamic control
   `recovery-prerollout-pose-query-v7-dynamic-always-commit-0001`. It must have all six v7 provenance artifacts, an
   unambiguous transcript, protected-file byte equivalence to frozen v1 dynamic baseline and synchronized
   runtime/baseline `<=1.20`.
2. Only then create exactly one v7 candidate
   `recovery-prerollout-pose-query-v7-dynamic-candidate-0001`. Every real alarm must have full rollback/witness,
   `export_pre_rollout_pose_query`, complete query evidence, no fallback and candidate/baseline runtime `<=1.20`; its transcript
   must additionally bind candidate policy and all exit markers.
3. Any failure is `PRE_ROLLOUT_POSE_QUERY_EXPORT_V7_AVAILABILITY_OR_RUNTIME_NO_GO`: do not run wrong/low, GT evaluator, or
   alter/retry v7 scientific components/output IDs.

## 6. Gate C：frozen matrix and scientific selection

Only Gate B PASS authorizes one always-control and one candidate each for wrong and low, giving six v7 frozen outputs. Then and
only then may the CPU evaluator read logical-base GT and compute prefix `0--14` Sim(3), tail `19--29` ATE/RPE effects
`(baseline-candidate)/baseline`. Candidate-ready requires:

- all three controls preserve four protected files byte-for-byte; every alarm has complete v6 algorithm and v7 provenance witness;
  zero restore/query/watchdog/leakage/protocol failure;
- at least 2/3 conditions have both ATE and translation-RPE positive, with both medians `>=+5%`; candidate runtime median `<=1.20`;
- all output/log/driver/transcript permissions and source/timeline hash bindings are intact.

Pass writes only an unexecuted blind-acquisition plan and returns
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V7_CANDIDATE_READY`. Otherwise returns
`PRE_ROLLOUT_POSE_QUERY_EXPORT_V7_FEASIBILITY_NO_GO`. Both end v7; a NO-GO does not authorize retuning this disclosed matrix.
