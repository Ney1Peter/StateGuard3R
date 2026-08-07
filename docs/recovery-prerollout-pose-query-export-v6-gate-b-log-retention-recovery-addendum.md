# Recovery pre-rollout pose-query export v6：Gate B 日志留存恢复附录

- 日期：2026-08-07
- 状态：**预注册；仅授权一个新的 control，尚未授权 candidate。**
- 目的：恢复一次已完成但审计证据不完整的 control 的日志留存，不改变或再次搜索 v6 机制。

## 0001 的不可采纳记录

`recovery-prerollout-pose-query-v6-dynamic-always-commit-0001` 已正常退出，输出目录及其 seven files 是
`0555/0444`；四个 protected files 与 frozen v1 dynamic baseline 逐字节相同，runtime 为
`3.949031349271536 / 3.4633694058284163 = 1.1402281670057521`。两次 GPU 2 preflight、main、postflight
logs 都存在且是 `0444`，postflight 为 exit 0。

但是 launcher 虽声明了 canonical `...-tmux-transcript.log`，tmux window 退出时没有保留 pane，而 transcript
从未写入。它不能事后重建为同一 run 的原始 tmux artifact。因此 0001 的唯一正式状态是
**`UNACCEPTED_AUDIT_INCOMPLETE`**：保留、不得覆盖、不得作为 Gate B PASS，也不得据其运行 candidate、wrong、low
或任何 GT evaluator。

## 唯一授权的恢复 run

只授权一个新的 direct-child output：

`recovery-prerollout-pose-query-v6-dynamic-always-commit-0002`

它仍然是 **always-commit control**，不是 candidate，也不会读取或利用 0001 的数值。除 canonical tmux transcript
留存外，所有运行语义必须与 0001 完全一致：同一 30-frame dynamic development manifest
`264b0c9e4f04f4da9e26a634ed008342d32c7b561d7b6ee4f9eb90d1688189c8`、checkpoint
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`、Detector-v3 config
`2c89d356cb6e33af536975def0106bfe9bd6368edaa1094e26414d1c51052748`、GPU 2 UUID
`GPU-d2be321e-2001-7e74-d0f0-3ee103fcd250`、CUDA visible device `2`、seed 0、size 512、beta 0.1、health v3、
watchdog 8 和 runner argv。

冻结 v6 executable component SHA-256 必须分别仍为：

- `scripts/run_recal3r_prerollout_pose_query_export_v6.py`:
  `af4d3d684f08b1dab16c14b9408041e70a9c35dd7375c43f1406387d9085c081`
- `src/stateguard3r/recal3r_prerollout_pose_query_runner_v6.py`:
  `a5e21e231bea2fe452a2082274a13725b760ed45cf687b31f43dd08e1ebd64f3`
- `src/stateguard3r/prerollout_pose_query_export_v6.py`:
  `ac7ff67981dbe9e083f81416a1f7d87fea2b7485134b9c6b5436422ca8d605bf`
- `src/stateguard3r/prerollout_pose_query_v6.py`:
  `3becaa5b4d7f793e869a10200493f2e18c4b079327c3958a0d3f52973c0785e1`

本附录是文档提交，因而 runner provenance 的 repository commit 可以不同于 0001；只有上述 executable component
bytes 才是机制身份，并且必须与 0001 相同。任何 component hash 漂移使 0002 fail closed，不得修改代码后重跑。

## 0002 的额外且唯一流程要求

在 tmux 启动 launcher **之前**，orchestrator 必须确认 output 与四个拟定 canonical logs 都不存在；随后只创建一个
空的 transcript file 并让 `tmux pipe-pane` 从启动时持续写入它。launcher 要求这个 transcript 正是空、可写的预建立
文件，并仍自行拒绝所有其他 output/log collision。tmux pane 必须保留至 transcript 的 SHA-256、内容和 `0444` mode
被验证；transcript 需包含 window、launcher、run id 和 launcher exit marker。完成后四个 canonical logs
`preflight/main/postflight/tmux-transcript` 必须都是 NUL-free `0444`。

两次 preflight 均需 GPU 2 free memory `>=12288 MiB`。output directory 必须 `0555`、每一文件 `0444`。control 只有在
四个 protected artifacts byte-equivalent、timeline path/hash binding、component hash、exit、permissions、canonical logs
以及 `runtime/baseline <=1.20` 全部通过时，才重新授权计划原有的唯一
`recovery-prerollout-pose-query-v6-dynamic-candidate-0001`。任何失败仍是 v6 Gate B short-circuit，禁止 candidate。
