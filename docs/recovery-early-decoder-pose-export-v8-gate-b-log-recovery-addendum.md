# Recovery early-decoder pose-token export v8：Gate B 日志恢复附录

- 日期：2026-08-07
- 状态：**预注册；只授权一个新的 dynamic always-control，尚未授权 candidate。**
- 原计划：`docs/recovery-early-decoder-pose-export-development-v8-execution-plan.md`
- Gate A：`docs/audits/recovery-early-decoder-pose-export-v8-gate-a.md`（PASS）

## 0001 的不可复用 preflight artifact

`recovery-early-decoder-pose-v8-dynamic-always-commit-0001` 没有启动 GPU forward。
其 preflight 在 two GPU-2 snapshots（均大于 12 GiB）和 all-new-path checks 成功后已经写入，
但 launcher 的 NUL-free self-check 被错误地放在 preflight write 与 `chmod 0444` 之间。该 shell
check 退出，因此没有执行 tmux pane creation、`pipe-pane`、CUDA command、main log、transcript、driver/result、
postflight 或 output directory。

现存 `logs/recovery-early-decoder-pose-v8-dynamic-always-commit-0001-preflight.log` 已检查为 NUL-free 并冻结为
`0444`。它是 incomplete preflight evidence，不能作为 Gate B control PASS，不能补写、覆盖、删除或事后构造
transcript；尤其不能据此启动 candidate/wrong/low/GT evaluator。

这不是模型、数据、detector、runtime 或质量结果：CUDA child 从未被 dispatch，因而没有对 v8 scientific
hypothesis 的正/负证据。

## 唯一恢复授权

只授权一次新的 direct-child output：

```text
recovery-early-decoder-pose-v8-dynamic-always-commit-0002
```

它仍是 `always-commit` control。除 canonical terminal-evidence ID 与 corrected preflight freeze order 外，
它必须与 Gate A 绑定机制完全相同：same StateGuard3R committed v8 components、clean ReCal3R
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`、same checkpoint/hash、30-frame dynamic manifest、Detector-v3
config、seed 0、size 512、beta `0.1`、health v3、watchdog 8、GPU 2 UUID and `CUDA_VISIBLE_DEVICES=2`。

在 actual `pipe-pane` dispatch 前，launcher must verify that `0002` output plus
preflight/main/postflight/transcript/driver/result paths are all new. It must persist two GPU 2 free-memory snapshots
(`>=12288 MiB`) into preflight, verify its NUL-free bytes **before** `chmod 0444`, then use a fresh pane in existing
`stateguard` session, configure live `pipe-pane`, and only then send the child command. Canonical preflight/main/
postflight/transcript and driver/result are all NUL-free `0444`; output root is `0555`, every output file `0444`.

`0002` may authorize exactly one dynamic candidate only if protected-file byte equivalence, full permissions/provenance,
timeline binding and runtime ratio `<=1.20` all pass. Any failure freezes v8 as
`EARLY_DECODER_POSE_EXPORT_V8_AVAILABILITY_OR_RUNTIME_NO_GO`; it does not authorize a second recovery or candidate.
