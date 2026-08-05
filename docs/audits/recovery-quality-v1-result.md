# Recovery-quality v1 正式结果

- 评测日期：2026-08-06
- 唯一正式结论：**`RECOVERY_QUALITY_NO_GO`**
- 范围：两条预注册、此前未产生 ReCal3R response 的 TUM RGB-D+GT 序列；每条序列的
  `clean`、`dynamic`、`wrong-order`、`low-overlap` 四个固定条件。
- 长时目标完成状态：**已完成并停止**。目标是完成固定的 2 × 4 × 3（baseline / always-commit /
  detector-policy）正式矩阵、八项后验评测及唯一聚合；它在唯一 seal 写入后以 NO-GO 正常结束，
  不因结果不利而重跑、调参或扩大 formal 数据。

## 结论

本轮完整、可审计地证明了 external transactional hold/replay 能按 alarm 执行，并且
`always-commit` 与原始 baseline 在所要求的模型产物上逐字节等价；但它**没有**在六个 corruption
trial 中改善 ATE 或平移 RPE。六个 event trial 的两项 effect 均为 `0.0`，且 policy 的运行时间
中位比为 `1.7210640716`（高于 1.20 上限）。因此不能声称该 policy 带来 recovery-quality 改善。

这不是 feasibility failure：所有 24 条 GPU forward、8 个 evaluator 和唯一 aggregation 都成功，
没有 restore failure。它是预注册质量门槛未通过的完整 NO-GO。

## 不可变证据链

| Artifact | SHA-256 |
| --- | --- |
| formal commitment | `001f41d9e8619d652ca1d9f0782d857257de30476c97064c7b67e2db0c590646` |
| scene A CPU input validation | `332ac3f347ac8595344e8dd627942c931f2f608a5cf20ae4d3004ec8a378c230` |
| scene B CPU input validation | `b0ab9220a7d87a782b06fac822ef2933266898580767b2f8bb2805769c62ff5e` |
| sole aggregate result | `88a6d50771211dae43ee8340bbcf7682a29f4dc7df968cf9be2e0946fd1db934` |
| sole aggregate attempt seal | `1a5259298cb84743cd379059e055689a47244e117ab2485b695a60a728aa8204` |

The commitment was written at `2026-08-06 01:46:17 CST`, before the formal forwards.  It binds
StateGuard3R commit `a115efbd26dce0b8966228ce5c558ed872fd30a3`, clean ReCal3R commit
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff`, the 3,173,761,006-byte checkpoint
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`, eight frozen manifests,
the Detector v3 config, and runner contract `cuda / size=512 / seed=0 / beta=0.1 / health=v2`.
The scene-B official archive is the sole new download and has SHA-256
`1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d`.

The aggregation wrote `result.json` and `attempt-seal.json` at `02:20:26 CST`.  This is the only
`recovery-quality-formal-aggregation-*` directory.  Its directory is `0555`, both files are
`0444`, the seal says `sealed: true`, and its recorded result hash matches the actual result.
All eight referenced evaluator files were independently re-hashed and are `0444`; all eight
match the commitment's scene, condition, and input-manifest inventory.

## 完成的正式矩阵

For each of the eight inputs, execution was strictly `baseline -> CPU alarm -> always-commit ->
detector-policy -> CPU evaluator`.  Thus the matrix contains 24 serial GPU forwards, all exit
code zero, and eight frozen evaluators.  Each evaluator reports byte equivalence of baseline and
always-commit for `checkpoint-load-audit.json`, `health.jsonl`, `predictions-summary.json`, and
`trajectory.json`.  Prefix-only, non-reflective Sim(3) alignment uses frames 0--14; all reported
ATE/RPE values below are the fixed tail frames 19--29 metrics.

| Scene | Condition | baseline / policy ATE (m) | baseline / policy translation RPE (m) | alarms | hold / replay | runtime ratio | ATE / RPE effect |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `walking_static` | clean | 0.01228458 / 0.01228458 | 0.00192187 / 0.00192187 | 15, 21 | 2 / 2 | 1.4841 | 0 / 0 |
| `walking_static` | dynamic | 0.01201392 / 0.01201392 | 0.00184376 / 0.00184376 | 15 | 1 / 1 | 1.9171 | 0 / 0 |
| `walking_static` | wrong-order | 0.01221281 / 0.01221281 | 0.00190543 / 0.00190543 | 15, 21 | 2 / 2 | 1.6134 | 0 / 0 |
| `walking_static` | low-overlap | 0.01163990 / 0.01163990 | 0.00177000 / 0.00177000 | 15, 17, 19 | 3 / 3 | 1.9900 | 0 / 0 |
| `walking_xyz` | clean | 0.01945846 / 0.01945846 | 0.01160378 / 0.01160378 | none | 0 / 0 | 2.0244 | 0 / 0 |
| `walking_xyz` | dynamic | 0.01845478 / 0.01845478 | 0.01162412 / 0.01162412 | 15, 17 | 2 / 2 | 1.7720 | 0 / 0 |
| `walking_xyz` | wrong-order | 0.01952298 / 0.01952298 | 0.01157589 / 0.01157589 | 16 | 1 / 1 | 1.6701 | 0 / 0 |
| `walking_xyz` | low-overlap | 0.02063401 / 0.02063401 | 0.01106629 / 0.01106629 | 15, 19 | 2 / 2 | 1.5588 | 0 / 0 |

`wrong-order` produced a rising-edge alarm and actual hold/replay in both scenes, satisfying that
behavioral gate.  The static clean run also alarmed; its unchanged metrics mean the clean
degradation gate passes, but this is still an important false-hold observation.  The policy's
rotation-RPE values are likewise byte-for-byte equal to baseline for every row (they are not a
decision metric).  The table does not omit the clean controls or any corruption trial.

## 预注册门禁

| Gate | Actual | Result |
| --- | ---: | --- |
| all evaluator provenance and baseline/always equivalence | 8 / 8 | PASS |
| no restore failure | 0 / 8 | PASS |
| wrong-order has hold and replay in each scene | 2 / 2 scenes | PASS |
| clean ATE and translation-RPE degradation | 0% in both clean trials | PASS |
| event trials | 6 | PASS (complete) |
| median ATE effect | 0.0, required >= 10% | **FAIL** |
| median translation-RPE effect | 0.0, required >= 10% | **FAIL** |
| positive-ATE event trials | 0 / 6, required >= 4 / 6 | **FAIL** |
| 10,000-replicate ATE bootstrap lower 95% | 0.0, required > 0 | **FAIL** |
| 10,000-replicate translation-RPE bootstrap lower 95% | 0.0, required > 0 | **FAIL** |
| policy / baseline runtime median | 1.7210640716, required <= 1.20 | **FAIL** |

Any one failed quality gate requires NO-GO; five fail here.  The aggregate decision is therefore
the protocol-required `RECOVERY_QUALITY_NO_GO`, not a discretionary interpretation.

## 运行异常及其处理

The first two direct CPU invocations of the scene-A clean alarm builder exited before publishing
an output with `ModuleNotFoundError: No module named 'scripts'`
(`logs/recovery-quality-formal-scene-a-clean-alarm-0001.log` and `-0002.log`).  They did not run
CUDA and created no alarm artifact.  The exact frozen code, input, threshold, policy, calibration,
and commitment were then used with the explicit process namespace
`PYTHONPATH=/data/wangzheng/Project2/StateGuard3R`; this published the one accepted alarm output
at `outputs/recovery-quality-formal-scene-a-clean-alarm-0001`.  This was a Python import-path
recovery only, not a code or experimental reconfiguration.  Both failed logs remain retained;
nothing successful was overwritten or rerun.

No further dataset or weight was downloaded.  Each GPU forward used one explicitly selected GPU
after a recorded free-memory check of at least 12 GiB, did not stop or alter another user's
process, and left no formal runner process after completion.

## 最终复核

Final CPU verification used a project-local pytest temporary root, as required by the runner's
output-isolation guard: `439 passed in 50.78s` in
`logs/recovery-quality-v1-final-cpu-suite-0002.log`.  `python -m compileall -q scripts tests`
passed, `git diff --check` passed, and a recursive post-run permission check found zero violations
of `0555` for formal directories and `0444` for formal files.  No StateGuard3R/ReCal3R formal
runner, alarm builder, evaluator, or aggregation process remained; no project-related listener
was present.

For transparency, the retained preceding log `...final-cpu-suite-0001.log` has 7 test failures
when pytest selected `/tmp/pytest-*`: those tests deliberately ask the runner to create an output
under that path, while the frozen runner correctly rejects outputs outside
`/data/wangzheng/Project2`.  The project-local rerun changed only the test temporary-directory
placement, created a new `0002` log, and passed all 439 tests; it did not change source, formal
input, commitment, policy, or any frozen artifact.

## 解释边界与后续约束

This result validates neither a geometric recovery benefit nor deployment readiness.  The alarm is
constructed from a completed baseline's causal per-frame health fields and then applied one frame
later to a separate forward.  It is therefore an auditable **offline causal shadow replay**, not a
single-path real-time deployment demonstration.  It also covers only two TUM sequences and the
three specified synthetic corruptions.

The two formal scenes are now disclosed and must not be used to retune this policy and then claim
confirmation.  The current long-term objective is closed: no more action is permitted under this
formal commitment.  A future investigation, if authorized, must be a separately preregistered
development-only diagnosis of why valid hold/replay leaves the trajectory unchanged, followed by
new unseen data and a new commitment before any renewed recovery-quality claim.
