# v19 official native final-mask hold: Gate-A NO-GO

- Date: 2026-08-14
- Decision: **`V19_IMPLEMENTATION_OR_INPUT_NO_GO` — terminal before CUDA.**
- Scope: test the pre-registered causal premise for the official native-mask
  wrapper.  No v19 capability, CUDA/tmux work, control, candidate, wrong/low,
  GT read, download or upstream modification occurred.

## What passed

The new scalar arm policy and instance-level native-mask wrapper are internally
consistent.  They prove that a native mask can be calculated first and, only on
an armed immediate-next frame, replaced by a zero final state mask while the
official loop retains its separate native memory update.  The static source
audit is bound to clean ReCal3R `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` and
model SHA-256 `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1`.

The allowed native-mask audit confirms the official call sequence:

```text
native _compute_recal3r_update_mask
-> update_mask2 = update_mask
-> state_feat uses update_mask1
-> mem uses update_mask2
-> calibration
```

It therefore supports the proposed *state-only* mechanism in isolation; it is
not a runtime authorization by itself.

## Falsified causal premise

V19 also required causal previous/current model-ready RGB overlap before the
current native rollout.  The pre-registration allowed only three instance
wrappers: `_downstream_head`, `_compute_recal3r_update_mask`, and
`_maybe_record_u_calibration_step`.  The independent pinned-source audit finds
the official order:

```text
_encode_image -> _recurrent_rollout -> _downstream_head
-> _compute_recal3r_update_mask -> _maybe_record_u_calibration_step
```

All three approved wrappers occur after the rollout.  None can observe or
derive the current model-ready RGB before it.  Creating the registered overlap
at any of these later points would violate the stated causal order.  The only
ways to obtain the required pre-rollout input are an additional encode/device
wrapper, a copied recurrent loop, or offline/future-frame precomputation;
each is expressly prohibited by v19.

The executable audit therefore reports:

```text
pre_rollout_rgb_hook_available: false
approved_wrappers_all_after_rollout: true
```

Gate A fails closed.  It does not silently remove overlap, shift it to after
rollout, or add another hook after the design has been registered.

## Verification and stop rule

With `CUDA_VISIBLE_DEVICES=''`, **15 tests passed in 0.17 seconds**.  They
cover scalar arm/consume/reset/last-frame behavior; native-mask-first and
zero-return wrapper behavior; exception restoration; same-frame
consume-plus-alarm rejection; held-state identity in a fake official loop; and
the independent causal-hook fail-closed audit.  `compileall` and
`git diff --check` passed.

ReCal3R remains clean and pinned.  No new input/weight/dependency was
downloaded, and existing v14/v15 untracked terminal files were untouched.
V19 is terminal: do not repair/re-run it, build its capability, start CUDA or
weaken/reorder its detector input.  Any next version must use fresh IDs and a
newly pre-registered causal observation boundary.
