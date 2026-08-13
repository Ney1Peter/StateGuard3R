# v18 one-shot native update-hold: Gate-A NO-GO

- Date: 2026-08-14
- Decision: **`V18_IMPLEMENTATION_OR_INPUT_NO_GO` — terminal before CUDA.**
- Scope: the committed v18 plan required an official-loop-only hook and a
  held-frame GPU identity witness for both native state and native memory.  The
  pinned source proves that this exact contract cannot be implemented without a
  prohibited route.  No v18 dynamic capability, CUDA/tmux execution, control,
  candidate, wrong/low condition, GT read, download or upstream modification
  occurred.

## Falsified feasibility premise

V18 deliberately rejected a copied recurrent loop in response to v17's
candidate-memory failure.  Its registered hook could wrap only the official
downstream head and `_maybe_record_u_calibration_step`, after the upstream
state/memory commits.  The static audit binds pinned ReCal3R source
`466c7cdf3acd2f589f1d82e5f6391966f19db9ff` and SHA-256
`32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1`.

In `forward_recurrent_lighter`, the native order is downstream prediction,
state write, memory write, then calibration.  Its permitted downstream call
has arguments `(head_input, shape)`, while its sole registered post-commit
callback has exactly `(frame_idx, state_prev, state_post)`.  `mem` is a local
variable of the upstream loop and is not passed to either callback.  Hence the
state GPU summary is observable, but the required held-memory GPU summary is
not.

The executable audit reports:

```text
state_gpu_witness_available:  true
memory_gpu_witness_available: false
forbidden routes needed for memory witness:
  copied_recurrent_loop
  stack_frame_reflection
  additional_native_method_wrapper
```

Each listed route is explicitly prohibited by the v18 plan.  Relaxing this
after the fact would make the v18 evidence non-falsifiable, so Gate A fails
closed rather than silently reducing the causal claim from state-and-memory to
state-only.

## Verification and stop rule

With `CUDA_VISIBLE_DEVICES=''`, the new v18 scalar arm and official-loop audit
suite passed **12 tests in 0.12 seconds**.  It covers one-arm, consume, clear,
last-frame identity and reset cancellation; input type rejection; pinned
official source/callback argument checks; and the expected fail-closed Gate-A
condition.  `compileall` and `git diff --check` also passed.

No v18 release artifact was created.  ReCal3R remains clean at its pin; no
dependency, data or weight was downloaded; existing v14/v15 untracked terminal
files were not touched.  This is terminal v18: do not repair its hook, weaken
the memory witness, build a v18 capability, run CUDA or reuse v18 IDs.  A
future experiment must use fresh versioned code/IDs and pre-register a distinct
causal evidence contract.
