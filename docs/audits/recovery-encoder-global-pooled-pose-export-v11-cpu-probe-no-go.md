# v11 encoder-global pose export: Gate-A CPU real-frame availability NO-GO

- Date: 2026-08-07
- Decision: **`ENCODER_GLOBAL_POOLED_POSE_EXPORT_V11_GATE_A_CPU_REAL_FRAME_AVAILABILITY_NO_GO`**
- Scope: the one-frame, CUDA-disabled Gate-A probe required by the frozen
  [v11 execution plan](../recovery-encoder-global-pooled-pose-export-development-v11-execution-plan.md).

## What was attempted

The probe read only dynamic-development manifest frame 0,
`rgb/1305031461.059662.png`.  With `CUDA_VISIBLE_DEVICES=''`, it successfully
called `load_images_for_eval`, which reported native `640x480` and loaded
`img=(1,3,384,512)`, `true_shape=(1,2)`.  It then loaded the already-present
official checkpoint strictly (`<All keys matched successfully>`) on CPU and
called the required frozen first step:

```text
_encode_image(image_tensor, true_shape)
```

It failed inside ReCal3R's native encoder before `_get_img_level_feat`,
`decoder_embed`, pose-head decoding, token capture, recurrence, detector, or
any output/evidence write:

```text
RuntimeError: expected scalar type Float but found Half
...
croco/models/curope/curope2d.py:39 -> _kernels.rope_2d(...)
croco/models/curope/curope.cpp: rope_2d_cpu
```

The originally attempted metadata guard expected `true_shape=(2,)`; the
actual loader contract is `(1,2)`.  That guard was corrected before the run
above.  The terminal failure is therefore not the guard, a dataset problem,
or a model/checkpoint loading problem.

## Why this is conclusive for v11 Gate A

The pinned upstream source itself makes its CPU encoder path inconsistent:

1. `croco/models/blocks.py` converts the attention `q` and `k` tensors to
   `torch.float16` immediately before native RoPE.
2. `croco/models/curope/curope.cpp::rope_2d_cpu` unconditionally obtains
   `tokens.accessor<float, 4>()`; it accepts CPU float tokens only.
3. The pinned call routes the forced half tensor to that CPU implementation,
   producing the observed exception.

Changing ReCal3R/CuRoPE, bypassing RoPE, or installing an in-memory CPU shim
would not be the frozen native `_encode_image` required by v11.  Replacing the
v11 real-frame CPU gate with a CUDA forward after this result would also alter
the pre-registered gate.  Neither is an admissible repair of v11.

The probe had `CUDA_VISIBLE_DEVICES=''`; a separate same-environment check
returned `cuda_available=False` and `cuda_initialized=False`.  No GPU forward,
v11 dispatcher invocation, dynamic control/candidate, wrong/low condition,
GT read, quality evaluation, download, or new weight occurred.  The success
evidence destination
`logs/recovery-encoder-global-pooled-pose-v11-dynamic-frame0-cpu-probe-0001.json`
does not exist, because the probe writes only after the complete token-only
path succeeds and no latent values were captured or serialized.

## Bound provenance

| Item | SHA-256 / value |
| --- | --- |
| ReCal3R commit | `466c7cdf3acd2f589f1d82e5f6391966f19db9ff` |
| checkpoint | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| `dust3r/model.py` | `32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1` |
| `croco/models/blocks.py` | `1ad00f2be14dcc38c6322011e134a9cba30ad445e0fe5ceb50fbb65b9febd807` |
| `croco/models/curope/curope.cpp` | `d715f5275600bc4f3f3fdc9aabec529a5954a9e6d161446346f580c68561927a` |
| v11 probe at execution | `edb8513d4a1fb71eeffef4b7afb21956ff072841fd930f92065826dd99b73d2d` |
| v11 primitive at execution | `bd409b831d34b30d5e15dd4147fe11c30c0ad29bf3109b60743aed1ef8d47ca5` |
| v11 runner at execution | `f90a413648eff58db220c57cbbf5f76d23fedf7e980e6963d591d0be530c5088` |

The tracked ReCal3R worktree was clean when the probe started.

## Required stop and successor boundary

v11 is terminal at Gate A.  Do not run its GPU-2 dynamic always-control,
candidate, wrong/low controls or candidates, dispatcher, quality evaluator,
or a new v11 run ID.  Do not alter its projection, frozen pose head, detector,
launcher, CPU gate, or upstream baseline to retry it.

Any successor must be pre-registered under a new ID and be mechanism-distinct
from v11's current-frame encoder-global latent.  It must also state a CPU
availability contract that is executable against the pinned upstream without
an encoder/RoPE substitution before it can request any CUDA work.
