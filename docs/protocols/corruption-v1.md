# Corruption manifest v1

The current persisted corruption-manifest schema is exactly:

```text
stateguard3r.corruption.v1
```

The generator always emits this version. The replay loader accepts only this
exact version: a missing version, `stateguard3r.corruption.v0`, and unknown
future versions are rejected instead of being upgraded or interpreted on a
best-effort basis.

## Coordinate contract

Version 1 makes the dynamic-occlusion coordinate semantics explicit. Every
`rectangle_occlusion` frame transform contains both:

```json
{
  "coordinate_space": "normalized",
  "coordinate_reference": "model_input_after_resize_and_center_crop"
}
```

The corresponding `dynamic_occlusion` label parameters record the same two
fields. The normalized rectangle therefore applies to the image tensor after
the model loader has resized and center-cropped it, immediately before model
inference. Replay applies the RGB fill only to an in-memory clone and converts
RGB values from `[0, 255]` to the model input range `[-1, 1]`; it never edits a
source image.

This field is the schema-breaking difference from the earlier v0 contract.
Removing it or changing its value does not produce a valid v1 manifest.

## Compatibility and stale artifacts

There is intentionally no implicit v0-to-v1 migration. A v0 manifest must be
regenerated from its original read-only source manifest and corruption specs so
that the coordinate contract is recorded and validated.

`outputs/synthetic-smoke-0001` is retained byte-for-byte as a stale development
artifact. Its manifest is labelled v0 even though it was produced while the
coordinate-reference field was being introduced, so it is not accepted by the
current loader and must not be cited as a current reproducible run. A fresh
`synthetic-smoke-0002` first validated v1 but used a zero-velocity occlusion.
`synthetic-smoke-0003` uses non-zero motion and is the current v1 pipeline
evidence; both remain synthetic fixtures, not ReCal3R experiment results.
