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

## Materialization evidence for `synthetic-smoke-0003`

The original 0003 CLI run generated the moving manifest and passed the strict
loader, but did not materialize image pixels. A follow-up CPU-only audit at
clean StateGuard3R commit `483ade139400bce27a5f94bc9c9a08a8b0d40c65`
loaded all 60 final ordered paths with ReCal3R's official
`load_images_for_eval` and passed them through `materialize_manifest_views`.
Every model input had shape `1x3x384x512`; CUDA remained uninitialized and no
raster file was written.

The audit retained its failed first attempt in the run record: an inline check
incorrectly called generic `load_images` with `crop=True`, which raised
`TypeError` before materialization and wrote no file. The successful retry used
the runner's real `load_images_for_eval(size=512, crop=True)` path.

The five moving rectangles produced these exact pixel bounds, with right and
bottom endpoints exclusive:

```text
frame 25: (128,  96, 384, 288)  49,152 pixels
frame 26: (148, 105, 405, 298)  49,601 pixels
frame 27: (168, 115, 425, 308)  49,601 pixels
frame 28: (189, 124, 446, 317)  49,601 pixels
frame 29: (209, 134, 466, 327)  49,601 pixels
```

For each frame, the rectangle interior was exactly `[1, -1, -1]` in normalized
model-input space and all pixels outside the rectangle were unchanged. Outputs
were independent clones, all loader-input tensor hashes remained unchanged,
and both Chateau source-file hashes were identical before and after replay.

The two-image fixture limits what the non-pixel corruption routes demonstrate.
Although `low_overlap_jump` selected source indices `50–54`, only 3/5 final
paths differed from their original positions. The reversed wrong-order indices
`[44,43,42,41,40]` changed 4/5 paths because its center frame maps to itself.
These checks validate manifest ordering and provenance only. The audit did not
run ReCal3R inference and is not evidence of real corruption response or
detection performance.
