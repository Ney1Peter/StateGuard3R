# v10 aspect-ratio spatial-grid design: preimplementation no-go

- Date: 2026-08-07
- Decision: **V10_ASPECT_RATIO_SPATIAL_GRID_PRE_GATE_A_NO_GO**

The pre-registered v10 aspect-ratio design changed v9's literal decoder input
from `(1,576,1024)` to `(1,768,1024)`.  That is the real 512×384/16×16 grid,
but the candidate still reads `dec[0]`, takes the same arithmetic mean, calls
the same `decoder_embed`, and captures at the same post-rollout/pre-memory
location as v9.

It is therefore a v9 token-count parameter correction, not a mechanism-distinct
successor.  The frozen v9 Gate-B stop rule forbids changing its shape/pool
parameter and retrying.

Before this review was completed, the isolated runner and its synthetic tests
were added in commits `f2e2840` and `060a9ca`.  They add only new v10 files and
do not alter any frozen v1--v9 source or artifact.  No v10 Gate-A audit,
launcher, CUDA forward, run ID, output, control, candidate, or quality
evaluation exists.  Those commits must remain an accurate chronology record;
they do not authorize the design to advance.

Do not add a v10 run ID, launcher, output, control, candidate, or quality
evaluation, and do not modify the 768 count, pooling, head, detector, or
logging to retry it.  This is a pre-Gate-A mechanism-boundary no-go, not an
experimental result.

The next admissible route is v11 encoder-global pre-retrieval export, whose
source is the frozen `model._get_img_level_feat(feat_i)` before recurrence and
does not read `dec[0]`.
