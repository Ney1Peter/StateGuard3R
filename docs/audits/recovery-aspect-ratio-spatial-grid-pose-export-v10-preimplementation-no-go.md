# v10 aspect-ratio spatial-grid design: preimplementation no-go

- Date: 2026-08-07
- Decision: **not implemented; no GPU forward or v10 artifact exists**

The pre-registered v10 aspect-ratio design changed v9's literal decoder input
from `(1,576,1024)` to `(1,768,1024)`.  That is the real 512×384/16×16 grid,
but the candidate still reads `dec[0]`, takes the same arithmetic mean, calls
the same `decoder_embed`, and captures at the same post-rollout/pre-memory
location as v9.

It is therefore a v9 token-count parameter correction, not a mechanism-distinct
successor.  The frozen v9 Gate-B stop rule forbids changing its shape/pool
parameter and retrying.  The v10 design is retained as an unimplemented
chronology record only; do not add a v10 run ID, launcher, output, control,
candidate or quality evaluation.

The next admissible route is v11 encoder-global pre-retrieval export, whose
source is the frozen `model._get_img_level_feat(feat_i)` before recurrence and
does not read `dec[0]`.
