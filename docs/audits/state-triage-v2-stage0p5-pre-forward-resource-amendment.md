# StateTriage3R v2 Stage 0.5 pre-forward resource-placement amendment

Date: 2026-08-24  
Status: recorded before any Stage 0.5 forward response

The Stage 0.5 execution plan named GPU 2.  Immediately before the first
forward, GPU 2 had 5,166 MiB free, while the frozen Stage 0 observer runs on
the same L20 reported a 6,366.72 MiB allocated peak.  GPU 2 was therefore not
an available placement under the user's "use a shared GPU when it has space"
constraint: it lacked the required space.

The 12 serial observer-only forwards will instead use `CUDA_VISIBLE_DEVICES=3`.
At the same check, physical GPU 3 was an NVIDIA L20 with 45,129 MiB free and
461 MiB already in use.  It is a shared, not exclusively reserved, placement.

This amendment changes neither the input inventory nor its hashes, observer or
runner source hashes, decision order/constants, output names, metrics, or
gates frozen in `state-triage-v2-stage0p5-gate-b.json`.  It must be cited in
the final audit.  A GPU change is not an additional run or a response retry.
