"""External transactional clone of pinned ReCal3R lighter recurrent execution.

The body below is line-for-line derived from the pinned ReCal3R
``forward_recurrent_lighter`` span 1661--1833.  It is deliberately maintained
outside the baseline checkout and refuses to run unless that exact source hash
and line-span hash are still present.  The only semantic additions are the
explicit state transaction around a candidate update and the causal hold/replay
controller; when every prior alarm is false, the added branch is commit-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .recal3r_state_v3 import (
    CausalHoldReplayController,
    StateClosure,
    StateTransaction,
    canonical_digest,
    capture_model_state,
    verify_pinned_lighter_source,
)


class TransactionalReCal3RError(RuntimeError):
    """Raised when the externally controlled lighter execution is invalid."""


def _clone_reset_mask(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        return detach().clone()
    if callable(clone):
        return clone()
    return value


@dataclass(frozen=True)
class TransactionalReCal3RResult:
    """Predictions plus an auditable commit timeline from external execution."""

    predictions: list[Mapping[str, Any]]
    views: Sequence[Mapping[str, Any]]
    state_closures: list[StateClosure]
    timeline: list[Mapping[str, Any]]
    dropped_transaction_frame_ids: list[int]
    source_provenance: Mapping[str, Any]


def _timeline_entry(
    transaction: StateTransaction,
    *,
    action: str,
    reason: str,
    prior_alarm: bool,
    replayed_transaction_frame_id: int | None,
    committed_state: StateClosure,
) -> dict[str, Any]:
    transaction_digest = transaction.digest()
    return {
        "frame_id": transaction.frame_id,
        "action": action,
        "reason": reason,
        "prior_alarm": prior_alarm,
        "replayed_transaction_frame_id": replayed_transaction_frame_id,
        "proposal_digest_sha256": canonical_digest(transaction_digest),
        "pre_state_digest_sha256": canonical_digest(transaction.pre_state.digest()),
        "proposed_state_digest_sha256": canonical_digest(transaction.post_state.digest()),
        "committed_state_digest_sha256": canonical_digest(committed_state.digest()),
    }


def run_transactional_recurrent_lighter(
    views: Sequence[Mapping[str, Any]],
    model: Any,
    device: Any,
    *,
    torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str],
    alarms: Sequence[bool],
    max_hold: int = 3,
    verify_source: bool = True,
) -> TransactionalReCal3RResult:
    """Run the hash-bound lighter path with causal transactional state control.

    ``alarms[t]`` is never used to decide frame ``t``.  At frame ``t > 0`` the
    controller sees only ``alarms[t - 1]``.  A held candidate restores its full
    pre-state/model/RNG closure; a later clear score replays the newest held
    post-state before processing that later frame.
    """

    if not views:
        raise TransactionalReCal3RError("views must not be empty")
    if len(alarms) != len(views):
        raise TransactionalReCal3RError("alarm count must equal view count")
    source_provenance = verify_pinned_lighter_source() if verify_source else {}
    controller = CausalHoldReplayController(alarms, max_hold=max_hold)

    # The following recurrent calculation mirrors pinned source lines
    # 1661--1833.  Transaction capture/restore is the only new control flow.
    model.config.model_update_type = canonicalize_model_update_type(
        getattr(model.config, "model_update_type", "cut3r")
    )
    model_update_type = model.config.model_update_type
    ress: list[Mapping[str, Any]] = []
    state_closures: list[StateClosure] = []
    timeline: list[Mapping[str, Any]] = []
    reset_mask: Any = False
    if model._uses_update_pressure_update() and hasattr(model, "update_pressure"):
        del model.update_pressure

    for i, _view in enumerate(views):
        replayed = controller.take_replay(i)
        replayed_transaction_frame_id: int | None = None
        if replayed is not None:
            replayed.post_model.restore(model, torch=torch)
            (
                state_feat,
                state_pos,
                init_state_feat,
                mem,
                init_mem,
            ) = replayed.post_state.restored()
            reset_mask = _clone_reset_mask(replayed.post_reset_mask)
            replayed_transaction_frame_id = replayed.frame_id

        view = to_gpu(_view, device)
        device = view["img"].device
        batch_size = view["img"].shape[0]
        img_mask = view["img_mask"].reshape(-1, batch_size)
        ray_mask = view["ray_mask"].reshape(-1, batch_size)
        imgs = view["img"].unsqueeze(0)
        ray_maps = view["ray_map"].unsqueeze(0)
        shapes = (
            view["true_shape"].unsqueeze(0)
            if "true_shape" in view
            else torch.tensor(view["img"].shape[-2:], device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .unsqueeze(0)
        )
        imgs = imgs.view(-1, *imgs.shape[2:])
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])
        shapes = shapes.view(-1, 2).to(imgs.device)
        img_masks_flat = img_mask.view(-1)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = model._encode_image(selected_imgs, selected_shapes)
        else:
            img_out, img_pos = None, None
        ray_maps = ray_maps.permute(0, 3, 1, 2)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = model._encode_ray_map(selected_ray_maps, selected_shapes_ray)
        else:
            ray_out, ray_pos = None, None
        shape = shapes
        if img_out is not None and ray_out is None:
            feat_i, pos_i = img_out[-1], img_pos
        elif img_out is None and ray_out is not None:
            feat_i, pos_i = ray_out[-1], ray_pos
        elif img_out is not None and ray_out is not None:
            feat_i, pos_i = img_out[-1] + ray_out[-1], img_pos
        else:
            raise TransactionalReCal3RError("ReCal3R frame has neither image nor raymap features")

        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()

        pre_state = StateClosure.capture(state_feat, state_pos, init_state_feat, mem, init_mem)
        pre_model = capture_model_state(model, torch=torch)
        pre_reset_mask = _clone_reset_mask(reset_mask)
        if model.pose_head_flag:
            global_img_feat_i = model._get_img_level_feat(feat_i)
            if i == 0 or reset_mask:
                pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i, pose_pos_i = None, None
        (
            new_state_feat,
            dec,
            self_attn_state,
            cross_attn_state,
            self_attn_img,
            cross_attn_img,
        ) = model._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=True,
        )
        del self_attn_state, self_attn_img, cross_attn_img
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
        if len(dec) != model.dec_depth + 1:
            raise TransactionalReCal3RError("pinned ReCal3R decoder depth changed")
        head_input = [
            dec[0].float(),
            dec[model.dec_depth * 2 // 4][:, 1:].float(),
            dec[model.dec_depth * 3 // 4][:, 1:].float(),
            dec[model.dec_depth].float(),
        ]
        res = model._downstream_head(head_input, shape, pos=pos_i)
        update = view.get("update", None)
        update_mask = view["img_mask"] & update if update is not None else view["img_mask"]
        update_mask = update_mask[:, None, None].float()
        prev_state_feat = state_feat
        if i == 0 or reset_mask:
            update_mask1 = update_mask
        elif model_update_type == "cut3r":
            update_mask1 = update_mask
        elif model_update_type == "ttt3r":
            update_mask1, _, _ = model._compute_state_update_mask(update_mask, cross_attn_state)
        else:
            update_mask1 = model._compute_recal3r_update_mask(
                update_mask, cross_attn_state, dec, prev_state_feat=state_feat
            )
        state_feat = new_state_feat * update_mask1 + state_feat * (1 - update_mask1)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        model._advance_recal3r_sequence_age(state_feat)
        model._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)
        res_cpu = to_cpu(res)
        ress.append(res_cpu)
        reset_mask = view["reset"]
        if reset_mask is not None:
            model._reset_update_pressure_if_needed(reset_mask)
            model._reset_recal3r_reference_state_if_needed(reset_mask, init_state_feat)
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)

        post_state = StateClosure.capture(state_feat, state_pos, init_state_feat, mem, init_mem)
        post_model = capture_model_state(model, torch=torch)
        transaction = StateTransaction(
            frame_id=i,
            pre_state=pre_state,
            post_state=post_state,
            pre_model=pre_model,
            post_model=post_model,
            pre_reset_mask=pre_reset_mask,
            post_reset_mask=_clone_reset_mask(reset_mask),
        )
        decision = controller.decide_candidate(transaction)
        if decision.action == "hold":
            transaction.pre_model.restore(model, torch=torch)
            state_feat, state_pos, init_state_feat, mem, init_mem = transaction.pre_state.restored()
            reset_mask = _clone_reset_mask(transaction.pre_reset_mask)
            committed_state = StateClosure.capture(state_feat, state_pos, init_state_feat, mem, init_mem)
        elif decision.action == "commit":
            committed_state = post_state
        else:
            raise TransactionalReCal3RError(f"unknown state policy action: {decision.action}")
        timeline_action = decision.action
        timeline_reason = decision.reason
        if replayed_transaction_frame_id is not None:
            timeline_action = f"release_replay_then_{decision.action}"
            timeline_reason = f"released_held_frame_{replayed_transaction_frame_id};{decision.reason}"
        timeline.append(
            _timeline_entry(
                transaction,
                action=timeline_action,
                reason=timeline_reason,
                prior_alarm=decision.prior_alarm,
                replayed_transaction_frame_id=replayed_transaction_frame_id,
                committed_state=committed_state,
            )
        )
        state_closures.append(committed_state)

    dropped = controller.drain()
    return TransactionalReCal3RResult(
        predictions=ress,
        views=views,
        state_closures=state_closures,
        timeline=timeline,
        dropped_transaction_frame_ids=[transaction.frame_id for transaction in dropped],
        source_provenance=source_provenance,
    )


__all__ = [
    "TransactionalReCal3RError",
    "TransactionalReCal3RResult",
    "run_transactional_recurrent_lighter",
]
