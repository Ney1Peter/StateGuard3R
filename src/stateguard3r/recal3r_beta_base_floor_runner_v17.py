"""Independent v17 recurrent loop with a one-shot native beta-base scope.

The loop mirrors the pinned ReCal3R lightweight recurrent topology.  Its only
candidate-specific model mutation is temporarily assigning ``model.beta_base``
to ``0.0`` for one native ``_compute_recal3r_update_mask`` invocation after a
previous raw-health alarm has committed.  It never alters the current alarm
frame's state, memory, prediction, native auxiliary field, or exported result.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

from .beta_base_floor_v17 import (
    FLOORED_BETA_BASE,
    NATIVE_BETA_BASE,
    BetaBaseFloorSelectionV17,
    arm_one_future_update_v17,
    select_one_shot_beta_base_v17,
)


LIGHTER_SOURCE = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py")
LIGHTER_SOURCE_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"
LIGHTER_SPAN_SHA256 = "03d3c534f5f59f6eb1fa1852ff1cbeb8e874e97b027cd863e48fb6347b100023"
UPDATE_MASK_SPAN_SHA256 = "4bddb325968b70f0ec9879a8288ba017523964fed1700f9eba78ae86156930ce"


class RecurrentBetaFloorV17Error(RuntimeError):
    """The isolated v17 native recurrent contract was violated."""


class CurrentFrameObserverV17(Protocol):
    """Observer boundary: raw current health is observed only after reset."""

    def observe(
        self,
        frame_id: int,
        prediction: Mapping[str, Any],
        model: Any,
        frame_context: Mapping[str, Any] | None,
    ) -> Any: ...

    def alarm(self, observation: Any) -> bool: ...

    def finalize(self, observation: Any) -> Mapping[str, Any]: ...

    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class RecurrentBetaFloorResultV17:
    predictions: list[Mapping[str, Any]]
    timeline: list[Mapping[str, Any]]
    source_provenance: Mapping[str, Any]
    recurrent_policy_runtime_seconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_native_v17_source() -> Mapping[str, Any]:
    """Bind the exact upstream equation and lightweight-loop source spans."""

    if _sha256(LIGHTER_SOURCE) != LIGHTER_SOURCE_SHA256:
        raise RecurrentBetaFloorV17Error("pinned ReCal3R model source differs")
    lines = LIGHTER_SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    update_span = "".join(lines[1198:1251]).encode("utf-8")
    lighter_span = "".join(lines[1660:1833]).encode("utf-8")
    if hashlib.sha256(update_span).hexdigest() != UPDATE_MASK_SPAN_SHA256:
        raise RecurrentBetaFloorV17Error("pinned ReCal3R beta equation span differs")
    if hashlib.sha256(lighter_span).hexdigest() != LIGHTER_SPAN_SHA256:
        raise RecurrentBetaFloorV17Error("pinned ReCal3R lighter span differs")
    return {
        "model_source": str(LIGHTER_SOURCE),
        "model_source_sha256": LIGHTER_SOURCE_SHA256,
        "native_beta_equation_lines": "1199-1251",
        "native_beta_equation_span_sha256": UPDATE_MASK_SPAN_SHA256,
        "lighter_lines": "1661-1833",
        "lighter_span_sha256": LIGHTER_SPAN_SHA256,
    }


def _plain_reset(value: Any) -> bool:
    """Resolve only a single-frame reset flag; reject ambiguous batches."""

    if value is None:
        return False
    if type(value) is bool:
        return value
    try:
        count = int(value.numel())
        if count != 1:
            raise RecurrentBetaFloorV17Error("v17 reset eligibility must contain one value")
        return bool(value.item())
    except AttributeError as error:
        raise RecurrentBetaFloorV17Error("v17 reset eligibility must be a scalar boolean") from error


def _native_beta_base(model: Any) -> float:
    try:
        value = float(model._beta_base())
    except (AttributeError, TypeError, ValueError) as error:
        raise RecurrentBetaFloorV17Error("model does not expose a scalar native beta base") from error
    if not math.isfinite(value) or value != NATIVE_BETA_BASE:
        raise RecurrentBetaFloorV17Error("native beta base differs from the pinned v17 value")
    return value


@contextmanager
def temporary_native_beta_floor_v17(
    model: Any,
    *,
    beta_base: float,
) -> Iterator[None]:
    """Limit the only v17 model mutation to one native mask calculation."""

    if beta_base not in {NATIVE_BETA_BASE, FLOORED_BETA_BASE}:
        raise RecurrentBetaFloorV17Error("v17 beta scope value is not preregistered")
    original_exists = hasattr(model, "beta_base")
    original_value = getattr(model, "beta_base", None)
    if _native_beta_base(model) != NATIVE_BETA_BASE:
        raise RecurrentBetaFloorV17Error("native beta base precondition differs")
    try:
        model.beta_base = beta_base
        observed = float(model._beta_base())
        if observed != beta_base:
            raise RecurrentBetaFloorV17Error("temporary native beta base was not installed")
        yield
    finally:
        if original_exists:
            model.beta_base = original_value
        else:
            try:
                del model.beta_base
            except AttributeError:
                pass
        if _native_beta_base(model) != NATIVE_BETA_BASE:
            raise RecurrentBetaFloorV17Error("native beta base was not restored")


def _mask_scope_evidence(
    selection: BetaBaseFloorSelectionV17,
    *,
    frame_id: int,
) -> Mapping[str, Any]:
    return {
        "frame_id": frame_id,
        "armed_before_mask": selection.armed_before,
        "armed_after_mask": selection.armed_after,
        "arm_consumed": selection.consumed,
        "arm_reset_cancelled": selection.reset_cancelled,
        "native_beta_base_before_mask": NATIVE_BETA_BASE,
        "native_beta_base_during_mask": selection.beta_base_for_mask,
        "native_beta_base_after_mask": NATIVE_BETA_BASE,
        "strict_native_mask_reduction_required": selection.consumed,
    }


def run_recurrent_beta_floor_native_v17(
    views: Sequence[Mapping[str, Any]],
    model: Any,
    device: Any,
    *,
    torch: Any,
    to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    canonicalize_model_update_type: Callable[[Any], str],
    before_frame: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    observer: CurrentFrameObserverV17 | None = None,
    verify_source: bool = True,
    synchronize: Callable[[], None] | None = None,
    require_cuda_candidate: bool = True,
) -> RecurrentBetaFloorResultV17:
    """Run the pinned lightweight topology with at most one later beta floor.

    ``observer is None`` is the control branch.  It does not construct an arm,
    opens no scalar scope, and exports direct native predictions.
    """

    if not views:
        raise RecurrentBetaFloorV17Error("v17 needs at least one RGB view")
    if observer is not None and require_cuda_candidate and getattr(device, "type", None) != "cuda":
        raise RecurrentBetaFloorV17Error("v17 candidate requires CUDA")
    provenance = pinned_native_v17_source() if verify_source else {}
    model.config.model_update_type = canonicalize_model_update_type(
        getattr(model.config, "model_update_type", "cut3r")
    )
    if model.config.model_update_type != "recal3r":
        raise RecurrentBetaFloorV17Error("v17 requires the pinned ReCal3R update type")
    _native_beta_base(model)
    predictions: list[Mapping[str, Any]] = []
    timeline: list[Mapping[str, Any]] = []
    elapsed, reset_mask, armed = 0.0, False, False

    for frame_id, raw_view in enumerate(views):
        frame_context = before_frame(frame_id, raw_view) if before_frame is not None else None
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        view = to_gpu(raw_view, device)
        device = view["img"].device
        batch_size = view["img"].shape[0]
        img_mask = view["img_mask"].reshape(-1, batch_size)
        ray_mask = view["ray_mask"].reshape(-1, batch_size)
        imgs = view["img"].unsqueeze(0).view(-1, *view["img"].shape[1:])
        ray_maps = view["ray_map"].unsqueeze(0)
        shapes = (
            view["true_shape"].unsqueeze(0)
            if "true_shape" in view
            else torch.tensor(view["img"].shape[-2:], device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .unsqueeze(0)
        ).view(-1, 2).to(imgs.device)
        selected_images, selected_shapes = imgs[img_mask.view(-1)], shapes[img_mask.view(-1)]
        image_out, image_pos = (
            model._encode_image(selected_images, selected_shapes)[0:2]
            if selected_images.size(0)
            else (None, None)
        )
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:]).permute(0, 3, 1, 2)
        selected_rays, selected_ray_shapes = ray_maps[ray_mask.view(-1)], shapes[ray_mask.view(-1)]
        ray_out, ray_pos = (
            model._encode_ray_map(selected_rays, selected_ray_shapes)[0:2]
            if selected_rays.size(0)
            else (None, None)
        )
        if image_out is not None and ray_out is None:
            feat_i, pos_i = image_out[-1], image_pos
        elif image_out is None and ray_out is not None:
            feat_i, pos_i = ray_out[-1], ray_pos
        elif image_out is not None and ray_out is not None:
            feat_i, pos_i = image_out[-1] + ray_out[-1], image_pos
        else:
            raise RecurrentBetaFloorV17Error("ReCal3R frame has no image/ray features")
        if frame_id == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat, init_mem = state_feat.clone(), mem.clone()
            if tuple(state_feat.shape) != (1, 768, 768) or tuple(mem.shape) != (1, 256, 1536):
                raise RecurrentBetaFloorV17Error("v17 persistent interfaces differ")
        global_img_feat_i = model._get_img_level_feat(feat_i) if model.pose_head_flag else None
        if model.pose_head_flag:
            pose_feat_i = (
                model.pose_token.expand(feat_i.shape[0], -1, -1)
                if frame_id == 0 or reset_mask
                else model.pose_retriever.inquire(global_img_feat_i, mem)
            )
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i, pose_pos_i = None, None
        new_state, native_sequence, self_state, cross_state, self_img, cross_img = model._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update"),
            return_attn=True,
        )
        del self_state, self_img, cross_img
        if len(native_sequence) != model.dec_depth + 1:
            raise RecurrentBetaFloorV17Error("v17 native sequence interface differs")
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, native_sequence[-1][:, 0:1])
        res = model._downstream_head(
            [
                native_sequence[0].float(),
                native_sequence[model.dec_depth * 2 // 4][:, 1:].float(),
                native_sequence[model.dec_depth * 3 // 4][:, 1:].float(),
                native_sequence[model.dec_depth].float(),
            ],
            shapes,
            pos=pos_i,
        )
        update = view.get("update")
        update_mask = (view["img_mask"] & update if update is not None else view["img_mask"])[
            :, None, None
        ].float()
        previous_state = state_feat
        current_reset = _plain_reset(view["reset"])
        previous_reset = _plain_reset(reset_mask)
        mask_selection: BetaBaseFloorSelectionV17 | None = None
        if current_reset and armed:
            mask_selection = select_one_shot_beta_base_v17(armed, reset=True)
            armed = mask_selection.armed_after
        if frame_id == 0 or previous_reset:
            update_mask1 = update_mask
        elif current_reset:
            update_mask1 = model._compute_recal3r_update_mask(
                update_mask, cross_state, native_sequence, prev_state_feat=state_feat
            )
        elif model.config.model_update_type == "cut3r":
            update_mask1 = update_mask
        elif model.config.model_update_type == "ttt3r":
            update_mask1, _, _ = model._compute_state_update_mask(update_mask, cross_state)
        elif observer is None:
            update_mask1 = model._compute_recal3r_update_mask(
                update_mask, cross_state, native_sequence, prev_state_feat=state_feat
            )
        else:
            mask_selection = select_one_shot_beta_base_v17(armed, reset=False)
            armed = mask_selection.armed_after
            if mask_selection.consumed:
                with temporary_native_beta_floor_v17(
                    model, beta_base=mask_selection.beta_base_for_mask
                ):
                    update_mask1 = model._compute_recal3r_update_mask(
                        update_mask, cross_state, native_sequence, prev_state_feat=state_feat
                    )
            else:
                update_mask1 = model._compute_recal3r_update_mask(
                    update_mask, cross_state, native_sequence, prev_state_feat=state_feat
                )
        state_feat = new_state * update_mask1 + state_feat * (1 - update_mask1)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        model._advance_recal3r_sequence_age(state_feat)
        model._maybe_record_u_calibration_step(frame_id, previous_state, state_feat)
        reset_value = view["reset"]
        if reset_value is not None:
            model._reset_update_pressure_if_needed(reset_value)
            model._reset_recal3r_reference_state_if_needed(reset_value, init_state_feat)
            reset_mask = reset_value[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        detector_evidence: Mapping[str, Any] = {}
        alarm = False
        if observer is not None:
            observation = observer.observe(frame_id, res, model, frame_context)
            alarm = bool(observer.alarm(observation))
            detector_evidence = dict(observer.finalize(observation)) | dict(
                observer.timeline_evidence(observation)
            )
            if alarm:
                armed = arm_one_future_update_v17(armed, alarm=True)
        if synchronize is not None:
            synchronize()
        elapsed += time.perf_counter() - started
        evidence = (
            _mask_scope_evidence(mask_selection, frame_id=frame_id)
            if mask_selection is not None
            else None
        )
        timeline.append(
            {
                "frame_id": frame_id,
                "action": "arm_one_future_native_beta_floor" if alarm else "commit",
                "reason": "current_online_detector_alarm" if alarm else (
                    "always_commit_control" if observer is None else "current_online_detector_clear"
                ),
                "current_alarm": alarm,
                "pending_transaction_count": 0,
                "beta_floor": evidence,
                **detector_evidence,
            }
        )
        predictions.append(to_cpu(res))
    return RecurrentBetaFloorResultV17(predictions, timeline, provenance, elapsed)


__all__ = [
    "CurrentFrameObserverV17",
    "LIGHTER_SOURCE",
    "LIGHTER_SOURCE_SHA256",
    "LIGHTER_SPAN_SHA256",
    "RecurrentBetaFloorResultV17",
    "RecurrentBetaFloorV17Error",
    "UPDATE_MASK_SPAN_SHA256",
    "pinned_native_v17_source",
    "run_recurrent_beta_floor_native_v17",
    "temporary_native_beta_floor_v17",
]
