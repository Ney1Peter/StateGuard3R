"""Own v16 recurrent ReCal3R loop with a post-alarm pressure-only write."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .bounded_update_pressure_v16 import PRESSURE_SHAPE, gpu_fingerprint_v16, inject_bounded_pressure_v16


LIGHTER_SOURCE = Path("/data/wangzheng/Project2/baselines/ReCal3R/src/dust3r/model.py")
LIGHTER_SOURCE_SHA256 = "32785a6f29fded66aa142207b29a33d7522f0c39aa068fe2175a956ec8dbc3c1"
LIGHTER_SPAN_SHA256 = "03d3c534f5f59f6eb1fa1852ff1cbeb8e874e97b027cd863e48fb6347b100023"


class RecurrentBoundedPressureV16Error(RuntimeError):
    """The independently pinned v16 recurrent loop has drifted."""


class CurrentFrameObserverV16(Protocol):
    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any] | None) -> Any: ...
    def alarm(self, observation: Any) -> bool: ...
    def finalize(self, observation: Any) -> Mapping[str, Any]: ...
    def timeline_evidence(self, observation: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RecurrentBoundedPressureResultV16:
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


def _pinned_lighter() -> Mapping[str, Any]:
    if _sha256(LIGHTER_SOURCE) != LIGHTER_SOURCE_SHA256:
        raise RecurrentBoundedPressureV16Error("pinned ReCal3R model source differs")
    span = "".join(LIGHTER_SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)[1660:1833]).encode("utf-8")
    if hashlib.sha256(span).hexdigest() != LIGHTER_SPAN_SHA256:
        raise RecurrentBoundedPressureV16Error("pinned ReCal3R lighter span differs")
    return {"model_source": str(LIGHTER_SOURCE), "model_source_sha256": LIGHTER_SOURCE_SHA256, "lighter_lines": "1661-1833", "lighter_span_sha256": LIGHTER_SPAN_SHA256}


def audit_v16_runner_contract(path: Path) -> Mapping[str, Any]:
    """Statically bind the only v16 post-native write and its causal order."""
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
        body = source[source.index("\ndef run_recurrent_bounded_pressure_v16(") + 1 :]
        state = body.index("state_feat = new_state * update_mask1 + state_feat * (1 - update_mask1)")
        memory = body.index("mem = new_mem * update_mask + mem * (1 - update_mask)", state)
        calibration = body.index("model._maybe_record_u_calibration_step(i, previous_state, state_feat)", memory)
        reset = body.index("model._reset_update_pressure_if_needed(reset_value)", calibration)
        observe = body.index("observation = observer.observe(i, res, model, frame_context)", reset)
        commit = body.index("observer.finalize(observation)", observe)
        inject = body.index("inject_bounded_pressure_v16(", commit)
        transfer = body.index("predictions.append(to_cpu(res))", inject)
    except (SyntaxError, ValueError) as error:
        raise RecurrentBoundedPressureV16Error("v16 lighter causal order is incomplete") from error
    if not state < memory < calibration < reset < observe < commit < inject < transfer:
        raise RecurrentBoundedPressureV16Error("v16 lighter causal order differs")
    imports: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = ("recovery", "manifest", "archive", "quarantine", "pointmap", "anchor", "registration", "prerollout", "decoder", "spatial", "encoder", "patch")
    def prohibited(name: str) -> bool:
        if re.search(r"(?:^|_)v(?:[1-9]|1[0-5])(?:_|$)", name):
            return True
        return any(word in name for word in forbidden)
    if any(prohibited(name) for name in imports):
        raise RecurrentBoundedPressureV16Error("v16 runner imports a prohibited component")
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run_recurrent_bounded_pressure_v16")
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "inject_bounded_pressure_v16"]
    if len(calls) != 1:
        raise RecurrentBoundedPressureV16Error("v16 must have exactly one pressure operator call")
    call = calls[0]
    direct_values = {node.id for node in call.args if isinstance(node, ast.Name)}
    if direct_values & {"res", "mem", "dec", "cross_state", "state_feat", "camera_pose"} or any(isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"} for node in ast.walk(call)):
        raise RecurrentBoundedPressureV16Error("v16 pressure operator receives a prohibited value")
    return {"runner_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(), "native_update_before_detector": True, "native_reset_before_detector": True, "detector_commit_before_pressure": True, "pressure_before_raw_cpu_transfer": True, "operator_has_no_pose_state_memory_latent_input": True}


def run_recurrent_bounded_pressure_v16(views: Sequence[Mapping[str, Any]], model: Any, device: Any, *, torch: Any, to_gpu: Callable[[Mapping[str, Any], Any], Mapping[str, Any]], to_cpu: Callable[[Mapping[str, Any]], Mapping[str, Any]], canonicalize_model_update_type: Callable[[Any], str], before_frame: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None, observer: CurrentFrameObserverV16 | None = None, verify_source: bool = True, synchronize: Callable[[], None] | None = None, require_cuda_candidate: bool = True) -> RecurrentBoundedPressureResultV16:
    if not views:
        raise RecurrentBoundedPressureV16Error("v16 needs at least one RGB view")
    if observer is not None and require_cuda_candidate and getattr(device, "type", None) != "cuda":
        raise RecurrentBoundedPressureV16Error("v16 candidate requires CUDA")
    provenance = _pinned_lighter() if verify_source else {}
    model.config.model_update_type = canonicalize_model_update_type(getattr(model.config, "model_update_type", "cut3r"))
    if model.config.model_update_type != "recal3r":
        raise RecurrentBoundedPressureV16Error("v16 requires the pinned ReCal3R update type")
    if hasattr(model, "update_pressure"):
        del model.update_pressure
    predictions: list[Mapping[str, Any]] = []
    timeline: list[Mapping[str, Any]] = []
    elapsed, reset_mask = 0.0, False
    for i, raw_view in enumerate(views):
        frame_context = before_frame(i, raw_view) if before_frame is not None else None
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        view = to_gpu(raw_view, device)
        device = view["img"].device
        batch_size = view["img"].shape[0]
        img_mask, ray_mask = view["img_mask"].reshape(-1, batch_size), view["ray_mask"].reshape(-1, batch_size)
        imgs, ray_maps = view["img"].unsqueeze(0).view(-1, *view["img"].shape[1:]), view["ray_map"].unsqueeze(0)
        shapes = (view["true_shape"].unsqueeze(0) if "true_shape" in view else torch.tensor(view["img"].shape[-2:], device=device).unsqueeze(0).repeat(batch_size, 1).unsqueeze(0)).view(-1, 2).to(imgs.device)
        selected_images, selected_shapes = imgs[img_mask.view(-1)], shapes[img_mask.view(-1)]
        image_out, image_pos = (model._encode_image(selected_images, selected_shapes)[0:2] if selected_images.size(0) else (None, None))
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:]).permute(0, 3, 1, 2)
        selected_rays, selected_ray_shapes = ray_maps[ray_mask.view(-1)], shapes[ray_mask.view(-1)]
        ray_out, ray_pos = (model._encode_ray_map(selected_rays, selected_ray_shapes)[0:2] if selected_rays.size(0) else (None, None))
        if image_out is not None and ray_out is None:
            feat_i, pos_i = image_out[-1], image_pos
        elif image_out is None and ray_out is not None:
            feat_i, pos_i = ray_out[-1], ray_pos
        elif image_out is not None and ray_out is not None:
            feat_i, pos_i = image_out[-1] + ray_out[-1], image_pos
        else:
            raise RecurrentBoundedPressureV16Error("ReCal3R frame has no image/ray features")
        if i == 0:
            state_feat, state_pos = model._init_state(feat_i, pos_i)
            model._init_recal3r_reference_state(state_feat)
            mem = model.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
            init_state_feat, init_mem = state_feat.clone(), mem.clone()
            if tuple(state_feat.shape) != (1, 768, 768) or tuple(mem.shape) != (1, 256, 1536):
                raise RecurrentBoundedPressureV16Error("v16 persistent interfaces differ")
        global_img_feat_i = model._get_img_level_feat(feat_i) if model.pose_head_flag else None
        if model.pose_head_flag:
            pose_feat_i = model.pose_token.expand(feat_i.shape[0], -1, -1) if i == 0 or reset_mask else model.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype)
        else:
            pose_feat_i, pose_pos_i = None, None
        new_state, dec, self_state, cross_state, self_img, cross_img = model._recurrent_rollout(state_feat, state_pos, feat_i, pos_i, pose_feat_i, pose_pos_i, init_state_feat, img_mask=view["img_mask"], reset_mask=view["reset"], update=view.get("update"), return_attn=True)
        del self_state, self_img, cross_img
        if len(dec) != model.dec_depth + 1:
            raise RecurrentBoundedPressureV16Error("v16 decoder interface differs")
        new_mem = model.pose_retriever.update_mem(mem, global_img_feat_i, dec[-1][:, 0:1])
        res = model._downstream_head([dec[0].float(), dec[model.dec_depth * 2 // 4][:, 1:].float(), dec[model.dec_depth * 3 // 4][:, 1:].float(), dec[model.dec_depth].float()], shapes, pos=pos_i)
        update = view.get("update")
        update_mask = (view["img_mask"] & update if update is not None else view["img_mask"])[:, None, None].float()
        previous_state = state_feat
        if i == 0 or reset_mask:
            update_mask1 = update_mask
        elif model.config.model_update_type == "cut3r":
            update_mask1 = update_mask
        elif model.config.model_update_type == "ttt3r":
            update_mask1, _, _ = model._compute_state_update_mask(update_mask, cross_state)
        else:
            update_mask1 = model._compute_recal3r_update_mask(update_mask, cross_state, dec, prev_state_feat=state_feat)
        state_feat = new_state * update_mask1 + state_feat * (1 - update_mask1)
        mem = new_mem * update_mask + mem * (1 - update_mask)
        model._advance_recal3r_sequence_age(state_feat)
        model._maybe_record_u_calibration_step(i, previous_state, state_feat)
        reset_value = view["reset"]
        if reset_value is not None:
            model._reset_update_pressure_if_needed(reset_value)
            model._reset_recal3r_reference_state_if_needed(reset_value, init_state_feat)
            reset_mask = reset_value[:, None, None].float()
            state_feat, mem = init_state_feat * reset_mask + state_feat * (1 - reset_mask), init_mem * reset_mask + mem * (1 - reset_mask)
        alarm, pressure, detector = False, None, {}
        if observer is not None:
            observation = observer.observe(i, res, model, frame_context)
            alarm = bool(observer.alarm(observation))
            detector = dict(observer.finalize(observation)) | dict(observer.timeline_evidence(observation))
            if alarm:
                state_before = gpu_fingerprint_v16(state_feat, torch=torch, label="state before pressure")
                mem_before = gpu_fingerprint_v16(mem, torch=torch, label="memory before pressure")
                pose_before = gpu_fingerprint_v16(res["camera_pose"], torch=torch, label="raw pose before pressure")
                committed, pressure = inject_bounded_pressure_v16(getattr(model, "update_pressure", None), alarm=True, device=device, dtype=state_feat.dtype, expected_shape=PRESSURE_SHAPE, torch=torch)
                model.update_pressure = committed
                if state_before != gpu_fingerprint_v16(state_feat, torch=torch, label="state after pressure") or mem_before != gpu_fingerprint_v16(mem, torch=torch, label="memory after pressure") or pose_before != gpu_fingerprint_v16(res["camera_pose"], torch=torch, label="raw pose after pressure"):
                    raise RecurrentBoundedPressureV16Error("v16 injection changed current raw state, memory, or pose")
                pressure = dict(pressure) | {"raw_current_state_gpu_fingerprint_unchanged": True, "raw_current_mem_gpu_fingerprint_unchanged": True, "raw_current_pose_gpu_fingerprint_unchanged": True, "raw_current_state_gpu_fingerprint": state_before, "raw_current_mem_gpu_fingerprint": mem_before, "raw_current_pose_gpu_fingerprint": pose_before}
        predictions.append(to_cpu(res))
        timeline.append({"frame_id": i, "action": "bounded_update_pressure_injection" if alarm else "commit", "reason": "current_online_detector_alarm" if alarm else ("always_commit_control" if observer is None else "current_online_detector_clear"), "current_alarm": alarm, "pending_transaction_count": 0, "pressure": pressure, "export_action": "export_real_camera_pose", **detector})
        if synchronize is not None:
            synchronize()
        elapsed += time.perf_counter() - started
    return RecurrentBoundedPressureResultV16(predictions, timeline, provenance, elapsed)


__all__ = ["CurrentFrameObserverV16", "RecurrentBoundedPressureResultV16", "RecurrentBoundedPressureV16Error", "audit_v16_runner_contract", "run_recurrent_bounded_pressure_v16"]
