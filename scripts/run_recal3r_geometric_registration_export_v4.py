#!/usr/bin/env python3
"""Run the preregistered v4 anchor-ORB-3D3D recovery candidate."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from scripts import run_recal3r_safe_anchor_export_v3 as v3


SCHEMA_VERSION = "stateguard3r.recal3r-geometric-registration-export-v4.v1"
_BASE_VALIDATE = v3._validate_args


def _parser() -> argparse.ArgumentParser:
    parser = v3.smoke._parser()
    parser.description = __doc__
    parser.set_defaults(health_profile="v3")
    parser.add_argument("--state-policy", choices=("always-commit", "detector-v3-incremental-anchor-orb-3d3d-registration-export"), required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--watchdog", type=int, default=8)
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _BASE_VALIDATE(args, parser)


def _self_provenance() -> dict[str, Any]:
    try:
        commit = v3.smoke._git_output(v3.ROOT, "rev-parse", "HEAD")
        changes = v3.smoke._git_output(v3.ROOT, "status", "--porcelain", "--untracked-files=no")
    except Exception as error:
        raise RuntimeError(f"cannot inspect v4 runner provenance: {error}") from error
    if changes:
        raise RuntimeError("StateGuard3R has tracked changes; v4 runner refuses execution")
    path = Path(__file__).resolve()
    return {"repository_root": str(v3.ROOT), "commit": commit, "tracked_worktree_clean": True, "script_path": str(path), "script_sha256": v3.smoke._sha256(path), "python_executable": sys.executable}


class _V4Observer(v3._V3Observer):
    """Reuse raw GPU health only; never invoke v3's pose fallback exporter."""

    def __init__(self, detector: Any, ignored_exporter: Any, *, timestamps: Sequence[Any], torch: Any, pose_encoding_to_camera: Any, camera_to_pose_encoding: Any) -> None:
        super().__init__(detector, ignored_exporter, timestamps=timestamps, torch=torch, pose_encoding_to_camera=pose_encoding_to_camera, camera_to_pose_encoding=camera_to_pose_encoding)
        from stateguard3r.geometric_registration_export_v4 import RealSafeGeometricAnchor

        self._geometric = RealSafeGeometricAnchor(camera_to_pose_encoding=camera_to_pose_encoding)
        self._rgb: dict[int, Any] = {}

    def observe(self, frame_id: int, prediction: Mapping[str, Any], model: Any, frame_context: Mapping[str, Any]) -> Any:
        rgb = frame_context.get("v4_rgb")
        if rgb is None:
            raise RuntimeError("v4 observer lacks current model-ready RGB")
        self._rgb[frame_id] = rgb
        return super().observe(frame_id, prediction, model, frame_context)

    def finalize(self, observation: Any, *, quarantined: bool, prediction: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        rgb = self._rgb.pop(observation.frame_id, None)
        if rgb is None:
            raise RuntimeError("v4 observer RGB was not finalized")
        self.records.append(observation.record)
        if quarantined:
            self._detector.quarantine(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="quarantine_current_rollback"))
            exported, action = self._geometric.register_alarm(observation.frame_id, rgb=rgb, prediction=prediction)
        else:
            self._detector.commit(observation.record)
            self.observed_ledger.append(replace(observation.record, decision="commit"))
            self._previous_safe_camera = self._decode(observation.raw_pose.detach().clone()).detach().clone()
            exported, action = self._geometric.commit_real(observation.frame_id, rgb=rgb, prediction=prediction)
        return exported, {"export_action": action.action, "anchor_frame_ids": None if action.anchor_frame_id is None else [action.anchor_frame_id], "registration": action.registration_evidence}


def _before(overlap: Any, original_before: Any) -> Any:
    def before(frame_id: int, view: Mapping[str, Any]) -> Mapping[str, Any]:
        value = dict(original_before(overlap, frame_id, view))
        image = view.get("img")
        if image is None or not callable(getattr(image, "detach", None)) or not callable(getattr(image, "clone", None)):
            raise RuntimeError("v4 model-ready RGB must expose detach/clone")
        value["v4_rgb"] = image.detach().clone()
        return value
    return before


def main(argv: Sequence[str] | None = None) -> int:
    # v3's runner is source-bound and generic over its observer.  The temporary
    # substitutions are process-local and restored even if v4 fails closed.
    old = (v3._parser, v3._validate_args, v3._V3Observer, v3.SCHEMA_VERSION, v3.v2._OnlineOverlap, v3._self_provenance)
    class _Overlap(v3.v2._OnlineOverlap):
        pass
    try:
        v3._parser, v3._validate_args, v3._V3Observer, v3.SCHEMA_VERSION, v3._self_provenance = _parser, _validate, _V4Observer, SCHEMA_VERSION, _self_provenance
        original_init, original_before = _Overlap.__init__, _Overlap.before_frame
        def init(self: Any) -> None:
            original_init(self)
            self.before_frame = _before(self, original_before)  # type: ignore[method-assign]
        _Overlap.__init__ = init  # type: ignore[method-assign]
        v3.v2._OnlineOverlap = _Overlap
        return v3.main(argv)
    finally:
        v3._parser, v3._validate_args, v3._V3Observer, v3.SCHEMA_VERSION, v3.v2._OnlineOverlap, v3._self_provenance = old


if __name__ == "__main__":
    raise SystemExit(main())
