#!/usr/bin/env python3
"""CUDA-hidden v20 frame-zero interface probe with a strict data ledger."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from stateguard3r.frame_zero_probe_capability_v20 import CAPSULE, CAPSULE_ID, CHECKPOINT, load_frame_zero_probe_capability_v20

BASELINE = ROOT.parent / "baselines" / "ReCal3R"
OUTPUT = ROOT / "logs" / "native-scalar-hold-v20-frame0-cpu-interface-probe-0001.json"


class ProbeV20Error(RuntimeError): pass


def _regular(path: Path, *, label: str) -> None:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o444: raise ProbeV20Error(f"{label} unsafe")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _snapshot(path: Path, resource: str) -> dict[str, str]:
    _regular(path, label=resource)
    info = path.stat()
    return {"resource": resource, "path": str(path), "sha256": _sha(path), "size_bytes": str(info.st_size), "inode": str(info.st_ino), "mtime_ns": str(info.st_mtime_ns), "mode_octal": "0444"}


def _write(value: Mapping[str, Any]) -> None:
    if os.path.lexists(OUTPUT): raise ProbeV20Error("v20 probe output target occupied")
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ProbeV20Error("v20 interface probe requires CUDA hidden")
    payload = load_frame_zero_probe_capability_v20(CAPSULE)
    rgb = Path(str(payload["rgb_path"]))
    ledger = [_snapshot(CAPSULE, "probe_capsule"), _snapshot(rgb, "frame_zero_rgb"), _snapshot(CHECKPOINT, "checkpoint")]
    if any(item["resource"] not in {"probe_capsule", "frame_zero_rgb", "checkpoint"} for item in ledger): raise ProbeV20Error("v20 probe ledger differs")
    sys.path[:0] = [str(BASELINE), str(BASELINE / "src")]
    import torch
    from dust3r.model import ARCroco3DStereo
    from dust3r.utils.image import load_images_for_eval
    original_forward, original_encoder = ARCroco3DStereo.forward_recurrent_lighter, ARCroco3DStereo._encode_image
    ARCroco3DStereo.forward_recurrent_lighter = lambda *_a, **_k: (_ for _ in ()).throw(ProbeV20Error("v20 probe forbids forward"))
    ARCroco3DStereo._encode_image = lambda *_a, **_k: (_ for _ in ()).throw(ProbeV20Error("v20 probe forbids encoder"))
    try:
        loaded = load_images_for_eval([str(rgb)], size=512, crop=True, square_ok=False, verbose=False)
        if len(loaded) != 1: raise ProbeV20Error("v20 probe official image loader differs")
        model = ARCroco3DStereo.from_pretrained(str(CHECKPOINT))
        image = loaded[0]["img"]
        # Bind the native persistent interfaces without model forward.
        register = model.register_tokens(torch.arange(model.state_size))
        state = model.decoder_embed_state(register.unsqueeze(0))
        memory = model.pose_retriever.mem.expand(1, -1, -1)
        update = torch.ones((1, 768, 1), dtype=torch.float32)
    finally:
        ARCroco3DStereo.forward_recurrent_lighter, ARCroco3DStereo._encode_image = original_forward, original_encoder
    if torch.cuda.is_initialized(): raise ProbeV20Error("v20 CPU probe initialized CUDA")
    report = {"schema_version": "stateguard3r.native-scalar-v20-cpu-interface-probe.v1", "run_id": "native-scalar-hold-v20-frame0-cpu-interface-probe-0001", "status": "passed", "cuda_visible_devices": "", "cuda_initialized": False, "model_forward_executed": False, "blocked_execution_routes": True, "probe_capsule_id": CAPSULE_ID, "data_access_ledger": ledger, "loaded_image": {"shape": list(image.shape), "dtype": str(image.dtype), "device": str(image.device)}, "interfaces": {"state_feature": {"shape": list(state.shape), "dtype": str(state.dtype), "device": str(state.device)}, "pose_memory": {"shape": list(memory.shape), "dtype": str(memory.dtype), "device": str(memory.device)}, "native_update_mask": {"shape": list(update.shape), "dtype": str(update.dtype), "device": str(update.device)}}}
    if report["interfaces"]["state_feature"]["shape"] != [1, 768, 768] or report["interfaces"]["pose_memory"]["shape"] != [1, 256, 1536] or report["interfaces"]["native_update_mask"]["shape"] != [1, 768, 1]: raise ProbeV20Error("v20 native interface differs")
    _write(report)
    return 0


if __name__ == "__main__": raise SystemExit(main())
