#!/usr/bin/env python3
"""Publish the sole v20 CUDA-hidden frame-zero capability."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.frame_zero_probe_capability_v20 import build_frame_zero_probe_capability_v20
if __name__ == "__main__": print(json.dumps(build_frame_zero_probe_capability_v20(), sort_keys=True))
