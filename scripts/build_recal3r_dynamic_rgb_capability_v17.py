#!/usr/bin/env python3
"""Construct the v17 RGB capability from the raw selector and its 30 RGB files."""

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stateguard3r.dynamic_rgb_capability_v17 import build_dynamic_rgb_capability_v17


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(build_dynamic_rgb_capability_v17(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
