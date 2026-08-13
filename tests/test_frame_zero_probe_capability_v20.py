from __future__ import annotations

from pathlib import Path

import pytest

from stateguard3r import frame_zero_probe_capability_v20 as capsule


def test_v20_frame_zero_capability_is_separate_from_runtime_listing() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "stateguard3r" / "frame_zero_probe_capability_v20.py").read_text(encoding="utf-8")
    for forbidden in ("rgb.txt", "dynamic_rgb", "_v17", "_v18", "_v19", "outputs/"):
        assert forbidden not in source
    assert capsule.CAPSULE_ID == "native-scalar-hold-v20-frame0-probe-0001"


def test_v20_frame_zero_builder_refuses_occupied_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "native-scalar-hold-v20-frame0-probe-0001.json"
    target.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(capsule, "CAPSULE", target)
    with pytest.raises(capsule.FrameZeroProbeCapabilityV20Error, match="occupied"):
        capsule.build_frame_zero_probe_capability_v20()


def test_v20_probe_script_has_no_forward_detector_or_dynamic_capability_import() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "probe_recal3r_native_scalar_interface_v20.py").read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES" in script
    assert "forbids forward" in script
    for forbidden in ("native_scalar_detector_v20", "official_native_scalar_hold_v20", "dynamic_rgb_capability_v20", "_v17", "_v18", "_v19"):
        assert forbidden not in script
