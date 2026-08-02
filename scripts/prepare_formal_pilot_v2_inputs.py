#!/usr/bin/env python3
"""Freeze the three cross-scene formal-v2 input manifests without a model call."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stateguard3r.corruption import build_corruption_manifest
from stateguard3r.visual_overlap import gt_depth_reprojection_overlap


DATASET = "rgbd_dataset_freiburg3_walking_static"
RAW_ROOT = ROOT.parent / "baselines" / "ReCal3R" / "data" / "tum" / DATASET
ARCHIVE = RAW_ROOT.with_suffix(".tgz")
ACQUISITION = ROOT / "outputs" / "formal-v2-data-0002" / "acquisition.json"
FRAME_COUNT, EVENT_START, SEED = 30, 15, 0
MAX_DELTA = Decimal("0.02")
INTRINSICS = {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6}
SPECS = (("holdout-dynamic", "dynamic_occlusion"), ("holdout-wrong", "wrong_order_segment"), ("holdout-low", "low_overlap_jump"))


class V2InputError(ValueError): pass


@dataclass(frozen=True)
class Entry:
    index: int; timestamp: Decimal; text: str; fields: tuple[str, ...]


def _require(ok: bool, text: str) -> None:
    if not ok: raise V2InputError(text)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def _parse(name: str, count: int) -> list[Entry]:
    entries=[]
    for line in (RAW_ROOT/name).read_text(encoding="utf-8").splitlines():
        text=line.strip()
        if not text or text.startswith("#"): continue
        fields=tuple(text.split()); _require(len(fields)==count,f"bad {name} row")
        try: timestamp=Decimal(fields[0])
        except InvalidOperation as e: raise V2InputError(f"bad {name} timestamp") from e
        _require(timestamp.is_finite(),f"nonfinite {name} timestamp")
        _require(not entries or timestamp>=entries[-1].timestamp,f"decreasing {name} timestamp")
        entries.append(Entry(len(entries),timestamp,text,fields))
    _require(entries,f"no {name} entries"); return entries


def _nearest(reference: Entry, candidates: Sequence[Entry], name: str) -> Entry | None:
    distances=[(abs(value.timestamp-reference.timestamp),value) for value in candidates]
    minimum=min(value for value,_ in distances); rows=[value for distance,value in distances if distance==minimum]
    return rows[0] if len(rows)==1 and minimum<=MAX_DELTA else None


def _path(entry: Entry) -> Path:
    value=RAW_ROOT/entry.fields[1]
    _require(value.is_file() and not value.is_symlink(),f"missing raw file {value}")
    return value


def _pose(entry: Entry) -> np.ndarray:
    tx,ty,tz,qx,qy,qz,qw=map(float,entry.fields[1:]); q=np.asarray([qw,qx,qy,qz]); q=q/np.linalg.norm(q); w,x,y,z=q
    matrix=np.eye(4); matrix[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]; matrix[:3,3]=[tx,ty,tz]; return matrix


def _depth(path: Path) -> np.ndarray:
    import cv2
    value=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    _require(value is not None and value.shape==(480,640),f"bad depth {path}"); return value


def _frame(rgb: Entry, depth: Entry, gt: Entry) -> dict[str, Any]:
    path=_path(rgb); depth_path=_path(depth); state=path.stat(); dstate=depth_path.stat()
    return {"rgb":rgb,"depth":depth,"gt":gt,"path":path,"depth_path":depth_path,"sha256":_sha(path),"depth_sha256":_sha(depth_path),"rgb_stat":state,"depth_stat":dstate}


def _valid_block(frames: Sequence[dict[str,Any]], start: int, length: int) -> bool:
    rows=frames[start:start+length]
    return len(rows)==length and all(row is not None for row in rows) and len({row["depth"].index for row in rows})==length and len({row["gt"].index for row in rows})==length


def _corruption(kind: str) -> dict[str, Any]:
    if kind=="dynamic_occlusion": return {"type":kind,"start":15,"end":19,"parameters":{"coordinate_space":"normalized","rectangle":{"x":.25,"y":.25,"width":.5,"height":.5},"velocity":{"dx":.04,"dy":.025},"fill":[255,0,0]}}
    if kind=="wrong_order_segment": return {"type":kind,"start":15,"end":18,"parameters":{"mode":"reverse"}}
    return {"type":kind,"start":15,"end":19,"parameters":{"source_start":30}}


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"),key=lambda p:len(p.parts),reverse=True): path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def prepare(output: Path) -> dict[str,Any]:
    output=output.resolve(strict=False); _require(output.parent==ROOT/"outputs" and not output.exists(),"output must be a new direct child of outputs")
    _require(ARCHIVE.is_file() and stat.S_IMODE(ARCHIVE.stat().st_mode)==0o444,"archive is not frozen")
    _require(RAW_ROOT.is_dir() and stat.S_IMODE(RAW_ROOT.stat().st_mode)==0o555,"raw root is not frozen")
    acq=json.loads(ACQUISITION.read_text()); _require(acq.get("status")=="PASS","acquisition audit missing")
    rgb,depth,gt=_parse("rgb.txt",2),_parse("depth.txt",2),_parse("groundtruth.txt",8)
    associated=[]
    for value in rgb:
        d,g=_nearest(value,depth,"depth"),_nearest(value,gt,"groundtruth")
        associated.append(_frame(value,d,g) if d is not None and g is not None else None)
    starts=[index for index in range(len(associated)-29) if _valid_block(associated,index,30)]
    bases=[]; used_rgb=set();used_depth=set();used_gt=set()
    for start in starts:
        rows=associated[start:start+30]
        if set(range(start,start+30))&used_rgb or {x["depth"].index for x in rows}&used_depth or {x["gt"].index for x in rows}&used_gt: continue
        bases.append(start); used_rgb.update(range(start,start+30)); used_depth.update(x["depth"].index for x in rows); used_gt.update(x["gt"].index for x in rows)
        if len(bases)==3: break
    _require(len(bases)==3,"cannot allocate three disjoint 30-frame windows")
    donor_starts=[]
    for start in range(len(associated)-4):
        rows=associated[start:start+5]
        if _valid_block(associated,start,5) and not(set(range(start,start+5))&used_rgb or {x["depth"].index for x in rows}&used_depth or {x["gt"].index for x in rows}&used_gt): donor_starts.append(start)
    base=bases[2]; ranked=[]
    for start in donor_starts:
        scores=[]
        for offset in range(5):
            left,right=associated[base+15+offset],associated[start+offset]
            scores.append(gt_depth_reprojection_overlap(_depth(left["depth_path"]),_depth(right["depth_path"]),_pose(left["gt"]),_pose(right["gt"]),INTRINSICS,stride=8).score)
        ranked.append((float(math.fsum(scores)/5),start,scores))
    score,donor,donor_scores=min(ranked,key=lambda row:(row[0],row[1]))
    staging=Path(tempfile.mkdtemp(prefix=f".{output.name}.",suffix=".staging",dir=output.parent)); published=False
    try:
        records=[]
        for (run_id,kind),start in zip(SPECS,bases,strict=True):
            pool=list(associated[start:start+30]);
            if kind=="low_overlap_jump": pool.extend(associated[donor:donor+5])
            run_dir=staging/run_id; run_dir.mkdir()
            frames=[]
            for pool_index,row in enumerate(pool):
                frames.append({"path":str(row["path"]),"source_pool_index":pool_index,"source_pool_role":"base" if pool_index<30 else "low_overlap_donor","raw_rgb_source_index":row["rgb"].index,"rgb_sha256":row["sha256"],"depth":{"source_entry_index":row["depth"].index,"sha256":row["depth_sha256"],"path":str(row["depth_path"])},"groundtruth":{"source_entry_index":row["gt"].index,"timestamp_text":row["gt"].fields[0],"translation_xyz":[float(x) for x in row["gt"].fields[1:4]],"quaternion_xyzw":[float(x) for x in row["gt"].fields[4:8]]}})
            source={"schema_version":"stateguard3r.tum-formal-v2-source.v1","dataset":DATASET,"run_id":run_id,"dataset_split":"holdout","corruption_type":kind,"source_is_read_only":True,"output_frame_count":30,"frame_count":len(frames),"frames":frames,"official_lineage":{"archive_sha256":_sha(ARCHIVE),"acquisition_sha256":_sha(ACQUISITION)},"association_policy":{"method":"unique_nearest_absolute_timestamp","max_absolute_delta_seconds":.02,"tie_policy":"reject"}}
            source_b=_json(source); manifest=build_corruption_manifest(source,[_corruption(kind)],seed=SEED,check_paths=True,source_base_dir=run_dir,source_manifest_path="source-manifest.json",source_manifest_sha256=hashlib.sha256(source_b).hexdigest(),sequence=run_id); input_b=_json(manifest)
            (run_dir/"source-manifest.json").write_bytes(source_b); (run_dir/"input-manifest.json").write_bytes(input_b)
            records.append({"run_id":run_id,"dataset_split":"holdout","corruption_type":kind,"base_rgb_source_indices":list(range(start,start+30)),"donor_rgb_source_indices":list(range(donor,donor+5)) if kind=="low_overlap_jump" else [],"raw_frame_sha256s":sorted(row["sha256"] for row in pool[:30] if row in associated[start:start+30]),"artifacts":{"source_manifest":{"path":f"{run_id}/source-manifest.json","sha256":hashlib.sha256(source_b).hexdigest(),"size_bytes":len(source_b)},"input_manifest":{"path":f"{run_id}/input-manifest.json","sha256":hashlib.sha256(input_b).hexdigest(),"size_bytes":len(input_b)}}})
        registry={"schema_version":"stateguard3r.formal-v2-inputs.v1","status":"pre_forward_blind_holdout_inputs","dataset":DATASET,"frame_count":30,"event_start":15,"runs":records,"offline_gt_depth_reprojection":{"intrinsics":INTRINSICS,"stride":8,"low_overlap_donor_start":donor,"mean_score":score,"per_pair_scores":scores if False else donor_scores,"usage":"offline_donor_selection_and_severity_only_not_online_ledger"},"cross_scene_proof":{"development_dataset":"rgbd_dataset_freiburg1_desk","holdout_dataset":DATASET,"disjoint":True},"acquisition":{"path":str(ACQUISITION),"sha256":_sha(ACQUISITION)},"archive":{"path":str(ARCHIVE),"sha256":_sha(ARCHIVE)}}
        (staging/"formal-v2-manifest.json").write_bytes(_json(registry)); _freeze(staging); os.replace(staging,output); published=True; return registry
    finally:
        if not published and staging.exists(): shutil.rmtree(staging)


def main(argv: Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(); parser.add_argument("output_dir",type=Path); args=parser.parse_args(argv)
    try: result=prepare(args.output_dir)
    except (V2InputError,OSError,ValueError) as error: parser.error(str(error))
    print(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
