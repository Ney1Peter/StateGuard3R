#!/usr/bin/env python3
"""CPU-only validation and immutable read lock for formal-v2 holdout inputs."""
from __future__ import annotations
import argparse, hashlib, json, os, stat, tempfile
from pathlib import Path
from typing import Any, Sequence
from stateguard3r.input_manifest import load_input_manifest

ROOT=Path(__file__).resolve().parents[1]; OUTPUTS=ROOT/"outputs"; RAW=ROOT.parent/"baselines"/"ReCal3R"/"data"/"tum"/"rgbd_dataset_freiburg3_walking_static"
RUNS=(("holdout-dynamic","dynamic_occlusion",5),("holdout-wrong","wrong_order_segment",4),("holdout-low","low_overlap_jump",5))
class V2InputValidationError(ValueError): pass
def req(v:bool,m:str)->None:
 if not v: raise V2InputValidationError(m)
def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def read(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());req(isinstance(x,dict),f"not object {p}");return x
def validate(root:Path)->dict[str,Any]:
 root=root.resolve();req(root.is_dir() and stat.S_IMODE(root.stat().st_mode)==0o555,"input root not frozen")
 registry=read(root/"formal-v2-manifest.json");req(registry.get("schema_version")=="stateguard3r.formal-v2-inputs.v1","schema")
 req(registry.get("status")=="pre_forward_blind_holdout_inputs","status")
 req(registry.get("cross_scene_proof",{}).get("disjoint") is True,"cross-scene proof")
 rows=registry.get("runs");req(isinstance(rows,list) and len(rows)==3,"run registry")
 all_rgb=set();checked=[]
 for run_id,kind,length in RUNS:
  row=next((x for x in rows if x.get("run_id")==run_id),None);req(isinstance(row,dict) and row.get("corruption_type")==kind,"run layout")
  d=root/run_id;req(d.is_dir() and stat.S_IMODE(d.stat().st_mode)==0o555,"run not frozen")
  source=d/"source-manifest.json"; inp=d/"input-manifest.json";req(source.is_file() and inp.is_file(),"missing manifest")
  req(stat.S_IMODE(source.stat().st_mode)==0o444 and stat.S_IMODE(inp.stat().st_mode)==0o444,"manifest not frozen")
  for name,path in (("source_manifest",source),("input_manifest",inp)):
   a=row.get("artifacts",{}).get(name,{});req(a.get("sha256")==sha(path) and a.get("size_bytes")==path.stat().st_size,"registry hash")
  source_json=read(source);req(source_json.get("dataset")==RAW.name and source_json.get("source_is_read_only") is True,"source identity")
  manifest=load_input_manifest(inp);req(len(manifest.frames)==30,"frame count")
  payload=read(inp); specs=payload.get("corruptions");req(isinstance(specs,list) and len(specs)==1,"corruptions")
  spec=specs[0];req(spec.get("type")==kind and spec.get("start")==15 and spec.get("end")==14+length,"event interval")
  for frame in manifest.frames:
   req(frame.path.is_file() and RAW in frame.path.parents,"frame outside raw root")
   digest=sha(frame.path);req(digest not in all_rgb,"raw RGB reused across holdout runs");all_rgb.add(digest)
  checked.append(run_id)
 return {"schema_version":"stateguard3r.formal-v2-input-cpu-validation.v1","status":"PASS","input_root":str(root),"runs":checked,"checks":{"cuda_hidden":os.environ.get("CUDA_VISIBLE_DEVICES")=="","strict_loader_replay":True,"cross_scene_disjoint":True,"no_holdout_model_output_read":True}}
def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument("input_root",type=Path);p.add_argument("output_dir",type=Path);a=p.parse_args(argv);out=a.output_dir.resolve(strict=False);req(out.parent==OUTPUTS and not out.exists(),"output")
 report=validate(a.input_root); report["validator"]={"path":str(Path(__file__).resolve()),"sha256":sha(Path(__file__).resolve())};out.mkdir();target=out/"cpu-validation.json";target.write_bytes((json.dumps(report,sort_keys=True,indent=2)+"\n").encode());target.chmod(0o444);out.chmod(0o555);print(json.dumps(report,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
