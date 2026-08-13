"""Static capability and native-scope checks for the isolated v17 graph."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "stateguard3r"
V17_ALLOWED_MODULES = frozenset(
    {
        "beta_base_floor_v17",
        "dynamic_rgb_capability_v17",
        "frame_zero_probe_capability_v17",
        "health",
        "online_detector_v17",
        "online_visual_overlap_v17",
        "recal3r_beta_base_floor_runner_v17",
        "timestamp_order_v17",
        "v17_source_import_audit",
    }
)
DISALLOWED_NAME_PARTS = (
    "recovery",
    "manifest",
    "archive",
    "quarantine",
    "pressure",
    "pointmap",
    "anchor",
    "registration",
    "prerollout",
    "decoder",
    "spatial",
    "encoder",
    "patch",
    "repair",
    "rollback",
)
NATIVE_MODEL_METHODS = frozenset(
    {
        "_advance_recal3r_sequence_age",
        "_beta_base",
        "_compute_recal3r_update_mask",
        "_compute_state_update_mask",
        "_downstream_head",
        "_encode_image",
        "_encode_ray_map",
        "_get_img_level_feat",
        "get_u_calibration_trace",
        "_init_recal3r_reference_state",
        "_init_state",
        "_maybe_record_u_calibration_step",
        "_recurrent_rollout",
        "_reset_recal3r_reference_state_if_needed",
        "_reset_update_pressure_if_needed",
    }
)


class V17SourceImportAuditError(ValueError):
    """The new v17 graph acquired a prohibited local capability."""


def _module_name(path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(PACKAGE.resolve(strict=True))
    except ValueError as error:
        raise V17SourceImportAuditError("v17 audit path is outside the package") from error
    if relative.suffix != ".py" or len(relative.parts) != 1:
        raise V17SourceImportAuditError("v17 audit accepts only direct package modules")
    return relative.stem


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise V17SourceImportAuditError(f"cannot parse v17 source {path.name}") from error


def _local_imports(tree: ast.Module) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and module:
                imported.append(module.split(".")[0])
            elif node.level:
                imported.extend(alias.name.split(".")[0] for alias in node.names)
            elif module.startswith("stateguard3r."):
                imported.append(module.split(".")[1])
        elif isinstance(node, ast.Import):
            imported.extend(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("stateguard3r.") and len(alias.name.split(".")) >= 2
            )
    return tuple(imported)


def _is_disallowed_local_name(name: str) -> bool:
    lowered = name.lower()
    return (
        name not in V17_ALLOWED_MODULES
        or any(part in lowered for part in DISALLOWED_NAME_PARTS)
        or bool(re.search(r"(?:^|_)v(?:[1-9]|1[0-6])(?:_|$)", lowered))
    )


def _has_prohibited_dynamic_import(tree: ast.Module) -> bool:
    """Reject dynamic local imports while allowing the fixed lazy cv2 load."""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_dynamic = (
            isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
        )
        if is_dynamic:
            if not node.args or not isinstance(node.args[0], ast.Constant):
                return True
            target = node.args[0].value
            if target != "cv2":
                return True
    return False


def _has_prohibited_artifact_literal(tree: ast.Module) -> bool:
    prohibited = ("/" + "outputs" + "/", "run" + ".json")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.lower().replace("\\", "/")
        if any(token in value for token in prohibited):
            return True
    return False


def audit_v17_runtime_import_graph(paths: Iterable[Path]) -> Mapping[str, Any]:
    """Walk every local import and reject historical/offline capabilities."""

    initial = tuple(Path(path).resolve(strict=True) for path in paths)
    if not initial:
        raise V17SourceImportAuditError("v17 import audit needs at least one source path")
    pending = list(initial)
    visited: dict[str, Path] = {}
    edges: dict[str, tuple[str, ...]] = {}
    while pending:
        path = pending.pop()
        name = _module_name(path)
        if _is_disallowed_local_name(name):
            raise V17SourceImportAuditError(f"v17 runtime module is prohibited: {name}")
        if name in visited:
            if visited[name] != path:
                raise V17SourceImportAuditError(f"v17 module name resolves ambiguously: {name}")
            continue
        tree = _parse(path)
        if _has_prohibited_dynamic_import(tree):
            raise V17SourceImportAuditError(f"v17 source uses a dynamic local import: {name}")
        if _has_prohibited_artifact_literal(tree):
            raise V17SourceImportAuditError(f"v17 source contains a prohibited artifact literal: {name}")
        imported = _local_imports(tree)
        if any(_is_disallowed_local_name(child) for child in imported):
            raise V17SourceImportAuditError(f"v17 source imports a prohibited local module: {name}")
        visited[name] = path
        edges[name] = tuple(sorted(set(imported)))
        pending.extend(PACKAGE / f"{child}.py" for child in imported)
    return {
        "audit_schema": "stateguard3r.v17-source-import-audit.v1",
        "module_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(visited.items())
        },
        "local_import_edges": {name: list(edges[name]) for name in sorted(edges)},
        "visited_modules": sorted(visited),
        "historical_local_capability": False,
        "offline_artifact_capability": False,
    }


def audit_v17_production_script(script: Path, runtime_paths: Iterable[Path]) -> Mapping[str, Any]:
    """Bind the v17 release script to its independently audited local graph."""

    script = Path(script).resolve(strict=True)
    try:
        relative = script.relative_to(ROOT / "scripts")
    except ValueError as error:
        raise V17SourceImportAuditError("v17 production script is outside scripts") from error
    if relative != Path("run_recal3r_beta_base_one_shot_v17.py"):
        raise V17SourceImportAuditError("v17 production script path differs")
    tree = _parse(script)
    imports = _local_imports(tree)
    if any(_is_disallowed_local_name(name) for name in imports):
        raise V17SourceImportAuditError("v17 production script imports a prohibited local module")
    if _has_prohibited_dynamic_import(tree):
        raise V17SourceImportAuditError("v17 production script uses a dynamic local import")
    source = script.read_text(encoding="utf-8")
    forbidden_tokens = (
        "formal-v3-runs",
        "development-dynamic",
        "input_manifest",
        "recovery-update-pressure",
        "bounded_update_pressure",
    )
    if any(token in source.lower() for token in forbidden_tokens):
        raise V17SourceImportAuditError("v17 production script contains a historical artifact literal")
    graph = audit_v17_runtime_import_graph(runtime_paths)
    allowed_script_imports = set(graph["visited_modules"]) | {"health"}
    if not set(imports) <= allowed_script_imports:
        raise V17SourceImportAuditError(
            "v17 production script import is absent from its audited runtime graph"
        )
    return {
        **graph,
        "production_script": str(script),
        "production_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "production_script_local_imports": sorted(set(imports)),
        "historical_production_artifact_capability": False,
    }


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise V17SourceImportAuditError(f"v17 runner function is missing: {name}")


def audit_v17_native_runner_contract(path: Path) -> Mapping[str, Any]:
    """Bind the sole temporary scalar scope and the pinned native ordering."""

    path = Path(path).resolve(strict=True)
    tree = _parse(path)
    function = _function(tree, "run_recurrent_beta_floor_native_v17")
    source = path.read_text(encoding="utf-8")
    try:
        mask = source.index("mask_selection = select_one_shot_beta_base_v17")
        scoped = source.index("with temporary_native_beta_floor_v17(", mask)
        state = source.index("state_feat = new_state * update_mask1", scoped)
        memory = source.index("mem = new_mem * update_mask", state)
        calibration = source.index("model._maybe_record_u_calibration_step", memory)
        reset = source.index("model._reset_update_pressure_if_needed", calibration)
        observe = source.index("observation = observer.observe", reset)
        commit = source.index("observer.finalize(observation)", observe)
        arm = source.index("arm_one_future_update_v17", commit)
        transfer = source.index("predictions.append(to_cpu(res))", arm)
    except ValueError as error:
        raise V17SourceImportAuditError("v17 native causal order is incomplete") from error
    if not mask < scoped < state < memory < calibration < reset < observe < commit < arm < transfer:
        raise V17SourceImportAuditError("v17 native causal order differs")
    strict_witness = source.find("_strict_mask_reduction_witness_v17", scoped, state)
    if strict_witness < 0:
        raise V17SourceImportAuditError("v17 runner lost its native strict-mask witness")
    if "get_u_calibration_last_state" in source:
        raise V17SourceImportAuditError("v17 runner reads a CPU calibration-state witness")
    model_calls = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
    }
    if not model_calls <= NATIVE_MODEL_METHODS:
        raise V17SourceImportAuditError("v17 runner calls an unapproved model method")
    if (
        "_reset_update_pressure_if_needed" not in model_calls
        or sum(
            1
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr == "_reset_update_pressure_if_needed"
        )
        != 1
    ):
        raise V17SourceImportAuditError("v17 runner lost its one native reset call")
    assigned_model_attrs = {
        node.targets[0].attr
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Attribute)
        and isinstance(node.targets[0].value.value, ast.Name)
        and node.targets[0].value.value.id == "model"
        and node.targets[0].value.attr == "config"
    }
    if assigned_model_attrs != {"model_update_type"}:
        raise V17SourceImportAuditError("v17 runner mutates an unapproved model field")
    scope_function = _function(tree, "temporary_native_beta_floor_v17")
    beta_assignments = [
        node
        for node in ast.walk(scope_function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "model"
        and node.targets[0].attr == "beta_base"
    ]
    if len(beta_assignments) != 2:
        raise V17SourceImportAuditError("v17 beta scope must assign only install and restore")
    forbidden_attributes = {
        node.attr
        for node in ast.walk(scope_function)
        if isinstance(node, ast.Attribute)
        and node.attr in {"cpu", "numpy", "tolist"}
    }
    if forbidden_attributes:
        raise V17SourceImportAuditError("v17 beta scope serializes a prohibited value")
    return {
        "runner_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "native_state_before_observer": True,
        "native_memory_before_observer": True,
        "native_calibration_before_observer": True,
        "native_reset_before_observer": True,
        "observer_commit_before_future_arm": True,
        "future_arm_before_raw_export": True,
        "only_candidate_model_mutation": "temporary_model_beta_base_scalar_scope",
        "native_reset_method_bound": "model._reset_update_pressure_if_needed",
        "native_strict_mask_witness": True,
        "alarm_frame_witness_origin": "post_native_commit_gpu_resident_state_memory_pose",
    }


__all__ = [
    "DISALLOWED_NAME_PARTS",
    "NATIVE_MODEL_METHODS",
    "V17_ALLOWED_MODULES",
    "V17SourceImportAuditError",
    "audit_v17_native_runner_contract",
    "audit_v17_production_script",
    "audit_v17_runtime_import_graph",
]
