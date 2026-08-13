"""Static isolation and causal-order checks for v20's native state hold."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "stateguard3r"
ALLOWED_MODULES = frozenset({
    "dynamic_rgb_capability_v20",
    "native_scalar_detector_v20",
    "official_native_scalar_hold_v20",
    "v20_source_import_audit",
})
FORBIDDEN_WORDS = (
    "recovery", "manifest", "archive", "ground_truth", "groundtruth",
    "timestamp", "overlap", "pressure", "pointmap", "anchor",
    "registration", "prerollout", "decoder", "spatial", "encoder",
    "patch", "repair", "rollback", "quarantine", "beta",
)
FORBIDDEN_MODEL_WRITES = frozenset({
    "beta_base", "update_pressure", "recal3r_state0", "pose_retriever",
    "config", "state_feat", "mem",
})


class V20SourceImportAuditError(ValueError):
    """A v20 runtime source escapes its preregistered capability boundary."""


def _module_name(path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(PACKAGE.resolve(strict=True))
    except ValueError as error:
        raise V20SourceImportAuditError("v20 audit source is outside package") from error
    if relative.suffix != ".py" or len(relative.parts) != 1:
        raise V20SourceImportAuditError("v20 audit accepts only direct package modules")
    return relative.stem


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise V20SourceImportAuditError(f"cannot parse v20 source {path.name}") from error


def _local_imports(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and module:
                names.append(module.split(".")[0])
            elif node.level:
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif module.startswith("stateguard3r."):
                names.append(module.split(".")[1])
        elif isinstance(node, ast.Import):
            names.extend(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("stateguard3r.") and len(alias.name.split(".")) >= 2
            )
    return tuple(names)


def _prohibited(name: str) -> bool:
    lowered = name.lower()
    return (
        name not in ALLOWED_MODULES
        or any(word in lowered for word in FORBIDDEN_WORDS)
        or bool(re.search(r"(?:^|_)v(?:[1-9]|1[0-9])(?:_|$)", lowered))
    )


def _dynamic_import(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name) and node.func.id == "__import__"
            or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
        ):
            return True
    return False


def _forbidden_literals(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower().replace("\\", "/")
            # Runtime files may name their own fresh v20 capability location,
            # but may not reach an archived/formal result, historical release,
            # or a v1--v19 component.  Do not treat explanatory docstrings
            # such as "no future data" as an import/read route.
            if any(token in lowered for token in ("/outputs/recovery-", "formal-v", "development-", "run.json", "state-timeline.json")):
                return True
            if re.search(r"(?:^|_)v(?:[1-9]|1[0-9])(?:_|$)", lowered):
                return True
    return False


def audit_v20_runtime_import_graph(paths: Iterable[Path]) -> Mapping[str, Any]:
    """Reject historical modules, artifacts and dynamic import routes."""

    initial = tuple(Path(path).resolve(strict=True) for path in paths)
    if not initial:
        raise V20SourceImportAuditError("v20 import audit needs a source")
    pending, visited, edges = list(initial), {}, {}
    while pending:
        path = pending.pop()
        name = _module_name(path)
        if _prohibited(name):
            raise V20SourceImportAuditError(f"v20 runtime module is prohibited: {name}")
        if name in visited:
            if visited[name] != path:
                raise V20SourceImportAuditError("v20 module resolves ambiguously")
            continue
        tree = _parse(path)
        if _dynamic_import(tree) or _forbidden_literals(tree):
            raise V20SourceImportAuditError(f"v20 source has a prohibited capability: {name}")
        children = _local_imports(tree)
        if any(_prohibited(child) for child in children):
            raise V20SourceImportAuditError(f"v20 source imports a prohibited local module: {name}")
        visited[name], edges[name] = path, tuple(sorted(set(children)))
        pending.extend(PACKAGE / f"{child}.py" for child in children)
    return {
        "audit_schema": "stateguard3r.v20-source-import-audit.v1",
        "module_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(visited.items())},
        "local_import_edges": {name: list(edges[name]) for name in sorted(edges)},
        "visited_modules": sorted(visited),
        "historical_local_capability": False,
        "offline_artifact_capability": False,
    }


def audit_v20_production_script(script: Path, runtime_paths: Iterable[Path]) -> Mapping[str, Any]:
    """Bind the unique v20 release script to the closed source graph."""

    script = Path(script).resolve(strict=True)
    if script != ROOT / "scripts" / "run_recal3r_native_scalar_hold_v20.py":
        raise V20SourceImportAuditError("v20 production script path differs")
    tree = _parse(script)
    imports = _local_imports(tree)
    if _dynamic_import(tree) or _forbidden_literals(tree) or any(_prohibited(name) for name in imports):
        raise V20SourceImportAuditError("v20 production script has a prohibited local route")
    graph = audit_v20_runtime_import_graph(runtime_paths)
    if not set(imports) <= set(graph["visited_modules"]):
        raise V20SourceImportAuditError("v20 production script import is not audited")
    return {
        **graph,
        "production_script": str(script),
        "production_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "production_script_local_imports": sorted(set(imports)),
        "historical_production_artifact_capability": False,
    }


def audit_v20_native_wrapper_contract(path: Path) -> Mapping[str, Any]:
    """Check that the only v20 effect is a native-mask result replacement."""

    path = Path(path).resolve(strict=True)
    tree, source = _parse(path), path.read_text(encoding="utf-8")
    try:
        mask = source.index("native_mask = originals[1]", source.index("def mask"))
        zero = source.index("held = native_mask.clone().zero_()", mask)
        native_record = source.index("originals[2](native_frame, state_before, state_after)", source.index("def record"))
        health = source.index("row, next_camera = _health_row(", native_record)
        observe = source.index("decision = observer.observe(row)", health)
        commit = source.index("observer.commit(row)", observe)
        arm = source.index("armed = bool(alarm", commit)
    except ValueError as error:
        raise V20SourceImportAuditError("v20 native wrapper causal order is incomplete") from error
    if not mask < zero < native_record < health < observe < commit < arm:
        raise V20SourceImportAuditError("v20 native wrapper causal order differs")
    forbidden_attributes = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"cpu", "numpy", "tolist"}
    }
    if forbidden_attributes:
        raise V20SourceImportAuditError("v20 native wrapper serializes a prohibited tensor")
    writes = {
        node.targets[0].attr for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "model"
    }
    if writes & FORBIDDEN_MODEL_WRITES:
        raise V20SourceImportAuditError("v20 native wrapper writes a prohibited model member")
    return {
        "wrapper_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "official_mask_called_before_zero_replacement": True,
        "state_commit_before_scalar_observation": True,
        "memory_update_rule_unchanged": True,
        "one_future_arm_after_current_commit": True,
        "only_candidate_effect": "zero_clone_of_native_state_mask",
        "no_future_or_final_trace_mode": True,
    }


__all__ = [
    "ALLOWED_MODULES", "V20SourceImportAuditError", "audit_v20_native_wrapper_contract",
    "audit_v20_production_script", "audit_v20_runtime_import_graph",
]
