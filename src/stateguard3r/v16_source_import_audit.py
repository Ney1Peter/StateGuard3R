"""Static local-import and prohibited-capability audit for the v16 graph."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "stateguard3r"
FORBIDDEN_MODULE_PARTS = (
    "recovery", "manifest", "archive", "quarantine", "pointmap",
    "anchor", "registration", "prerollout", "decoder", "spatial", "encoder", "patch",
)
ALLOWED_V16_MODULES = frozenset({
    "bounded_update_pressure_v16",
    "recal3r_bounded_update_pressure_runner_v16",
    "online_detector_v16",
    "online_visual_overlap_v16",
    "timestamp_order_v16",
    "dynamic_rgb_capability_v16",
    "health",
    "v16_source_import_audit",
})


class V16SourceImportAuditError(ValueError):
    """The v16 runtime graph acquired a prohibited local capability."""


def _module_name(path: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(PACKAGE.resolve(strict=True))
    except ValueError as error:
        raise V16SourceImportAuditError("v16 audit path is outside the package") from error
    if relative.suffix != ".py" or len(relative.parts) != 1:
        raise V16SourceImportAuditError("v16 audit accepts only direct package modules")
    return relative.stem


def _local_imports(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise V16SourceImportAuditError(f"cannot parse v16 source {path.name}") from error
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
            imported.extend(alias.name.split(".")[1] for alias in node.names if alias.name.startswith("stateguard3r.") and len(alias.name.split(".")) >= 2)
    return tuple(imported)


def _forbidden_local_name(name: str) -> bool:
    lowered = name.lower()
    if name not in ALLOWED_V16_MODULES or any(token in lowered for token in FORBIDDEN_MODULE_PARTS):
        return True
    return bool(re.search(r"(?:^|_)v(?:[1-9]|1[0-5])(?:_|$)", name))


def _source_has_prohibited_literal(path: Path) -> bool:
    """Reject artifact/legacy import literals, including dynamic-import bypasses."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.lower().replace("\\", "/")
        if "/outputs/" in value or "run.json" in value or "input_manifest" in value:
            return True
        # A module name uses dot-separated Python identifiers.  Schema names
        # such as ``stateguard3r.dynamic-rgb-capability-v16.v1`` are data, not
        # dynamic imports, and must not be confused with one.
        match = re.fullmatch(r"stateguard3r\.([A-Za-z_]\w*)", value)
        if match:
            local = match.group(1)
            if _forbidden_local_name(local):
                return True
    return False


def audit_v16_runtime_import_graph(paths: Iterable[Path]) -> Mapping[str, Any]:
    """Walk recursive local imports and reject legacy, offline, and unknown nodes."""
    initial = tuple(Path(path).resolve(strict=True) for path in paths)
    if not initial:
        raise V16SourceImportAuditError("v16 import audit needs at least one source path")
    pending = list(initial)
    visited: dict[str, Path] = {}
    edges: dict[str, tuple[str, ...]] = {}
    while pending:
        path = pending.pop()
        name = _module_name(path)
        if _forbidden_local_name(name):
            raise V16SourceImportAuditError(f"v16 runtime module is prohibited: {name}")
        if name in visited:
            if visited[name] != path:
                raise V16SourceImportAuditError(f"v16 module name resolves ambiguously: {name}")
            continue
        if _source_has_prohibited_literal(path):
            raise V16SourceImportAuditError(f"v16 source contains a prohibited artifact literal: {name}")
        imported = _local_imports(path)
        if any(_forbidden_local_name(child) for child in imported):
            raise V16SourceImportAuditError(f"v16 source imports a prohibited local module: {name}")
        visited[name] = path
        edges[name] = tuple(sorted(set(imported)))
        pending.extend(PACKAGE / f"{child}.py" for child in imported)
    missing = sorted(ALLOWED_V16_MODULES - {"dynamic_rgb_capability_v16", "health", "v16_source_import_audit"} - set(visited))
    return {
        "audit_schema": "stateguard3r.v16-source-import-audit.v1",
        "module_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(visited.items())},
        "local_import_edges": {name: list(edges[name]) for name in sorted(edges)},
        "visited_modules": sorted(visited),
        "unused_allowed_runtime_modules": missing,
        "legacy_recovery_or_input_import": False,
        "offline_annotation_or_artifact_path": False,
    }


def audit_v16_production_script(script: Path, runtime_paths: Iterable[Path]) -> Mapping[str, Any]:
    """Bind the runner script's local imports to the independently audited graph."""
    script = Path(script).resolve(strict=True)
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise V16SourceImportAuditError("cannot parse v16 production script") from error
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("stateguard3r."):
            imports.append(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[1] for alias in node.names if alias.name.startswith("stateguard3r.") and len(alias.name.split(".")) >= 2)
    if any(_forbidden_local_name(name) for name in imports):
        raise V16SourceImportAuditError("v16 production script imports a prohibited local module")
    # Evidence-schema strings are not imports.  Inspect a dotted literal only
    # when it is the module argument of a real dynamic-import call.
    for node in ast.walk(tree):
        dynamic = (
            isinstance(node, ast.Call)
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
        )
        if dynamic:
            match = re.fullmatch(r"stateguard3r\.([A-Za-z_]\w*)", node.args[0].value)
            if match and _forbidden_local_name(match.group(1)):
                raise V16SourceImportAuditError("v16 production script contains a prohibited dynamic local import")
    graph = audit_v16_runtime_import_graph(runtime_paths)
    return {
        **graph,
        "production_script": str(script),
        "production_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "production_script_local_imports": sorted(set(imports)),
    }


__all__ = ["ALLOWED_V16_MODULES", "FORBIDDEN_MODULE_PARTS", "V16SourceImportAuditError", "audit_v16_production_script", "audit_v16_runtime_import_graph"]
