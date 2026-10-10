"""Check every public Python declaration and its applicable contract facets.

Discovery never imports runtime, optional, or GPU modules. Re-exports resolve to
one source owner; a reviewed, source-bound manifest records which facets apply.
Unknown export forms and unclassified new declarations fail closed. Prose still
requires review: this checker verifies contract coverage, not scientific truth.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from tools.render_python_api_doc import PACKAGE, ROOT, public_api_modules
except ModuleNotFoundError:
    from render_python_api_doc import PACKAGE, ROOT, public_api_modules

RUFF_DOC_RULES = "D100,D101,D102,D103,D104,D105,D107"
MANIFEST = ROOT / "manifests/python_api_contracts.json"
# Applicability is explicit and reviewed, rather than guessed from prose length.
PROFILES = {
    "accessor": {"behavior"},
    "constant": {"behavior"},
    "type": {"behavior"},
    "module": {"behavior"},
    "exception": {"behavior"},
    "lifecycle": {"behavior", "errors", "ownership"},
    "record": {"behavior", "values", "ownership"},
    "validation": {"behavior", "values", "errors"},
    "transformation": {"behavior", "values", "errors", "ownership"},
    "execution": {"behavior", "values", "errors", "ownership", "backends"},
    "resource": {"behavior", "values", "errors", "ownership"},
}
PLACEHOLDER = re.compile(
    r"\b(?:TODO|FIXME|TBD|XXX|placeholder|undocumented)\b", re.IGNORECASE
)
DEFINITION = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Declaration:
    """One authoritative source declaration, independent of its public aliases."""

    path: Path
    name: str
    kind: str
    node: ast.AST

    def key(self, source_root: Path) -> str:
        """Bind contract inventory to the logical source file and declaration."""
        return f"{self.path.relative_to(source_root)}:{self.name}"

    @property
    def docstring(self) -> str | None:
        """Read source documentation only for nodes with a Python docstring."""
        if isinstance(self.node, (*DEFINITION, ast.Module)):
            return ast.get_docstring(self.node)
        return None


class ExportResolutionError(ValueError):
    """An export cannot be statically bound to a supported source declaration."""


def _module_source(source_root: Path, name: str) -> Path | None:
    """Locate first-party Python source without importing dependencies."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name):
        raise ExportResolutionError(f"invalid module path {name!r}")
    prefix = source_root.joinpath(*name.split("."))
    if not prefix.resolve().is_relative_to(source_root.resolve()):
        raise ExportResolutionError(f"module path escapes source root: {name!r}")
    for path in (prefix.with_suffix(".py"), prefix / "__init__.py"):
        if path.is_file():
            if not path.resolve().is_relative_to(source_root.resolve()):
                raise ExportResolutionError(f"module source escapes root: {name!r}")
            return path
    return None


def _literal(value: ast.AST | None) -> object:
    """Accept literals and the existing literal frozenset convention only."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) == 1
        and not value.keywords
    ):
        return frozenset(ast.literal_eval(value.args[0]))
    return ast.literal_eval(value)


def _type_union(value: ast.AST | None) -> bool:
    """Recognize explicit PEP 604 aliases without evaluating imported types."""
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.BitOr)
        and all(
            isinstance(item, (ast.Name, ast.Attribute)) or _type_union(item)
            for item in (value.left, value.right)
        )
    )


class _UnboundExport(ExportResolutionError):
    """No package attribute exists; import-from may next try a child module."""

    def __init__(self, module: str, symbol: str, path: Path) -> None:
        self.module = module
        self.symbol = symbol
        super().__init__(
            f"unresolved export {module}.{symbol} in {path.name}; "
            "use an explicit local definition/import, simple alias, or supported literal lazy map"
        )


def _bound_names(node: ast.AST) -> set[str]:
    """Find possible module bindings without descending into function scopes."""
    if isinstance(node, DEFINITION):
        return {node.name}
    if isinstance(node, ast.Lambda):
        return set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    return set().union(*(_bound_names(child) for child in ast.iter_child_nodes(node)))


def _false_type_checking_guard(tree: ast.Module, node: ast.If) -> bool:
    """Ignore typing-only declarations only when the guard is proven false."""
    test = node.test
    # Also cover `typing.TYPE_CHECKING = True` before a subsequent import of
    # the flag; an unshadowed local import is not then evidence of False.
    if any(
        isinstance(item, ast.Attribute)
        and item.attr == "TYPE_CHECKING"
        and isinstance(item.ctx, (ast.Store, ast.Del))
        for statement in tree.body
        if statement.lineno < node.lineno and not isinstance(statement, DEFINITION)
        for item in ast.walk(statement)
    ):
        return False
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        binding = _binding(tree, test.id, node.lineno)
        if (
            isinstance(binding, ast.ImportFrom)
            and binding.module == "typing"
            and binding.level == 0
        ):
            return any(
                alias.name == "TYPE_CHECKING"
                and (alias.asname or alias.name) == test.id
                for alias in binding.names
            )
        return (
            isinstance(binding, (ast.Assign, ast.AnnAssign))
            and isinstance(binding.value, ast.Constant)
            and binding.value.value is False
        )
    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
    ):
        binding = _binding(tree, test.value.id, node.lineno)
        imported = isinstance(binding, ast.Import) and any(
            alias.name == "typing" and (alias.asname or alias.name) == test.value.id
            for alias in binding.names
        )
        if not imported:
            return False
        # A direct write to the module flag defeats the typing convention.
        return not any(
            isinstance(item, ast.Attribute)
            and item.attr == "TYPE_CHECKING"
            and isinstance(item.value, ast.Name)
            and item.value.id == test.value.id
            and isinstance(item.ctx, (ast.Store, ast.Del))
            for statement in tree.body
            if statement.lineno < node.lineno and not isinstance(statement, DEFINITION)
            for item in ast.walk(statement)
        )
    return False


def _binding(
    tree: ast.Module, symbol: str, before: int | None = None
) -> ast.AST | None:
    """Honor the latest runtime binding and reject unknown conditional writes."""
    for node in reversed(tree.body):
        if before is not None and node.lineno >= before:
            continue
        if isinstance(node, ast.If) and _false_type_checking_guard(tree, node):
            # An else arm does run; unknown bindings there still fail closed.
            if any(symbol in _bound_names(item) for item in node.orelse):
                raise ExportResolutionError(
                    f"unsupported conditional binding for {symbol}"
                )
            continue
        names = _bound_names(node)
        if symbol not in names and "*" not in names:
            continue
        if isinstance(node, (*DEFINITION, ast.Import, ast.ImportFrom)):
            if "*" in names:
                raise ExportResolutionError(
                    f"wildcard import may rebind {symbol}; use an explicit import"
                )
            return node
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == symbol
                for target in targets
            ):
                if isinstance(node, ast.AnnAssign) and node.value is None:
                    continue  # Annotation-only statements do not replace a value.
                return node
        raise ExportResolutionError(
            f"unsupported conditional/dynamic binding for {symbol} at line {node.lineno}"
        )
    return None


def _absolute_module(target: str, package: str) -> str:
    """Convert relative module syntax with an actionable static diagnostic."""
    try:
        return (
            importlib.util.resolve_name(target, package)
            if target.startswith(".")
            else target
        )
    except (ValueError, ImportError) as error:
        raise ExportResolutionError(
            f"invalid module target {target!r} in {package}"
        ) from error


def _checked_literal(tree: ast.Module, binding: ast.Assign | ast.AnnAssign) -> object:
    """Do not mistake a shadowed literal constructor for a constant expression."""
    value = binding.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and _binding(tree, value.func.id, binding.lineno) is not None
    ):
        raise ValueError(f"literal constructor {value.func.id} is shadowed")
    return _literal(value)


def _check_map_uses(
    tree: ast.Module, binding: ast.AST, name: str, before: int | None
) -> None:
    """Reject mutation/aliasing of literal lazy maps outside their loader body."""
    for node in tree.body:
        if node.lineno <= binding.lineno or (
            before is not None and node.lineno >= before
        ):
            continue
        if isinstance(node, DEFINITION):
            continue
        if not any(
            isinstance(item, ast.Name) and item.id == name for item in ast.walk(node)
        ):
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "list"
            and len(node.value.args) == 1
            and not node.value.keywords
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == name
            and _binding(tree, "list", node.lineno) is None
        ):
            continue
        raise ExportResolutionError(
            f"unsupported use/mutation of literal {name} at line {node.lineno}"
        )


def _resolve_export(
    source_root: Path,
    module_name: str,
    symbol: str,
    seen: set[tuple[str, str, int | None]],
    *,
    before: int | None = None,
) -> Declaration:
    """Resolve latest imports/aliases, literal lazy maps and explicit values."""
    key = (module_name, symbol, before)
    if key in seen:
        raise ExportResolutionError(f"cyclic export {module_name}.{symbol}")
    seen.add(key)
    path = _module_source(source_root, module_name)
    if path is None:
        raise ExportResolutionError(f"no first-party source for {module_name}.{symbol}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = (
        module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    )
    node = _binding(tree, symbol, before)
    if isinstance(node, DEFINITION):
        return Declaration(
            path,
            symbol,
            "class" if isinstance(node, ast.ClassDef) else "function",
            node,
        )
    if isinstance(node, ast.ImportFrom):
        alias = next(
            alias for alias in node.names if (alias.asname or alias.name) == symbol
        )
        target = _absolute_module("." * node.level + (node.module or ""), package)
        try:
            return _resolve_export(
                source_root,
                target,
                alias.name,
                seen,
                before=node.lineno if target == module_name else None,
            )
        except _UnboundExport as error:
            # Importing a package attribute beats a same-named child module.
            # Only a genuinely absent target attribute permits module fallback;
            # an unresolved alias inside that target must remain an error.
            if error.module != target or error.symbol != alias.name:
                raise
            child = _module_source(source_root, f"{target}.{alias.name}")
            if child is None:
                raise
            return Declaration(
                child, "<module>", "module", ast.parse(child.read_text())
            )
    if isinstance(node, ast.Import):
        alias = next(
            alias
            for alias in node.names
            if (alias.asname or alias.name.split(".")[0]) == symbol
        )
        target = alias.name if alias.asname else alias.name.split(".")[0]
        child = _module_source(source_root, target)
        if child is None:
            raise ExportResolutionError(f"no first-party module contract for {target}")
        return Declaration(child, "<module>", "module", ast.parse(child.read_text()))
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        if isinstance(value, ast.Name):
            return _resolve_export(
                source_root, module_name, value.id, seen, before=node.lineno
            )
        if _type_union(value):
            return Declaration(path, symbol, "type", node)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.attr == "dtype"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value in ("bool", "float32", "float64")
            and not value.keywords
        ):
            numpy_binding = _binding(tree, value.func.value.id, node.lineno)
            if isinstance(numpy_binding, ast.Import) and any(
                alias.name == "numpy"
                and (alias.asname or alias.name) == value.func.value.id
                for alias in numpy_binding.names
            ):
                return Declaration(path, symbol, "dtype", node)
        try:
            _checked_literal(tree, node)
        except (ValueError, TypeError) as error:
            raise ExportResolutionError(
                f"unresolved export {module_name}.{symbol}: unsupported assignment at line {node.lineno}; "
                "use a simple alias, literal constant, explicit type union or supported NumPy dtype"
            ) from error
        return Declaration(path, symbol, "constant", node)

    for map_name in ("_EXPORTS", "_LAZY_EXPORTS", "_PUBLIC_MODULES"):
        binding = _binding(tree, map_name, before)
        if binding is None:
            continue
        if not isinstance(binding, (ast.Assign, ast.AnnAssign)):
            raise ExportResolutionError(
                f"{module_name}.{map_name} must be a literal mapping"
            )
        _check_map_uses(tree, binding, map_name, before)
        try:
            mapping = _checked_literal(tree, binding)
        except (ValueError, TypeError) as error:
            raise ExportResolutionError(
                f"{module_name}.{map_name} must be a literal mapping"
            ) from error
        if map_name == "_PUBLIC_MODULES":
            if not isinstance(mapping, (tuple, list, set, frozenset)):
                raise ExportResolutionError(
                    f"{module_name}.{map_name} must be a literal module collection"
                )
            if symbol in mapping:
                child = _module_source(source_root, f"{module_name}.{symbol}")
                if child is None:
                    raise ExportResolutionError(
                        f"missing lazy submodule {module_name}.{symbol}"
                    )
                return Declaration(
                    child, "<module>", "module", ast.parse(child.read_text())
                )
            continue
        if not isinstance(mapping, dict):
            raise ExportResolutionError(
                f"{module_name}.{map_name} must be a literal mapping"
            )
        if symbol not in mapping:
            continue
        target = mapping[symbol]
        if map_name == "_EXPORTS" and isinstance(target, str):
            target_module, attribute = _absolute_module(target, package), symbol
        elif (
            map_name == "_LAZY_EXPORTS"
            and isinstance(target, tuple)
            and len(target) == 2
            and all(isinstance(part, str) and part for part in target)
        ):
            target_module, attribute = (
                _absolute_module("." + target[0], package),
                target[1],
            )
        else:
            raise ExportResolutionError(
                f"{module_name}.{map_name}[{symbol!r}] has unsupported target {target!r}"
            )
        return _resolve_export(source_root, target_module, attribute, seen)
    raise _UnboundExport(module_name, symbol, path)


def _public_member(name: str) -> bool:
    """Include explicit public names and Python magic operations."""
    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _check_class_bindings(declaration: Declaration) -> None:
    """Reject hidden callable/conditional members without rejecting typed fields.

    Explicit public methods/classes are inspected below. Literal class constants
    and annotated data fields are not callable declarations. Unannotated aliases,
    factories, lambdas, known callable aliases even when annotated, and conditional
    public bindings need an explicit method definition rather than silent omission.
    """
    assert isinstance(declaration.node, ast.ClassDef)
    callable_names = {
        node.name for node in declaration.node.body if isinstance(node, DEFINITION)
    }
    dataclass_record = any(
        (isinstance(target, ast.Name) and target.id == "dataclass")
        or (isinstance(target, ast.Attribute) and target.attr == "dataclass")
        for decorator in declaration.node.decorator_list
        for target in (
            decorator.func if isinstance(decorator, ast.Call) else decorator,
        )
    )
    for node in declaration.node.body:
        if isinstance(node, DEFINITION):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = set().union(*(_bound_names(target) for target in targets))
            value = node.value
            if isinstance(node, ast.AnnAssign) and value is None:
                continue  # An annotation alone does not replace a live method.
            known_callable = (
                bool(names & callable_names)
                or isinstance(value, ast.Lambda)
                or isinstance(value, ast.Name)
                and value.id in callable_names
                or isinstance(value, ast.Call)
                and (
                    isinstance(value.func, ast.Name)
                    and value.func.id in {"staticmethod", "classmethod", "property"}
                )
            )
            if known_callable:
                callable_names.update(names)
            public = sorted(name for name in names if _public_member(name))
            if not public:
                continue
            if isinstance(node, ast.AnnAssign) and not known_callable:
                annotation = node.annotation
                if isinstance(annotation, ast.Subscript):
                    annotation = annotation.value
                class_variable = (
                    isinstance(annotation, ast.Name)
                    and annotation.id == "ClassVar"
                    or isinstance(annotation, ast.Attribute)
                    and annotation.attr == "ClassVar"
                )
                if value is None or dataclass_record and not class_variable:
                    continue  # Ordinary dataclass fields/default factories are data.
                if isinstance(value, ast.Name):
                    module_tree = ast.parse(declaration.path.read_text())
                    binding = _binding(module_tree, value.id, declaration.node.lineno)
                    if isinstance(binding, (ast.Assign, ast.AnnAssign)):
                        try:
                            _checked_literal(module_tree, binding)
                        except (TypeError, ValueError):
                            pass
                        else:
                            continue
            if not known_callable:
                try:
                    ast.literal_eval(value)
                except (TypeError, ValueError):
                    pass
                else:
                    continue
            raise ExportResolutionError(
                f"{declaration.name}.{public[0]}: unsupported public class assignment; "
                "define the method/class explicitly, or annotate a genuine data field"
            )
        public = sorted(name for name in _bound_names(node) if _public_member(name))
        if public:
            raise ExportResolutionError(
                f"{declaration.name}.{public[0]}: unsupported conditional/dynamic class member; "
                "declare public methods and data fields directly in the class body"
            )


def _members(declaration: Declaration) -> Iterator[Declaration]:
    """Yield only explicitly defined public/magic members, not inherited ones."""
    yield declaration
    if not isinstance(declaration.node, ast.ClassDef):
        return
    _check_class_bindings(declaration)
    for node in declaration.node.body:
        if not isinstance(node, DEFINITION):
            continue
        if not _public_member(node.name):
            continue
        suffix = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr in (
                    "setter",
                    "deleter",
                ):
                    suffix = f"@{decorator.attr}"
        member = Declaration(
            declaration.path,
            f"{declaration.name}.{node.name}{suffix}",
            "class" if isinstance(node, ast.ClassDef) else "function",
            node,
        )
        yield from _members(member)


def _record_declaration(
    declarations: dict[str, Declaration],
    declaration: Declaration,
    source_root: Path,
    failures: list[str],
) -> None:
    """Reject different live objects that would collide in a source-owner key."""
    key = declaration.key(source_root)
    previous = declarations.get(key)
    if previous is not None and (
        previous.kind != declaration.kind
        or getattr(previous.node, "lineno", None)
        != getattr(declaration.node, "lineno", None)
    ):
        failures.append(
            f"{key}: conflicting live definitions at one authoritative owner; use distinct source names"
        )
        return
    declarations[key] = declaration


def public_declarations(
    package: Path | None = None,
) -> tuple[dict[str, str], dict[str, Declaration], list[str]]:
    """Return alias-to-owner bindings, unique declarations and resolution errors."""
    package = PACKAGE if package is None else package
    exports, declarations, failures = {}, {}, []
    for module in public_api_modules(package):
        for symbol in module.exports:
            exported = f"{module.name}.{symbol}"
            try:
                owner = _resolve_export(package.parent, module.name, symbol, set())
            except ExportResolutionError as error:
                failures.append(f"{exported}: {error}")
                continue
            exports[exported] = owner.key(package.parent)
            try:
                for declaration in _members(owner):
                    _record_declaration(
                        declarations, declaration, package.parent, failures
                    )
            except ExportResolutionError as error:
                failures.append(f"{exported}: {error}")
    return exports, declarations, failures


def _doc_error(doc: str | None) -> str | None:
    """Reject missing documentation and explicit placeholders without size tests."""
    if not doc or not doc.strip():
        return "missing docstring"
    if PLACEHOLDER.search(doc) or doc.strip().strip(". ").lower() in {
        "pass",
        "...",
        "documentation goes here",
    }:
        return "placeholder documentation"
    return None


def missing_public_docstrings(package: Path | None = None) -> tuple[str, ...]:
    """Check source docstrings and fail closed on unknown public export forms."""
    package = PACKAGE if package is None else package
    failures = []
    declarations: dict[str, Declaration] = {}
    # Keep the exported alias in diagnostics, including aliases of the same owner.
    for module in public_api_modules(package):
        for symbol in module.exports:
            try:
                owner = _resolve_export(package.parent, module.name, symbol, set())
            except ExportResolutionError as error:
                failures.append(f"{module.name}.{symbol}: {error}")
                continue
            try:
                for declaration in _members(owner):
                    _record_declaration(
                        declarations, declaration, package.parent, failures
                    )
                    if declaration.kind in ("constant", "type", "dtype"):
                        continue  # Their substantive docs are required by the manifest.
                    if error := _doc_error(declaration.docstring):
                        suffix = declaration.name.removeprefix(owner.name)
                        failures.append(
                            f"{module.name}.{symbol}{suffix} -> {declaration.path.relative_to(package.parent)}: {error}"
                        )
            except ExportResolutionError as error:
                failures.append(f"{module.name}.{symbol}: {error}")
    return tuple(sorted(set(failures)))


def _reference_text(root: Path, reference: str) -> str:
    """Resolve a repository Markdown label to its own nonempty contract section."""
    filename, separator, anchor = reference.partition("#")
    path = root / filename
    if (
        not separator
        or not anchor
        or not filename.startswith("docs/")
        or ".." in Path(filename).parts
    ):
        raise ValueError("contract reference must be docs/path.md#explicit-label")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"contract document escapes repository root: {filename}")
    if not path.is_file():
        raise ValueError(f"missing contract document {filename}")
    text = path.read_text(encoding="utf-8")
    label = re.search(rf"^\({re.escape(anchor)}\)=\s*$", text, re.MULTILINE)
    if label is None:
        raise ValueError(f"missing contract label {reference}")
    section = text[label.end() :]
    heading = re.match(r"\s*(#{1,6})[^\n]*\n", section)
    if heading is None:
        raise ValueError(f"contract label must precede a heading: {reference}")
    section = section[heading.end() :]
    # Every facet owns its text; content from the next facet cannot fill a hole.
    section = re.split(
        r"^#{1,6}\s|^\([^)]+\)=\s*$", section, maxsplit=1, flags=re.MULTILINE
    )[0].strip()
    if error := _doc_error(section):
        raise ValueError(f"{reference}: {error}")
    return section


def public_contract_errors(
    package: Path | None = None, manifest: Path | None = None, root: Path | None = None
) -> tuple[str, ...]:
    """Require a source-bound, debt-free contract record for every declaration."""
    package = PACKAGE if package is None else package
    manifest = MANIFEST if manifest is None else manifest
    root = ROOT if root is None else root
    exports, declarations, failures = public_declarations(package)
    try:
        policy = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return (f"cannot read public contract manifest: {error}",)
    if policy.get("schema_version") != 1 or policy.get("debt") != []:
        failures.append(
            "public contract manifest requires schema_version=1 and an empty debt list"
        )
    recorded_exports = policy.get("exports", {})
    for exported in sorted(exports.keys() | recorded_exports.keys()):
        if exports.get(exported) != recorded_exports.get(exported):
            failures.append(f"{exported}: missing/stale authoritative export binding")
    records = policy.get("declarations", {})
    for key in sorted(declarations.keys() | records.keys()):
        declaration, record = declarations.get(key), records.get(key)
        if declaration is None:
            failures.append(
                f"{key}: stale contract record; remove retired declarations"
            )
            continue
        if record is None:
            failures.append(
                f"{key}: missing contract classification; choose applicable facets and authoritative references"
            )
            continue
        profile = record.get("profile")
        if profile not in PROFILES:
            failures.append(f"{key}: unknown contract profile {profile!r}")
            continue
        if record.get("kind") != declaration.kind:
            failures.append(f"{key}: stale declaration kind")
        facets = record.get("facets", {})
        for facet in sorted(PROFILES[profile] - facets.keys()):
            failures.append(f"{key}: missing applicable {facet} contract")
        for facet, reference in facets.items():
            try:
                if reference == "docstring":
                    if error := _doc_error(declaration.docstring):
                        raise ValueError(error)
                    if facet != "behavior":
                        raise ValueError(
                            "non-behavior facets require an explicit authoritative section"
                        )
                elif isinstance(reference, str):
                    _reference_text(root, reference)
                else:
                    raise ValueError("contract reference must be a string")
            except ValueError as error:
                failures.append(f"{key}: {facet}: {error}")
    return tuple(sorted(set(failures)))


def main(argv: list[str] | None = None) -> int:
    """Check complete contract coverage and pinned Ruff on published modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff", default="ruff", help="path to the Ruff executable")
    args = parser.parse_args(argv)
    errors = (*missing_public_docstrings(), *public_contract_errors())
    for error in errors:
        print(error, file=sys.stderr)
    command = [
        args.ruff,
        "check",
        "--select",
        RUFF_DOC_RULES,
        *(str(module.source.relative_to(ROOT)) for module in public_api_modules()),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    return 1 if errors or result.returncode else 0


if __name__ == "__main__":
    sys.exit(main())
