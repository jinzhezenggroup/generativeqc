"""Render the complete public method catalog for Sphinx.

The native ABI manifest intentionally owns only native method IDs. This renderer
combines that manifest with public composite selectors and the automatic Libxc
semilocal resolver so the user documentation reflects every public entry path
without turning documentation into another capability registry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
MANIFEST = ROOT / "manifests/public_methods.json"
LIBXC_CATALOG = PYTHON / "generativeqc_compiler/xc/libxc_bulk_catalog.json"


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported public method manifest schema")
    if not isinstance(payload.get("methods"), list):
        raise TypeError("public method manifest requires methods")
    if not isinstance(payload.get("composite_methods", []), list):
        raise TypeError("public method manifest composite_methods must be a list")
    return payload


def _automatic_libxc_rows() -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    if str(PYTHON) not in sys.path:
        sys.path.insert(0, str(PYTHON))

    from generativeqc_compiler.xc.libxc_blacklist import LIBXC_SEMILOCAL_BLACKLIST
    from generativeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

    catalog = json.loads(LIBXC_CATALOG.read_text(encoding="utf-8"))
    families = {
        record["name"]: record["family"]
        for record in catalog["registrations"]
        if record.get("graph_status") == "imported"
    }
    automatic = set(AUTO_BULK_COMPONENTS)
    available = sorted(
        (
            (name, families.get(name, "unknown"))
            for name in automatic
            if name not in LIBXC_SEMILOCAL_BLACKLIST
        ),
        key=lambda row: (row[1], row[0]),
    )
    blocked = sorted(
        (
            (name, families.get(name, "unknown"), reason)
            for name, reason in LIBXC_SEMILOCAL_BLACKLIST.items()
            if name in automatic
        ),
        key=lambda row: (row[1], row[0]),
    )
    return available, blocked


def public_method_doc_dependencies() -> tuple[Path, ...]:
    """Track both catalog data and resolution logic for incremental Sphinx builds."""
    paths = {
        Path(__file__).resolve(),
        MANIFEST,
        LIBXC_CATALOG,
        PYTHON / "generativeqc/ks.py",
        PYTHON / "generativeqc/_generated_methods.py",
    }
    for package in ("method", "xc"):
        directory = PYTHON / "generativeqc_compiler" / package
        for pattern in ("*.py", "*.json"):
            paths.update(directory.rglob(pattern))
    return tuple(sorted(paths))


def _compiler_dft_rows() -> list[tuple[str, str, str]]:
    """Discover named compositions and aliases through the public KS resolver."""
    if str(PYTHON) not in sys.path:
        sys.path.insert(0, str(PYTHON))

    from generativeqc.ks import public_dft_selectors, resolve_ks_method

    rows = []
    for selector in public_dft_selectors():
        try:
            method, _ = resolve_ks_method(selector)
        except (ValueError, NotImplementedError):
            continue
        rows.append((selector, method.identifier, method.spin))
    return rows


def _code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "—"


def render_public_methods_markdown() -> str:
    payload = _load_manifest()
    methods = sorted(payload["methods"], key=lambda method: method["abi_id"])
    composites = payload.get("composite_methods", [])
    automatic, blocked = _automatic_libxc_rows()

    lines = [
        "# Public methods",
        "",
        "This page is rendered automatically by Sphinx from the public native method",
        "manifest, compiler-discovered DFT names, public composite selectors, and",
        "the automatic Libxc semilocal inventory. It is not a handwritten support list.",
        "",
        "Backend-, basis-, grid-, spin-, and property-specific admission checks still",
        "apply at execution time and fail closed when a requested combination is not",
        "qualified.",
        "",
        "## Registered native methods",
        "",
        "These selectors have stable native ABI method IDs.",
        "",
        "| Method | Family | Properties | Batch | Aliases | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method in methods:
        properties = _code_list(method["properties"])
        aliases = _code_list(method.get("aliases", []))
        family = method["family"].replace("_", " ")
        batch = "yes" if method["supports_batch"] else "no"
        status = (
            "available"
            if method["properties"]
            else "unavailable — "
            + method.get("unavailable_reason", "no executable provider")
        )
        lines.append(
            f"|`{method['name']}` | {family} | {properties} | {batch} | "
            f"{aliases} | {status} |"
        )

    # Do not duplicate compatibility selectors already shown in either manifest
    # table. Compiler discovery owns any newly admitted names and aliases.
    listed = {
        selector
        for method in (*methods, *composites)
        for selector in (method["name"], *method.get("aliases", []))
    }
    lines.extend(
        [
            "",
            "## Compiler-discovered DFT selectors",
            "",
            "These additional named compositions and aliases pass the same public",
            "KS resolver used by Calculator, without requiring another native ABI ID.",
            "Names already shown in the native or composite tables are omitted here.",
            "Backend, basis, grid and derivative admission still apply at execution.",
            "",
            "| Method | MethodIR identifier | Spin |",
            "| --- | --- | --- |",
        ]
    )
    for selector, identifier, spin in _compiler_dft_rows():
        if selector not in listed:
            lines.append(f"|`{selector}` | `{identifier}` | {spin} |")

    lines.extend(
        [
            "",
            "## Public composite selectors",
            "",
            "These selectors resolve to compiler-owned MethodIR compositions rather",
            "than receiving a second native ABI ID.",
            "",
            "| Method | Composition | Spin | Properties | Batch | Aliases |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method in composites:
        lines.append(
            f"|`{method['name']}` | `{method['identifier']}` | "
            f"{method['spin']} | {_code_list(method['properties'])} | "
            f"{'yes' if method['supports_batch'] else 'no'} | "
            f"{_code_list(method['aliases'])} |"
        )

    lines.extend(
        [
            "",
            "## Automatic Libxc semilocal MethodIR registrations",
            "",
            "The automatic CPU Libxc route is default-allow for imported non-curated",
            "LDA/GGA/meta-GGA registrations whose required ingredients are supported,",
            "except for explicit functional-specific blacklist entries. These names",
            "are MethodIR/KS registrations, not native ABI method IDs.",
            "",
            f"Current automatically admitted registrations: **{len(automatic)}**.",
            "",
            "| Libxc identifier | Family | Spin | Backend | Properties |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, family in automatic:
        lines.append(f"|`{name}` | {family} | RKS / UKS | CPU | `energy` |")

    lines.extend(
        [
            "",
            "## Explicit Libxc blacklist",
            "",
            "These imported registrations are intentionally excluded from the",
            "automatic public route until their functional-specific production defect",
            "is resolved.",
            "",
            f"Current explicit blacklist entries in the automatic inventory: **{len(blocked)}**.",
            "",
            "| Libxc identifier | Family | Reason |",
            "| --- | --- | --- |",
        ]
    )
    for name, family, reason in blocked:
        lines.append(f"|`{name}` | {family} | {reason} |")

    lines.append("")
    return "\n".join(lines)


def render_public_methods_source(app: Any, docname: str, source: list[str]) -> None:
    """Populate the public-method page and register every source dependency."""
    if docname != "public_methods":
        return
    for dependency in public_method_doc_dependencies():
        app.env.note_dependency(str(dependency))
    source[0] = render_public_methods_markdown()


if __name__ == "__main__":
    print(render_public_methods_markdown())
