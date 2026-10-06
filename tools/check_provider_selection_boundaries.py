"""Freeze provider-selection API debt at scientific/compiler ownership boundaries.

Direct vendor calls are guarded by check_vendor_boundaries.py. This companion
guard catches a different failure mode: method/schedule APIs that select an
implementation by name even when the vendor call itself lives elsewhere.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
import typing
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_electronic_structure_boundaries import _cpp_code

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("manifests/maintenance/provider_selection_boundaries.json")
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu", ".inl"}
CLASSIFICATIONS = {
    "provider",
    "runtime_abi",
    "infrastructure",
    "diagnostic",
    "migration",
}

# These are implementation-selection surfaces, not generic words such as
# "provider" or "backend". The list is intentionally narrow: adding a new
# vendor-named switch must require an explicit architecture review instead of
# evading the guard through a slightly different spelling.
SELECTOR_IDENTIFIERS = frozenset(
    {
        "reduction_provider",
        "matrix_gemm",
        "df_matrix_gemm",
        "lambda_matrix_gemm",
        "conventional_matrix_gemm",
        "use_cublas",
        "use_cublaslt",
        "use_cutensor",
        "use_cutlass",
        "use_cub",
        "use_cusolver",
        "use_cusparse",
        "use_nccl",
    }
)
_SELECTOR = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, SELECTOR_IDENTIFIERS))) + r")\b"
)


def _selector_references(source: str) -> Counter[str]:
    """Count selector identifiers in native code, ignoring comments/literals."""

    code = _cpp_code(source)
    return Counter(match.group() for match in _SELECTOR.finditer(code))


def _python_selector_references(source: str) -> Counter[str]:
    """Count Python identifiers plus non-docstring selector strings.

    String keys are included because a provider selector hidden in a schedule
    dictionary or emitted generator fragment is still an API surface. Docstrings
    are descriptive text and do not count.
    """

    result: Counter[str] = Counter()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in SELECTOR_IDENTIFIERS:
            result[token.string] += 1

    tree = ast.parse(source)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            result.update(match.group() for match in _SELECTOR.finditer(node.value))
    return result


def source_inventory(root: Path) -> dict[str, dict[str, int]]:
    """Collect production selector surfaces in stable path order."""

    paths = [
        path
        for directory in (root / "src", root / "include")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in NATIVE_SUFFIXES
    ]
    paths += list((root / "python/generativeqc_compiler").rglob("*.py"))
    paths += list((root / "tools").glob("generate*.py"))

    inventory: dict[str, dict[str, int]] = {}
    for path in sorted(paths):
        references = (
            _python_selector_references(path.read_text())
            if path.suffix == ".py"
            else _selector_references(path.read_text())
        )
        if references:
            inventory[path.relative_to(root).as_posix()] = dict(
                sorted(references.items())
            )
    return inventory


def audit_provider_selection_boundaries(root: Path = ROOT) -> dict[str, typing.Any]:
    """Reject new selector surfaces and require retired migration debt to vanish."""

    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid provider-selection boundary manifest schema")

    entries = manifest["files"]
    inventory = source_inventory(root)
    errors: list[str] = []
    for path in sorted(set(entries) | set(inventory)):
        actual = inventory.get(path, {})
        entry = entries.get(path)
        if entry is None:
            errors.append(
                f"{path}: unclassified provider-selection identifiers: "
                + ", ".join(actual)
            )
            continue
        if not actual:
            errors.append(f"{path}: remove retired provider-selection entry")
            continue

        classification = entry.get("classification")
        if (
            classification not in CLASSIFICATIONS
            or not entry.get("reason")
            or not entry.get("contract")
        ):
            errors.append(f"{path}: classification, contract and reason are required")
        elif (
            not entry["contract"].startswith("#")
            and not (root / entry["contract"]).is_file()
        ):
            errors.append(f"{path}: contract does not exist: {entry['contract']}")

        expected = entry.get("selectors", {})
        if not expected or any(
            name not in SELECTOR_IDENTIFIERS
            or type(count) is not int
            or count < 1
            for name, count in expected.items()
        ):
            errors.append(
                f"{path}: selector counts must name guarded identifiers with positive integers"
            )
            continue

        for name in sorted(set(expected) | set(actual)):
            if name not in expected:
                errors.append(f"{path}: unclassified provider selector {name}")
            elif name not in actual:
                errors.append(f"{path}: remove retired selector {name}")
            elif actual[name] != expected[name]:
                errors.append(
                    f"{path}: {name} count {actual[name]} != classified "
                    f"{expected[name]}; migrate new uses or retire removed debt"
                )

    return {"errors": errors, "inventory": inventory}


def main() -> int:
    """Run the checked inventory; --json includes current selector counts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit_provider_selection_boundaries()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for error in report["errors"]:
            print(error, file=sys.stderr)
        print(
            f"Checked {len(report['inventory'])} provider-selection files; "
            f"{len(report['errors'])} errors"
        )
    return bool(report["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
