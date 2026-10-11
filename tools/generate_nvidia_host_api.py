#!/usr/bin/env python3
"""Regenerate provider-free CUDA host API declarations from NVIDIA SDK headers.

Only checked-in declarations are consumed by manylinux wheels: CUDA provider
development packages and Clang are never runtime or wheel-build dependencies.
SDK refreshes fail closed on missing symbols or unsupported C declarators.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "src/runtime/nvidia_host_api/generativeqc_nvidia_host_functions.h"
API = re.compile(r"\b(cublas(?!Lt)[A-Z]\w*|cusolver(?:Dn|Get)[A-Z]\w*)\b")
# Generated declarations must be live source, not examples or disabled comments.
NON_CODE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',
    re.DOTALL,
)
DECL = re.compile(
    r"\b(cublasStatus_t|cusolverStatus_t)\s+(cublas\w+|cusolver\w+)\s*\((.*?)\);",
    re.DOTALL,
)
ALIAS = re.compile(
    r"^#define\s+(cublas\w+|cusolver\w+)\s+(cublas\w+|cusolver\w+)\s*$",
    re.MULTILINE,
)
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}


def called_apis(source: str) -> set[str]:
    """Find every call candidate with bounded, memoized comment traversal."""
    # Candidates remain independent: an API-looking name inside comment text
    # must not consume a later real call. Index delimiter ends so overlapping
    # candidate suffixes cannot repeatedly search an unterminated comment.
    newlines = [match.start() for match in re.finditer("\n", source)]
    block_ends = [match.start() for match in re.finditer(r"\*/", source)]
    suffix_ends: dict[int, int] = {}

    def skip_trivia(position: int) -> int:
        visited: list[int] = []
        while position not in suffix_ends:
            visited.append(position)
            if position >= len(source):
                break
            if source[position].isspace():
                position += 1
            elif source.startswith("//", position):
                index = bisect_left(newlines, position + 2)
                position = newlines[index] + 1 if index < len(newlines) else len(source)
            elif source.startswith("/*", position):
                index = bisect_left(block_ends, position + 2)
                position = (
                    block_ends[index] + 2 if index < len(block_ends) else len(source)
                )
            else:
                break
        end = suffix_ends.get(position, position)
        for start in visited:
            suffix_ends[start] = end
        return end

    return {
        match.group(1)
        for match in API.finditer(source)
        if source.startswith("(", skip_trivia(match.end()))
    }


def project_uses(root: Path) -> set[str]:
    """Discover called APIs, excluding separate xTB and SDK-header owners."""
    used: set[str] = set()
    for folder in (
        root / "src",
        root / "include",
        root / "python/generativeqc_compiler",
    ):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix not in SOURCE_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            if rel.startswith(("src/xtb/", "src/runtime/nvidia_host_api/")):
                continue
            used.update(called_apis(path.read_text(encoding="utf-8", errors="replace")))
    return used


def declaration_source(source: str) -> str:
    """Mask comments and literals without joining previously separated tokens."""
    source = NON_CODE.sub(lambda match: re.sub(r"[^\n]", " ", match.group()), source)
    if re.search(
        r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif|undef)\b", source, re.MULTILINE
    ):
        raise ValueError("generated NVIDIA declarations must be unconditional")
    return source


def declarations(source: str) -> set[str]:
    return {match.group(2) for match in DECL.finditer(declaration_source(source))}


def aliases(source: str) -> dict[str, str]:
    return dict(ALIAS.findall(declaration_source(source)))


def check_usage(root: Path, target: Path) -> None:
    """Cheap SDK-free check; run before spending time compiling CUDA wheels."""
    source = target.read_text(encoding="utf-8")
    declared = declarations(source)
    declared.update(
        name for name, actual in aliases(source).items() if actual in declared
    )
    missing = sorted(project_uses(root) - declared)
    if missing:
        raise ValueError(
            "NVIDIA wheel header lacks called APIs: "
            + ", ".join(missing)
            + "; run tools/generate_nvidia_host_api.py --sdk-include PATH --write"
        )


def run_clang(clang: str, includes: list[Path], header: str, *, ast: bool) -> str:
    flags = [f"-I{path}" for path in includes]
    if ast:
        prefix = "cublas" if header == "cublas_v2.h" else "cusolver"
        command = [
            clang,
            "-x",
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            *flags,
            "-Xclang",
            "-ast-dump=json",
            "-Xclang",
            f"-ast-dump-filter={prefix}",
        ]
    else:
        command = [clang, "-x", "c++", "-std=c++17", "-E", "-dM", *flags]
    result = subprocess.run(
        [*command, "-"],
        input=f"#include <{header}>\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if result.returncode:
        raise ValueError(
            f"Clang could not parse NVIDIA {header}; install matching SDK headers:\n"
            + result.stderr[-6000:]
        )
    return result.stdout


def ast_objects(output: str) -> list[dict[str, Any]]:
    """Clang's ast-dump-filter emits separate adjacent JSON objects."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    position = 0
    while position < len(output):
        while position < len(output) and output[position].isspace():
            position += 1
        if position == len(output):
            break
        parsed, position = decoder.raw_decode(output, position)
        if not isinstance(parsed, dict):
            raise ValueError("expected Clang declaration object")  # noqa: TRY004 - malformed SDK output
        objects.append(parsed)
    return objects


def sdk_functions(
    clang: str, includes: list[Path], header: str
) -> dict[str, tuple[str, ...]]:
    prefix = "cublas" if header == "cublas_v2.h" else "cusolver"
    found: dict[str, tuple[str, ...]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for entry in node:
                visit(entry)
            return
        if not isinstance(node, dict):
            return
        name = node.get("name", "")
        if node.get("kind") == "FunctionDecl" and name.startswith(prefix):
            signature = node.get("type", {}).get("qualType", "")
            result = signature.split(" (", 1)[0]
            if result not in ("cublasStatus_t", "cusolverStatus_t"):
                return
            if node.get("variadic") or "..." in signature:
                raise ValueError(f"unsupported variadic NVIDIA prototype: {name}")
            if "__attribute__" in signature:
                raise ValueError(f"unsupported NVIDIA calling convention: {name}")
            arguments = tuple(
                child["type"]["qualType"]
                for child in node.get("inner", [])
                if child.get("kind") == "ParmVarDecl"
            )
            if any("(" in arg or "[" in arg for arg in arguments):
                raise ValueError(
                    f"unsupported function-pointer/array parameter: {name}"
                )
            previous = found.setdefault(name, (result, *arguments))
            if previous != (result, *arguments):
                raise ValueError(f"conflicting NVIDIA SDK prototypes: {name}")
        visit(node.get("inner", []))

    for obj in ast_objects(run_clang(clang, includes, header, ast=True)):
        visit(obj)
    if not found:
        raise ValueError(f"NVIDIA SDK yielded no functions from {header}")
    return found


def canonical_type(source: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*([*&])\s*", r"\1", source.strip()))


def render(decls: dict[str, tuple[str, ...]], maps: dict[str, str]) -> str:
    output = [
        "#pragma once",
        "",
        "// Generated by tools/generate_nvidia_host_api.py from NVIDIA CUDA headers.",
        "// Do not hand-edit function signatures; regenerate using the pinned SDK.",
        "// CUDA provider libraries remain lazy runtime dependencies, not wheel link inputs.",
        "",
        "// clang-format off",
    ]
    for name in sorted(decls):
        signature = decls[name]
        params = ", ".join(
            f"{canonical_type(kind)} arg{i}" for i, kind in enumerate(signature[1:])
        )
        output.append(f"{signature[0]} {name}({params or 'void'});")
    output.append("")
    for name, target in sorted(maps.items()):
        output.append(f"#define {name} {target}")
    output.extend(["// clang-format on", ""])
    return "\n".join(output)


def generate(root: Path, target: Path, clang: str, includes: list[Path]) -> str:
    current = target.read_text(encoding="utf-8")
    needed = project_uses(root) | declarations(current) | set(aliases(current))
    sdk_aliases: dict[str, str] = {}
    functions: dict[str, tuple[str, ...]] = {}
    for header in ("cublas_v2.h", "cusolverDn.h"):
        functions.update(sdk_functions(clang, includes, header))
        sdk_aliases.update(aliases(run_clang(clang, includes, header, ast=False)))
    wanted: dict[str, tuple[str, ...]] = {}
    chosen_aliases: dict[str, str] = {}
    for name in sorted(needed):
        resolved = sdk_aliases.get(name, name)
        if resolved not in functions:
            raise ValueError(f"{name}: missing official CUDA declaration ({resolved})")
        wanted[resolved] = functions[resolved]
        if resolved != name:
            chosen_aliases[name] = resolved
    return render(wanted, chosen_aliases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-include",
        action="append",
        type=Path,
        default=[],
        help="NVIDIA toolkit, cuBLAS or cuSOLVER include path (repeatable)",
    )
    parser.add_argument("--clang", default="clang++")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-usage", action="store_true", help="fast SDK-free CI check"
    )
    mode.add_argument("--write", action="store_true", help="regenerate from NVIDIA SDK")
    mode.add_argument(
        "--check-sdk", action="store_true", help="verify checked-in SDK snapshot"
    )
    args = parser.parse_args()
    try:
        if args.check_usage:
            check_usage(ROOT, GENERATED)
        else:
            if not args.sdk_include:
                raise ValueError(
                    "SDK refresh requires --sdk-include (repeat all header paths)"
                )
            generated = generate(ROOT, GENERATED, args.clang, args.sdk_include)
            if args.write:
                GENERATED.write_text(generated, encoding="utf-8")
            elif generated != GENERATED.read_text(encoding="utf-8"):
                raise ValueError("generated NVIDIA ABI is stale; run with --write")
            check_usage(ROOT, GENERATED)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"NVIDIA host ABI: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
