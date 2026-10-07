"""Conservative native zero-materialization inventory, without native execution.

Only a closed scalar/loop subset earns a write-support certificate. All other
zero-allocation candidates remain unknown, including aggregate members. This is
an advisory source audit, not a C++ compiler, numerical sparsity test or profiler.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tools.audit_native_complexity import (
        SOURCE_SUFFIXES,
        _mask_comments_and_literals,
        _matching,
        _skip_space,
    )
    from tools.audit_native_work import _calls, _functions, _split
except ModuleNotFoundError:
    from audit_native_complexity import (  # type: ignore[import-not-found]
        SOURCE_SUFFIXES,
        _mask_comments_and_literals,
        _matching,
        _skip_space,
    )
    from audit_native_work import (  # type: ignore[import-not-found]
        _calls,
        _functions,
        _split,
    )

_ID = r"[A-Za-z_]\w*"
_INTEGER = r"(?:std::size_t|size_t|int|unsigned(?:\s+long)?)"
_VECTOR = re.compile(rf"std::vector\s*<\s*(?:double|float)\s*>\s+({_ID})\s*\(")
_ASSIGN = re.compile(rf"\b({_ID}(?:\.{_ID})*)\s*\.assign\s*\(")
_ZERO = re.compile(r"0(?:\.0*)?(?:[fF])?\Z")


class Unsupported(ValueError):
    """A source construct falls outside the closed proof subset."""


def _expr(text: str) -> ast.expr:
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError) as error:
        raise Unsupported("unsupported arithmetic expression") from error
    if any(
        not isinstance(
            part,
            (
                ast.Expression,
                ast.Name,
                ast.Load,
                ast.Constant,
                ast.BinOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Call,
            ),
        )
        for part in ast.walk(node)
    ):
        raise Unsupported("unsupported arithmetic expression")
    return node


def _key(node: ast.expr) -> str:
    return ast.dump(node, include_attributes=False)


def _expand(
    node: ast.expr,
    helpers: dict[str, tuple[list[str], ast.expr]],
    bindings: dict[str, ast.expr] | None = None,
    depth: int = 0,
) -> ast.expr:
    if depth > 32:
        raise Unsupported("helper expansion limit")
    bindings = bindings or {}
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            _expand(node.left, helpers, bindings, depth + 1),
            node.op,
            _expand(node.right, helpers, bindings, depth + 1),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords or node.func.id not in helpers:
            raise Unsupported("unknown arithmetic helper")
        parameters, body = helpers[node.func.id]
        if len(parameters) != len(node.args):
            raise Unsupported("helper arity mismatch")
        values = [_expand(arg, helpers, bindings, depth + 1) for arg in node.args]
        return _expand(body, helpers, dict(zip(parameters, values)), depth + 1)
    raise Unsupported("unknown arithmetic helper or noninteger expression")


def _helpers(
    clean: str, namespace: tuple[str, ...]
) -> dict[str, tuple[list[str], ast.expr]]:
    if re.search(r"\busing\b", clean):
        return {}
    functions = _functions(clean)
    counts = Counter(f.name for f in functions)
    result = {}
    for function in functions:
        if (
            counts[function.name] != 1
            or function.namespace != namespace
            or not re.fullmatch(_INTEGER, function.signature)
        ):
            continue
        if re.search(
            rf"\b(?!return\b|throw\b)[\w:<>]+\s+{re.escape(function.name)}\s*\([^;{{}}]*\)\s*(?:noexcept\s*)?;",
            clean,
        ):
            continue
        parameters = _split(function.parameters)
        parsed = [
            re.fullmatch(rf"(?:const\s+)?{_INTEGER}\s+({_ID})", p) for p in parameters
        ]
        body = re.fullmatch(
            r"\s*return\s+([^;]+);\s*", clean[function.body : function.end]
        )
        if not body or not parsed or not all(parsed):
            continue
        try:
            names = [p[1] for p in parsed if p]
            expression = _expr(body[1])
            callees = {
                id(p.func) for p in ast.walk(expression) if isinstance(p, ast.Call)
            }
            if any(
                isinstance(p, ast.Name) and id(p) not in callees and p.id not in names
                for p in ast.walk(expression)
            ):
                continue
            result[function.name] = (names, expression)
        except Unsupported:
            continue
    return result


def _factors(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _factors(node.left) + _factors(node.right)
    raise Unsupported("allocation is not a symbolic monomial")


def _resolve_scalars(
    node: ast.expr, aliases: dict[str, ast.expr], depth: int = 0
) -> ast.expr:
    """Resolve immutable extent aliases under the mathematical-integer precondition."""
    if depth > 32:
        raise Unsupported("scalar alias expansion limit")
    if isinstance(node, ast.Name) and node.id in aliases:
        return _resolve_scalars(aliases[node.id], aliases, depth + 1)
    if isinstance(node, ast.BinOp):
        left = _resolve_scalars(node.left, aliases, depth + 1)
        right = _resolve_scalars(node.right, aliases, depth + 1)
        if isinstance(left, ast.Constant) and isinstance(right, ast.Constant):
            if isinstance(node.op, ast.Add):
                return ast.Constant(left.value + right.value)
            if isinstance(node.op, ast.Sub):
                return ast.Constant(left.value - right.value)
            return ast.Constant(left.value * right.value)
        if isinstance(node.op, ast.Sub) and _key(left) == _key(right):
            return ast.Constant(0)
        if (
            isinstance(node.op, (ast.Add, ast.Sub))
            and isinstance(right, ast.Constant)
            and right.value == 0
        ):
            return left
        if (
            isinstance(node.op, ast.Add)
            and isinstance(left, ast.Constant)
            and left.value == 0
        ):
            return right
        if isinstance(node.op, ast.Mult):
            if any(
                isinstance(part, ast.Constant) and part.value == 0
                for part in (left, right)
            ):
                return ast.Constant(0)
            if isinstance(left, ast.Constant) and left.value == 1:
                return right
            if isinstance(right, ast.Constant) and right.value == 1:
                return left
        return ast.BinOp(left, node.op, right)
    return node


def _polynomial(node: ast.expr, depth: int = 0) -> dict[tuple[str, ...], int]:
    """Bounded integer polynomial normal form for extent equality and cancellation."""
    if depth > 32:
        raise Unsupported("extent polynomial expansion limit")
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return {(): node.value} if node.value else {}
    if isinstance(node, ast.Name):
        return {(node.id,): 1}
    if isinstance(node, ast.BinOp):
        left, right = (
            _polynomial(node.left, depth + 1),
            _polynomial(node.right, depth + 1),
        )
        result: dict[tuple[str, ...], int] = {}
        if isinstance(node.op, ast.Mult):
            if len(left) * len(right) > 256:
                raise Unsupported("extent polynomial term limit")
            for first, first_value in left.items():
                for second, second_value in right.items():
                    powers = tuple(sorted(first + second))
                    if len(powers) > 32:
                        raise Unsupported("extent polynomial degree limit")
                    result[powers] = result.get(powers, 0) + first_value * second_value
        elif isinstance(node.op, (ast.Add, ast.Sub)):
            result = left.copy()
            sign = 1 if isinstance(node.op, ast.Add) else -1
            for powers, value in right.items():
                result[powers] = result.get(powers, 0) + sign * value
        else:
            raise Unsupported("unknown extent arithmetic")
        result = {powers: value for powers, value in result.items() if value}
        if len(result) > 256:
            raise Unsupported("extent polynomial term limit")
        return result
    raise Unsupported("unknown extent polynomial")


def _growth_degree(node: ast.expr) -> int:
    return max((len(powers) for powers in _polynomial(node)), default=0)


def _axes(node: ast.expr, dimension: str, rank: int) -> list[ast.expr]:
    if rank == 1:
        return [node]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = node.left
        if (
            isinstance(left, ast.BinOp)
            and isinstance(left.op, ast.Mult)
            and isinstance(left.right, ast.Name)
            and left.right.id == dimension
        ):
            return _axes(left.left, dimension, rank - 1) + [node.right]
    raise Unsupported("index is not canonical row-major arithmetic")


def _certificate(
    clean: str,
    function: Any,
    start: int,
    end: int,
    name: str,
    count: str,
    helpers: dict[str, tuple[list[str], ast.expr]],
) -> dict[str, Any]:
    if "." in name:
        raise Unsupported("aggregate member type/default constructor not resolved")
    if re.search(r"(?m)^\s*#\s*(?!include\b|pragma\b)", clean):
        raise Unsupported("preprocessor control or macros require a C++ frontend")
    if re.search(r"\b(static|thread_local)\b", clean[function.body : start]):
        raise Unsupported("nonfresh storage")
    parameter_matches = [
        re.fullmatch(rf"(?:const\s+)?{_INTEGER}\s+({_ID})", p)
        for p in _split(function.parameters)
    ]
    if not parameter_matches or not all(parameter_matches):
        raise Unsupported("only integral scalar parameters are certified")
    scalars = {p[1] for p in parameter_matches if p}
    if name in scalars:
        raise Unsupported("shadowed allocation name")
    aliases: dict[str, ast.expr] = {}
    prefix = clean[function.body : start]
    # Only immutable integral scalar declarations may precede the fresh owner.
    declaration = re.compile(
        rf"\s*const\s+(?:auto|{_INTEGER})\s+({_ID})\s*=\s*([^;]+);"
    )
    position = 0
    while prefix[position:].strip():
        match = declaration.match(prefix, position)
        if not match or match[1] in scalars:
            raise Unsupported("unparsed setup or alias before allocation")
        node = _expand(_expr(match[2]), helpers)
        if any(isinstance(p, ast.Name) and p.id not in scalars for p in ast.walk(node)):
            raise Unsupported("unknown setup dimension")
        aliases[match[1]] = node
        scalars.add(match[1])
        position = match.end()
    factors = _factors(_expand(_expr(count), helpers))
    rank = len(factors)
    if rank not in {2, 3, 4} or len(set(factors)) != 1 or factors[0] not in scalars:
        raise Unsupported("only homogeneous rank-2/3/4 symbolic shapes are certified")
    dimension = factors[0]
    dense_extent = _resolve_scalars(ast.Name(dimension, ast.Load()), aliases)

    def extent(expression: str) -> ast.expr:
        return _resolve_scalars(_expr(expression), aliases)

    def full_loop(loop: tuple[str, str, str]) -> bool:
        lower, operator, upper = loop
        return (
            not _polynomial(extent(lower))
            and operator == "<"
            and _polynomial(extent(upper)) == _polynomial(dense_extent)
        )

    support = []
    writes = []
    range_conditions: set[str] = set()
    returned = False

    def walk(begin: int, limit: int, loops: dict[str, tuple[str, str, str]]) -> None:
        nonlocal returned
        cursor = begin
        while cursor < limit:
            cursor = _skip_space(clean, cursor)
            if cursor >= limit:
                return
            if returned:
                raise Unsupported("code after return")
            if clean[cursor] == "{":
                closing = _matching(clean, cursor, "{", "}")
                if closing < 0 or closing >= limit:
                    raise Unsupported("unmatched block")
                walk(cursor + 1, closing, loops)
                cursor = closing + 1
                continue
            if re.match(r"for\b", clean[cursor:]):
                opening = _skip_space(clean, cursor + 3)
                if clean[opening : opening + 1] != "(":
                    raise Unsupported("unparsed loop")
                closing = _matching(clean, opening, "(", ")")
                header = clean[opening + 1 : closing]
                match = re.fullmatch(
                    rf"\s*{_INTEGER}\s+({_ID})\s*=\s*(0|{_ID})\s*;\s*\1\s*(<=|<)\s*({_ID})\s*;\s*(?:\+\+\1|\1\+\+)\s*",
                    header,
                )
                if not match or match[1] in scalars | loops.keys():
                    raise Unsupported("noncanonical or shadowed loop")
                variable, lower, operator, upper = match.groups()
                if lower != "0" and lower not in scalars:
                    raise Unsupported("unknown loop lower bound")
                if upper not in scalars | loops.keys():
                    raise Unsupported("unknown loop upper bound")
                body = _skip_space(clean, closing + 1)
                # A dedicated bounded statement parser keeps chained unbraced loops.
                body_end = statement_end(body, limit)
                walk(body, body_end, {**loops, variable: (lower, operator, upper)})
                cursor = body_end
                continue
            semicolon = clean.find(";", cursor, limit)
            if semicolon < 0:
                raise Unsupported("unparsed statement")
            statement = clean[cursor : semicolon + 1].strip()
            if statement == f"return {name};" and not loops:
                returned = True
                cursor = semicolon + 1
                continue
            write = re.fullmatch(
                rf"{re.escape(name)}\s*\[([^\[\]]+)\]\s*(?:=|\+=|-=)\s*([^;]+);",
                statement,
            )
            if not write:
                raise Unsupported("unknown call, alias, mutation or control flow")
            # RHS is deliberately restricted to side-effect-free arithmetic in
            # known integral scalars/loop variables or numeric literals.
            rhs = _expr(write[2])
            if any(isinstance(p, ast.Call) for p in ast.walk(rhs)):
                raise Unsupported("unsupported write RHS")
            rhs_names = {p.id for p in ast.walk(rhs) if isinstance(p, ast.Name)}
            if not rhs_names <= scalars | loops.keys():
                raise Unsupported("unknown RHS scalar")
            axes = _axes(_expand(_expr(write[1]), helpers), dimension, rank)
            domain = []
            used = []
            for axis in axes:
                offset = "0"
                coordinate = axis
                if (
                    isinstance(axis, ast.BinOp)
                    and isinstance(axis.op, ast.Add)
                    and isinstance(axis.left, ast.Name)
                ):
                    offset, coordinate = axis.left.id, axis.right
                if not isinstance(coordinate, ast.Name) or coordinate.id not in loops:
                    raise Unsupported("unsupported index axis")
                if not _polynomial(extent(offset)):
                    offset = "0"
                variable = coordinate.id
                lower, operator, upper = loops[variable]
                if offset != "0":
                    if (
                        lower != "0"
                        or operator != "<"
                        or upper not in aliases
                        or _polynomial(extent(upper))
                        != _polynomial(extent(f"{dimension} - {offset}"))
                    ):
                        raise Unsupported("unproved occupied/virtual offset partition")
                elif lower != "0" and (
                    lower not in scalars or upper != dimension or operator != "<"
                ):
                    raise Unsupported("unproved axis bounds")
                if _polynomial(extent(lower)) == _polynomial(extent(upper)):
                    raise Unsupported("degenerate empty loop domain")
                if offset != "0":
                    range_conditions.add(f"0 <= {offset} <= {dimension}")
                elif lower != "0":
                    range_conditions.add(f"0 <= {lower} <= {dimension}")
                elif upper in scalars and upper != dimension:
                    range_conditions.add(f"0 <= {upper} <= {dimension}")
                domain.append(
                    {
                        "variable": variable,
                        "offset": offset,
                        "lower": lower,
                        "operator": operator,
                        "upper": upper,
                    }
                )
                if variable not in used:
                    used.append(variable)
            if set(used) != loops.keys():
                raise Unsupported("unused loop dimension/repeated reductions")
            triangle = [v for v in used if loops[v][2] in loops]
            if triangle:
                if (
                    rank != 2
                    or len(used) != 2
                    or len(triangle) != 1
                    or domain[0]["variable"] != used[0]
                    or not full_loop(loops[used[0]])
                    or loops[used[1]] != ("0", "<=", used[0])
                    or any(d["offset"] != "0" for d in domain)
                ):
                    raise Unsupported("unsupported dependent domain")
                term = f"{dimension}*({dimension}+1)/2"
                kind, degree = "lower-triangle", 2 * _growth_degree(dense_extent)
            else:
                if any(loops[v][1] != "<" or loops[v][2] not in scalars for v in used):
                    raise Unsupported("unsupported dependent domain")
                term = "*".join(
                    loops[v][2]
                    if loops[v][0] == "0"
                    else f"({loops[v][2]}-{loops[v][0]})"
                    for v in used
                )
                degree = sum(
                    _growth_degree(extent(f"{loops[v][2]} - {loops[v][0]}"))
                    for v in used
                )
                full = (
                    len(used) == rank
                    and all(full_loop(loops[v]) for v in used)
                    and all(d["offset"] == "0" for d in domain)
                )
                kind = "dense" if full else "cartesian-or-diagonal"
            support.append(
                {
                    "axes": domain,
                    "elements": term,
                    "growth_degree": degree,
                    "kind": kind,
                }
            )
            writes.append(
                {"line": clean.count("\n", 0, cursor) + 1, "index": write[1].strip()}
            )
            cursor = semicolon + 1

    def statement_end(begin: int, limit: int) -> int:
        if clean[begin : begin + 1] == "{":
            closing = _matching(clean, begin, "{", "}")
            if closing < 0 or closing >= limit:
                raise Unsupported("unmatched loop body")
            return closing + 1
        if re.match(r"for\b", clean[begin:]):
            opening = clean.find("(", begin, limit)
            closing = _matching(clean, opening, "(", ")")
            return statement_end(_skip_space(clean, closing + 1), limit)
        semicolon = clean.find(";", begin, limit)
        if semicolon < 0:
            raise Unsupported("unparsed loop body")
        return semicolon + 1

    walk(end, function.end, {})
    if not support:
        raise Unsupported("no admitted writes")
    unique = {json.dumps(d, sort_keys=True): d for d in support}
    domains = list(unique.values())
    dense = any(d["kind"] == "dense" for d in domains)
    bound = " + ".join(d["elements"] for d in domains)
    single = len(domains) == 1
    return {
        "classification": "dense-write-domain"
        if dense
        else ("exact-structured-write-support" if single else "write-domain-union"),
        "dense_elements": "*".join(factors),
        "dense_growth_degree": rank * _growth_degree(dense_extent),
        "extent_aliases": {
            key: ast.unparse(_resolve_scalars(value, aliases))
            for key, value in aliases.items()
        },
        "written_elements": bound,
        "written_count_kind": "exact-address-domain" if single else "union-upper-bound",
        "written_growth_degree": max(d["growth_degree"] for d in domains),
        "domains": domains,
        "writes": writes,
        "expansion_ratio": f"({'*'.join(factors)})/({bound})"
        if single and not dense
        else None,
        "strict_reduction_proved": False,
        "strict_reduction_conditions": [f"{dimension} > 1"]
        if single and domains[0]["kind"] == "lower-triangle"
        else [],
        "recommendation": None
        if dense or not single
        else "Review block/diagonal/packed storage and all consumers before changing the dense ABI.",
        "conditions": sorted(range_conditions)
        + [
            "Integral dimensions and extents are nonnegative; scalar conversions preserve mathematical values and index/allocation arithmetic does not overflow.",
            "This certificate ends at the producer return; later mutation and numerical nonzeros are not inferred.",
        ],
    }


def audit_native(source: str, path: str = "<memory>") -> list[dict[str, Any]]:
    """Inventory two-argument zero vector constructors/assigns, certifying a subset."""
    clean = _mask_comments_and_literals(source)
    parsed = re.sub(r"(?m)^[ \t]*#.*", lambda m: " " * len(m[0]), clean)
    findings = []
    functions = _functions(parsed)
    helpers_by_namespace = {
        namespace: _helpers(parsed, namespace)
        for namespace in {f.namespace for f in functions}
    }
    callers: dict[str, list[dict[str, Any]]] = {}
    for caller in functions:
        for call in _calls(parsed, caller):
            callers.setdefault(call.name, []).append(
                {"function": caller.name, "line": clean.count("\n", 0, call.start) + 1}
            )
    for function in functions:
        helpers = helpers_by_namespace[function.namespace]
        for pattern in (_VECTOR, _ASSIGN):
            for match in pattern.finditer(clean, function.body, function.end):
                opening = clean.index("(", match.start())
                closing = _matching(clean, opening, "(", ")")
                args = _split(clean[opening + 1 : closing])
                if closing < 0 or len(args) != 2 or not _ZERO.fullmatch(args[1]):
                    continue
                end = _skip_space(clean, closing + 1)
                if clean[end : end + 1] != ";":
                    continue
                name = match[1]
                finding = {
                    "path": path,
                    "line": clean.count("\n", 0, match.start()) + 1,
                    "function": function.name,
                    "buffer": name,
                    "allocation_expression": args[0],
                    "classification": "unknown",
                    "recommendation": None,
                    "barriers": {
                        "producer": {"line": clean.count("\n", 0, match.start()) + 1},
                        "consumer": {
                            "status": "not-analyzed",
                            "reason": "Return/escape does not establish downstream layout requirements.",
                            "same_file_call_candidates": callers.get(function.name, []),
                        },
                        "abi_layout": {
                            "status": "dense-vector-candidate",
                            "evidence": clean[match.start() : end + 1].strip(),
                        },
                        "structured_ir": {
                            "status": "not-analyzed",
                            "reason": "No TensorIR integration in this standalone slice.",
                        },
                    },
                }
                try:
                    owner_start = match.start()
                    if pattern is _ASSIGN:
                        fresh = re.search(
                            rf"std::vector\s*<\s*(?:double|float)\s*>\s+{re.escape(name)}\s*;\s*\Z",
                            clean[function.body : match.start()],
                        )
                        if not fresh or "." in name:
                            raise Unsupported(
                                "assign freshness/type or aggregate ABI not resolved"
                            )
                        owner_start = function.body + fresh.start()
                    finding.update(
                        _certificate(
                            clean,
                            function,
                            owner_start,
                            end + 1,
                            name,
                            args[0],
                            helpers,
                        )
                    )
                except (Unsupported, RecursionError) as error:
                    finding["unknown_reason"] = str(error)
                findings.append(finding)
    return sorted(findings, key=lambda f: (f["line"], f["buffer"]))


def audit_tree(
    root: Path, paths: tuple[str, ...] = ("src", "include")
) -> dict[str, Any]:
    """Retain byte identities for every source and every consumed scanner module."""
    root = root.resolve()
    sources: dict[str, str] = {}
    findings = []
    candidates: set[Path] = set()
    resolved_selections: set[str] = set()
    alias_topology_unverified = False
    for path in paths:
        selected = root / path
        target = selected.resolve()
        alias_topology_unverified |= selected != target
        resolved_selections.add(target.relative_to(root).as_posix())
        if not target.exists():
            raise ValueError(f"missing input path: {path}")
        for candidate in [target] if target.is_file() else target.rglob("*"):
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise ValueError(f"unresolved input path: {candidate}") from error
            resolved.relative_to(root)
            alias_topology_unverified |= candidate != resolved
            candidates.add(resolved)
    for source in sorted(candidates):
        relative = source.relative_to(root).as_posix()
        if (
            not source.is_file()
            or source.suffix not in SOURCE_SUFFIXES
            or relative.startswith("src/xtb/native/")
        ):
            continue
        data = source.read_bytes()
        sources[relative] = hashlib.sha256(data).hexdigest()
        findings.extend(audit_native(data.decode("utf-8", errors="replace"), relative))

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "--literal-pathspecs", "-C", str(root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    scanners = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in (
            "audit_structured_materialization.py",
            "audit_native_complexity.py",
            "audit_native_work.py",
        )
    }
    dirty = git("status", "--porcelain")
    # Endpoint status cannot prove the intermediate symlink topology unchanged.
    # Known changes still take precedence over that uncertainty or a failed query.
    source_dirty: bool | None = (
        None if dirty is None or alias_topology_unverified else False
    )
    status_checks = (
        (sorted(set(paths) | resolved_selections | sources.keys()), ()),
        (sorted(sources), ("--ignored",)),
    )
    for status_paths, extra_options in status_checks:
        for begin in range(0, len(status_paths), 32):
            status = git(
                "status",
                "--porcelain",
                "--untracked-files=all",
                *extra_options,
                "--",
                *status_paths[begin : begin + 32],
            )
            if status:
                source_dirty = True
            elif status is None and source_dirty is not True:
                source_dirty = None
    return {
        "schema": "generativeqc.native-structured-materialization.v1",
        "advisory_only": True,
        "provenance": {
            "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "working_tree_dirty": None if dirty is None else bool(dirty),
            "scanned_source_dirty": source_dirty,
            "alias_topology_unverified": alias_topology_unverified,
            "source_hashes": sources,
            "scanner_hashes": scanners,
            "scanned_source_digest": hashlib.sha256(
                json.dumps(sources, sort_keys=True).encode()
            ).hexdigest(),
            "scanner_digest": hashlib.sha256(
                json.dumps(scanners, sort_keys=True).encode()
            ).hexdigest(),
            "source_roots": list(paths),
        },
        "scanned_files": len(sources),
        "counts": dict(Counter(f["classification"] for f in findings)),
        "limitations": [
            "Closed lexical subset, not a C++ frontend or whole-program proof.",
            "Unknown candidates carry no representation recommendation; absence is not proof.",
            "Address domains describe possible writes, not measured or guaranteed numerical nonzeros.",
            "No runtime bytes, timings, speedup, production role or dense-oracle size gate is inferred.",
        ],
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--path", action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_tree(
        args.root, tuple(args.path) if args.path else ("src", "include")
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
