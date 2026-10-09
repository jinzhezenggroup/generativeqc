"""Advisory work audit: allocation, synchronization and repeated producers.

This is a bounded, lexical C++ analysis, not a C++ compiler or a profiler. Only
unambiguous same-file free-function edges are followed (at most four calls).
Unknown dispatch, capacity, aliasing and purity remain unknown. Python exact
materialization support uses the standard-library AST in work_audit_python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.audit_native_complexity import (
        LoopSpan,
        _loop_spans,
        _matching,
        _skip_space,
    )
except ModuleNotFoundError:
    from audit_native_complexity import (  # type: ignore[import-not-found]
        LoopSpan,
        _loop_spans,
        _matching,
        _skip_space,
    )

_IDENT = r"[A-Za-z_]\w*"
_CALL = re.compile(rf"\b({_IDENT}(?:::{_IDENT})*)\s*\(")
_SCALAR = r"(?:double|float|int|unsigned|long|bool|std::size_t|size_t)"
_ALLOCATORS = frozenset(
    [
        "malloc",
        "calloc",
        "aligned_alloc",
        "cudaMalloc",
        "cudaMallocManaged",
        "cudaMallocAsync",
        "cudaHostAlloc",
        "cudaMallocHost",
    ]
)
_HOST_REALLOCATORS = frozenset(["realloc"])
_DEVICE_RELEASES = frozenset(["cudaFree", "cudaFreeAsync"])
_HOST_RELEASES = frozenset(["free", "cudaFreeHost"])
_SYNCS = frozenset(
    ["cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"]
)
_TRANSFERS = frozenset(
    ["cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpy2D", "cudaMemcpy2DAsync"]
)
_CONTROL = frozenset(
    [
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "sizeof",
        "alignof",
        "decltype",
        "static_assert",
        "return",
        "throw",
    ]
)


@dataclass(frozen=True)
class Function:
    name: str
    start: int
    body: int
    end: int
    parameters: str
    signature: str
    namespace: tuple[str, ...] = ()


@dataclass(frozen=True)
class Call:
    name: str
    start: int
    end: int
    arguments: str


@dataclass(frozen=True)
class Site:
    kind: str
    start: int
    expression: str
    count: str | None
    owner_start: int | None = None


def _compact(text: str) -> str:
    return " ".join(text.split())


def _split(text: str) -> list[str]:
    """Split ordinary argument lists, retaining nested calls/initializers."""
    output: list[str] = []
    start = depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            output.append(text[start:index].strip())
            start = index + 1
    output.append(text[start:].strip())
    return output


def _functions(clean: str) -> list[Function]:
    output: list[Function] = []
    scopes = []
    for scope in re.finditer(
        r"\b(namespace|struct|class)\s*([A-Za-z_][\w:]*)?[^;{}]*\{", clean
    ):
        opening = clean.index("{", scope.start())
        closing = _matching(clean, opening, "{", "}")
        if closing >= 0:
            scopes.append((scope.group(1), scope.group(2), opening, closing))
    for match in _CALL.finditer(clean):
        if match.group(1) in _CONTROL:
            continue
        containing = [scope for scope in scopes if scope[2] < match.start() < scope[3]]
        if any(scope[0] in {"class", "struct"} for scope in containing):
            continue
        namespace = tuple(
            scope[1] for scope in containing if scope[0] == "namespace" and scope[1]
        )
        opening = clean.index("(", match.start())
        closing = _matching(clean, opening, "(", ")")
        if closing < 0:
            continue
        body = _skip_space(clean, closing + 1)
        # Read-only qualification is admitted, constructors/initializer lists,
        # trailing returns and function-try blocks are intentionally unsupported.
        qualifier = re.match(r"(?:const\s*)?(?:noexcept\s*)?", clean[body:])
        assert qualifier is not None
        body += qualifier.end()
        if body >= len(clean) or clean[body] != "{":
            continue
        start = max(clean.rfind(mark, 0, match.start()) for mark in ";{}") + 1
        prefix = clean[start : match.start()].strip()
        if not prefix or re.search(r"[=().\[\]]|\b(return|throw|else)\b", prefix):
            continue
        if not re.fullmatch(r"[\w\s:<>,*&~#]+", prefix):
            continue
        end = _matching(clean, body, "{", "}")
        if end < 0:
            continue
        output.append(
            Function(
                match.group(1),
                match.start(),
                body + 1,
                end,
                clean[opening + 1 : closing],
                prefix,
                namespace,
            )
        )
    # A declaration parsed inside another body (e.g. local lambda syntax) must
    # never become a separately resolved free-function call.
    return [f for f in output if not any(g.body <= f.start < g.end for g in output)]


def _calls(clean: str, function: Function) -> list[Call]:
    result = []
    for match in _CALL.finditer(clean, function.body, function.end):
        if match.group(1) in _CONTROL:
            continue
        previous = clean[: match.start()].rstrip()[-1:]
        if previous in {".", ">"}:
            continue
        opening = clean.index("(", match.start())
        closing = _matching(clean, opening, "(", ")")
        if closing < 0 or closing > function.end:
            continue
        if _shadowed(match.group(1), function, clean, match.start()):
            continue
        result.append(
            Call(
                match.group(1), match.start(), closing + 1, clean[opening + 1 : closing]
            )
        )
    return result


def _count_expression(
    expression: str,
    function: Function,
    clean: str,
    position: int,
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Admit scalar syntax only; an allocator/iterator/move is never a size.

    Explicit scalar declarations establish scalar type. Auto is followed only
    through an admitted arithmetic initializer; arbitrary calls remain unknown.
    """
    expression = expression.strip()
    expression = re.sub(rf"static_cast\s*<\s*{_SCALAR}\s*>\s*\(", "(", expression)
    if re.search(r"[.:[\]{}&]|->|[A-Za-z_]\w*\s*\(", expression):
        return False
    if not re.fullmatch(r"[\w\s()+*/%<>|^-]+", expression):
        return False
    names = set(re.findall(_IDENT, re.sub(r"\b\d+(?:[uUlL]+)?\b", "0", expression)))
    before = function.parameters + ";" + clean[function.body : position]
    for name in names:
        if name in seen:
            return False
        # Unknown or non-scalar local shadows invalidate the binding.
        declarations = list(
            re.finditer(
                rf"\b([A-Za-z_][\w:]*(?:<[^;{{}}]+>)?)\s+[&*]?\s*{re.escape(name)}\s*(?=[=;,){{])",
                before,
            )
        )
        if any(
            match.group(1) not in {"auto", "const"}
            and re.fullmatch(_SCALAR, match.group(1)) is None
            for match in declarations
        ):
            return False
        if re.search(rf"\b{_SCALAR}\s+{re.escape(name)}\b", before):
            continue
        alias = re.search(rf"\bauto\s+{re.escape(name)}\s*=\s*([^;]+);", before)
        if alias is None or not _count_expression(
            alias.group(1),
            function,
            clean,
            max(
                function.body,
                alias.start() - len(function.parameters) - 1 + function.body,
            ),
            seen | {name},
        ):
            return False
    return True


def _vectors(clean: str, function: Function) -> dict[str, tuple[int, str | None]]:
    result: dict[str, tuple[int, str | None]] = {}
    for match in re.finditer(r"\bstd::vector\s*<", clean[function.body : function.end]):
        start = function.body + match.start()
        opening = clean.index("<", start)
        closing = _matching(clean, opening, "<", ">")
        if closing < 0:
            continue
        prefix = clean[
            max(function.body, clean.rfind(";", function.body, start) + 1) : start
        ]
        if re.search(r"\b(static|thread_local)\b", prefix):
            continue
        cursor = closing + 1
        while cursor < function.end:
            tail = re.match(rf"\s*({_IDENT})\s*([;,(])", clean[cursor:])
            if tail is None:
                break
            name, delimiter = tail.groups()
            position = cursor + tail.start(1)
            count = None
            admitted = True
            if delimiter == "(":
                args_start = cursor + tail.end() - 1
                args_end = _matching(clean, args_start, "(", ")")
                if args_end < 0:
                    break
                arguments = _split(clean[args_start + 1 : args_end])
                count = arguments[0]
                if any(
                    re.search(r"\.(?:c?begin|c?end)\s*\(|\bstd::move\b", argument)
                    for argument in arguments
                ):
                    admitted = False
                if re.fullmatch(_IDENT, count):
                    admitted = admitted and bool(
                        re.search(
                            rf"\b(?:{_SCALAR}|auto)\s+{re.escape(count)}\b",
                            function.parameters + "\n" + clean[function.body : start],
                        )
                    )
                cursor = _skip_space(clean, args_end + 1)
                delimiter = clean[cursor : cursor + 1]
            else:
                cursor += tail.end() - 1
            if count and re.fullmatch(_IDENT, count):
                alias = re.search(
                    rf"\bauto\s+{re.escape(count)}\s*=\s*([^;]+)",
                    clean[function.body : start],
                )
                if alias and re.search(
                    r"\.(?:c?begin|c?end)\s*\(|\bstd::move\b", alias.group(1)
                ):
                    admitted = False
            if count:
                primitive_fill = len(arguments) == 2 and bool(
                    re.fullmatch(r"[-+]?\d+(?:\.\d*)?[fFlLuU]*", arguments[1])
                )
                admitted = admitted and (
                    primitive_fill
                    or _count_expression(count, function, clean, position)
                )
            if admitted:
                result[name] = (position, count)
            if delimiter != ",":
                break
            cursor += 1
    return result


def _fresh_vector_members(clean: str, function: Function) -> dict[str, int]:
    """Resolve vector members only for a visibly fresh local aggregate."""
    structures: dict[str, set[str]] = {}
    for match in re.finditer(rf"\bstruct\s+({_IDENT})\s*{{", clean):
        if (
            len(
                re.findall(
                    rf"\b(?:struct|class)\s+{re.escape(match.group(1))}\b", clean
                )
            )
            != 1
        ):
            continue
        opening = clean.index("{", match.start())
        closing = _matching(clean, opening, "{", "}")
        if closing < 0:
            continue
        fields = set(
            re.findall(
                rf"(?<!static )std::vector\s*<[^;]+>\s+({_IDENT})\s*;",
                clean[opening:closing],
            )
        )
        if re.search(r"\bstatic\s+std::vector\s*<", clean[opening:closing]):
            continue
        if fields and not re.search(
            rf"\b{re.escape(match.group(1))}\s*\(", clean[opening:closing]
        ):
            structures[match.group(1)] = fields
    result = {}
    for typename, fields in structures.items():
        for match in re.finditer(
            rf"(?<!\w){re.escape(typename)}\s+({_IDENT})\s*;",
            clean[function.body : function.end],
        ):
            start = function.body + match.start()
            if re.search(
                r"\b(static|thread_local)\s*$",
                clean[max(function.body, start - 24) : start],
            ):
                continue
            result.update({f"{match.group(1)}.{field}": start for field in fields})
    return result


def _sites(clean: str, function: Function, calls: list[Call]) -> list[Site]:
    result = []
    vectors = _vectors(clean, function)
    for name, (position, count) in vectors.items():
        if count and not re.fullmatch(r"0[uUlL]*", count) and "std::move" not in count:
            result.append(
                Site(
                    "host-allocation",
                    position,
                    f"std::vector {name}({count})",
                    count,
                    position,
                )
            )
    fresh = {name: position for name, (position, _) in vectors.items()}
    fresh.update(_fresh_vector_members(clean, function))
    for match in re.finditer(
        rf"\b({_IDENT}(?:\.{_IDENT})?)\s*\.\s*(assign|resize|reserve)\s*\(",
        clean[function.body : function.end],
    ):
        target, _method = match.groups()
        position = function.body + match.start()
        if target not in fresh or fresh[target] >= position:
            continue
        # Only the first capacity-affecting operation on fresh empty storage.
        between = clean[fresh[target] : position]
        if re.search(
            rf"\b{re.escape(target)}\s*\.\s*(assign|resize|reserve|push_back|emplace_back|swap)\s*\(",
            between,
        ):
            continue
        root = target.split(".")[0]
        residual = between[between.find(";") + 1 :]
        if "." in target:
            member = target.split(".")[1]
            residual = re.sub(
                rf"\b{re.escape(root)}\.(?!{re.escape(member)}\b)[A-Za-z_]\w*",
                "other_field",
                residual,
            )
        if re.search(rf"\b{re.escape(root)}\b", residual):
            continue
        escaped = any(
            re.search(rf"\b{re.escape(root)}\b", call.arguments)
            for call in calls
            if fresh[target] < call.start < position
        )
        assigned = re.search(
            rf"\b{re.escape(target)}\s*=(?!=)|\b{re.escape(root)}\s*=(?!=)|(?:&|\*)\s*[A-Za-z_]\w*\s*=\s*{re.escape(root)}\b",
            between,
        )
        # Member calls are not graph edges, but may mutate a vector argument.
        if any(
            re.search(rf"\([^;]*\b{re.escape(root)}\b[^;]*\)", statement)
            for statement in between.split(";")[1:]
        ):
            escaped = True
        if escaped or assigned:
            continue
        if target in vectors and vectors[target][1] not in {None, "", "0"}:
            continue
        opening = clean.index("(", position)
        closing = _matching(clean, opening, "(", ")")
        if closing < 0:
            continue
        arguments = _split(clean[opening + 1 : closing])
        count = arguments[0]
        if not _count_expression(count, function, clean, position):
            continue
        if any(
            re.search(r"\.(?:c?begin|c?end)\s*\(", argument) for argument in arguments
        ):
            continue
        if re.fullmatch(_IDENT, count) and not re.search(
            rf"\b(?:{_SCALAR}|auto)\s+{re.escape(count)}\b",
            function.parameters + "\n" + clean[function.body : position],
        ):
            continue
        if count and not re.fullmatch(r"0[uUlL]*", count) and "std::move" not in count:
            result.append(
                Site(
                    "host-allocation",
                    position,
                    _compact(clean[position : closing + 1]),
                    count,
                    fresh[target],
                )
            )
    for call in calls:
        name = call.name.removeprefix("std::")
        kind = None
        if name in _ALLOCATORS:
            # cudaMallocHost allocates page-locked *host* memory; the broad
            # cudaMalloc* prefix would misclassify this as a device buffer.
            kind = (
                "device-allocation"
                if name in {"cudaMalloc", "cudaMallocManaged", "cudaMallocAsync"}
                else "host-allocation"
            )
        elif name in _HOST_REALLOCATORS:
            kind = "host-reallocation"
        elif name in _DEVICE_RELEASES:
            kind = "device-release"
        elif name in _HOST_RELEASES:
            kind = "host-release"
        elif name in _SYNCS:
            kind = "synchronization"
        elif name in _TRANSFERS:
            kind = "transfer"
        if kind:
            result.append(
                Site(kind, call.start, _compact(clean[call.start : call.end]), None)
            )
    return sorted(result, key=lambda site: site.start)


def _context(loops: list[LoopSpan], position: int, clean: str) -> list[dict[str, Any]]:
    return [
        {
            "line": clean.count("\n", 0, loop.start) + 1,
            "variable": loop.variable,
            "header": _compact(clean[loop.start : loop.body_start]),
            "constant_extent": loop.constant_extent,
        }
        for loop in loops
        if loop.body_start <= position < loop.body_end
    ]


def _finding(
    rule: str,
    path: str,
    clean: str,
    function: Function,
    position: int,
    evidence: list[str],
    **details: Any,
) -> dict[str, Any]:
    return {
        "rule_id": rule,
        "path": path,
        "line": clean.count("\n", 0, position) + 1,
        "function": function.name,
        "column": position - clean.rfind("\n", 0, position),
        "evidence": evidence,
        "confidence": "structural-site",
        "disposition": "needs-role-and-runtime-review",
        "action": "Confirm endpoint reachability and resource ownership before changing this site.",
        "details": details,
    }


def _repeated_loop_blocks(
    clean: str, path: str, function: Function, loops: list[LoopSpan]
) -> list[dict[str, Any]]:
    """Identical source producer blocks, explicitly without an alias proof.

    A repeat is a review candidate, never a reusable-producer or ratio proof.
    Only closed arithmetic loop blocks with one buffer target are considered.
    """
    groups: dict[str, list[LoopSpan]] = defaultdict(list)
    for loop in loops:
        if not function.body <= loop.start < loop.end <= function.end:
            continue
        block = clean[loop.start : loop.end]
        if not any(
            child.start > loop.start and child.end <= loop.end for child in loops
        ):
            continue
        calls = {match.group(1) for match in _CALL.finditer(block)}
        if calls - {"for"} or re.search(
            r"\b(if|while|break|continue|return|goto)\b", block
        ):
            continue
        writes = re.findall(r"\b([A-Za-z_]\w*)\s*\[[^;]+?\]\s*(?:\+=|-=|=(?!=))", block)
        reads = set(re.findall(r"\b([A-Za-z_]\w*)\s*\[", block)) - set(writes)
        if len(set(writes)) != 1 or len(reads) < 2 or "+=" not in block:
            continue
        groups[_compact(block)].append(loop)
    result = []
    for copies in groups.values():
        if len(copies) < 2:
            continue
        # Report maximal equal producers once, not each nested subloop again.
        if any(
            other[0].start < copies[0].start
            and copies[0].end <= other[0].end
            and len(other) == len(copies)
            for other in groups.values()
            if other != copies
        ):
            continue
        item = _finding(
            "repeated-producer-loop",
            path,
            clean,
            function,
            copies[0].start,
            [
                _compact(clean[copies[0].start : copies[0].end]),
                "Identical arithmetic producer loop appears at multiple source locations.",
                "Inputs, aliases, intervening mutation and both paths' execution must be checked before claiming reuse.",
            ],
            source_copies=len(copies),
            repeated_lines=[clean.count("\n", 0, copy.start) + 1 for copy in copies],
            executed_to_logical_ratio=None,
            count_kind="syntactic-copies-not-runtime-count",
        )
        item["confidence"] = "identical-source-block"
        item["disposition"] = "needs-alias-lifetime-and-execution-proof"
        item["action"] = (
            "Check common input identity and retained-workspace policy; retain legitimate bounded recomputation."
        )
        result.append(item)
    return result


def _shadowed(name: str, owner: Function, clean: str, position: int) -> bool:
    if re.search(rf"\b{re.escape(name)}\b", owner.parameters):
        return True
    before = clean[owner.body : position]
    return bool(
        re.search(
            rf"\b(?:auto|[A-Za-z_]\w*)\s+[&*]?\s*{re.escape(name)}\s*[=;({{]|\(\s*\*\s*{re.escape(name)}\s*\)",
            before,
        )
    )


def _call_target(
    call: Call,
    owner: Function,
    unique: dict[str, Function],
    clean: str,
    functions: list[Function],
) -> Function | None:
    target = unique.get(call.name)
    if target is None or target.namespace != owner.namespace:
        return None
    # Declaration-only overloads, qualified prototypes and callable objects
    # outside parsed bodies are unresolved. Only this definition is admitted.
    for match in re.finditer(rf"\b{re.escape(call.name)}\s*\(", clean):
        if match.start() != target.start and not any(
            f.body <= match.start() < f.end for f in functions
        ):
            return None
    if _shadowed(call.name, owner, clean, call.start):
        return None
    before = clean[owner.body : call.start]
    if re.search(
        rf"\b(?:auto|[A-Za-z_]\w*)\s+[&*]?\s*{re.escape(call.name)}\s*[=;({{]|\(\s*\*\s*{re.escape(call.name)}\s*\)",
        before,
    ):
        return None
    return target


def audit_native(
    text: str, path: str = "<memory>", *, max_call_depth: int = 4
) -> list[dict[str, Any]]:
    if not 0 <= max_call_depth <= 8:
        raise ValueError("max_call_depth must be between zero and eight")
    clean, loops = _loop_spans(text)
    functions = _functions(clean)
    by_name: dict[str, list[Function]] = defaultdict(list)
    for function in functions:
        by_name[function.name].append(function)
    unique = {name: rows[0] for name, rows in by_name.items() if len(rows) == 1}
    calls = {f: _calls(clean, f) for f in functions}
    sites = {f: _sites(clean, f, calls[f]) for f in functions}
    findings = []
    for function in functions:
        findings.extend(_repeated_loop_blocks(clean, path, function, loops))
        for site in sites[function]:
            context = _context(loops, site.start, clean)
            if context and (
                site.owner_start is None or _context(loops, site.owner_start, clean)
            ):
                findings.append(
                    _finding(
                        f"loop-{site.kind}",
                        path,
                        clean,
                        function,
                        site.start,
                        [
                            site.expression,
                            "Visible allocation/boundary site is lexically inside a loop; execution and positive extent are not measured.",
                        ],
                        loop_context=context,
                        requested_elements=site.count,
                        count_kind="static-site-not-runtime-count",
                        call_path=[],
                    )
                )

        # Follow only same-file uniquely named functions, with explicit bounded
        # paths and cycle protection. Branch reachability remains a hypothesis.
        def follow(
            owner: Function,
            trail: list[dict[str, Any]],
            seen: set[str],
            caller: Function = function,
        ) -> None:
            if len(trail) > max_call_depth:
                return
            for site in sites[owner]:
                findings.append(
                    _finding(
                        f"loop-callee-{site.kind}",
                        path,
                        clean,
                        caller,
                        trail[0]["offset"],
                        [
                            site.expression,
                            "Same-file call path reaches this site from a lexical loop; conditional execution and scientific role require review.",
                        ],
                        allocation_site={
                            "line": clean.count("\n", 0, site.start) + 1,
                            "function": owner.name,
                            "column": site.start - clean.rfind("\n", 0, site.start),
                        },
                        requested_elements=site.count,
                        count_kind="static-path-not-runtime-count",
                        loop_context=_context(loops, trail[0]["offset"], clean),
                        call_path=[
                            {k: v for k, v in edge.items() if k != "offset"}
                            for edge in trail
                        ],
                    )
                )
            if len(trail) == max_call_depth:
                return
            for edge in calls[owner]:
                target = _call_target(edge, owner, unique, clean, functions)
                if target is not None and edge.name not in seen:
                    follow(
                        target,
                        trail
                        + [
                            {
                                "function": edge.name,
                                "line": clean.count("\n", 0, edge.start) + 1,
                                "offset": edge.start,
                            }
                        ],
                        seen | {edge.name},
                    )

        if max_call_depth:
            for call in calls[function]:
                if (
                    _context(loops, call.start, clean)
                    and _call_target(call, function, unique, clean, functions)
                    is not None
                    and call.name != function.name
                ):
                    follow(
                        unique[call.name],
                        [
                            {
                                "function": call.name,
                                "line": clean.count("\n", 0, call.start) + 1,
                                "offset": call.start,
                            }
                        ],
                        {function.name, call.name},
                    )
    # Multiple internal routes can converge on a single site. Keep one shortest
    # witness per loop call + site, rather than inflating counts with graph paths.
    result = {}
    for finding in findings:
        key = (
            finding["rule_id"],
            finding["line"],
            finding["column"],
            json.dumps(finding["details"].get("allocation_site")),
            finding["evidence"][0],
            json.dumps(finding["details"].get("outer_multiplier")),
        )
        if key not in result or len(finding["details"].get("call_path", [])) < len(
            result[key]["details"].get("call_path", [])
        ):
            result[key] = finding
    return list(result.values())


def fingerprint(finding: dict[str, Any]) -> str:
    """Semantic site anchor, stable under line motion (occurrences added by tree audit)."""

    def semantic(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: semantic(item)
                for key, item in value.items()
                if key not in {"line", "column", "return_line", "repeated_lines"}
            }
        if isinstance(value, list):
            return [semantic(item) for item in value]
        if isinstance(value, str):
            return _compact(value)
        return value

    anchor = [
        finding["rule_id"],
        finding["path"],
        finding["function"],
        re.sub(r"^line \d+:\s*", "", _compact(finding["evidence"][0])),
        semantic(finding.get("details", {})),
    ]
    return hashlib.sha256(json.dumps(anchor, sort_keys=True).encode()).hexdigest()[:20]


def audit_tree(
    root: Path,
    *,
    paths: tuple[str, ...] = ("src", "include", "python"),
    max_call_depth: int = 4,
) -> dict[str, Any]:
    try:
        from tools.work_audit_python import audit_python
    except ModuleNotFoundError:
        from work_audit_python import audit_python  # type: ignore[import-not-found]
    try:
        from tools.audit_structured_materialization import (
            audit_tree as structured_audit,
        )
    except ModuleNotFoundError:
        from audit_structured_materialization import (  # type: ignore[import-not-found]
            audit_tree as structured_audit,
        )
    root = root.resolve()
    findings = []
    scanned = Counter()

    def audit_source(relative: str, data: bytes) -> None:
        text = data.decode("utf-8", errors="replace")
        if Path(relative).suffix == ".py":
            findings.extend(audit_python(text, relative))
            scanned["python_ast"] += 1
        else:
            findings.extend(audit_native(text, relative, max_call_depth=max_call_depth))
            scanned["native_lexical"] += 1

    # One authoritative strict selection/provenance boundary. Every analyzer
    # consumes the same captured bytes; a later path replacement cannot redirect
    # a second read or attach a source hash to different content.
    structured = structured_audit(
        root, paths, include_python_sources=True, _source_visitor=audit_source
    )
    for candidate in structured["findings"]:
        findings.append(
            {
                "rule_id": "native.structured-zero-materialization",
                "path": candidate["path"],
                "line": candidate["line"],
                "function": candidate["function"],
                "evidence": [
                    f"{candidate['buffer']}: zero allocation {candidate['allocation_expression']}; {candidate['classification']}"
                ],
                "confidence": "unknown"
                if candidate["classification"] == "unknown"
                else "high",
                "disposition": "review-required",
                "details": candidate,
            }
        )
    occurrences: Counter[str] = Counter()
    for finding in sorted(
        findings, key=lambda item: (item["path"], item["line"], item.get("column", 0))
    ):
        anchor = fingerprint(finding)
        occurrences[anchor] += 1
        finding["fingerprint"] = f"{anchor}-{occurrences[anchor]}"

    provenance = dict(structured["provenance"])
    provenance["analyzer_digest"] = provenance["scanner_digest"]
    diagnostics = [
        finding["details"]["materialization_diagnostic"]
        for finding in findings
        if "materialization_diagnostic" in finding.get("details", {})
    ]
    diagnostics.extend(
        boundary["materialization_diagnostic"]
        for boundary in structured["production_boundaries"]
    )
    return {
        "schema": "generativeqc.work-audit.v1",
        "advisory_only": True,
        "provenance": provenance,
        "materialization_diagnostics": diagnostics,
        "production_boundaries": structured["production_boundaries"],
        "scanned_files": dict(scanned),
        "max_call_depth": max_call_depth,
        "counts": dict(sorted(Counter(f["rule_id"] for f in findings).items())),
        "limitations": [
            "Static sites/paths are not runtime counts or latency measurements.",
            "Native parsing is a bounded lexical subset, not a complete C++ AST.",
            "Only unambiguous same-file free-function calls are resolved; virtual/template/operator dispatch and external calls are unknown.",
            "Nonfresh vector growth is omitted because capacity is unknown; default/zero-size vector declarations are not allocations.",
            "No inferred role exemptions and no claim that all findings are production regressions.",
            "Generated native code embedded in Python string templates is not parsed; scan generated output explicitly.",
            "No source-level finding proves CUDA residency, executed byte counts, or complete endpoint cost.",
        ],
        "findings": sorted(
            findings,
            key=lambda f: (f["path"], f["line"], f["rule_id"], f["fingerprint"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--path", action="append", dest="paths")
    parser.add_argument("--format", choices=("json", "text"), default="text")
    parser.add_argument("--max-call-depth", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--compare",
        type=Path,
        help="Prior JSON receipt; report added/removed fingerprints, never fail the build.",
    )
    args = parser.parse_args(argv)
    report = audit_tree(
        args.root,
        paths=tuple(args.paths or ("src", "include", "python")),
        max_call_depth=args.max_call_depth,
    )
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        previous = {item["fingerprint"] for item in baseline["findings"]}
        current = {item["fingerprint"] for item in report["findings"]}
        report["comparison"] = {
            "added": sorted(current - previous),
            "removed": sorted(previous - current),
            "unchanged": len(current & previous),
        }
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"Advisory work audit: {dict(report['scanned_files'])}; {len(report['findings'])} static findings"
        )
        print(json.dumps(report["counts"], sort_keys=True))
        if not args.summary_only:
            for finding in report["findings"]:
                print(
                    f"{finding['path']}:{finding['line']}: {finding['rule_id']} [{finding['fingerprint']}] {finding['evidence'][0]}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
