"""Audit native source for avoidable high-order scalar contraction patterns."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"}
)
DEFAULT_EXCLUDES = ("src/xtb/native",)
_TYPE = (
    r"(?:const\s+)?(?:"
    r"(?:std::)?(?:size_t|u?int(?:8|16|32|64)_t)|"
    r"unsigned(?:\s+\w+)?|int|long(?:\s+\w+)?|auto)"
)
_FOR = re.compile(r"\bfor\s*\(")
_LOOP_VAR = re.compile(_TYPE + r"\s+([A-Za-z_]\w*)\s*=")
_ALIAS = re.compile(_TYPE + r"\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")
_ACCESS = re.compile(r"[A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*\s*\[([^\[\]]+)\]")
_REDUCTION = re.compile(
    r"(?P<lhs>[A-Za-z_][^;=\n]*?)\s*(?P<op>\+=|-=)\s*(?P<rhs>[^;]+);"
)


@dataclass(frozen=True)
class LoopSpan:
    start: int
    end: int
    body_start: int
    body_end: int
    variable: str | None
    depth: int = 1


@dataclass(frozen=True)
class LoopFinding:
    path: str
    line: int
    depth: int
    variables: tuple[str, ...]
    classification: str
    lhs: str | None = None


def _mask_comments_and_literals(text: str) -> str:
    chars = list(text)
    out = list(text)
    index = 0
    state = "normal"
    quote = ""
    while index < len(chars):
        char = chars[index]
        following = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "normal":
            if char == "/" and following == "/":
                out[index] = out[index + 1] = " "
                index += 2
                state = "line-comment"
                continue
            if char == "/" and following == "*":
                out[index] = out[index + 1] = " "
                index += 2
                state = "block-comment"
                continue
            if char in {'"', "'"}:
                quote = char
                out[index] = " "
                index += 1
                state = "literal"
                continue
            index += 1
            continue
        if state == "line-comment":
            if char == "\n":
                state = "normal"
            else:
                out[index] = " "
            index += 1
            continue
        if state == "block-comment":
            if char == "*" and following == "/":
                out[index] = out[index + 1] = " "
                index += 2
                state = "normal"
            else:
                if char != "\n":
                    out[index] = " "
                index += 1
            continue
        if char == "\\":
            out[index] = " "
            if index + 1 < len(chars) and chars[index + 1] != "\n":
                out[index + 1] = " "
            index += 2
        elif char == quote:
            out[index] = " "
            index += 1
            state = "normal"
        else:
            if char != "\n":
                out[index] = " "
            index += 1
    return "".join(out)


def _matching(text: str, opening: int, left: str, right: str) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == left:
            depth += 1
        elif text[index] == right:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _keyword_at(text: str, index: int, keyword: str) -> bool:
    if not text.startswith(keyword, index):
        return False
    following = index + len(keyword)
    return following >= len(text) or not (
        text[following].isalnum() or text[following] == "_"
    )


def _statement_end(text: str, index: int) -> int:
    index = _skip_space(text, index)
    if index >= len(text):
        return len(text)
    if text[index] == "{":
        closing = _matching(text, index, "{", "}")
        return len(text) if closing < 0 else closing + 1
    for keyword in ("for", "while", "if", "switch"):
        if not _keyword_at(text, index, keyword):
            continue
        opening = text.find("(", index)
        closing = _matching(text, opening, "(", ")")
        if closing < 0:
            return len(text)
        end = _statement_end(text, closing + 1)
        if keyword == "if":
            alternative = _skip_space(text, end)
            if _keyword_at(text, alternative, "else"):
                end = _statement_end(text, alternative + len("else"))
        return end
    parentheses = brackets = braces = 0
    for position in range(index, len(text)):
        char = text[position]
        if char == "(":
            parentheses += 1
        elif char == ")":
            parentheses -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == "{":
            braces += 1
        elif char == "}":
            if braces == 0:
                return position
            braces -= 1
        elif char == ";" and parentheses == brackets == braces == 0:
            return position + 1
    return len(text)


def _loop_spans(text: str) -> tuple[str, list[LoopSpan]]:
    clean = _mask_comments_and_literals(text)
    loops: list[LoopSpan] = []
    for match in _FOR.finditer(clean):
        opening = clean.find("(", match.start())
        closing = _matching(clean, opening, "(", ")")
        if closing < 0:
            continue
        body_start = _skip_space(clean, closing + 1)
        body_end = _statement_end(clean, body_start)
        variable_match = _LOOP_VAR.search(clean[opening + 1 : closing])
        loops.append(
            LoopSpan(
                match.start(),
                body_end,
                body_start,
                body_end,
                variable_match.group(1) if variable_match else None,
            )
        )
    resolved: list[LoopSpan] = []
    for loop in loops:
        depth = 1 + sum(
            parent.start < loop.start
            and parent.body_start <= loop.start < parent.body_end
            for parent in loops
        )
        resolved.append(
            LoopSpan(
                loop.start,
                loop.end,
                loop.body_start,
                loop.body_end,
                loop.variable,
                depth,
            )
        )
    return clean, resolved


def _dependencies(
    expression: str,
    loop_variables: tuple[str, ...],
    aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    result = {
        variable
        for variable in loop_variables
        if re.search(rf"\b{re.escape(variable)}\b", expression)
    }
    for name in set(re.findall(r"\b[A-Za-z_]\w*\b", expression)):
        if name in aliases and name not in seen:
            result.update(
                _dependencies(
                    expression=aliases[name],
                    loop_variables=loop_variables,
                    aliases=aliases,
                    seen=seen | {name},
                )
            )
    return frozenset(result)


def _access_indices(
    expression: str,
    aliases: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> list[str]:
    result = [match.group(1) for match in _ACCESS.finditer(expression)]
    for name in set(re.findall(r"\b[A-Za-z_]\w*\b", expression)):
        if name in aliases and name not in seen:
            result.extend(_access_indices(aliases[name], aliases, seen | {name}))
    return result


def _matrix_chain_lhs(
    body: str,
    context: str,
    loop_variables: tuple[str, ...],
) -> str | None:
    aliases = {match.group(1): match.group(2) for match in _ALIAS.finditer(context)}
    for reduction in _REDUCTION.finditer(body):
        lhs = reduction.group("lhs").strip()
        rhs = reduction.group("rhs")
        lhs_dependencies = _dependencies(lhs, loop_variables, aliases)
        rhs_dependencies = _dependencies(rhs, loop_variables, aliases)
        access_dependencies = [
            _dependencies(index, loop_variables, aliases)
            for index in _access_indices(rhs, aliases)
        ]
        access_dependencies = [item for item in access_dependencies if item]
        rank2_accesses = [item for item in access_dependencies if len(item) <= 2]
        if (
            len(lhs_dependencies) <= 2
            and len(rhs_dependencies) >= 4
            and len(rank2_accesses) >= 3
            and set().union(*rank2_accesses) >= set(loop_variables[-4:])
            and not any(len(item) >= 3 for item in access_dependencies)
        ):
            return lhs
    return None


def audit_text(
    text: str,
    *,
    path: str = "<memory>",
    minimum_depth: int = 4,
) -> tuple[LoopFinding, ...]:
    """Return leaf high-order loop nests and classify avoidable matrix chains."""

    if minimum_depth < 2:
        raise ValueError("minimum_depth must be at least two")
    _, loops = _loop_spans(text)
    leaves = [
        loop
        for loop in loops
        if loop.depth >= minimum_depth
        and not any(
            child.start > loop.start and child.end <= loop.end for child in loops
        )
    ]
    findings: list[LoopFinding] = []
    for inner in leaves:
        parents = sorted(
            (
                loop
                for loop in loops
                if loop.start < inner.start
                and loop.body_start <= inner.start < loop.body_end
            ),
            key=lambda loop: loop.start,
        )
        chain = parents + [inner]
        variables = tuple(loop.variable for loop in chain if loop.variable is not None)
        lhs = None
        if len(set(variables)) >= minimum_depth:
            body = text[inner.body_start : inner.body_end]
            context = text[chain[-minimum_depth].start : inner.body_end]
            lhs = _matrix_chain_lhs(body, context, variables)
        findings.append(
            LoopFinding(
                path=path,
                line=text.count("\n", 0, inner.start) + 1,
                depth=inner.depth,
                variables=variables,
                classification=(
                    "matrix-chain-candidate" if lhs is not None else "high-order-loop"
                ),
                lhs=lhs,
            )
        )
    return tuple(findings)


def audit_tree(
    source_root: Path,
    *,
    minimum_depth: int = 4,
    excludes: tuple[str, ...] = (),
) -> tuple[int, tuple[LoopFinding, ...]]:
    root = source_root.resolve()
    scanned = 0
    findings: list[LoopFinding] = []
    paths = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.suffix in SOURCE_SUFFIXES
    )
    for path in paths:
        relative = path.relative_to(root.parent).as_posix()
        if any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in excludes
        ):
            continue
        scanned += 1
        findings.extend(
            audit_text(
                path.read_text(errors="replace"),
                path=relative,
                minimum_depth=minimum_depth,
            )
        )
    return scanned, tuple(findings)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("src"))
    parser.add_argument("--minimum-depth", type=int, default=4)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--include-vendored", action="store_true")
    parser.add_argument("--fail-on-matrix-chain", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    excludes = () if args.include_vendored else DEFAULT_EXCLUDES
    scanned, findings = audit_tree(
        args.source_root,
        minimum_depth=args.minimum_depth,
        excludes=excludes,
    )
    candidates = tuple(
        item for item in findings if item.classification == "matrix-chain-candidate"
    )
    payload = {
        "schema": "generativeqc.native-complexity-audit.v1",
        "scanned_files": scanned,
        "high_order_loops": [asdict(item) for item in findings],
        "matrix_chain_candidates": [asdict(item) for item in candidates],
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"native complexity audit: {scanned} files, "
            f"{len(findings)} high-order loop nests"
        )
        for item in findings:
            suffix = f" -> {item.lhs}" if item.lhs is not None else ""
            print(
                f"{item.path}:{item.line}: depth={item.depth} "
                f"{item.classification}{suffix}"
            )
        if not findings:
            print("no high-order native loop nests found")
    if args.fail_on_matrix_chain and candidates:
        print(
            "avoidable rank-2 matrix-chain scalar reductions detected; "
            "use TensorIR/shared dense-linalg lowering",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
