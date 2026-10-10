"""Extract limited C++ test contracts without coupling to formatting or comments.

Prefer compiling production headers and translation units directly. These helpers
exist for host-only probes of CUDA implementations that cannot be compiled by
an ordinary host compiler. They deliberately reject missing or ambiguous names.
"""

from __future__ import annotations

import re


def _code_only(source: str) -> str:
    """Mask comments and C++ literals while retaining offsets and newlines."""
    masked = list(source)
    length = len(source)

    def erase(start: int, end: int) -> None:
        for index in range(start, end):
            if source[index] != "\n":
                masked[index] = " "

    index = 0
    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = length if end < 0 else end
            erase(index, end)
            index = end
        elif source.startswith("/*", index):
            close = source.find("*/", index + 2)
            if close < 0:
                raise ValueError("unterminated C++ block comment")
            end = close + 2
            erase(index, end)
            index = end
        elif source.startswith('R"', index):
            opener = re.match(r'R"([^\s\\()]{0,16})\(', source[index:])
            if opener is None:
                index += 1
                continue
            delimiter = opener.group(1)
            closing = ")" + delimiter + '"'
            end = source.find(closing, index + len(opener.group(0)))
            if end < 0:
                raise ValueError("unterminated C++ raw string")
            end += len(closing)
            erase(index, end)
            index = end
        elif (
            source[index] == "'"
            and index > 0
            and index + 1 < length
            and source[index - 1].isdigit()
            and source[index + 1].isalnum()
        ):
            # C++14 digit separator, not a character literal.
            index += 1
        elif source[index] in ('"', "'"):
            quote = source[index]
            end = index + 1
            while end < length:
                if source[end] == "\\":
                    end += 2
                elif source[end] == quote:
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise ValueError("unterminated C++ string/character literal")
            erase(index, end)
            index = end
        else:
            index += 1
    return "".join(masked)


def _closing(code: str, opening: int, left: str, right: str) -> int:
    if code[opening] != left:
        raise ValueError(f"expected {left!r} at {opening}")
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == left:
            depth += 1
        elif code[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unclosed C++ {left!r} at {opening}")


def _function_matches(
    source: str, name: str, *, declaration: bool
) -> list[tuple[int, int]]:
    code = _code_only(source)
    matches = []
    for match in re.finditer(r"\b" + re.escape(name) + r"\s*\(", code):
        # These probes support ordinary named functions with their return type
        # on the name's line. A call (including a call inside an if condition)
        # must not become a contract when its real definition is absent.
        start = max(code.rfind(token, 0, match.start()) for token in "\n;{}") + 1
        prefix = code[start : match.start()].strip()
        prefix = re.sub(r"\b__launch_bounds__\s*\([^()]*\)", "", prefix).strip()
        if (
            not prefix
            or prefix.endswith("::")
            or re.search(
                r"\b(?:return|co_return|if|else|while|for|switch|case|throw)\b", prefix
            )
            or re.fullmatch(r"[\w\s:<>,*&]+", prefix) is None
            or re.search(r"[A-Za-z_]", prefix) is None
        ):
            continue
        opening = code.find("(", match.start(), match.end())
        closing = _closing(code, opening, "(", ")")
        # Consume only supported function suffixes. In particular, a closing
        # parenthesis belonging to an enclosing expression is never a suffix.
        boundary = closing + 1
        while boundary < len(code):
            if code[boundary].isspace():
                boundary += 1
                continue
            qualifier = re.match(
                r"(?:noexcept|const|volatile|override|final)\b|&&?", code[boundary:]
            )
            if qualifier is None:
                break
            boundary += qualifier.end()
            if qualifier.group() == "noexcept":
                while boundary < len(code) and code[boundary].isspace():
                    boundary += 1
                if boundary < len(code) and code[boundary] == "(":
                    boundary = _closing(code, boundary, "(", ")") + 1
        if boundary >= len(code):
            continue
        marker = code[boundary]
        if marker != (";" if declaration else "{"):
            continue
        matches.append((start, boundary))
    return matches


def _unique(matches: list[tuple[int, int]], name: str) -> tuple[int, int]:
    if len(matches) != 1:
        raise ValueError(
            f"expected one C++ contract for {name!r}, found {len(matches)}"
        )
    return matches[0]


def cpp_function_declaration(source: str, name: str) -> str:
    """Return the real, unique declaration without its terminating semicolon.

    The return type must share a line with the function name, as in our internal
    C++ headers. The declaration supplies a host test double's ABI directly.
    """
    start, end = _unique(_function_matches(source, name, declaration=True), name)
    result = source[start:end].strip()
    if not result or result.startswith("#"):
        raise ValueError(f"invalid declaration for {name!r}")
    return result


def cpp_function_definition(
    source: str, name: str, *, include_template: bool = False
) -> str:
    """Extract exactly one named implementation for a CPU-executed CUDA shim.

    Signature whitespace/parameter changes are immaterial; body braces in
    comments, ordinary strings or C++ raw literals do not terminate the body.
    """
    start, opening = _unique(_function_matches(source, name, declaration=False), name)
    code = _code_only(source)
    end = _closing(code, opening, "{", "}") + 1
    if include_template:
        template = list(re.finditer(r"\btemplate\s*<", code[:start]))
        if not template:
            raise ValueError(f"missing template for {name!r}")
        candidate = template[-1].start()
        if any(token in code[candidate:start] for token in ";{}"):
            raise ValueError(f"nonadjacent template for {name!r}")
        start = candidate
    return source[start:end]


def cpp_record_definition(source: str, name: str) -> str:
    """Extract one struct's genuine definition, ignoring forward declarations."""
    code = _code_only(source)
    found = []
    for match in re.finditer(r"\bstruct\s+" + re.escape(name) + r"\b", code):
        begin = match.end()
        while begin < len(code) and code[begin] not in ";{":
            begin += 1
        if begin >= len(code) or code[begin] != "{":
            continue
        # An elaborated type in a parameter/variable declaration is not the
        # record's definition, even if a function or lambda body follows it.
        # Support only an optional final specifier and ordinary base classes.
        suffix = code[match.end() : begin]
        if (
            re.fullmatch(r"\s*(?:final\b\s*)?(?::\s*[A-Za-z_][\w\s:<>,]*)?", suffix)
            is None
        ):
            continue
        found.append((match.start(), _closing(code, begin, "{", "}") + 1))
    start, end = _unique(found, name)
    return source[start:end]


def cpp_if_block(source: str, condition_prefix: str) -> str:
    """Extract one braced production if-body, matching a stable condition prefix.

    This is reserved for host-only execution of CUDA admission/rollback code.
    It skips single-statement guards and ignores comments, literals and spacing.
    """
    code = _code_only(source)
    target = "".join(condition_prefix.split())
    matches = []
    for match in re.finditer(r"\bif\s*\(", code):
        opening = code.find("(", match.start(), match.end())
        closing = _closing(code, opening, "(", ")")
        condition = "".join(code[opening + 1 : closing].split())
        if not condition.startswith(target):
            continue
        brace = closing + 1
        while brace < len(code) and code[brace].isspace():
            brace += 1
        if brace >= len(code) or code[brace] != "{":
            continue
        matches.append((match.start(), _closing(code, brace, "{", "}") + 1))
    start, end = _unique(matches, f"if({condition_prefix}...)")
    return source[start:end]
