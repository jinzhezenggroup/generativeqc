#!/usr/bin/env python3
"""Check source-owned native contracts and render the declaration reference.

This is a deliberately bounded declaration scanner, not a C++ compiler. It
recognizes the public header's explicit C ABI and non-template C++ wrapper
forms, rejects unrecognized public forms, and never imports a native module.
The reviewed manifest binds signatures, C++ public record fields, and applicable facets;
a changed signature or new declaration requires an explicit contract review.
Structured facets establish coverage, not scientific truth: reviewers must
still compare their content to implementation and capability authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = (
    "include/generativeqc/generativeqc.h",
    "include/generativeqc/generativeqc.hpp",
    "include/generativeqc/ks.hpp",
)
MANIFEST = "manifests/native_api_contracts.json"
REFERENCE = "docs/reference/native_symbols.md"
FACETS = {"behavior", "inputs", "outputs", "lifetime", "errors", "execution", "units"}
# Match literals before comments, so URLs and comment-like string contents survive.
LEX = re.compile(
    r'R"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r"|/\*[\s\S]*?\*/|//[^\n]*|[A-Za-z_]\w*|\d+(?:\.\d*)?"
    r"|::|&&|\[\[|\]\]|->|==|!=|<=|>=|\S"
)


class ContractError(ValueError):
    """An unsupported declaration or missing/invalid contract."""


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass
class Declaration:
    name: str
    kind: str
    signature: str
    start: int
    comment: str
    fields: tuple[str, ...] = ()
    header: str = ""
    field_declarations: tuple[str, ...] = ()

    @property
    def anchor(self) -> str:
        prefix = "native-c-" if self.kind == "c-function" else "native-cpp-"
        name = self.name.replace("~", "destructor-").replace(
            "operator=", "copy-assignment"
        )
        return prefix + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def preprocessor_directives(source: str) -> list[tuple[int, int, str]]:
    """Find directives without mistaking comments or string payloads for code."""
    lexical = list(LEX.finditer(source))
    masked = list(source)
    strings = []
    for match in lexical:
        value = match.group()
        if value.startswith(("/*", "//")):
            masked[match.start() : match.end()] = [
                "\n" if char == "\n" else " " for char in value
            ]
        elif value.startswith(('"', "'", 'R"')):
            strings.append((match.start(), match.end()))
    code = "".join(masked)
    directives = []
    for match in re.finditer(r"^[ \t]*\#(?:[^\n\\]|\\[^\n]|\\\n)*", code, re.MULTILINE):
        marker = code.index("#", match.start(), match.end())
        if not any(start <= marker < end for start, end in strings):
            directives.append((match.start(), match.end(), match.group()))
    return directives


def validate_macro_forms(source: str) -> None:
    """Reject macro aliases/generators outside the literal public-header grammar.

    Otherwise preprocessing could hide a declaration from this static scanner.
    The only nonliteral macro admitted is the existing export annotation itself;
    an unfamiliar form requires explicit scanner support, not silent exclusion.
    """
    exports = {
        "__declspec ( dllexport )",
        "__declspec ( dllimport )",
        '__attribute__ ( ( visibility ( "default" ) ) )',
    }
    for _, _, raw_directive in preprocessor_directives(source):
        directive = raw_directive.replace("\\\n", " ")
        match = re.fullmatch(r"\s*#\s*define\s+([A-Za-z_]\w*)(.*)", directive)
        if not match:
            continue
        name, suffix = match.groups()
        if suffix.startswith("("):
            raise ContractError(
                f"unsupported function-like public-header macro: {name}"
            )
        value = suffix.strip()
        if name == "GENERATIVEQC_API":
            if " ".join(executable_tokens(value)) not in exports:
                raise ContractError("unsupported GENERATIVEQC_API export definition")
        elif value and not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*", value):
            raise ContractError(f"unsupported nonliteral public-header macro: {name}")


def tokens(source: str) -> list[Token]:
    """Lex strings atomically, removing comments and preprocessor directives."""
    preprocessor = preprocessor_directives(source)
    return [
        Token(m.group(), m.start(), m.end())
        for m in LEX.finditer(source)
        if not m.group().startswith(("/*", "//"))
        and not any(a <= m.start() < b for a, b, _ in preprocessor)
    ]


def executable_tokens(source: str) -> tuple[str, ...]:
    """String-aware comment stripping used to prove header code unchanged.

    Unlike declaration scanning, this retains preprocessor tokens as executable
    interface content, including macro definitions and conditional compilation.
    """
    return tuple(
        m.group()
        for m in LEX.finditer(source)
        if not m.group().startswith(("/*", "//"))
    )


def adjacent_comment(source: str, start: int) -> str:
    """Collect only contiguous adjacent documentation blocks, never random prose."""
    prefix = source[:start].rstrip()
    blocks = []
    while prefix.endswith("*/"):
        pos = prefix.rfind("/*")
        if pos < 0 or not prefix[pos:].startswith("/**"):
            break
        blocks.append(prefix[pos + 3 : -2])
        prefix = prefix[:pos].rstrip()
    return "\n".join(
        re.sub(r"^\s*\* ?", "", b, flags=re.MULTILINE).strip() for b in reversed(blocks)
    )


def paired(ts: list[Token], start: int, left: str, right: str) -> int:
    level = 0
    for i in range(start, len(ts)):
        level += ts[i].text == left
        level -= ts[i].text == right
        if level == 0:
            return i
    raise ContractError(f"unclosed {left} at offset {ts[start].start}")


def signature(ts: list[Token]) -> str:
    return " ".join(t.text for t in ts)


def scan_c(source: str) -> list[Declaration]:
    validate_macro_forms(source)
    ts = tokens(source)
    result = []
    covered = set()
    for i, token in enumerate(ts):
        if token.text != "GENERATIVEQC_API":
            continue
        end = i + 1
        while end < len(ts) and ts[end].text not in {";", "{"}:
            end += 1
        declaration = ts[i:end]
        # Return type is intentionally bounded; typedef function pointers,
        # variables, attributes, and macro-generated declarations fail closed.
        text = signature(declaration)
        match = re.fullmatch(
            r"GENERATIVEQC_API (?:uint32_t|generativeqc_status|void|const char \*) "
            r"(generativeqc_\w+) \( ([^{};]*) \)",
            text,
        )
        if end == len(ts) or ts[end].text != ";" or not match:
            raise ContractError(f"unsupported C public declaration: {text}")
        if any(
            t.text in {"(", ")"}
            for t in declaration[
                declaration.index(next(t for t in declaration if t.text == "("))
                + 1 : -1
            ]
        ):
            raise ContractError(f"unsupported nested C declarator: {text}")
        covered.update(range(i, end + 1))
        result.append(
            Declaration(
                match[1],
                "c-function",
                text + " ;",
                token.start,
                adjacent_comment(source, token.start),
            )
        )
    for i, token in enumerate(ts):
        if i not in covered and token.text in {
            "(",
            "__attribute__",
            "__attribute",
            "__declspec",
        }:
            raise ContractError(
                "unsupported C declaration outside literal GENERATIVEQC_API form: "
                + signature(ts[max(0, i - 3) : i + 8])
            )

    return result


def operation_name(owner: str | None, head: list[Token]) -> str:
    texts = [t.text for t in head]
    p = texts.index("(")
    if p == 0:
        raise ContractError("unnamed public C++ operation")
    name = texts[p - 1]
    if p > 1 and texts[p - 2] == "~":
        name = "~" + name
    elif p > 1 and texts[p - 2] == "operator":
        name = "operator" + name
    if p == 1 and name != owner:
        raise ContractError(
            f"unsupported C++ return/declarator form: {signature(head)}"
        )
    close = paired(head, p, "(", ")")
    if any(t.text in {"(", ")"} for t in head[p + 1 : close]):
        raise ContractError(f"unsupported nested C++ declarator: {signature(head)}")
    if not re.fullmatch(r"~?[A-Za-z_]\w*|operator=", name):
        raise ContractError(f"unsupported public C++ operation: {signature(head)}")
    if any(t in texts[:p] for t in ("template", "friend", "using", "typedef")):
        raise ContractError(f"unsupported public C++ declaration: {signature(head)}")
    if owner and name in {owner, "operator="}:
        params = texts[p + 1 : texts.index(")", p)]
        if "&&" in params and owner in params:
            name += "-move"
        elif "&" in params and owner in params:
            name += "-copy"
    return "generativeqc::" + (owner + "::" if owner else "") + name


def scan_cpp(source: str) -> list[Declaration]:
    validate_macro_forms(source)
    ts = tokens(source)
    if any(token.text == "friend" for token in ts):
        raise ContractError(
            "unsupported C++ friend declaration, including ADL-visible private friends"
        )
    if [t.text for t in ts[:3]] != ["namespace", "generativeqc", "{"]:
        raise ContractError("unexpected public C++ namespace/header form")
    end = paired(ts, 2, "{", "}")
    if end != len(ts) - 1:
        raise ContractError("unrecognized declarations outside generativeqc namespace")
    result = []

    def consume(
        start: int, stop: int, owner: str | None = None, public: bool = True
    ) -> tuple[list[str], list[str]]:
        i = start
        fields = []
        field_declarations = []
        while i < stop:
            if ts[i].text in {"public", "protected", "private"}:
                if i + 1 >= stop or ts[i + 1].text != ":":
                    raise ContractError("invalid access specifier")
                public = ts[i].text == "public"
                i += 2
                continue
            if owner is None and ts[i].text in {"class", "struct"}:
                name = ts[i + 1].text
                brace = i + 2
                while brace < stop and ts[brace].text != "{":
                    brace += 1
                if brace == stop:
                    raise ContractError("unsupported forward public C++ type")
                close = paired(ts, brace, "{", "}")
                if ts[close + 1].text != ";":
                    raise ContractError("unsupported public C++ type suffix")
                declaration = Declaration(
                    "generativeqc::" + name,
                    "cpp-type",
                    signature(ts[i:brace]),
                    ts[i].start,
                    adjacent_comment(source, ts[i].start),
                )
                result.append(declaration)
                fields_found, field_sources = consume(
                    brace + 1, close, name, ts[i].text == "struct"
                )
                declaration.fields = tuple(fields_found)
                declaration.field_declarations = tuple(field_sources)
                i = close + 2
                continue
            j = i
            paren = None
            while j < stop:
                if ts[j].text == "[[":
                    j = paired(ts, j, "[[", "]]") + 1
                    continue
                if ts[j].text == "(":
                    paren = j
                    j = paired(ts, j, "(", ")") + 1
                    # Constructor initializer lists include calls/braced data;
                    # their executable content is not an API signature.
                    while j < stop and ts[j].text not in {":", "{", ";"}:
                        j += 1
                    break
                if ts[j].text in {"{", ";"}:
                    break
                j += 1
            if j == stop:
                raise ContractError("unterminated public C++ declaration")
            head = ts[i:j]
            if paren is not None:
                if public:
                    result.append(
                        Declaration(
                            operation_name(owner, head),
                            "cpp-operation",
                            signature(head),
                            ts[i].start,
                            adjacent_comment(source, ts[i].start),
                        )
                    )
                if ts[j].text == ":":
                    # Skip each constructor initializer's balanced () or {}.
                    j += 1
                    while j < stop and ts[j].text != "{":
                        if ts[j].text == "(":
                            j = paired(ts, j, "(", ")")
                        j += 1
                if ts[j].text == "{":
                    j = paired(ts, j, "{", "}") + 1
                    if j < stop and ts[j].text == ";":
                        j += 1
                else:
                    j += 1
            else:
                if owner is None:
                    raise ContractError(
                        f"unsupported namespace declaration: {signature(head)}"
                    )
                if public:
                    text = signature(head)
                    if any(
                        t.text
                        in {"using", "typedef", "template", "class", "struct", ":"}
                        for t in head
                    ) or not re.fullmatch(r"[\w:<> ,]+\s+[A-Za-z_]\w*", text):
                        raise ContractError(f"unsupported public record field: {text}")
                    fields.append(head[-1].text)
                if ts[j].text == "{":
                    j = paired(ts, j, "{", "}") + 1
                    if j >= stop or ts[j].text != ";":
                        raise ContractError("unsupported record initializer")
                j += 1
                if public:
                    field_declarations.append(signature(ts[i:j]))
            i = j
        return fields, field_declarations

    consume(3, end)
    return result


def discover(root: Path) -> list[Declaration]:
    declarations = []
    for path in HEADERS:
        scanner = scan_c if path.endswith(".h") else scan_cpp
        found = scanner((root / path).read_text(encoding="utf-8"))
        for declaration in found:
            declaration.header = path
        declarations.extend(found)
    names = [d.name for d in declarations]
    if len(set(names)) != len(names):
        raise ContractError(
            "duplicate native declaration identity; classify new overload explicitly"
        )
    if len({d.anchor for d in declarations}) != len(declarations):
        raise ContractError("duplicate native reference anchor")
    return declarations


def contract_facets(declaration: Declaration) -> dict[str, str]:
    markers = re.findall(r"^@native-contract (.+)$", declaration.comment, re.MULTILINE)
    if markers != [declaration.name]:
        raise ContractError(
            f"{declaration.name}: missing exact source-owned @native-contract"
        )
    result = {}
    for match in re.finditer(
        r"^@(\w+)\s+([^@]+?)(?=^@|\Z)", declaration.comment, re.MULTILINE | re.DOTALL
    ):
        if match[1] not in FACETS:
            raise ContractError(
                f"{declaration.name}: unknown contract facet {match[1]}"
            )
        body = match[2].strip()
        if (
            match[1] in result
            or not body
            or re.search(
                r"\b(?:TODO|TBD|FIXME|placeholder)\b|^(?:A |The )?documented (?:entry point|API)\.?$",
                body,
                re.IGNORECASE,
            )
        ):
            raise ContractError(f"{declaration.name}: unusable {match[1]} contract")
        result[match[1]] = body
    return result


def reviewed_entry(declaration: Declaration, facets: list[str]) -> dict:
    return {
        "kind": declaration.kind,
        "header": declaration.header,
        "signature_sha256": hashlib.sha256(declaration.signature.encode()).hexdigest(),
        "required_facets": sorted(facets),
        "fields": list(declaration.fields),
        "field_declarations": list(declaration.field_declarations),
    }


def validate(root: Path) -> list[Declaration]:
    declarations = discover(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ContractError("unsupported native contract inventory schema")
    entries = manifest["declarations"]
    names = {d.name for d in declarations}
    if names != entries.keys():
        raise ContractError(
            f"native declaration inventory drift: new={sorted(names - entries.keys())}; "
            f"removed={sorted(entries.keys() - names)}"
        )
    for d in declarations:
        entry = entries[d.name]
        required = entry["required_facets"]
        if not required or "behavior" not in required or not set(required) <= FACETS:
            raise ContractError(f"{d.name}: invalid reviewed facet policy")
        if entry != reviewed_entry(d, required):
            raise ContractError(
                f"{d.name}: signature/field drift requires contract review"
            )
        actual = contract_facets(d)
        if not set(required) <= actual.keys():
            raise ContractError(
                f"{d.name}: missing facets {sorted(set(required) - actual.keys())}"
            )
        if d.fields:
            for field in d.fields:
                if not re.search(r"\b" + re.escape(field) + r"\b", d.comment):
                    raise ContractError(
                        f"{d.name}: record field {field} lacks source contract"
                    )
    return declarations


def render(declarations: list[Declaration]) -> str:
    lines = [
        "# Native symbol contracts",
        "",
        "Generated from the source-owned public header comments by",
        "`python tools/check_native_api_contracts.py --write-reference`.",
        "Do not edit this file directly. The reviewed declaration/facet inventory",
        "lives in `manifests/native_api_contracts.json`.",
        "",
        "See [native conventions](native_api.md) for descriptor initialization,",
        "capability authorities, serialization, and status handling.",
        "",
        "## Index",
        "",
    ]
    for d in declarations:
        lines.append(f"- [{d.name}](#{d.anchor})")
    for d in declarations:
        lines.extend(
            [
                "",
                f"({d.anchor})=",
                f"## {d.name}",
                "",
                f"Source: [{Path(d.header).name}](../../{d.header})",
                "",
                "```cpp",
                d.signature,
                "```",
                "",
            ]
        )
        comment = re.sub(r"^@native-contract .+\n?", "", d.comment, flags=re.MULTILINE)
        comment = re.sub(
            r"^@(\w+)\s+",
            lambda m: "\n**" + m[1].capitalize() + ".** ",
            comment,
            flags=re.MULTILINE,
        )
        comment = re.sub(r"\\p\s+([\w]+)", r"`\1`", comment)
        lines.append(comment.strip())
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--write-reference", action="store_true")
    args = parser.parse_args()
    try:
        declarations = validate(args.root)
        expected = render(declarations)
        reference = args.root / REFERENCE
        if args.write_reference:
            reference.write_text(expected, encoding="utf-8")
        elif (
            not reference.exists() or reference.read_text(encoding="utf-8") != expected
        ):
            raise ContractError("native reference is stale; run with --write-reference")
    except (ContractError, KeyError, json.JSONDecodeError) as exc:
        print(f"native API contract check failed: {exc}")
        return 1
    counts = {
        kind: sum(d.kind == kind for d in declarations)
        for kind in ("c-function", "cpp-type", "cpp-operation")
    }
    print(
        "Native API contracts covered: "
        + ", ".join(f"{n} {k}" for k, n in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
