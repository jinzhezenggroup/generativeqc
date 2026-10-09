#!/usr/bin/env python3
"""Generate a native CLI catalog from compiler MethodIR, with strict admission.

Only build-time Python is needed. Representable methods without qualified native
lowerers are listed with reasons, never silently projected to a different energy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.dft._generated_native_semilocal import (
    SEMILOCAL_FAMILIES,
)
from generativeqc_compiler.method._generated_xc_aliases import (
    METHOD_ALIASES,
)
from generativeqc_compiler.method.ks_execution import (
    compile_ks_execution_plan,
)
from generativeqc_compiler.method.spec import (
    METHOD_CATALOG,
    resolve_method,
)
from generativeqc_compiler.xc._generated_split_hybrids import (
    SPLIT_HYBRIDS,
)


@dataclass(frozen=True)
class Row:
    name: str
    source: str
    identity: str
    spin: int
    domain: str
    components: tuple[tuple[str, float], ...]
    exchange: tuple[tuple[int, float, float], ...]
    cpu: bool
    cuda: bool
    reason: str


def _match_curated(plan: object) -> dict | None:
    actual = dict(plan.semilocal.functional.components)
    omega = plan.semilocal.functional.range_omega
    for record in SEMILOCAL_FAMILIES:
        if omega != Fraction(record["range_omega"]):
            continue
        expected = {name: Fraction(value) for name, value in record["components"]}
        if actual.keys() != expected.keys():
            continue
        if record["coefficient_policy"] == "native-scales":
            if any(value < 0 for value in actual.values()):
                continue
        elif actual != expected:
            continue
        return record
    return None


def _match_split(plan: object) -> bool:
    if plan.semilocal.functional.range_omega or len(plan.exchange) != 1:
        return False
    term = plan.exchange[0]
    if term.operator != "full-range" or term.omega:
        return False
    actual = dict(plan.semilocal.functional.components)
    return any(
        actual == {name: Fraction(value) for name, value in record["components"]}
        and term.coefficient == Fraction(record["exact_exchange"])
        for record in SPLIT_HYBRIDS.values()
    )


def _admit(plan: object) -> tuple[bool, bool, str, str]:
    if plan.method.basis is not None or plan.post_scf:
        return (
            False,
            False,
            "",
            "basis/dispersion/gCP correction is not natively composed",
        )
    if plan.nonlocal_correlation is not None:
        return (
            False,
            False,
            "",
            "nonlocal contribution needs a complete native composition",
        )
    if any(term.operator != "full-range" or term.omega for term in plan.exchange):
        return False, False, "", "range exchange needs a complete native composition"
    if len(plan.exchange) > 1:
        return False, False, "", "multiple exact-exchange terms are not supported"

    record = _match_curated(plan)
    if record is not None:
        if record["exchange_policy"] == "canonical":
            return (
                False,
                False,
                "",
                "canonical semilocal family requires its full method graph",
            )
        if plan.exchange and record["exchange_policy"] == "none":
            return False, False, "", "semilocal family has no exact-exchange lowerer"
        if any(value < 0 for _, value in plan.semilocal.functional.components):
            return False, False, "", "negative semilocal scaling is unavailable"
        cuda = bool(record["cuda_ks"])
        if plan.exchange:
            exact = record["cuda_global_hybrid_exact_exchange"]
            if exact is None or plan.exchange[0].coefficient != Fraction(exact):
                cuda = False
            if record["coefficient_policy"] == "native-scales":
                components = dict(plan.semilocal.functional.components)
                x_name, c_name = (entry[0] for entry in record["components"])
                if (
                    components[x_name] != 1 - plan.exchange[0].coefficient
                    or components[c_name] != 1
                ):
                    cuda = False
        return True, cuda, str(record["scf_domain"]), ""

    if _match_split(plan):
        return False, True, "libxc-7.0/split-global-hybrid-v1", ""
    return False, False, "", "no qualified native semilocal lowerer for this graph"


def _row(identifier: str, spin: str) -> Row:
    method = resolve_method(identifier, spin=spin)
    plan = compile_ks_execution_plan(method)
    cpu, cuda, domain, reason = _admit(plan)
    components = ()
    exchange = ()
    if cpu or cuda:
        components = tuple(
            (name, float(coefficient))
            for name, coefficient in plan.semilocal.functional.components
        )
        exchange = tuple(
            (1, float(term.coefficient), float(term.omega)) for term in plan.exchange
        )
    return Row(
        name=f"{identifier.lower()}-{'rks' if spin == 'unpolarized' else 'uks'}",
        source=method.identifier,
        identity=method.identity,
        spin=1 if spin == "unpolarized" else 2,
        domain=domain,
        components=components,
        exchange=exchange,
        cpu=cpu,
        cuda=cuda,
        reason=reason,
    )


def rows() -> tuple[Row, ...]:
    identifiers = {name.lower(): name for name in METHOD_CATALOG}
    for alias in sorted(METHOD_ALIASES):
        identifiers.setdefault(alias.lower(), alias)
    result: dict[str, Row] = {}
    for identifier in identifiers.values():
        for spin in ("unpolarized", "polarized"):
            item = _row(identifier, spin)
            if item.name in result:
                raise ValueError(f"duplicate MethodIR CLI selector {item.name!r}")
            result[item.name] = item
    return tuple(result[name] for name in sorted(result))


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def render(items: tuple[Row, ...]) -> str:
    components: list[tuple[str, float]] = []
    exchanges: list[tuple[int, float, float]] = []
    rows_cpp = []
    for row in items:
        offset_c, offset_e = len(components), len(exchanges)
        components.extend(row.components)
        exchanges.extend(row.exchange)
        carrier = (
            "GENERATIVEQC_METHOD_PBE_RKS"
            if row.spin == 1
            else "GENERATIVEQC_METHOD_PBE_UKS"
        )
        fields = (
            _q(row.name),
            _q(row.source),
            _q(row.identity),
            _q(row.domain),
            carrier,
            f"{row.spin}u",
            f"{offset_c}u",
            f"{len(row.components)}u",
            f"{offset_e}u",
            f"{len(row.exchange)}u",
            "true" if row.cpu else "false",
            "true" if row.cuda else "false",
            _q(row.reason),
        )
        rows_cpp.append("    {" + ", ".join(fields) + "},")
    output = [
        "// Generated by tools/generate_cli_method_catalog.py from canonical MethodIR.",
        "// Build-time only; the installed binary does not need Python or JSON.",
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        '#include "generativeqc/generativeqc.h"',
        "",
        "namespace generativeqc::cli::method_generated {",
        "struct Component { std::string_view name; double coefficient; };",
        "struct Exchange { std::int32_t kind; double coefficient; double omega; };",
        "struct Method {",
        "  std::string_view name, source, identity, domain;",
        "  generativeqc_method carrier;",
        "  std::uint32_t spin, component_offset, component_count;",
        "  std::uint32_t exchange_offset, exchange_count;",
        "  bool cpu, cuda;",
        "  std::string_view reason;",
        "};",
        f"inline constexpr std::array<Component, {len(components)}> kComponents{{{{",
    ]
    output.extend(
        f"    {{{_q(name)}, {coefficient!r}}}," for name, coefficient in components
    )
    output.extend(
        [
            "}};",
            f"inline constexpr std::array<Exchange, {len(exchanges)}> kExchanges{{{{",
        ]
    )
    output.extend(
        f"    {{{kind}, {coefficient!r}, {omega!r}}},"
        for kind, coefficient, omega in exchanges
    )
    output.extend(
        [
            "}};",
            f"inline constexpr std::array<Method, {len(items)}> kMethods{{{{",
            *rows_cpp,
            "}};",
            "inline constexpr const Method* find_method(std::string_view name) noexcept {",
            "  for (const auto& entry : kMethods)",
            "    if (entry.name == name) return &entry;",
            "  return nullptr;",
            "}",
            "}  // namespace generativeqc::cli::method_generated",
            "",
        ]
    )
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render(rows())
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
