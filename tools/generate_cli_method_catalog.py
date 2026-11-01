"""Generate the shared native catalog from compiler MethodIR, with strict admission.

Only build-time Python is needed. Representable methods without qualified native
lowerers are listed with reasons, never silently projected to a different energy.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.dft._generated_native_semilocal import (
    SEMILOCAL_FAMILIES,
)
from generativeqc_compiler.method._generated_xc_aliases import (
    METHOD_ALIASES,
)
from generativeqc_compiler.method.bulk_ks import resolve_bulk_ks
from generativeqc_compiler.method.dispersion import (
    D3Spec,
    D4Spec,
    DispersionCorrectionPrimitive,
)
from generativeqc_compiler.method.ks_execution import (
    compile_ks_execution_plan,
)
from generativeqc_compiler.method.spec import (
    METHOD_CATALOG,
    d4_composite_method_identifiers,
    resolve_method,
)
from generativeqc_compiler.xc._generated_split_hybrids import (
    SPLIT_HYBRIDS,
)
from generativeqc_compiler.xc.automatic_semilocal import (
    automatic_composition_name,
    automatic_scf_domain,
)
from generativeqc_compiler.xc.spec import POINTWISE_BULK_COMPONENTS


@dataclass(frozen=True)
class Row:
    name: str
    source: str
    identity: str
    spin: int
    domain: str
    components: tuple[tuple[str, float], ...]
    exchange: tuple[tuple[int, float, float], ...]
    fock_coefficients: tuple[float, ...]
    cpu: bool
    cuda: bool
    reason: str
    range_omega: float = 0.0
    nonlocal_correlation: tuple[int, float, float, float] = (0, 0.0, 0.0, 0.0)
    dispersion: tuple[float | int, ...] = (0,) + (0.0,) * 16
    bound_basis: bool = False


def _match_curated(plan: Any) -> Any:
    actual = dict(plan.semilocal.functional.components)
    omega = plan.semilocal.functional.range_omega
    for record in SEMILOCAL_FAMILIES:
        if omega != Fraction(record["range_omega"]):
            continue
        expected = {name: Fraction(value) for name, value in record["components"]}
        if record["coefficient_policy"] == "native-scales":
            if actual.keys() != expected.keys():
                continue
            if any(value < 0 for value in actual.values()):
                continue
        elif actual != expected:
            continue
        return record
    return None


def _match_split(plan: Any) -> bool:
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


def _canonical_r2scan3c(plan: Any) -> bool:
    """Only the full pinned basis/D4/gCP graph may select its native owner."""
    canonical = resolve_method("R2SCAN-3c", spin=plan.method.spin)
    return (
        plan.method.basis == canonical.basis
        and plan.method.primitives == canonical.primitives
    )


def _admit(plan: Any) -> tuple[bool, bool, str, str]:
    if _canonical_r2scan3c(plan):
        record = _match_curated(plan)
        # Both correction terms have native owners on the selected backend.
        return True, bool(record["cuda_ks"]), str(record["scf_domain"]), ""
    if plan.method.basis is not None:
        return (
            False,
            False,
            "",
            "basis/dispersion/gCP correction is not natively composed",
        )
    if plan.post_scf and (
        len(plan.post_scf) != 1
        or not isinstance(plan.post_scf[0], DispersionCorrectionPrimitive)
        or not isinstance(plan.post_scf[0].specification, (D3Spec, D4Spec))
    ):
        return False, False, "", "no complete native post-SCF correction owner"
    record = _match_curated(plan)
    if record is not None:
        if record["exchange_policy"] == "canonical":
            canonical = compile_ks_execution_plan(
                resolve_method(record["canonical_method"], spin=plan.method.spin)
            )
            if (
                plan.semilocal.semantic_payload()
                != canonical.semilocal.semantic_payload()
                or plan.exchange != canonical.exchange
                or plan.nonlocal_correlation != canonical.nonlocal_correlation
            ):
                return (
                    False,
                    False,
                    "",
                    "canonical semilocal family requires its full method graph",
                )
            # The complete canonical graph is already lowered by the native KS
            # owner. Carry SR/LR and VV10 together; omitting either changes SCF.
            return (
                True,
                bool(record["cuda_nonlocal_correlation"]),
                str(record["scf_domain"]),
                "",
            )
        if plan.nonlocal_correlation is not None:
            return False, False, "", "no qualified complete native nonlocal composition"
        if any(term.operator != "full-range" or term.omega for term in plan.exchange):
            return False, False, "", "no qualified complete native range composition"
        if len(plan.exchange) > 1:
            return False, False, "", "multiple exact-exchange terms are not supported"
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
    if (
        not plan.nonlocal_correlation
        and not plan.post_scf
        and not plan.semilocal.functional.range_omega
        and len(plan.exchange) <= 1
        and all(
            term.operator == "full-range" and not term.omega for term in plan.exchange
        )
    ):
        name = automatic_composition_name(
            plan.semilocal.functional.components,
            plan.exchange[0].coefficient if plan.exchange else Fraction(0),
        )
        if name is not None:
            return True, True, automatic_scf_domain(name), ""
    return False, False, "", "no qualified native semilocal lowerer for this graph"


def _row(identifier: str, spin: str) -> Row:
    suffix = "rks" if spin == "unpolarized" else "uks"
    source = identifier
    cpu = cuda = False
    domain = identity = ""
    components: tuple[tuple[str, float], ...] = ()
    exchange: tuple[tuple[int, float, float], ...] = ()
    fock_coefficients: tuple[float, ...] = ()
    range_omega = 0.0
    nonlocal_correlation = (0, 0.0, 0.0, 0.0)
    dispersion = (0,) + (0.0,) * 16
    bound_basis = False
    try:
        method = resolve_method(identifier, spin=spin)
        source = method.identifier
        plan = compile_ks_execution_plan(method)
        cpu, cuda, domain, reason = _admit(plan)
        bound_basis = _canonical_r2scan3c(plan)
        # Evaluate provenance only on valid representations. Generated Libxc
        # inventories may contain legal metadata combinations without a
        # serializable native XC point domain. Those remain visible, disabled.
        identity = method.identity
        if cpu or cuda:
            components = tuple(
                (name, float(coefficient))
                for name, coefficient in plan.semilocal.functional.components
            )
            exchange = tuple(
                (
                    {"full-range": 1, "short-range": 2, "long-range": 3}[term.operator],
                    float(term.coefficient),
                    float(term.omega),
                )
                for term in plan.exchange
            )
            range_omega = float(plan.semilocal.functional.range_omega)
            if plan.nonlocal_correlation is not None:
                nlc = plan.nonlocal_correlation
                nonlocal_correlation = (
                    {"vv10": 1, "rvv10": 2}[nlc.spec.variant],
                    float(nlc.spec.b),
                    float(nlc.spec.c),
                    float(nlc.coefficient),
                )
            if plan.post_scf:
                spec = plan.post_scf[0].specification
                if (
                    isinstance(spec, D4Spec)
                    and spec.profile != "standard"
                    and not bound_basis
                ):
                    raise NotImplementedError(
                        "named D4 requires the standard EEQ profile"
                    )
                kind = (
                    3
                    if isinstance(spec, D4Spec)
                    else (1 if spec.damping == "bj" else 2)
                )
                fields = (
                    "s6",
                    "s8",
                    "s9",
                    "a1",
                    "a2",
                    "rs6",
                    "rs8",
                    "alp",
                    "cn_cutoff",
                    "pair_cutoff",
                    "atm_cutoff",
                    "pair_switch_width",
                    "atm_switch_width",
                    "ga",
                    "gc",
                    "charge_cn_cutoff",
                )
                dispersion = (kind,) + tuple(
                    float(getattr(spec, field, 0) or 0) for field in fields
                )
            fock_coefficients = tuple(
                float(term.fock_coefficient) for term in plan.exchange
            )
    except (TypeError, ValueError, NotImplementedError) as error:
        reason = f"MethodIR is not natively representable: {error}"
        cpu = cuda = False
        domain = identity = ""
        components = ()
        exchange = ()
    return Row(
        name=f"{identifier.lower()}-{suffix}",
        source=source,
        identity=identity,
        spin=1 if spin == "unpolarized" else 2,
        domain=domain,
        components=components,
        exchange=exchange,
        fock_coefficients=fock_coefficients,
        cpu=cpu,
        cuda=cuda,
        reason=reason,
        range_omega=range_omega,
        nonlocal_correlation=nonlocal_correlation,
        dispersion=dispersion,
        bound_basis=bound_basis,
    )


def rows() -> tuple[Row, ...]:
    identifiers = {name.lower(): name for name in METHOD_CATALOG}
    for alias in sorted(METHOD_ALIASES):
        identifiers.setdefault(alias.lower(), alias)
    for name in d4_composite_method_identifiers():
        identifiers.setdefault(name.lower(), name)
    result: dict[str, Row] = {}
    for identifier in identifiers.values():
        for spin in ("unpolarized", "polarized"):
            item = _row(identifier, spin)
            if item.name in result:
                raise ValueError(f"duplicate MethodIR CLI selector {item.name!r}")
            result[item.name] = item
            prefix = "libxc:" if spin == "unpolarized" else "libxc-uks:"
            result[prefix + identifier.lower()] = replace(
                item, name=prefix + identifier.lower()
            )
    # Atomic selectors share the same semantic specialization policy as recipes.
    for functional_name in sorted(POINTWISE_BULK_COMPONENTS):
        for spin in ("unpolarized", "polarized"):
            try:
                resolution = resolve_bulk_ks(functional_name, spin=spin, backend="cpu")
            except ValueError:
                continue
            prefix = "libxc:" if spin == "unpolarized" else "libxc-uks:"
            name = prefix + functional_name.lower()
            cpu, cuda, domain, reason = _admit(resolution.plan)
            result[name] = Row(
                name=name,
                source=resolution.method.identifier,
                identity=resolution.method.identity,
                spin=1 if spin == "unpolarized" else 2,
                # The installed catalog must carry the same numerical contract
                # as the point worker. Density-only LDA has separate RKS/UKS
                # layouts in its own domain; using the generic work domain
                # rejects CLI preparation and bypasses native Python selection.
                domain=domain,
                components=tuple(
                    (component, float(coefficient))
                    for component, coefficient in resolution.plan.semilocal.functional.components
                ),
                exchange=(),
                fock_coefficients=(),
                cpu=cpu,
                # The same native program represents both backends. Actual
                # CUDA preparation still requires a linked static provider and
                # applies its precision/shape/spin admission; catalog selection
                # must not fall back to Python lowering when a package omits it.
                cuda=cuda,
                reason=reason,
            )
    return tuple(result[name] for name in sorted(result))


def _q(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _bound_basis_cpp() -> list[str]:
    """Compile the existing audited snapshot, without another basis registry.

    The native template retains source shell/column order and raw decimal
    values. Its normalization is shared with every ordinary system constructor.
    """
    from generativeqc_compiler.method.basis_binding import validate_basis_snapshot

    # Reuse the canonical stdlib-only basis reader without importing the runtime
    # package's eager numerical exports. A private namespace preserves relative
    # imports without replacing or partially initializing ``generativeqc``.
    package_name = "_generativeqc_cli_codegen"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "python" / "generativeqc")]
        package.__package__ = package_name
        sys.modules[package_name] = package
    load_basis = importlib.import_module(f"{package_name}.basis").load_basis

    binding = resolve_method("R2SCAN-3c").basis
    basis = validate_basis_snapshot(
        binding,
        load_basis(ROOT / "python/generativeqc/data/r2scan3c/def2-mtzvpp-h-ar.json"),
    )
    elements, shells, exponents, coefficients = [], [], [], []
    for element in basis.elements:
        start = len(shells)
        for shell in element.shells:
            exponent_start = len(exponents)
            coefficient_start = len(coefficients)
            exponents.extend(float(value) for value in shell.exponents)
            for column in shell.coefficients:
                coefficients.extend(float(value) for value in column)
            shells.append(
                (
                    shell.angular_momentum,
                    exponent_start,
                    len(shell.exponents),
                    coefficient_start,
                    len(shell.coefficients),
                )
            )
        elements.append((element.atomic_number, 0, start, len(shells) - start, 0, 0))
    lines = [
        f"inline constexpr char kBoundBasisName[] = {_q(binding.name)};",
        f"inline constexpr char kBoundBasisIdentity[] = {_q(binding.basis_identity)};",
    ]
    for name, kind, records in (
        ("Elements", "generativeqc_basis_element_template_v1", elements),
        ("Shells", "generativeqc_basis_shell_template_v1", shells),
        ("Exponents", "double", exponents),
        ("Coefficients", "double", coefficients),
    ):
        lines.append(
            f"inline constexpr std::array<{kind}, {len(records)}> kBoundBasis{name}{{{{"
        )
        lines.extend(
            "    "
            + (
                "{" + ", ".join(map(str, record)) + "}"
                if isinstance(record, tuple)
                else repr(record)
            )
            + ","
            for record in records
        )
        lines.append("}};")
    lines.extend(
        [
            "inline constexpr generativeqc_basis_template_descriptor_v1 kBoundBasis{",
            "    sizeof(generativeqc_basis_template_descriptor_v1), GENERATIVEQC_ABI_VERSION,",
            "    kBoundBasisElements.data(), kBoundBasisElements.size(),",
            "    kBoundBasisShells.data(), kBoundBasisShells.size(),",
            "    kBoundBasisExponents.data(), kBoundBasisExponents.size(),",
            "    kBoundBasisCoefficients.data(), kBoundBasisCoefficients.size(), nullptr, 0};",
        ]
    )
    return lines


def render(items: tuple[Row, ...]) -> str:
    components: list[tuple[str, float]] = []
    exchanges: list[tuple[int, float, float, float]] = []
    rows_cpp = []
    for row in items:
        offset_c, offset_e = len(components), len(exchanges)
        components.extend(row.components)
        exchanges.extend(
            (*term, fock)
            for term, fock in zip(row.exchange, row.fock_coefficients, strict=True)
        )
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
            repr(row.range_omega),
            "{" + ", ".join(repr(value) for value in row.nonlocal_correlation) + "}",
            "{"
            + ", ".join(repr(value) for value in row.dispersion)
            + (", 1" if row.bound_basis else ", 0")
            + "}",
            "&kBoundBasis" if row.bound_basis else "nullptr",
            "kBoundBasisName" if row.bound_basis else '""',
            "kBoundBasisIdentity" if row.bound_basis else '""',
        )
        rows_cpp.append("    {" + ", ".join(fields) + "},")
    output = [
        "// Generated by tools/generate_cli_method_catalog.py from canonical MethodIR.",
        "// Build-time only; the installed binary does not need Python or JSON.",
        "#pragma once",
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        '#include "methods/named_ks_catalog.hpp"',
        "",
        "namespace generativeqc::methods::named_ks::generated {",
        *_bound_basis_cpp(),
        f"inline constexpr std::array<generativeqc_ks_semilocal_component, {len(components)}> kComponents{{{{",
    ]
    output.extend(
        f"    {{{_q(name)}, {coefficient!r}}}," for name, coefficient in components
    )
    output.extend(
        [
            "}};",
            f"inline constexpr std::array<generativeqc_ks_exchange_term, {len(exchanges)}> kExchanges{{{{",
        ]
    )
    output.extend(
        f"    {{{kind}, {coefficient!r}, {omega!r}, {fock!r}}},"
        for kind, coefficient, omega, fock in exchanges
    )
    output.extend(
        [
            "}};",
            f"inline constexpr std::array<Method, {len(items)}> kMethods{{{{",
            *rows_cpp,
            "}};",
            "inline constexpr const Method* find_method(std::string_view name) noexcept {",
            "  for (const auto& entry : kMethods)",
            "    if (entry.name == name ||",
            '        (name.starts_with("libxc-rks:") && entry.name.starts_with("libxc:") &&',
            "         name.substr(10) == entry.name.substr(6))) return &entry;",
            "  return nullptr;",
            "}",
            "}  // namespace generativeqc::methods::named_ks::generated",
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
