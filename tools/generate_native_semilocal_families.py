"""Generate native curated semilocal metadata for C++ and Python consumers."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/native_semilocal_families.json"
CPP_OUTPUT = ROOT / "src/dft/semilocal_family.hpp"
PYTHON_OUTPUT = ROOT / "python/generativeqc_compiler/xc/_generated_native_semilocal.py"
SCHEMA = "generativeqc.native-semilocal-families.v3"
FAST_PATH_FIELDS = (
    "component_scaling",
    "mixed_ao_precision",
    "mixed_density_precision",
    "response",
    "graph_replay",
)
FAST_PATH_STATUS_CPP = {
    "unavailable": "CudaXcCapability::Unavailable",
    "qualification-required": "CudaXcCapability::QualificationRequired",
    "qualified": "CudaXcCapability::Qualified",
}


def load_manifest(path: Path = MANIFEST) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported native semilocal family manifest schema")
    raw = payload.get("families")
    if not isinstance(raw, list) or not raw:
        raise ValueError("native semilocal family manifest requires families")
    families = tuple(raw)
    codes = [item.get("code") for item in families]
    if codes != list(range(len(families))):
        raise ValueError("native semilocal family codes must be contiguous from zero")
    names = [item.get("name") for item in families]
    if len(set(names)) != len(names):
        raise ValueError("native semilocal family names must be unique")
    allowed_coefficients = {"exact", "native-scales"}
    allowed_exchange = {"none", "any", "canonical"}
    allowed_stationary_kernels = {"lda", "pbe", "r2scan", "composed", "wb97mv"}
    for item in families:
        fast_paths = item.get("cuda_fast_paths")
        if not isinstance(fast_paths, dict) or set(fast_paths) != set(FAST_PATH_FIELDS):
            raise ValueError(
                "native semilocal CUDA fast-path capability census is incomplete"
            )
        for field in FAST_PATH_FIELDS:
            if fast_paths[field] not in FAST_PATH_STATUS_CPP:
                raise ValueError(f"unsupported CUDA fast-path status for {field}")
        components = item.get("components")
        if not isinstance(components, list) or not 1 <= len(components) <= 4:
            raise ValueError("native semilocal family requires 1-4 components")
        if item.get("coefficient_policy") not in allowed_coefficients:
            raise ValueError("unsupported native semilocal coefficient policy")
        if item.get("exchange_policy") not in allowed_exchange:
            raise ValueError("unsupported native semilocal exchange policy")
        if item.get("stationary_kernel") not in allowed_stationary_kernels:
            raise ValueError("unsupported native semilocal stationary kernel")
        for field in (
            "cuda_ks",
            "requires_gradient",
            "requires_tau",
            "stationary_ecp_gradient",
            "native_range_exchange",
            "native_nonlocal_correlation",
            "cuda_nonlocal_correlation",
            "molecular_nonlocal_domain",
            "incremental_xc",
            "stationary_second_order",
        ):
            if type(item.get(field)) is not bool:
                raise TypeError(f"native semilocal {field} must be bool")
        if item["requires_tau"] and not item["requires_gradient"]:
            raise ValueError("tau-dependent native semilocal family requires gradients")
        Fraction(item["range_omega"])
        if item.get("cuda_global_hybrid_exact_exchange") is not None:
            fraction = Fraction(item["cuda_global_hybrid_exact_exchange"])
            if not 0 < fraction <= 1:
                raise ValueError(
                    "CUDA global-hybrid exchange fraction must lie in (0,1]"
                )
        if (
            item["cuda_nonlocal_correlation"]
            and not item["native_nonlocal_correlation"]
        ):
            raise ValueError("CUDA nonlocal admission requires native nonlocal support")
        if (
            item["molecular_nonlocal_domain"]
            and not item["native_nonlocal_correlation"]
        ):
            raise ValueError(
                "molecular nonlocal domain requires native nonlocal support"
            )
        for component in components:
            if not isinstance(component, list) or len(component) != 2:
                raise ValueError(
                    "native semilocal components require name/coefficient pairs"
                )
            if not isinstance(component[0], str) or not component[0]:
                raise ValueError("native semilocal component name must be nonempty")
            Fraction(component[1])
        if item["exchange_policy"] == "canonical" and not item.get("canonical_method"):
            raise ValueError("canonical exchange policy requires canonical_method")
    return families


def _cpp_number(value: str) -> str:
    return repr(float(Fraction(value)))


def cpp_fast_path_capabilities(item: dict[str, Any]) -> str:
    return ", ".join(
        FAST_PATH_STATUS_CPP[item["cuda_fast_paths"][field]]
        for field in FAST_PATH_FIELDS
    )


def emit_cpp(families: tuple[dict[str, Any], ...] | None = None) -> str:
    families = load_manifest() if families is None else families
    enum_rows = "\n".join(f"  {item['symbol']} = {item['code']}," for item in families)
    records = []
    for item in families:
        component_ids = [name for name, _ in item["components"]]
        coefficients = [coefficient for _, coefficient in item["components"]]
        ids = component_ids + [None] * (4 - len(component_ids))
        coeffs = coefficients + ["0"] * (4 - len(coefficients))
        ids_cpp = ", ".join(
            "nullptr" if name is None else json.dumps(name) for name in ids
        )
        coeffs_cpp = ", ".join(_cpp_number(value) for value in coeffs)
        records.append(
            "\n".join(
                (
                    f"    {{SemilocalFamily::{item['symbol']},",
                    f"     {json.dumps(item['name'])},",
                    f"     {json.dumps(item['scf_domain'])},",
                    f"     {item['domain_version']}U,",
                    f"     {'true' if item['cuda_ks'] else 'false'},",
                    f"     {'true' if item['requires_gradient'] else 'false'},",
                    f"     {'true' if item['requires_tau'] else 'false'},",
                    f"     {'true' if item['stationary_ecp_gradient'] else 'false'},",
                    "     "
                    + (
                        "-1.0"
                        if item["cuda_global_hybrid_exact_exchange"] is None
                        else _cpp_number(item["cuda_global_hybrid_exact_exchange"])
                    )
                    + ",",
                    f"     {'true' if item['native_range_exchange'] else 'false'},",
                    f"     {'true' if item['native_nonlocal_correlation'] else 'false'},",
                    f"     {'true' if item['cuda_nonlocal_correlation'] else 'false'},",
                    f"     {'true' if item['molecular_nonlocal_domain'] else 'false'},",
                    f"     {'true' if item['incremental_xc'] else 'false'},",
                    f"     {'true' if item['stationary_second_order'] else 'false'},",
                    f"     {{{cpp_fast_path_capabilities(item)}}},",
                    f"     {{{ids_cpp}}},",
                    f"     {{{coeffs_cpp}}},",
                    f"     {len(component_ids)}U,",
                    f"     {_cpp_number(item['range_omega'])},",
                    "     "
                    + (
                        "true"
                        if item["coefficient_policy"] == "native-scales"
                        else "false"
                    )
                    + "},",
                )
            )
        )
    record_text = "\n".join(records)
    return f"""#pragma once

// Generated by tools/generate_native_semilocal_families.py.
// Do not edit by hand; manifests/native_semilocal_families.json is canonical.

#include <array>
#include <cstdint>
#include <stdexcept>

#include "dft/xc_capabilities.hpp"

namespace generativeqc::dft {{

/** Native curated semilocal execution identity shared by CPU and CUDA KS.
 *
 * These stable codes are execution selectors for already-admitted native
 * implementations. They do not grant method/backend capability by themselves.
 */
enum class SemilocalFamily : std::uint32_t {{
{enum_rows}
}};

struct SemilocalFamilyMetadata {{
  SemilocalFamily family;
  const char* name;
  const char* scf_domain;
  std::uint32_t domain_version;
  bool cuda_ks;
  bool requires_gradient;
  bool requires_tau;
  bool stationary_ecp_gradient;
  double cuda_global_hybrid_exact_exchange;
  bool native_range_exchange;
  bool native_nonlocal_correlation;
  bool cuda_nonlocal_correlation;
  bool molecular_nonlocal_domain;
  bool incremental_xc;
  bool stationary_second_order;
  CudaXcFastPathCapabilities cuda_fast_paths;
  std::array<const char*, 4> component_ids;
  std::array<double, 4> component_coefficients;
  std::uint32_t component_count;
  double range_omega;
  bool component_coefficients_are_native_scales;
}};

// clang-format off
inline constexpr std::array<SemilocalFamilyMetadata, {len(families)}> kSemilocalFamilyMetadata{{{{
{record_text}
}}}};
// clang-format on

constexpr std::uint32_t semilocal_family_code(SemilocalFamily family) noexcept {{
  return static_cast<std::uint32_t>(family);
}}

constexpr const SemilocalFamilyMetadata* semilocal_family_metadata_from_code(
    std::uint32_t code) noexcept {{
  if (code >= kSemilocalFamilyMetadata.size()) return nullptr;
  const auto& metadata = kSemilocalFamilyMetadata[code];
  return semilocal_family_code(metadata.family) == code ? &metadata : nullptr;
}}

constexpr const SemilocalFamilyMetadata& semilocal_family_metadata(
    SemilocalFamily family) noexcept {{
  return kSemilocalFamilyMetadata[semilocal_family_code(family)];
}}

constexpr const char* semilocal_family_name(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).name;
}}

constexpr const char* semilocal_family_scf_domain(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).scf_domain;
}}

constexpr std::uint32_t semilocal_family_domain_version(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).domain_version;
}}

constexpr bool semilocal_family_has_cuda_ks(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).cuda_ks;
}}

constexpr bool semilocal_family_requires_gradient(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).requires_gradient;
}}

constexpr bool semilocal_family_requires_tau(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).requires_tau;
}}

constexpr bool semilocal_family_has_stationary_ecp_gradient(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).stationary_ecp_gradient;
}}

constexpr double semilocal_family_cuda_global_hybrid_exact_exchange(
    SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).cuda_global_hybrid_exact_exchange;
}}

constexpr bool semilocal_family_supports_range_exchange(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).native_range_exchange;
}}

constexpr bool semilocal_family_supports_nonlocal_correlation(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).native_nonlocal_correlation;
}}

constexpr bool semilocal_family_supports_cuda_nonlocal_correlation(
    SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).cuda_nonlocal_correlation;
}}

constexpr bool semilocal_family_uses_molecular_nonlocal_domain(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).molecular_nonlocal_domain;
}}

constexpr bool semilocal_family_supports_incremental_xc(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).incremental_xc;
}}

constexpr bool semilocal_family_supports_stationary_second_order(SemilocalFamily family) noexcept {{
  return semilocal_family_metadata(family).stationary_second_order;
}}

inline SemilocalFamily semilocal_family_from_code(std::uint32_t code) {{
  if (const auto* metadata = semilocal_family_metadata_from_code(code)) return metadata->family;
  throw std::invalid_argument("unknown native KS semilocal family code");
}}

static_assert(semilocal_family_code(SemilocalFamily::{families[0]["symbol"]}) == 0U);
static_assert(semilocal_family_code(SemilocalFamily::{families[-1]["symbol"]}) + 1U ==
              kSemilocalFamilyMetadata.size());

}}  // namespace generativeqc::dft
"""


def emit_python(families: tuple[dict[str, Any], ...] | None = None) -> str:
    families = load_manifest() if families is None else families
    rows = []
    for item in families:
        component_text = (
            "(\n"
            + "".join(
                f"            ({json.dumps(name)}, {json.dumps(coefficient)}),\n"
                for name, coefficient in item["components"]
            )
            + "        )"
        )
        fields = [
            f'        "symbol": {json.dumps(item["symbol"])},',
            f'        "code": {item["code"]},',
            f'        "name": {json.dumps(item["name"])},',
            f'        "scf_domain": {json.dumps(item["scf_domain"])},',
            f'        "domain_version": {item["domain_version"]},',
            f'        "cuda_ks": {bool(item["cuda_ks"])!r},',
            f'        "requires_gradient": {bool(item["requires_gradient"])!r},',
            f'        "requires_tau": {bool(item["requires_tau"])!r},',
            f'        "stationary_kernel": {json.dumps(item["stationary_kernel"])},',
            f'        "stationary_ecp_gradient": {bool(item["stationary_ecp_gradient"])!r},',
            '        "cuda_global_hybrid_exact_exchange": '
            + (
                "None"
                if item["cuda_global_hybrid_exact_exchange"] is None
                else json.dumps(item["cuda_global_hybrid_exact_exchange"])
            )
            + ",",
            f'        "native_range_exchange": {bool(item["native_range_exchange"])!r},',
            f'        "native_nonlocal_correlation": {bool(item["native_nonlocal_correlation"])!r},',
            f'        "cuda_nonlocal_correlation": {bool(item["cuda_nonlocal_correlation"])!r},',
            f'        "molecular_nonlocal_domain": {bool(item["molecular_nonlocal_domain"])!r},',
            f'        "incremental_xc": {bool(item["incremental_xc"])!r},',
            f'        "stationary_second_order": {bool(item["stationary_second_order"])!r},',
            '        "cuda_fast_paths": {',
            *(
                f'            "{field}": {json.dumps(item["cuda_fast_paths"][field])},'
                for field in FAST_PATH_FIELDS
            ),
            "        },",
            f'        "components": {component_text},',
            f'        "range_omega": {json.dumps(item["range_omega"])},',
            f'        "coefficient_policy": {json.dumps(item["coefficient_policy"])},',
            f'        "exchange_policy": {json.dumps(item["exchange_policy"])},',
        ]
        if item.get("canonical_method") is not None:
            fields.append(
                f'        "canonical_method": {json.dumps(item["canonical_method"])},'
            )
        rows.append("    {\n" + "\n".join(fields) + "\n    },")
    domains = {}
    for item in families:
        version = item["domain_version"]
        domain = item["scf_domain"]
        previous = domains.setdefault(version, domain)
        if previous != domain:
            raise ValueError("one native domain version cannot name two SCF domains")
    domain_text = (
        "{\n"
        + "".join(
            f"    {version}: {json.dumps(domain)},\n"
            for version, domain in sorted(domains.items())
        )
        + "}"
    )
    return (
        '"""Generated native curated semilocal metadata. Do not edit by hand."""\n\n'
        "# Generated by tools/generate_native_semilocal_families.py from\n"
        "# manifests/native_semilocal_families.json.\n\n"
        "from typing import TypedDict\n\n\n"
        "class _SemilocalFamilyFields(TypedDict):\n"
        '    """Validated fields shared by all curated native semilocal families."""\n\n'
        "    symbol: str\n"
        "    code: int\n"
        "    name: str\n"
        "    scf_domain: str\n"
        "    domain_version: int\n"
        "    cuda_ks: bool\n"
        "    requires_gradient: bool\n"
        "    requires_tau: bool\n"
        "    stationary_kernel: str\n"
        "    stationary_ecp_gradient: bool\n"
        "    cuda_global_hybrid_exact_exchange: str | None\n"
        "    native_range_exchange: bool\n"
        "    native_nonlocal_correlation: bool\n"
        "    cuda_nonlocal_correlation: bool\n"
        "    molecular_nonlocal_domain: bool\n"
        "    incremental_xc: bool\n"
        "    stationary_second_order: bool\n"
        "    cuda_fast_paths: dict[str, str]\n"
        "    components: tuple[tuple[str, str], ...]\n"
        "    range_omega: str\n"
        "    coefficient_policy: str\n"
        "    exchange_policy: str\n\n\n"
        "class _SemilocalFamilyRecord(_SemilocalFamilyFields, total=False):\n"
        '    """Canonical method binding is present only for canonical exchange policy."""\n\n'
        "    canonical_method: str\n\n\n"
        "SEMILOCAL_FAMILIES: tuple[_SemilocalFamilyRecord, ...] = (\n"
        + "\n".join(rows)
        + "\n)\n\n"
        + 'SEMILOCAL_FAMILY_CODES = frozenset(item["code"] for item in SEMILOCAL_FAMILIES)\n'
        + 'SEMILOCAL_FAMILY_BY_CODE = {item["code"]: item for item in SEMILOCAL_FAMILIES}\n'
        + f"SCF_DOMAIN_BY_VERSION = {domain_text}\n"
    )


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cpp = emit_cpp()
    python = emit_python()
    if args.check:
        stale = []
        if CPP_OUTPUT.read_text(encoding="utf-8") != cpp:
            stale.append(str(CPP_OUTPUT.relative_to(ROOT)))
        if PYTHON_OUTPUT.read_text(encoding="utf-8") != python:
            stale.append(str(PYTHON_OUTPUT.relative_to(ROOT)))
        if stale:
            raise SystemExit(
                "stale generated native semilocal metadata: " + ", ".join(stale)
            )
        return 0
    _write(CPP_OUTPUT, cpp)
    _write(PYTHON_OUTPUT, python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
