"""CUDA exact-shell first-derivative package inventory and source emission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

from .derivative_aot_registry import (
    AOT_DIRECT_CONTRACTION_CONTRACT,
    AOT_DIRECT_OUTPUT_CONTRACT,
    AOT_SPIN_CONTRACT,
    DerivativeAotPackageKey,
    DerivativeShellAotKey,
    make_shell_key,
)
from .ir import KernelConsumer
from .range_separation import CoulombKernel, CoulombKernelFamily

if TYPE_CHECKING:
    from .production_profile import ResolvedProductionProfile


@dataclass(frozen=True, slots=True)
class CudaDerivativeShellPackage:
    """One target-scoped exact-shell first-derivative package."""

    key: DerivativeShellAotKey
    target: str
    route: str

    @property
    def package_key(self) -> DerivativeAotPackageKey:
        return DerivativeAotPackageKey(
            backend=self.key.backend,
            target=self.target,
            scientific_identity=self.key.identity,
            output_contract=AOT_DIRECT_OUTPUT_CONTRACT,
            spin_contract=AOT_SPIN_CONTRACT,
            contraction_contract=AOT_DIRECT_CONTRACTION_CONTRACT,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "scientific_identity": self.key.identity,
            "scientific_identity_payload": self.key.to_payload(),
            "package_identity": self.package_key.identity,
            "package_identity_payload": self.package_key.to_payload(),
            "target": self.target,
            "route": self.route,
        }


def cuda_derivative_shell_packages(
    profile: ResolvedProductionProfile,
    radials: tuple[CoulombKernel, ...],
) -> tuple[CudaDerivativeShellPackage, ...]:
    """Bind one accepted Direct force profile to full/SR/LR derivative identities."""

    if profile.portable:
        return ()
    if any(
        radial.family
        not in (CoulombKernelFamily.SHORT_RANGE, CoulombKernelFamily.LONG_RANGE)
        for radial in radials
    ):
        raise ValueError("CUDA range shell packages require explicit SR/LR radials")
    full = CoulombKernel(CoulombKernelFamily.FULL_RANGE, 0.0)
    packages: list[CudaDerivativeShellPackage] = []
    for selection in profile.selections:
        if KernelConsumer.FORCE not in selection.consumers:
            continue
        for radial in (full, *radials):
            key = make_shell_key(radial, selection.spec, backend="cuda")
            packages.append(
                CudaDerivativeShellPackage(
                    key=key,
                    target=profile.target.architecture,
                    route=(
                        "generated-full-force"
                        if radial.family == CoulombKernelFamily.FULL_RANGE
                        else "bounded-range-shell-aot"
                    ),
                )
            )
    return tuple(
        sorted(
            packages,
            key=lambda item: (
                item.key.shell_class,
                CoulombKernelFamily(item.key.radial.family).value,
                item.key.radial.omega,
            ),
        )
    )


def package_inventory_identity(
    packages: tuple[CudaDerivativeShellPackage, ...],
) -> str:
    """Return one stable digest for the complete generated package inventory."""

    return canonical_hash([package.to_payload() for package in packages])


def _range_tag(radial: CoulombKernel) -> str:
    return {
        CoulombKernelFamily.SHORT_RANGE: "Short",
        CoulombKernelFamily.LONG_RANGE: "Long",
    }[CoulombKernelFamily(radial.family)]


def emit_cuda_derivative_shell_aot_header(
    profile: ResolvedProductionProfile,
    radials: tuple[CoulombKernel, ...],
) -> str:
    """Emit target-gated exact-shell SR/LR dispatch plus full package metadata."""

    packages = cuda_derivative_shell_packages(profile, radials)
    if not packages:
        raise ValueError("CUDA derivative shell AOT requires a non-portable force profile")
    target_arch = profile.target.architecture
    target_code = (
        profile.target.compute_capability_major * 100
        + profile.target.compute_capability_minor * 10
    )
    rows = []
    for package in packages:
        radial = package.key.radial
        rows.append(
            "    {"
            f'"{package.key.shell_name}", {package.key.shell_class}U, '
            f'"{CoulombKernelFamily(radial.family).value}", {radial.omega:.17g}, '
            f'"{package.key.identity}", "{package.package_key.identity}", '
            f'"{package.route}"'
            "},"
        )

    range_packages = tuple(
        package
        for package in packages
        if package.key.radial.family != CoulombKernelFamily.FULL_RANGE
    )
    by_radial: dict[
        tuple[CoulombKernelFamily, float], list[CudaDerivativeShellPackage]
    ] = {}
    for package in range_packages:
        radial = package.key.radial
        by_radial.setdefault(
            (CoulombKernelFamily(radial.family), radial.omega), []
        ).append(package)
    radial_blocks = []
    operator_names = {
        CoulombKernelFamily.SHORT_RANGE: "Short",
        CoulombKernelFamily.LONG_RANGE: "Long",
    }
    for (family, omega), items in sorted(
        by_radial.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        omega_milli = round(omega * 1000)
        if abs(omega - omega_milli / 1000.0) > 1e-15:
            raise ValueError("CUDA shell AOT currently requires millibohr omega identity")
        cases = []
        for package in sorted(items, key=lambda item: item.key.shell_class):
            shell_class = package.key.shell_class
            cases.append(
                f"""      case {shell_class}U:
        contract_two_electron_force_quartet_subtile_range_shell_aot_scaled<
            Unrestricted, {shell_class}U,
            generativeqc::integrals::CoulombRange::{_range_tag(package.key.radial)},
            {omega_milli}>(
            batch, task_count, task, screening_tolerance, schwarz_bounds, density, active, forces,
            exchange_coefficient, subtile, lane);
        return true;"""
            )
        radial_blocks.append(
            f"""  if (radial_operator == DirectRangeOperator::{operator_names[family]} &&
      omega == {omega:.17g}) {{
    switch (shell_class) {{
{chr(10).join(cases)}
      default:
        return false;
    }}
  }}"""
        )

    inventory_id = package_inventory_identity(packages)
    return f"""#pragma once

#include <cstddef>
#include <cstdint>

#include "scf/cuda/direct_bounded_fallback.hpp"
#include "scf/cuda/direct_force_quartet.cuh"

namespace generativeqc::scf::cuda_execution {{

struct DerivativeShellAotMetadata {{
  const char* shell_name;
  unsigned shell_class;
  const char* radial_family;
  double omega;
  const char* scientific_identity;
  const char* package_identity;
  const char* route;
}};

inline constexpr const char* kDerivativeShellAotTarget = "{target_arch}";
inline constexpr const char* kDerivativeShellAotInventoryIdentity = "{inventory_id}";
inline constexpr DerivativeShellAotMetadata kDerivativeShellAotInventory[] = {{
{chr(10).join(rows)}
}};
inline constexpr std::size_t kDerivativeShellAotInventoryCount =
    sizeof(kDerivativeShellAotInventory) / sizeof(kDerivativeShellAotInventory[0]);

template <bool Unrestricted>
__device__ inline bool contract_packaged_derivative_shell_aot(
    unsigned shell_class, DirectRangeOperator radial_operator, double omega, DeviceBatch batch,
    const std::uint32_t* task_count, const ActiveShellQuartetTile* task,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, double exchange_coefficient, std::size_t subtile,
    unsigned lane) {{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == {target_code}
{chr(10).join(radial_blocks)}
#else
  (void)shell_class;
  (void)radial_operator;
  (void)omega;
  (void)batch;
  (void)task_count;
  (void)task;
  (void)screening_tolerance;
  (void)schwarz_bounds;
  (void)density;
  (void)active;
  (void)forces;
  (void)exchange_coefficient;
  (void)subtile;
  (void)lane;
#endif
  return false;
}}

}}  // namespace generativeqc::scf::cuda_execution
"""


def package_inventory_payload(
    profile: ResolvedProductionProfile,
    radials: tuple[CoulombKernel, ...],
) -> dict[str, object]:
    packages = cuda_derivative_shell_packages(profile, radials)
    return {
        "target": profile.target.architecture,
        "profile": profile.profile,
        "inventory_identity": package_inventory_identity(packages),
        "packages": [package.to_payload() for package in packages],
    }
