# Native C++ SDK: explicit Kohn-Sham methods

The installed `libgenerativeqc` executes native DFT SCF without starting
Python. Python remains a build-time dependency for generated kernels, not a
requirement of the installed C/C++ runtime.

## Native manifest names

`generativeqc::resolve_method("pbe-rks")` and
`generativeqc::default_method_descriptor("pbe-rks")` use the
**generated native ABI registry**. Names such as `pbe0-rks`, `b3lyp-rks`
and `wb97m-v` are compiler-owned scientific compositions; they are not
native ABI IDs. Do not guess an ABI number for them.

## Complete named KS programs

Use `generativeqc::prepare_named_ks` for compiler catalog names. With an existing
`Context` and `System`, a complete corrected calculation is:

```cpp
auto controls = generativeqc::default_method_descriptor("pbe-rks");
auto calculation = generativeqc::prepare_named_ks(
    context, system, controls, "pbe0-d3(bj)-rks");
auto result = calculation.execute(GENERATIVEQC_PROPERTY_ENERGY);
```

The `pbe-rks`/`pbe-uks` descriptor supplies a stable spin transport carrier and
solver controls. The named program supplies every scientific component. Query
`Calculation::supported_properties()` before requesting forces. The helper also
accepts grid controls, nonlocal/dispersion byte budgets and an optional exact
MethodIR identity. It copies inputs during preparation.

C clients use `generativeqc_calculation_prepare_named_ks_v1` or
`generativeqc_batch_prepare_named_ks_v1` with a `generativeqc_named_ks_descriptor`.
The ordinary execution/destruction APIs operate on the returned handles.
Unknown graphs, identity/carrier mismatches and insufficient resources fail
without creating a live owner. Scientific fields in `descriptor.ks_options`
are supplied by the named graph; grid, scheduling, nonlocal capacity and supported
size-tagged execution extensions remain caller controls. A null `ks_options`
selects level-3 quadrature and a 256 MiB nonlocal budget. The explicit dispersion
budget bounds all retained correction owners; batches conservatively divide it
equally among systems. No Python interpreter, reference engine or runtime compiler
participates in this endpoint.

For a method with a defining basis, use the named system constructor first:

```cpp
generativeqc_named_ks_descriptor named{
    sizeof(named), GENERATIVEQC_ABI_VERSION, "r2scan-3c-rks", nullptr, 256ull << 20};
generativeqc::System system(context, atoms, named);
generativeqc::Calculation calculation(context, system, controls, named);
```

Its C equivalent is `generativeqc_system_create_named_ks_v1`. The system descriptor
supplies atoms, charge and multiplicity with empty shell/primitive arrays; the
method owns representation and basis data. Named calculation/batch preparation
also verifies the actual normalized orbital basis, including systems created
through other APIs. A matching name/checksum alone cannot bypass this check.
Currently r²SCAN-3c supplies the pinned H–Ar spherical def2-mTZVPP template and
complete CPU/CUDA energy (r²SCAN + special-profile D4 + gCP). The correction budget
reserves gCP storage before admitting D4. Native force admission follows the
prepared electronic context; CUDA evaluates gCP on device. See the
[r²SCAN-3c capability contract](r2scan3c.md) for the bounded supported domains.

## Explicit PBE0 energy in C++

```cpp
#include <array>
#include <iostream>
#include "generativeqc/ks.hpp"

int main() {
  generativeqc_context_descriptor ctx_desc{
      sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION, 0,
      GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc::Context context(ctx_desc);

  const std::array<generativeqc_atom, 2> atoms{{
      {1, 0.0, 0.0, -0.7}, {1, 0.0, 0.0, 0.7}}};
  const std::array<generativeqc_primitive, 6> primitives{{
      {3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423},
      {0.168855404, 0.4446345422}, {3.425250914, 0.1543289673},
      {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}};
  const std::array<generativeqc_shell, 2> shells{{
      {0, 0, 0, 3}, {1, 0, 3, 3}}};
  generativeqc_system_descriptor sys_desc{
      sizeof(generativeqc_system_descriptor), GENERATIVEQC_ABI_VERSION,
      atoms.data(), static_cast<std::uint32_t>(atoms.size()),
      shells.data(), static_cast<std::uint32_t>(shells.size()),
      primitives.data(), static_cast<std::uint32_t>(primitives.size()),
      0, 1, GENERATIVEQC_BASIS_CARTESIAN};
  generativeqc::System system(context, sys_desc);

  // Explicit MethodIR-equivalent PBE0: 75% PBE exchange,
  // 100% PBE correlation, 25% full-range exact exchange.
  generativeqc::KsComposition pbe0(
      GENERATIVEQC_METHOD_PBE_RKS, "semilocal-scaled-v1/pbe-spin-c2-1e-18", 1);
  pbe0.set_grid({1, 64, 12, 24, 3, 1.0e-12, 256})
      .add_semilocal("GGA_C_PBE", 1.0)
      .add_semilocal("GGA_X_PBE", 0.75)
      .add_exact_exchange(GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, 0.25);

  auto descriptor = generativeqc::default_method_descriptor("pbe-rks");
  descriptor.max_iterations = 200;
  descriptor.energy_tolerance = 1.0e-12;
  descriptor.density_tolerance = 1.0e-10;
  auto calculation = pbe0.prepare(context, system, descriptor);
  auto result = calculation.execute(GENERATIVEQC_PROPERTY_ENERGY);
  std::cout << result.energy << " Eh\n";
}
```

The C++ `KsComposition` owns its strings and exchange terms; its
`prepare()` call supplies an immutable snapshot to the native calculation.
Callers need not manage temporary C descriptor pointers. The example uses a
**small reference grid** for H₂, not a universal production quadrature.
Select and converge the grid for your real system.

A native client can also use the C ABI's
`generativeqc_method_from_name` / `generativeqc_method_get_name` to query
stable provider identities. Every native DFT request remains subject to
runtime validation of its physical composition, backend and basis.

## Scalar ECP systems

`System` has an overload accepting the system descriptor, an atom-major
`std::span<const std::int32_t>` of core counts and a
`std::span<const generativeqc_ecp_term>` of residual terms. It checks the core
array length, copies inputs through `generativeqc_system_create_ecp`, and keeps
the all-electron constructor's RAII lifetime and fail-closed native admission.
See [ECP conventions and capabilities](ecp.md).

## Density fitting and derivative boundary

For energy-only calculations, configure
`generativeqc_method_descriptor::density_fitting_mode` and optionally supply
`density_fitting_auxiliary_basis`, just as the native CLI does. CPU and CUDA
support depend on their qualified native provider and execution context.

The public native `Calculation::supported_properties()` reports capabilities
of the exact prepared context. The method-global registry does not grant a
blanket force bit. Both RKS and UKS, including an empty beta population, use
one token-verified stationary owner. The current named-program inventory is:

| Electronic program | CPU Direct/DF | CUDA Direct/DF |
| --- | --- | --- |
| LDA, PBE, PBE0, B3LYP, r²SCAN, WB97M-V | native forces | native forces with linked profiles |
| PBE50 | native forces | no CUDA SCF provider |
| M06-2X, MN15 | no CPU SCF provider | native forces with linked profiles |

Generated aliases and supported D3/D4 compositions use the same electronic
program and native correction owners. The canonical r²SCAN-3c graph includes
both D4 and gCP; see its [contract](r2scan3c.md). Imported CPU LDA/GGA/meta-GGA
programs with a validated first-order point evaluator also use the native
moving-grid force owner. Unsupported functional graphs remain unavailable.

Direct forces require packaged first derivatives and orbital s/p/d shells.
CPU builds may disable that inventory with
`GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF`, which revokes the prepared
capability. DF supports orbital grids through f and auxiliary integrals through
g in Cartesian or real-spherical representation. Integral through-g support
does not admit orbital g grids. Direct ECP forces reuse the checked residual
operator derivative and ionic nuclear charges; ECP+DF energy remains unavailable.

WB97M-V differentiates its full semilocal, range and VV10 model. DF preserves
the existing full-range fitted J/K plus independent exact LR convention;
explicit fitted LR uses the matching retained metric derivative. No source is
replaced with a different approximation. CUDA requires strict FP64,
device-fused XC and three partition iterations for this composite. Separate XC
and nonlocal grids are supported. General CUDA FP64 profiles require three
partition iterations; application/mixed-precision routes retain their separate
qualification rules.

Historical molecular-size ceilings (8 atoms, 64 AOs, 128 primitives, 8192 VV10
points), CPU Direct primitive/grid-visit ceilings, and stationary CUDA arena
atom/AO/primitive ceilings are removed. Dynamic storage and checked integer
products govern capacity. Per-page task limits, geometry scratch, explicit
source/nonlocal/device budgets, ECP quadrature convergence and actual operator
inventory still apply. D4's retained EEQ provider has its own fixed pivot
capacity; this electronic-force change does not enlarge that provider.
The CPU source-publication capacity is a numeric-buffer bound, not a complete
process RSS bound.

The force owner consumes the final occupation-weighted D and energy-weighted W.
Direct J/K share generated primitive derivatives. DF differentiates both
three-center integrals and its metric. CUDA consumes resident density pointers
on the owning provider streams, copies compact source vectors on device and
performs the complete reduction before publishing the force. Independent LR
providers retain their own stream/lifetime and synchronize at the source seam.

`generativeqc_batch_execute` requires every force-requested member to qualify;
selective execution can request forces for individual admitted members. Failed,
nonconverged or stale-token items cannot publish partial forces. The Python
frontend queries `generativeqc_ks_batch_supported_properties_v1` after checking
model identity. Conservative method metadata cannot override a successful
prepared capability query; missing native capability cannot enable an
unqualified fallback. Resource budgets still apply.

Independent PySCF energies, moving-grid analytic forces and final D/W, plus
reconverged finite differences, qualify the numerical paths. Native tests also
cover byte floors, failure cleanup, stale leases and ragged replay. A capability
query is an execution contract, not a promise of quadrature accuracy or
convergence for every molecule.


A CUDA DF context prepared with a NumericCap capacity promise retains its
energy path but does not advertise or execute forces: the derivative peak is
not included in that promise. Query the actual prepared properties; ordinary
DF force contexts without this promise retain their existing admission.
