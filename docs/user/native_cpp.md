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

## Density fitting and derivative boundary

For energy-only calculations, configure
`generativeqc_method_descriptor::density_fitting_mode` and optionally supply
`density_fitting_auxiliary_basis`, just as the native CLI does. CPU and CUDA
support depend on their qualified native provider and execution context.

The public native `Calculation::supported_properties()` reports
**context-qualified** DFT analytic forces for a narrow, all-electron,
strict-FP64 **CPU density-fitted PBE/PBE0 RKS** execution domain. The
method-global registry remains energy-only: exact source coefficients,
spin, backend, fitting provider, basis/ECP and correction terms determine
admission for the prepared calculation. Request
`GENERATIVEQC_PROPERTY_FORCES` only after querying actual prepared properties
(the property constant is `GENERATIVEQC_PROPERTY_FORCES`).

The same native, token-verified stationary-force consumer is used by
`generativeqc_batch_execute` when **every member** of the prepared batch
qualifies and a force output buffer is supplied. Mixed/unsupported native
DFT force contexts fail before execution; a malformed or nonconverged item
does not publish partial forces for that item. CPU Direct, CUDA, UKS,
range-separated, ECP, nonlocal and correction-bearing contexts remain
unqualified for native DFT forces, even when Python has a separately
supported DFT-force path.

For Python `PreparedBatch.execute(properties=("energy", "forces"))`,
the frontend now first queries each member's private native prepared-property
record (`generativeqc_ks_batch_supported_properties_v1`). If all members
qualify, it executes through `generativeqc_batch_execute` and copies the
native force buffers; no Python stationary gradient is run for that request.
Unsupported members and older native binaries retain their separately
qualified Python stationary-force path. A failed native capability query other
than explicit `NOT_IMPLEMENTED` is an error, not permission to fall back.
Capability is evaluated after the existing Python model/warm-state identity
checks and does not bypass resource budgets.

The CPU PBE0 force qualification must still satisfy the independent
reconverged finite-difference acceptance in #2151/#2222. A native
force-capability flag is not evidence of an independently validated
method/size/backend beyond its explicitly admitted context.
