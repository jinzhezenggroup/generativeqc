# Decision: reuse native scaled point workers for mixed global-hybrid response

Status: implemented (low-level CPU qualification; public admission unchanged)
Date: 2026-11-04

## Problem

The shared full-range molecular source owner could compose K with the native
B3 semilocal graph, but the generic interior PBE expression is not the native
SCF numerical domain. Native PBE uses scaled density/gradient coordinates,
stable tail arithmetic and a C2 spin continuation. Differentiating a different
interior expression would lose method identity precisely in difficult tails.
The native directional point worker already supports physical X/C weights,
but its Python transport previously admitted only unscaled XC without K.

## Decision

Add a narrow identity-bound semilocal point-partial provider contract to the
existing compiler grid owner. A provider returns unweighted energy density,
Cartesian feature potentials and their JVP. The compiler still owns bounded
tiling, field/output validation, grid-weight pullbacks and scientific identity;
it never imports the runtime. Keep the existing compiler-AD B3 and WB97M-V
providers unchanged. Do not admit native-point/nonlocal combinations yet.

The runtime binds the matching CPU LDA/PBE snapshot and calls its actual SCF
and directional point workers. Semilocal scales come from the authenticated
snapshot; full-range K is neither folded into nor removed by the XC worker.
The same composite J/K/XC response, frozen nuclear RHS, integral sources and
stationary executor handle PBE0 and custom fractions. No PBE0-specific
response or Hessian equation is added. No displaced SCF or external engine
enters production.

Expose scaled private CPU RKS/UKS response transports and migrate the Python
consumer directly. Retain the frozen unscaled C ABI with unit weights, since
that ABI is an explicit compatibility/integrity contract rather than a Python
import alias. Reject invalid scales before evaluation. Preserve the native
vacuum/empty-spin rejection instead of introducing a cutoff or regularizer.

## Rejected alternatives

- Merely whitelisting PBE in the interior scalar graph would differentiate a
  different tail/spin numerical model from native SCF.
- Mirroring native PBE algebra in another Python worker would create a second
  scientific owner and drift risk.
- Calling a reference engine in the production point provider would violate
  the native endpoint contract even if numerical tests agreed.
- Scaling a combined X+C response cannot express distinct X/C weights.

## Evidence and limits

Independent Libxc point energies, Cartesian potentials and multistep potential
differences cover both spin layouts and distinct X/C weights. Gates remain
2e-11 for energy/potentials and 2e-8 for Hessian actions. Identity mismatch,
insufficient native tile budget, undefined vacuum directions and closed owners
reject. Native tests retain 30 independent high-precision RKS directions and
48 UKS directions, plus scaled X/C linearity, input-domain and frozen-ABI tests.

Complete H2/HeH+ reference-energy Hessian tests also cover canonical PBE0 and
37/100 exchange with the pre-existing fixed raw-matrix and directional gates.
They match basis, moving quadrature, converged density and energy before the
derivative comparison. These are small all-s source-checkout fixtures, not Ni,
installed-endpoint, production-scale, angular or public admission evidence.
Native/host tests do not imply real-GPU execution. #2224 and #2843 still need
their component/state, angular, lifecycle and endpoint qualification gates.

## Revisit when

Other native numerical point domains or native-point/nonlocal combinations
provide equally authenticated derivative workers and independent molecular
qualification. Widen the provider domain only with those proofs.

## References

- Issues #2224 and #2843.
- [Full-range composition](2026-11-04-full-range-composite-second-order.md).
- [CPU coefficient provenance](2026-11-04-cpu-hybrid-coefficient-provenance.md).
- `docs/developer/dft_hessian.md`.
