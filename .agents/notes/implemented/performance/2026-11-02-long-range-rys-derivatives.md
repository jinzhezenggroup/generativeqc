# Decision: Share range-separated geometry with Rys DF derivatives

Status: implemented and qualified on sampled A100 and RTX 5090 endpoints
Date: 2026-11-02

## Problem

After [raw-panel batching](2026-11-02-spectral-raw-panels.md), a fresh A100
diagnostic attributes 15.38 seconds of the 17.86-second fitted long-range force
response to three-center derivative contraction. The selector disabled Rys
whenever omega was nonzero, even for classes with an existing Rys program,
while retaining schedules selected by the production class policy.

## Decision

Extend the compiler-owned geometry emitter to accept an explicit range
quadrature. Both polynomial and Rys lowering use the same change of variable
`t = sqrt(theta) * u`, with `theta = omega^2 / (omega^2 + rho)`. Evaluate the
Rys rule at `theta*T`, multiply the geometry's `sx` and `sy` by theta, and
multiply the common prefactor by `sqrt(theta)`. The roots remain in u
coordinates: scaling them again would apply theta twice. The existing
`omega/hypot(omega,sqrt(rho))` evaluation avoids squaring a large omega.

The range-aware geometry is passed the launch's operator parameter. The
existing production class selector may then use Rys for either operator.
Unsupported classes, including auxiliary g, retain polynomial lowering.
Legacy policy remains a bounded rollback. The strict arithmetic policy and
the prohibition on mixed-precision long-range derivatives remain unchanged.
This changes neither the default exact-LR route nor the explicit fitted-LR
and shell selectors. It introduces no device-model-specific constants.

## Validation and provenance

The emitted host evaluator compares all 23 Rys classes with independent
libcint contracted/spherical derivatives for omega `1e-3`, `0.3`, `1.2`, and
`1e3`, at `8e-11` absolute/relative gates. Existing Coulomb, quadrature-boundary
and high-precision center-differentiation controls remain. The shell suite
passes 323 cases using the shared cached native compiler fixture; four
additional moment-contract cases also passed before the fixture migration.

The candidate source identity is
`c677c168ee521dca849488207785e4f96bbb3e2e1b05808359c160eb8df96b7d`.
OFF-library controls pass on A100 (LSF 224844) and RTX 5090 (Slurm 15691):
independent long-range shell/group/packet/tail derivatives through f, the 64
bounded spectral-panel controls, and 16 Python spectral/auxiliary-g molecular
cases. The 5090 job also reports zero compute-sanitizer errors on the long-range
shell controls. Additional legacy/auto Coulomb shell controls pass in Slurm
15693. Compiler ownership, native complexity and formatting checks pass.

Complete clean endpoint measurements are separate from diagnostic traces:

| GPU / job | #2932 cold / warm median (s) | Panels + Rys cold / warm median (s) |
| --- | --- | --- |
| A100-PCIE-40GB / LSF 224845 | 89.0669 / 25.1360 | 82.4840 / 17.0932 |
| A100-PCIE-40GB / LSF 224851, reversed order | 91.5580 / 25.3211 | 83.4969 / 17.2009 |
| RTX 5090 / Slurm 15692 | 282.0303 / 66.8804 | 269.8115 / 55.2717 |

Both arms use DF full-range J/K, fitted long-range exchange, shell derivatives,
96 atoms, def2-SVP/def2-SVP-JKFIT and README default grids. Each population has
one cold constructor/prepare/SCF/force/export call and three same-owner replays
from its own frozen cold density. Native work remains 28 cold Focks and one per
warm call; grids and VV10 work are identical between implementations. All
cold/warm sample pairs against the original agree within `1.38e-10` maximum
force-component error, with identical energies in these paired populations.

The same A100 job measures the candidate-code mixed route (DF full-range J/K,
exact long-range exchange) at 111.4524 / 14.5114 seconds, and stock GPU4PySCF DF
at 88.1772 / 29.4709 seconds. The mixed route is not a master checkout and is
not a fully Direct calculation. The fitted route's warm endpoint is still
17.8% slower than the mixed route, despite its 32.0% improvement over #2932.
All native/reference sample pairs pass the README observable gates; for the
fitted candidate the maximum errors across both A100 jobs are `8.31e-10`
hartree/atom and `2.37e-7`
hartree/bohr. These default-grid static-geometry timings do not establish a
grid-converged production ranking or a no-regression claim on untested GPUs.
The reversed-order job repeats the result: mixed 112.5843 / 14.5956 seconds,
GPU4PySCF 86.0701 / 29.7092 seconds, and a 32.1% fitted-route warm reduction.

A separate matched A100 trace (LSF 224856) compares panel-only and panels+Rys.
Long-range derivative contraction falls from 15.3987 to 8.1815 seconds, and
inclusive LR response falls from 17.8836 to 10.6279 seconds. Raw generation is
unchanged at approximately 0.82 seconds, 57 panels and 8,542,322,688 bytes.
The remaining shell time includes 4.5713 seconds for A<B bra classes (excluding
auxiliary g) and 1.8914 seconds for auxiliary g. These classes remain polynomial;
expanding Rys coverage is a separate, independently qualified follow-up.

The first diagnostic attempt (224846) failed its source-identity check before
running an endpoint: the frozen panel snapshot's recursive `build*` exclusion
omitted `src/api/build_identity.hpp.in`. The retry restored only that unchanged
inventory entry and verified the exact original `9e1d10...` source identity
before use; its repair receipt and the failed attempt are retained. This did
not affect either clean endpoint population or the qualified candidate source.

Development receipts live in ignored
`.artifacts/wb97mv-spectral-panels-20261102/`.

## Revisit when

Extend Rys to further angular classes only with independent derivative gates
and complete endpoint evidence. Do not replace the truncated metric adjoint
with a naive fitted-factor or pseudoinverse Gram derivative: discarded metric
directions remain part of the physical response.
