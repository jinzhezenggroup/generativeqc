# Decision: adopt native AO reuse on the user's accepted cold point estimate

Status: implemented
Date: 2026-10-08

## Problem

The default-off candidate in
[the frozen-master root-cause audit](2026-10-08-pbe0-ao-reuse-and-csr-admission.md)
passes independent numerical gates and removes duplicated primitive/axis work.
Its six-pair complete E/F median improves from 115.437826 to 111.128942 seconds
(3.732645%), but the 95% bootstrap gain interval crosses zero. That audit
therefore initially declines default promotion.

## Decision and acceptance basis

The user explicitly accepts the observed 3.73% point estimate and asks to adopt
the optimization. Change the native CMake default to
`GENERATIVEQC_CUDA_AO_RADIAL_REUSE=ON`; retain explicit `OFF` for the original
scalar schedule. This is a user-approved acceptance decision, **not a claim
that the original positive-confidence-bound gate passed**. The interval remains
[-3.116631%, +13.435876%], with every original observation and Fock history
retained. GPU4PySCF parity is not achieved.

Native SCF and force-grid collocation share the generated four/ten-jet family.
One/twenty-jet fallback, the Python compiler/JIT default and explicit artifact
identity selectors remain unchanged. Production force producer selection,
cutoff, crossover, grid, screening and solver settings are unchanged. The
forced CSR/tile-256 experiment is not adopted with this decision.

Keep the separate CSR admission correctness fix: optional map storage must not
erase the native derivative allowance. Its dense-fallback recovery is not
claimed as a performance gain.

## Invariants

- Reuse only identical primitive/axis DAG values; preserve mathematical roots,
  Cartesian/spherical projection and accumulation order.
- Do not infer a constant cold gain, same-density SCF speed ratio or parity from
  a noisy median. Retain complete histories and independent E/F acceptance.
- Explicitly disabled native builds and scalar/JIT generation remain available;
  source identities must not alias the reusable schedule to scalar output.
- No dense-discovery cutoff or resource guard is weakened to claim a gain.

## Evidence and consequences

The prior candidate's native AO fixtures, sanitizers, 6/48-atom E/F gates and
twelve primary 96-atom observations are retained in the linked audit. Matched
AO NCU duration is 910.144 versus 256.736 us, with fewer executed instructions
despite lower achieved occupancy. Six-pair force medians are 22.799250 versus
20.111963 seconds. None of these observations changes the complete cold
confidence interval or proves benefits for every method/backend.

A fresh default build must omit the AO option override and verify CMake resolves
it to `ON`, then exercise the native AO fixtures and public E/F endpoint. Reuse
the compiler cache, preserve all frozen baseline/candidate sources, and keep
the default-source smoke sample outside the original statistical population.

### Fresh default-source validation

Slurm 6640 performs the fresh Release/SM120 build without an AO option override
and confirms `GENERATIVEQC_CUDA_AO_RADIAL_REUSE:BOOL=ON` in CMakeCache. The
generated grid source SHA256 is
`c7ac46319552ac5971414e0b5a556348eda90f933b5a1c5af6c26b447248379a`, identical
to the qualified opt-in candidate's generated grid bytes. The fresh native
library SHA256 is
`41c37449059bd72734b3eab587892aeb07a06392aac5aeb4fab77c6ad863d43c`.
The 72 actual-native AO checks pass with maximum error 1.705303e-13; memcheck
and synccheck each report zero errors. Compiler cache version/stats and build
commands remain retained.

The first endpoint wrapper mistakenly requested a nonexistent six-atom
reference filename. Preserve that failed harness observation; do not treat it
as a failed scientific gate or a timing sample. Slurm 6642 corrects only the
reference paths and SHA-verifies the same unrecompiled native library. Complete
independent 6/48/96-atom E/F gates all pass. Maximum energy/force errors across
these gates are 6.139090e-12 Eh / 2.493839e-11 Eh/bohr. The single 96-atom smoke
has 18 Focks, 111.665615 s complete and 18.801898 s force; it remains separate
from the original six-pair cold profit population.

Final host coverage is 294 passed with one opt-in NVCC skip; native compilation
and actual-device coverage above exercise the new default. Compiler structure
checks 494 modules with zero dependency errors; Ruff and diff checks pass.

Revisit selection if independently accepted workloads regress or resource
pressure makes reuse inappropriate. Further work must address the much larger
SCF J/K and force-domain costs; the accepted 3.73% does not close that gap.
