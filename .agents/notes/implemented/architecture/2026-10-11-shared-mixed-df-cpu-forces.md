# Decision: share mixed DF source semantics across CPU and CUDA

Status: implemented
Date: 2026-10-11

## Problem

The CUDA mixed-DF force qualification in PR #2246 does not alone qualify CPU.
The existing CPU stationary consumer evaluates exact SR and LR independently;
using those loops unchanged for a fitted primary would silently differentiate a
different Hamiltonian. Conversely, treating all fitted consumers as having no
quartet work would hide the exact LR correction's actual primitive traversal.

## Decision

Use one backend-neutral graph/provider admission predicate and one compiler-owned
external source inventory. Keep the old CUDA predicate and inventory spelling as
thin compatibility facades. Native prepared-provider identity validation, DF
primary response, physical exchange signs/spin factors and five-row source
recomposition share the same mixed bridge for CPU and CUDA.

CPU supplies only a compact, compiler-generated weighted LR derivative. The
stationary integral TensorIR uses the same same-spin exchange algebra, with
coefficient `a_long-a_short`; the existing generated range executor owns the LR
integrals and atom-center scatter. Native code resolves the primary/correction
Fock identities and composes the physical SR/LR rows. CUDA retains its resident
density and separate lazy LR derivative lease, stream event and device capacity
admission. CPU does not prepare that CUDA-only owner or invent a device reserve.

Keep complete CPU semilocal/nonlocal/grid/Pulay/nuclear assembly in the existing
stationary consumer. Mixed DF elides Direct H/J/full-K/SR loops but accounts for
one full ordered LR quartet pass. Host capacity includes one range-provider
family rather than both SR and LR. Existing finite semantic work limits still
reject before derivative compilation/execution. DF response resources remain
explicitly outside this partial additional-host inventory.

## Rejected alternatives

- Exact SR plus exact LR instead of fitted K: changes the energy Hamiltonian.
- A Python copy of native source recomposition: duplicates signs, spin factors
  and provider identity assumptions across backends.
- Promote CPU d/f by removing guards: the existing CPU range consumer/AOT and
  numeric inventory are qualified only for one-component s/p AOs. Generated
  mathematics alone is not complete-endpoint qualification.
- Move integral execution/allocation into compiler planning: violates compiler
  ownership and makes host-only generation depend on a native runtime.
- PySCF fallback: the independent oracle is test-only, never production work.

## Invariants

- Source total is fitted full-range exchange plus the exact weighted LR
  correction. The five rows have the same physical semantics on both backends.
- Value-side range provider, geometry, omega, screening, spin and DF identities
  must match. Every publication checks the live state before and after response.
- A zero correction still cannot recover independent SR/LR rows and fails closed.
- CPU retains its s/p, eight-atom, sixteen-AO and work-bounded domain. CUDA
  through-f qualification and additional lazy-LR reserve remain unchanged.
- No functional-name dispatch or reference-engine dependency is added.

## Evidence

`tests/python/test_mixed_rsh_df_forces.py` shares complete CPU/CUDA acceptance:
independently reconverged PySCF full-range DF J/K plus exact LR; both quadratures
and auxiliary centers rebuilt for every displacement. Gates are 2e-8 Hartree
energy, 3e-6 Hartree/Bohr forces, 2e-9 translation residual and 2e-8 warm/rebound
force agreement, with 2e-4 Bohr finite differences. CPU cases cover RKS H2,
UKS H3 in Cartesian/spherical bases, plus s/p spherical water.

The actual shared native bridge is also host-compiled with RKS/UKS providers,
CPU/CUDA backend mismatches, missing corrections, state revocation, cold/warm
CUDA reserves and bounded failures. Compiler weight tests independently check
the correction coefficient and spin convention. Metadata-only work tests prove
that exactly one LR pass is admitted for mixed DF. Raw source-matched validation
and compiler-cache evidence is retained in `.artifacts/wb97m-df-cpu/`.

Final CPU/host/compiler plus existing public DF qualification: **142 passed,
26 skipped** in a CPU-only Release build. Five complete CPU endpoints pass the
independent acceptance gates; CPU d/f cases stay outside admission. Shared-path
CUDA and existing public DF regressions: **45 passed, 23 deselected**, under a
finite twenty-minute Slurm allocation on n1 (`node1`, RTX 5090), including seven
complete mixed endpoints. The compiler ownership checker covers 513 modules
with no dependency errors. Native public contracts, provider-selection
boundaries, public Python docstrings, scoped Ruff/clang-format and diff checks
pass. Verified ccache launchers are present in both native build command streams;
cache statistics and source/patch checksums distinguish caching from numerical
qualification. No speedup or complete-force memory claim follows from this work.

## Revisit when

An independently qualified bounded CPU range-derivative consumer and inventory
cover multicomponent d/f orbitals, or a native backend-neutral derivative Fock
provider can replace the CPU generated-tile binding without changing semantics.
