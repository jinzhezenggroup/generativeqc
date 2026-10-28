# Decision: complete native meta-GGA forces and CUDA r²SCAN-3c

Status: implemented
Date: 2026-10-25

## Problem

The complete named KS owner could assemble common energies, but r²SCAN CPU
stationary derivatives, complete WB97M-V derivatives and CUDA gCP still required
separate Python consumers or explicit refusal. Energy admission alone cannot
promise complete CLI/C API forces.

This supersedes the force/CUDA exclusions in
[complete named KS ownership](2026-10-25-complete-named-ks-owners.md) and
[native basis-bound r²SCAN-3c](2026-10-25-native-bound-r2scan3c.md).

## Decision

Reuse audited scientific programs behind the native prepared owner:

- CPU moving-grid geometry consumes existing r²SCAN/WB97M-V point derivatives,
  including the kinetic-density Hessian contraction. Generated family metadata
  supplies CPU profile admission; Python only asks whether the actual prepared
  context has a native force provider.
- WB97M-V CPU range response packages the existing shared Cartesian derivative
  emitter at build time. Its omega comes from canonical MethodIR; ordered
  exchange cotangents are contracted in 256-quartet pages. The CPU VV10 owner
  composes AO, pair-point and partition-weight response on the same active domain.
- WB97M-V CUDA binds the complete MethodIR source inventory, including distinct
  SR/LR exchange and nonlocal AO/grid/weight slots. The existing fused RSH
  derivative provider and resident final-SCF rho/gradient feed device sources.
  The shared ordered reducer publishes one complete gradient. No dense D/W
  export or Python/host derivative composition is introduced.
- CUDA gCP shares the audited CPU pair equation and tables. One lane visits all
  partners in canonical pair orientation; half pair energies avoid double
  counting. A deterministic device reduction avoids atomics and quadratic
  storage. Energy-only requests omit radial derivative work.

Independent XC/nonlocal grids reuse one tiled AO/geometry owner. Pending
geometry drains before topology changes; initial V4 grid setup may precede AO
setup and therefore must not call a topology-validating drain without pending
work. Grid topology/radii and full nonlocal point extent are explicit.

## Bounds and lifetime

WB97M-V native forces require all-electron Direct, at most 8 atoms/64 AOs and
8192 nonlocal points. CPU also requires at most 128 primitives, angular momentum
through d, and the existing 2,000,000 primitive-record/100,000,000 grid-pair-visit
Direct gates. CUDA initially admits strict FP64 s/p, device-fused execution and
three partition iterations. The small CUDA r²SCAN s/p/d domain admits both
spins with at most 8 atoms/64 AOs/128 primitives. Larger compatibility contexts
retain their explicit previous consumer rather than widening admission by name.

CUDA gCP admits 1..4096 atoms: O(N) retained storage and O(N²) visits. Charge
its device buffers and host component receipts before allocating D4. Complete
force capacity includes VV10 effective fields, six seed rows, energy, error
slots and pair workspace; both the nonlocal sub-budget and whole-force
allowance remain hard gates. The pure VV10 capacity layout lives in an inline
shape-only header so CPU builds and isolated capacity probes need no CUDA or
public runtime dependency.

Final SCF token, owner, generation and stream dependencies remain authenticated.
Failed executions publish neither partial forces nor correction receipts. Every
gCP replay overwrites its device inputs and result/status storage. Energy-only,
failed-geometry and recovery replays must not leak an earlier derivative result.

## Rejected alternatives

- Numerical differentiation or Python postprocessing as the native force API:
  these would not provide the requested analytic, complete native endpoint.
- A second range, meta-GGA or gCP equation solely for this consumer: shared
  audited algebra makes both maintenance and scientific qualification clearer.
- Host gCP inside a CUDA named method: this would silently change the backend
  contract. Existing compatibility consumers may explicitly report CPU gCP.
- Promoting arbitrary DF/ECP/large composite contexts from a small-system gate:
  bounded capacity does not prove bounded work or complete derivative coverage.

## Evidence and oracle details

`test_native_meta_forces.py` checks RKS/UKS H2 and water, independent smaller,
equal and larger nonlocal product grids, native capability, prohibited Python
force fallback, failed replay and recovery. Independent PySCF 2.14/Libxc 7 gates
are 2e-8 Hartree and 3e-7 Hartree/Bohr; translation is 1e-10. Explicit pinned
STO-3G input coefficients are shared with the oracle, whose integrals/SCF/XC/VV10
and gradients remain independent. Named PySCF STO-3G rounds coefficients
differently and compares a different Hamiltonian at this tolerance.

V4 Treutler checks cover initial topology and switching XC/nonlocal grids. The
existing scoped independent-oracle adapter explicitly matches the documented
rho >= 1e-10 molecular VV10 domain. Unmodified CPU PySCF uses 1e-8; on the
coarse H2 grid this alone changes energy by 2.04e-8 Hartree. This is not grounds
to relax the numerical gate or change the production density policy.

`test_native_r2scan3c.py` uses independent PySCF, dftd4 4.3 and gCP references
with the exact defining basis and asserts native prepared force admission.
`test_gcp_r2scan3c.cpp` checks MB16-43/06 mixed elements, all 48 coordinates at
two difference steps (2e-8 gradient gate), CPU/device agreement, coincidence
failure, recovery and insufficient capacity. `test_cli_cuda_named.py` checks
complete native CLI energies/forces with Python absent from PATH, including
two-step reconverged energy differences. Existing source-order, compiled status,
ABI and compiler-boundary checks remain required.

CUDA qualification uses n1, CUDA 12.9, sm_120 RTX 5090, verified sccache 0.16.0,
and finite Slurm `main` / `gpu:5090:1` allocations. Transferred overlays are
SHA-256 verified; local logs/manifests/cache statistics are retained under
`.artifacts/native-cli/`. These are implementation-correctness checks, not
quadrature-convergence or endpoint-performance claims.

Final qualification on n1 used Slurm job 14253, assigned visible ordinal `3`,
with a 20-minute limit: 21 CUDA Python cases, all 14 compiled native profile
source/lifetime and actual-resource gates, the complete named CLI suite, and
all ten existing CLI stationary endpoints passed. Mixed-element gCP's largest
two-step difference error was 3.23e-12 Hartree/Bohr. New H2 r²SCAN-3c/WB97M-V
complete force difference errors were at most 4.99e-8 Hartree/Bohr. Local final
merged CPU/compiler checks passed 91 tests plus six subtests, in addition to
105 focused ownership/scalar-order tests and six native/CLI CTest suites.
Sphinx warnings-as-errors and the complete pre-commit hooks passed.
After merging the branch fixes and adding descriptor/prepared-grid capacity
parity, Slurm job 14254 (assigned ordinal `2`, ten-minute limit) passed all
22 CUDA Python cases. Distinct ragged-grid descriptor sizing fails explicitly
when this shape ABI lacks element topology; the prepared query remains exact.

The requested CUDA library, CLI and qualification targets build successfully.
An attempted all-target CUDA build additionally exposed existing MP2/UMP2
contract-test targets missing CUDA include paths (`cuda_runtime_api.h`);
that unrelated all-target build is not claimed as passing.

## Revisit when

Independent complete-endpoint evidence justifies larger/nonlocal/DF/ECP domains,
or an explicit global resource inventory covers the additional retained owners.
Preserve complete source coverage, same-token publication and independent oracles
when extending admission or changing scheduling.
