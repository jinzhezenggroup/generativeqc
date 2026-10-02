# Decision: reuse shell-bounded Hermite storage for exact range exchange

Status: implemented by PR #1711; large-system performance qualification continues
Date: 2026-10-02

## Problem and evidence

The default through-f Cartesian source already knows all four shell angular
momenta, but its SR/LR branch calls `primitive_eri_cartesian<MaximumAngular>`.
That initializes six generic Hermite arrays for every primitive quartet,
including low-angular work. Each generic array has 4 x 6 x 10 scalar entries;
the six-array inventory is 1,440 doubles for values or 1,440 Dual3 values for
geometry derivatives (46,080 bytes before other recurrence storage).

A fresh Release/sm_120 build of master `b9c626d15`, CUDA 12.9, measured under
Slurm 12067 on RTX 5090, isolates this relevant endpoint cost. Six water atoms,
spherical def2-TZVP, diagnostic moving grid 24 x 8 x 16 per atom, direct FP64,
energy/density/screening thresholds 1e-11/1e-9/1e-12:

- Cold SCF: 31.991 s / 17 iterations; cold forces: 25.202 s, including 16.650 s
  force preparation/compilation.
- Warm complete SCF + analytic forces: 10.097 s / one iteration; forces: 8.207 s.
- Changed geometry: 29.884 s / 11 iterations.
- Separate fixed-final-density CUDA-event timing: J 259.198 ms, full K
  259.090 ms, long K 1,530.883 ms. J/K are replayed separately by this diagnostic;
  these times must not be added to estimate the fused physical SCF endpoint.
- Warm integral derivatives account for about 8.1 s; combined resident geometry
  and VV10 work drains in about 0.12 s. The asynchronous VV10 enqueue timer is
  not the VV10 device execution time.

The immutable baseline binary, raw component report, trace and ccache receipts
are retained locally under `.artifacts/wb97m-large/`. The baseline library was
copied before candidate compilation; its SHA-256 is retained alongside it.
These are diagnostic measurements at six atoms, not large-system acceptance or
GPU4PySCF parity evidence.

## Candidate

Extract the existing shell-bounded Hermite contraction into one generated
helper and let the Cartesian SR/LR source call it with the existing radial
identity. Full-range special low-order paths retain their selection; its
higher-order path calls the same extracted helper with Full/omega=0.

The new Hermite inventory is
`3 * ((a+1)*(b+1)*(a+b+2) + (c+1)*(d+1)*(c+d+2))` scalars: 12 for ssss, 54 for
ppss, and 768 for ffff. These are logical workspace/initialization counts,
not measured device traffic or register allocations. No ERI, shell quartet,
primitive quartet, radial evaluation or density contraction is removed.

## Invariants and retained fallback

- Use the same compiler-owned Hermite and Cartesian contraction mathematics,
  FP64 coefficients, range moments, omega, normalization and AD seeds.
- Keep direct positive SR interval evaluation; do not replace it with Full-LR.
- Keep geometry/density screening, provider/resource selection, sparse public
  projection, summation order and complete SCF/force convergence policies.
- Keep the public-AO generic source for constrained owner capacity and other
  existing fallback cases. No retained allocation or public option is added.
- Do not infer physical work counts from zero canonical counters: shell and
  canonical execution routes have different observers.

## Qualification

The host-executed generated-helper check covers all 55 canonical s/p/d/f shell
classes, six general component samples plus pure x/y/z-axis quartets, Full/SR/LR,
separate double values and all four Dual3 atom seeds: 25,245 comparisons against
the retained generic Cartesian evaluator. Pure-axis ffff reaches pair power six
and the terminal Hermite boundary read at t=7.
It validates the storage specialization, not independent scientific accuracy or
CUDA performance. Fourteen existing code-generation guards and compiler/SCF/
electronic-structure/CUDA-ownership checks also pass.

Before promotion, require real-device full/SR/LR matrices and derivatives against
independent CPU ERIs, independent GPU4PySCF energy/analytic-force gates, and
matched baseline/candidate complete cold/warm/changed-geometry timings. Include
a larger system, semantic quartet/radial counts and the bounded fallback gates.
Use the existing 1e-8 Eh / 1e-7 Eh/Bohr endpoint gates. The current six-atom
profile alone proves neither a candidate speedup nor the requested large-system
advantage.

## Subsequent device evidence

Slurm 12072 on node3 / RTX 5090 passed canonical full/SR/LR matrices, all-center
range derivatives, and three independent complete RKS TZVP/TZVPD and UKS NH2
WB97M-V tests, including reconverged directional energies. At six atoms the
candidate warm SCF plus analytic-force endpoint is 6.814 s versus baseline
10.097 s (one warm repeat each). GPU4PySCF is 6.798 s. All original/moved/fixed
force samples pass independent gates, with maximum energy/force errors
3.38e-12 Eh and 1.34e-9 Eh/Bohr. This is a 1.48x baseline improvement on the
diagnostic grid, not a large-system or GPU4PySCF advantage. Cold process clocks
share persistent compiler caches and do not compare clean-cache compilation.

The twelve-atom comparison initially found a 4.47635e-5 Eh/Bohr force discrepancy
from GPU4PySCF 1.8.1. Unmodified baseline and candidate agree within 1.17e-12.
Independent reconverged GPU4PySCF directional energies converge to the native
analytic result, while GPU4PySCF's analytic direction remains off by 6.32e-5.
CPU PySCF/Libcint independently reconverges both geometries and validates every
candidate sample to 4.78e-12 Eh / 9.08e-10 Eh/Bohr. The discrepancy belongs to
the original reference analytic path; its exact component is not isolated yet.
Do not weaken the acceptance gate or silently accept that reference force.
Raw inputs, node3 component reports and binary identities remain in the ignored
`.artifacts/wb97m-large/` checkout artifacts; the CPU/reference diagnosis is
retained under `.artifacts/lr-moments/` in the separate radial candidate checkout.

## Rejected alternatives and next decision

Enabling the opt-in bounded value provider is a separate scheduling change and
was previously measured slower; this patch does not re-enable it. Replacing the
64-point range quadrature with a new special-function evaluator would change
numerics and is deliberately a separate proposal. Assess this storage reduction
at the complete consumer before adding another optimization.
