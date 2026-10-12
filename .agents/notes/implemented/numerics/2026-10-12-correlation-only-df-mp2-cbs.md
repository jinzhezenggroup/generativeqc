# Decision: correlation-only DF-MP2 for GPU DF-CBS E+F

Status: implemented
Date: 2026-10-12

## Problem

Legacy CUDA RI-MP2 has a DF-HF reference and no analytic-force consumer. The
native DF-CCSD(T) method instead uses conventional exact RHF. Combining these
would subtract different reference/correlation models in the small-basis
correction. Conventional CC force-domain guards also do not describe the
separate DF force owner.

## Decision

Add an explicit correlation-only CUDA mode, without changing existing ABI
layouts or the legacy selector. Reuse compiler-owned ordered MP2 algebra and
its automatic factor/spectrum reverse. A factors-only source demand avoids CC
core-block and amplitude producers for standalone MP2. The physical source
pullback and conventional RHF frame/nuclear response are shared with DF-CCSD(T).

CBS auxiliary identities bind the model. The small-basis group completes CC
Lambda/source response before borrowing that same reference/source for MP2.
CC blocks/amplitudes are retired; parent cotangents and optional preconditioner
remain charged while the child executes. There is no second SCF, metric or MO
source build. HF forces include nuclear repulsion, not just electronic response.

The correlation source and nuclear sink admit orbital-through-f and
auxiliary-through-g, including the g shells in cc-pVTZ-RI. This is a scoped
`correlation_df` capability, not an expansion of the generic CUDA DF exports or
legacy DF-HF. Both preflight and shell expansion share this consumer policy;
checking only preflight admits the model and then incorrectly rejects it during
native system construction.
The native system factory also had an orbital-through-f guard. A private
source-only auxiliary factory admits g without changing the public generic
factory or constructing a CPU reference context. Preparation rejects these
handles as orbitals and for generic DF-HF, including batch preparation.

## Rejected alternatives

- Reinterpret `density_fitting="cuda"`: changes an existing reference contract.
- Subtract legacy RI-MP2 from exact-reference DF-CCSD(T): inconsistent composite.
- Numerical forces or PySCF in production: violates the native analytic endpoint.
- Inherit the conventional 12/56-AO guards: unrelated to the DF owner.
- Claim arbitrary QM7/CBS scalability from three-index storage: generated MP2
  adjoint scratch is still `O(nocc² nvir²)` and the source is `O(nAO² naux)`.

## Invariants

Both correlation consumers use the same source identity, geometry, auxiliary,
metric threshold and conventional reference. E-only omits all reverse phases.
MP2 never needs CC amplitudes, CC/Lambda convergence or a full NMO^4 producer.
Repeated spectrum cotangents are summed for fixed-canonical audits. Molecular
response uses full oo/vv Fock cotangents from the noncanonical MP2 amplitude
equation, not diagonal epsilon seeds alone. Otherwise even nondegenerate
same-space rotation stationarity can fail; dividing frame cotangents by
same-space gaps is not a safe fix at degeneracy. Full Fock seeds replace, never
add to, diagonal spectrum seeds. MP2 diagnostics do not inherit CC
iterations/triples. Failures publish neither primary nor companion outputs.
Clearing vectors must release their capacity before subtracting reservations;
initializer-list assignment alone is not a deallocation.

## Evidence

`tests/python/test_df_mp2_equations.py` compares compiler energy and directional
factor/spectrum VJPs to independent scalar ordered-MP2 expressions.
`tests/python/test_df_mp2_native.py` compares complete molecular GPU E+F and
DF-CBS with independently reconverged PySCF correlation-only energies, including
degenerate methane orbitals, source reuse and atomic failure publication.
Real-GPU executions require finite-time Slurm allocations.

The integrated GPU suite passes all eight cases, including physical
cc-pVDZ/cc-pVTZ water DF-CBS, g auxiliary handle role rejection, same-source
companions, and native analytic derivatives against independently reconverged
energies. Source-only auxiliary handles are rejected both in single calculation
and batch preparation when supplied as orbitals or to legacy DF-HF MP2.

`benchmarks/results/df-cbs-qm7-20261012/` retains source-matched complete
QM7 ethane DF-CBS E+F (58/144 orbital AOs, 39.23 s) and a C7H16 DF-MP2 E+F
endpoint (178 AOs, 112.38 s). The maximum independent directional errors are
5.05e-9 and 7.98e-9 Eh/Bohr. The ethane endpoint qualifies the actual DF force
owner beyond conventional 56-AO admission; the C7H16 endpoint qualifies MP2,
not a full large-molecule CBS calculation. Planned capacities are not measured
allocator peaks, and these single endpoint observations are not speedup claims.

## Revisit when

Source/scratch admission or actual endpoint timing, rather than speculative
microbenchmarks, makes blocked MP2 response or a shared multi-RHS RHF response
worth its additional ownership and numerical qualification burden.
