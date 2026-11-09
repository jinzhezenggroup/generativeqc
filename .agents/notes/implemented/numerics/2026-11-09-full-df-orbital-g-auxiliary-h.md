# Decision: close full-DF CUDA orbital-g/auxiliary-h energies and forces

Status: implemented
Date: 2026-11-09

## Problem

Issue #3055 requires unmodified augmented heavy-atom QZ orbital and JKFIT
bases, not just removal of the first overlap preflight error. Orbital g and
auxiliary h must reach every value, spherical expansion, nuclear derivative,
stationary reference, correlation response and weighted force sink. Existing
auxiliary-h SCF values and opt-in orbital-g one-electron generation did not
establish that closure. This supersedes the QZ portion of the
[diagnostic-stage proposal](../../proposed/2026-11-09-full-df-augmented-cbs-closure.md).

## Decision

- Keep the generic CUDA orbital bound through f. A distinct `full_df` basis
  consumer describes the closed-shell CUDA full-DF MP2/DF-CCSD(T) dependencies:
  S/T/V and three-center orbital values/first derivatives through g, and
  auxiliary metric/three-center values/first derivatives through h. This
  reference uses RI-JK; it must not require unsupported conventional ERIs.
- Extend the existing compiler-owned Gaussian-moment DAG, not a second native
  recurrence. The explicit g/g/h value variant needs F0 through F13; first
  derivatives need F0 through F14. A derivative may raise one orbital center
  to h or one metric center to i internally. Neither is public h/i admission.
  Keep the lower-domain evaluators separately bounded and preserve their
  arithmetic. Dispatch must inspect both orbital angular indices.
- Generate the existing one-electron value and derivative family through g.
  The compact weighted gradient owner uses six expansion terms and an explicit
  private-view stride. Retain the three-term Direct-HF representation and its
  public ABI. Source/gradient resource planners account for the wider compact
  representation before allocation.
- Give Cartesian one-electron exports their own host packing mode through g.
  Unlike a DF-only value source, one-electron schedules still need shell-pair
  metadata. Skipping that state produced a misleading allocation refusal;
  widening a maximum alone would not have fixed it. Both single and batch
  exporters use the same packing mode.
- Retain the complete bounded generic polynomial source/weighted sink when
  optional raw Hermite reuse or shell-specialized derivative schedules do not
  cover g orbitals or h auxiliaries. Skip unsupported optional storage rather
  than swallowing actual invalid-input errors. High derivative selection must
  include orbital g even when the auxiliary basis contains only low-l shells.
- Keep nonzero-range high-l, unrestricted, exact-reference, DFT and higher
  derivative admission independent. Native method gates still reject orbital
  g without the complete full-DF reference. Metadata loading is not execution
  qualification. There is no production PySCF/reference-oracle fallback.

## Rejected alternatives

Widening the generic capability maximum would incorrectly promote conventional
ERI and unrelated consumers. Truncating auxiliary h or dropping diffuse/high-l
shells changes the Hamiltonian and basis identity. Extending the optional
Hermite cache or a specialized force schedule is unnecessary for correctness
when the complete generic consumer remains bounded; neither is assumed to
cover a domain merely because it can allocate storage. A force reconstructed
from production finite differences is not an analytic-force implementation.

## Invariants and qualification

The permanent allocated tests compare independent libcint Cartesian and
spherical raw g/g/h values, every center derivative and translations, including
diffuse/tight exponents and coincident centers. Generic S/T/V weighted sinks
are checked with nonsymmetric independent weights, four schedules, explicit
budget rejection and finite differences. A synthetic orbital-g/low-l-auxiliary
endpoint protects derivative policy selection independently of auxiliary h.

Molecular qualification uses unmodified spherical aug-QZ/aug-QZ-JKFIT for
all-electron CN- full-DF RHF/MP2 and a separately converged all-electron CH+
full-DF CCSD(T) endpoint with RHF/MP2 companions. Independent energies are
reconverged at the original and displaced geometries with PySCF/libcint;
directional force gates use both 1e-4 and 3e-5 Bohr. The component energy gate
is 5e-9 Hartree, force-slope gate 3e-6 Hartree/Bohr, and translation gate
3e-7 Hartree/Bohr. T/Q CBS retains separate DZ and TZ corrections, each
component E/F, actual HF/MP2 weights, root diagnostics and exact basis decimals.

The initial CH+ geometry with H at (0.3,-0.4,2.0) Bohr exposed a distinct
stationary native DZ HF root, despite a small SCF residual; QZ agreed with the
independent reference. The CBS gate correctly rejected that comparison rather
than relaxing tolerances. The retained test uses H at (0.3,-0.4,2.4) Bohr,
where the independently converged DZ/TZ/QZ roots are compared separately.
An SCF convergence flag is not a ground-state or root-identity guarantee.
The failed development comparison is retained with the qualification evidence.

Source/binary identities, exact input snapshots, component outputs, independent
slopes, complete endpoint timings, source/transform work, compiler-cache stats
and sampled device-process memory live in
`benchmarks/results/full-df-aug-qz-3055-20261109/`. GPU runs use finite Slurm
allocations on nodes other than the maintained local node3. NVML's 100 ms
process-memory sample includes CUDA/context/library overhead and may miss
short peaks; a native numeric capacity plan is not an observed memory peak.
The correctness build uses CUDA fast-compile settings and does not establish
release-build performance or a speedup claim.

## Consequences and revisit conditions

The high-l generic polynomial route trades throughput for explicit complete
coverage and bounded ownership. Optimize it only with full endpoint evidence,
semantic work counts and the same numerical gates, retaining a bounded fallback
where resident-storage assumptions fail. Heavy-atom aug-5Z remains unsupported:
external orbital h/auxiliary i need their own values, derivatives, response and
molecular qualification before any Q/5 claim. Hydrogen's lower angular maxima
must not be used to infer heavy-atom 5Z support.

## References

- Issue #3055; PR #3057.
- `docs/user/ccsdt_cbs.md` for current public behavior and the staged 5Z plan.
- `tests/python/test_full_df_cbs_native.py` for allocated independent endpoints.
- `docs/maintainer/performance_engineering.md` for complete endpoint evidence.

## Paired stationary staging reservation

The shared one-electron gradient packer retains six expansion slots even for
lower-angular shells. The Python stationary planner must reserve its native
91-byte per-AO host metadata allowance, not the legacy Direct-HF three-slot
58-byte allowance. Two host copies make the previous under-reservation 66 bytes
per AO, potentially allowing optional grid storage to consume bytes needed by
the paired integral provider. Keep all caller host/device caps unchanged.
The host-compiled parity probe now includes the real basis header instead of
copying only the obsolete three-slot constant. Boundary admission tests retain
exact-fit success and one-byte-short refusal at the corrected actual usage.

The dry capacity qualification tool independently reconstructs this layout and
must charge the same six slots. Its previous hard-coded three-slot oracle was
missed in the initial reservation repair. Keep that independent fail-closed
check rather than importing the production result as its own expected value.
The capacity-suite mutation now restores the obsolete three-slot reservation
and requires rejection; complete report and provider-admission tests cover
the same dependency chain without increasing the caller's byte limits.
