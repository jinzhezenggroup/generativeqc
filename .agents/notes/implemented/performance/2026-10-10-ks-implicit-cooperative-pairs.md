# Decision: implicit triangular addressing for cooperative one-electron response

Status: implemented as an explicitly selected composed path; Hcore alone is not promoted
Date: 2026-10-10

## Problem

The prepared stationary KS bridge hard-codes component lanes because its
Direct owner has no triangular AO-pair index list. The historical generated
nuclear-center schedule assigns one AO pair per warp and is already qualified
for RHF/UHF. A shell-owned adapter avoids that list but serializes component
pairs within a warp; its five-repeat 48-atom complete endpoints fail the 2%
gate. See `../../rejected/2026-10-10-ks-shell-nucleus-cooperative.md`.

## Decision

Preserve one AO pair per warp without storing its indices. Decode the complete
lower-triangular ordinal from the known AO count. Floating-point square root
only estimates the first integer index; exact triangular inequalities correct
the estimate before any basis access. The second index is an integer remainder.
The existing normalized AO-to-shell metadata then supplies the scientific
inputs to unchanged `contract_pair_nucleus_cooperative`.

Both production Direct packers enumerate complete per-system shell triangles:
`src/scf/cuda/topology.cpp` and `src/scf/cuda/rhf_bucket.cpp`. Scientific
screening is a later integral-consumer decision, not a hole in this one-electron
metadata. Consequently the implicit AO triangle preserves the old component
kernel's mathematical domain and calls, rather than bypassing an admitted
one-electron screen. The standalone bridge has the same full-matrix contract.

Schedule 3 with two null AO-list pointers and zero pair count selects the new
adapter. Explicit AO-list schedule 3 remains unchanged; partial lists reject.
Validate the atom-offset/AO-shell prerequisites, matrix products and finite
grid dimension before submitting. The prepared bridge honors the existing
cooperative selector for Hcore, retains component lanes for overlap-only
Pulay, and keeps explicit `shell_warp` as the previous complete control.
Standalone cooperative callers no longer pack either pair list, but keep
their existing conservative resource admission bound.

## Invariants

No new scientific algebra, derivative DAG, precision, screen, threshold,
public ABI or production reference dependency. No retained allocation, upload,
prefix queue or shell scan. Nuclear ownership and all original warp collectives
remain in the existing generated primitive consumer. Inactive systems and
tail warp tasks skip uniformly. CUDA status is observed without clearing it.
Matrix allocation and execution grids remain bounded; source hits and compiler
caches are reused rather than disabled or cleared.

## Evidence and qualification

The actual decoder/kernel body is host-compiled, with exhaustive small AO
triangles and all 32 lanes at batch sizes one/three, inactive neighbors and
tail warps. Integer boundary checks include 64-bit triangular ordinals through
`nbf=INT32_MAX`, without allocating those matrices. Actual launch code is
host-compiled with fake submits to test prerequisites, explicit-list dispatch,
partial lists, grid rejection and sticky launch failure.

The GPU gate must independently compare Cartesian/spherical nonsymmetric
S/T/V weights through f and complete PBE0/def2-SVP E/F against the retained
full-density GPU4PySCF oracle. Formal timing is the same-binary shell-warp versus
implicit-cooperative ablation, five interleaved repeats per side at 48/96 atoms
and warm/moved-warm geometries. Require >2% and twice-summed-relative-MAD,
unchanged semantic work/resource bounds and every-repeat numerical acceptance.
Keep complete cold/moved setups but do not claim matched-work cold gains or
byte-identical independently converged native seeds. Separate profilers and
sanitizers do not enter the clean timing population.

V3 native sources and raw evidence are retained in ignored
`.artifacts/shell-nucleus-stage/` and n1
`/data/jzzeng/qc-next-hotspot-20261009-125a4e33f/evidence/shell-nucleus-v3/`.
The frozen library basis is the #2166 v7 composition on `125a4e33f`, plus the
exact changed native sources. This is not a timed whole-current-master binary.
Review/master port validation must preserve that distinction. Unrelated master
increments do not trigger a duplicate full qualification matrix.

Both five-repeat populations finished and were independently verified against
the retained full-density references. The original two-owner population
(Slurm 6869) passed every numerical/work/resource check, but its 48-atom warm
median gain of 3.31623% did not exceed the 3.78049% noise floor. Its 48-atom
moved-warm gain was 2.49886%; the 96-atom gains were 5.66074% and 5.66900%.

A separate matched-owner population (Slurm 6872) uses the same publicly frozen
native density/coordinate bytes in both arms. This comparison is legal because
the prepared Hcore selector is read per call, not captured by a Fock lease.
Before/after native seed blobs remain byte-identical for both geometries. The
48-atom warm gain is 1.98585%, strictly below the unchanged 2% gate; moved-warm
passes at 3.44225%. The 96-atom gains are 5.94958% and 5.62673%, both passing.
All 43 complete calls per size pass numerical and semantic/resource checks;
maximum 96-atom errors are 1.10e-11 Eh and 3.79e-11 Eh/Bohr. These populations
do not qualify V3 for promotion. In particular, the 48-atom failure is not
rounded into a pass, and the passing 96-atom size does not authorize a PR.

The read-only native warm-state ABI is used for seed capture because the public
Python checkpoint path does not resolve this PBE0 accuracy model. The failed
checkpoint attempt is retained outside the clean populations. Slurm allocates
CPUs but its TaskPlugin does not bind this process: the observed allowed CPUs
are 0--127. Single-core oversubscription was not verified and is not an
explanation for the first population's noise.

Slurm 6876 traces one matched-owner warm call per arm at each size and collects
an actual integral-class force census separately. These intrusive traces are
diagnostic only; they are excluded from every promotion timing population.

## Qualified composition and default boundary

The materially changed class-major order-seven producer qualifies the composed
explicit Hcore/force selection, not the failed Hcore-only population. Slurm 6921
uses the actual `88cfa7017` native build, one library and two captured force
owners. All 46 complete calls per size pass independent retained-reference,
physical residual, work/resource and per-owner native seed byte checks. The
48-atom warm/moved-warm gains are 4.46405%/3.67395%; the 96-atom gains are
5.96079%/6.08424%. Each strictly exceeds its unchanged robust noise floor.
See [the composed decision](2026-10-10-order-seven-class-major-force.md) and
[source-matched evidence](../../../../benchmarks/results/pbe0-implicit-class-force-20261010/README.md).

The existing global one-electron policy defaults to cooperative scheduling,
but the prepared KS bridge previously always used component lanes. Merely
honoring that default would silently promote the failed Hcore-only change
when cooperative two-electron derivatives are disabled. Therefore this bridge
requires an explicit mapping knob to select implicit cooperative Hcore; an
unset knob preserves its previous schedule. The standalone cooperative
contract remains unchanged. A compiled test uses the actual policy and bridge
admission for unset, explicit cooperative, shell, serial and invalid values.
This default-preservation followup leaves every explicitly measured kernel
and schedule unchanged; it is separately built/checked, not relabelled as the
timed library or used to justify another clean timing population.

## Revisit when

If the one-pair-per-warp schedule fails complete endpoints, investigate source
work/reduction costs with a checked causal trace. Do not promote only the easier
size, relabel the failed shell adapter, lower the timing/noise gate, or add a
large resident index list merely to reproduce its old address arithmetic.

## References

- `.agents/notes/rejected/2026-10-10-ks-shell-nucleus-cooperative.md`
- `.agents/notes/implemented/performance/2026-09-20-generated-nucleus-cooperative-one-electron-default.md`
- `.agents/notes/implemented/performance/2026-10-01-default-screened-through-f.md`
- `docs/developer/one_electron_derivatives.md`
