# Decision: qualify CPU d/f mixed-DF forces with the shared range recurrence

Status: implemented
Date: 2026-10-11

## Problem

The initial shared CPU/CUDA source bridge retained an s/p-only CPU range
consumer. Removing its angular guards would compile and retain a provider per
Cartesian component, potentially 20^4 providers per radial operator through f,
and would mishandle real-spherical public AO expansions. The fitted force
consumer also unnecessarily prepared the complete s/p/d Direct dispatcher just
to evaluate nuclear repulsion.

This supersedes the CPU angular-domain decision in
[the initial shared-source note](2026-10-11-shared-mixed-df-cpu-forces.md), not
its scientific mixed-provider contract or excluded DF-response resource scope.

## Decision

- Lower the compiler-owned Direct Cartesian scalar source to host execution
  by changing qualifiers only. Its Hermite coefficients, packed simplex,
  Dual3 algebra and bounded SR/LR moment derivative seeds are the same owners
  used by CUDA. Guard the execution qualifiers so CPU JIT needs no CUDA SDK.
- Retain one generic operator/omega-frozen CPU artifact, not one component
  artifact. The native synchronous consumer reads already normalized packed
  public AO expansions and immediately contracts their primitive derivatives.
  It never calls PySCF or displaces a production energy.
- Visit the full unscreened eightfold quartet domain once. TensorIR supplies
  every ordered RKS/UKS density cotangent; sum distinct permutation weights
  before differentiation. Undo diagonal multiplicity rather than multiplying
  an arbitrary exchange weight by eight. Admission independently counts the
  exact expanded primitive visits over this canonical domain.
- Seed all slots on one physical atom together and recover the final atom by
  translation. Primitive qualification retains independent slots. A quartet
  with all slots on one atom has exactly zero physical derivative; its
  primitive validation visits remain counted even without a recurrence.
  Report visits and actual atom-seeded recurrence evaluations separately.
- Fitted primary integrals use a nuclear-only generated consumer. Their
  one-electron, Pulay and fitted J/K response still come from the common native
  CPU/CUDA bridge and its five-source composer. Mixed DF traverses exact LR
  only, with the graph's long-minus-short coefficient.
- Admit all-electron CPU mixed forces through f, at most 8 atoms, 64 AOs,
  128 primitives and 32,000,000 canonical expanded primitive visits. Public
  mixed weights use 512-quartet tiles; charge the eightfold density binding,
  publication arrays and a conservative 256 KiB shared scalar workspace per
  radial source in the existing additional-host inventory. Direct primary
  consumers retain their smaller domain and work cap.
- Preserve the existing Direct s/p packaged-AOT consumer and its conservative
  component/group plan capacity. Only mixed fitted consumers unconditionally
  select the generic compact host owner; absent/unmatched AOT and d-shell
  Direct diagnostics have the same generic fallback.
- Bundle and identity-hash the complete native header closure for installed
  JIT execution. Use explicit finite cached C++ compilation with strict FP64
  contraction and the repository's C++20 header contract.

## Rejected alternatives

Blindly lifting s/p guards neither implements spherical pullbacks nor bounds
code/plan proliferation. Compiling all f component subsets instead would
retain the same compile amplification. A second CPU range recurrence or a
CPU/PySCF force oracle would split scientific ownership. A full derivative
AO-rank-four tensor is unnecessary for this compact external-weight consumer.

## Invariants

Full-range J/K remains fitted and the LR correction remains exact Direct;
this is not omega-dependent RI-K. Coefficients and spin factors remain owned
by MethodIR/TensorIR and the resolved native Fock strategy. Backend schedule
differences do not introduce functional-name branches. No numerical screening
is added. Failure leaves native publication and work counters unchanged.

The additional numeric inventory and semantic counters explicitly exclude
native fitted-response internals, Python/compiler/loaded-code objects and
opaque runtime storage. They are not whole-force or process-memory claims.

## Evidence

`tests/python/test_stationary_rsh_cpu.py` independently checks SR/LR dddd and
ffff derivatives against normalized libcint energy differences in Cartesian
and real-spherical bases, including signed contraction coefficients and
repeated centers. Full-range partition and fixed-density s regressions remain.
`tests/python/test_stationary_cpu_work.py` independently enumerates unequal
contractions and expansions, constant provider capacities and exact rejection
boundaries. On water the spherical SVP/TZVP range schedules admit respectively
398,786 and 4,701,375 primitive visits, plus three nuclear pairs; the Cartesian
counts are 326,255 and 2,692,535, respectively.

`tests/python/test_mixed_rsh_df_forces.py` removes CPU d/f skips and adds
Cartesian RKS and spherical UKS SVP/TZVP complete endpoints. The independent
mixed-DF oracle reconverges each displacement and rebuilds orbital/auxiliary
centers and both quadratures. Existing energy, force, translation, warm and
moved/rebound gates remain unchanged. Source-matched qualification logs,
cached compiler commands and compiler stack-use evidence are retained under
`.artifacts/wb97m-df-cpu-df/`; no speedup is claimed.

All eleven complete CPU endpoints pass, including Cartesian RKS and spherical
UKS SVP/TZVP. The CPU/bridge/existing-DF/compiler suite reports 96 passes and
28 skips, with one expected configuration failure: the packaged CPU AOT export
test requires AOT ON, whereas this qualification build deliberately has it OFF.
No AOT-export behavior is changed to hide that mismatch. Applicable AOT-off
selection contracts pass separately (4 passes); range AOT compatibility,
primitive d/f, failure-publication and work-bound contracts pass (50 passes,
3 deselections). A final complete UKS spherical TZVP recheck passes after the
compatibility restoration. Host compiler/IR regressions report 80 passes and
7 deselections; provider/API/documentation contracts report 38 passes, one skip
and 22 subtest passes. The final count-only bridge/admission checks report
39 passes. These overlapping suites are not summed as distinct tests.

CUDA rebuild and Slurm validation on n1 (`node1`, RTX 5090, finite twenty-minute
allocation) pass: 31 tests, one skip and 62 deselections, including all eleven
complete mixed endpoints and independently checked CUDA SR/LR primitives.
Compiler ownership covers 514 modules with zero dependency errors. GCC stack
evidence reports a 59,520-byte maximum Coulomb frame and a 46,800-byte primitive
frame, below the conservative combined 256 KiB workspace allowance. Complete
endpoint qualification durations are retained in the logs, but no comparative speedup or complete
force memory claim is made.

## Revisit when

A native backend-neutral prepared derivative Fock owner can absorb host
scheduling without changing the graph-weighted mixed contract, or a qualified
source-driven shell packet can reuse the radial recurrence across components
while preserving this bounded fallback and independent acceptance gates.
