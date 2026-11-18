# Decision: order-preserving host GGA response SIMD with arena reuse

Status: implemented; fixed-grid component qualified, not complete-endpoint acceptance
Date: 2026-11-13

## Problem

The complete positive-Full-moment cohort still times out at the unchanged
1800-second cap. Its frozen integral preparation now completes, but only two
nuclear response columns finish. The fixed-grid response component's dominant
costs are AO field reduction and potential pullback, not scalar XC evaluation.
The retained diagnosis and rejected BLAS proposal are in
`../../proposed/2026-11-13-batched-host-response-field-contractions.md`.

Ordinary BLAS gives an attractive interior timing but changes reduction order:
108 of 792 near-vacuum controls switch the linked production branch. This is a
scientific failure, not an error that can be accepted by relaxing interior error
tolerances. The original scalar graph, density threshold, minority-spin branches,
small-gradient policy, and fail-closed behavior must remain unchanged.

## Decision

`src/dft/host_gga_ordered.hpp` is a native contraction execution template, not a
new XC formula or a new compiler algebra. On supported GCC/Clang x86 builds,
runtime dispatch selects an AVX2-targeted function without global ISA flags.
All callers compile with strict floating point and contraction disabled.

Field lanes are separate rho/x/y/z components. Each lane visits every ordered
row/column pair with the original multiplication/addition grouping. The rho
lane selects its original two-AO product rather than doubling the gradient
expression. Primal and tangent accumulators remain separate, and both gradient
terms use the full possibly nonsymmetric matrices. Pullback lanes are separate
output columns; point order, three gradient additions, coefficient/AO product
grouping, and weight multiplication are unchanged. There is no horizontal
reduction, FMA, half-coefficient multiplication, triangle copying, or symmetry
repair.

Before coefficient inputs become live, their already admitted numeric panel
provides four packing doubles per AO. No additional allocation, persistent
storage, resource identity, or optional admission API is introduced. A panel too
small for that pack retains the original scalar field schedule. Unsupported
CPUs/toolchains retain scalar fields and pullback. This is a capacity fallback,
not a scientific fallback. Scratch query, allocation lifetime, transactional
publication, provider validation, and exact primal-cache authentication remain
unchanged. Every action still recomputes primal fields and compares every cached
XC input bit; owner pointer equality is not substituted for authentication.

## Rejected alternatives

- Naive BLAS scheduling is not admitted despite its faster interior component
  timing. The model must not be clipped/smoothed or given a looser vacuum gate.
- Point-lane transposition preserves examined outputs but makes field reduction
  slower and allocates an extra tile panel. Its roughly 2.85-second cached
  component is not a useful production promotion over this packed schedule.
- The first four-component-lane prototype allocated four doubles per AO outside
  the reported workspace. Its numerical evidence qualifies the arithmetic,
  not that allocation contract; production instead reuses the admitted arena.
- Global `-march=native`, FMA, fast-math, and assuming AVX2 on all CPUs are not
  acceptable portability or reproducibility shortcuts.

## Evidence and limitations

`host-ordered-response-source-20261113` binds source identity
`af5af70b907622fb4eb79accb3148ccfbe6f1de6cf4f9ec81b257b5869209265`
and the matching rebuilt CPU core SHA256
`f366ff2c6c94ea7eb1f960671dde4f6188d6bccd4072b51c6909059fbef26f00`.
Its same-grid component uses the frozen native occupied C to reconstruct actual
RKS D: 81 AOs, six atoms, 87458 v4 points, 256-point tiles. Six interleaved
streamed/cached actions are bitwise equal to the frozen incumbent and retain
all point/tile/cache work. Mandatory numeric scratch remains 785672 bytes.

Streamed complete component medians are 3.625787140 / 1.494106501 seconds
(incumbent / current source). Warm cached medians, using the two actions after
preparation, are 3.5166686745 / 1.379788724 seconds. Cached field reduction is
0.6126474125 seconds, pullback 0.3337796075, AO reader 0.4066969615, and scalar/
feature work 0.006349335. Each bilinear sweep still visits 573811938 AO-pair/
point terms. These are complete fixed-grid component times, not complete JK,
GMRES solve, nuclear column, Hessian, or optimizer timings.

The permanent controls cover bounded packing with canaries, zero/short/exact
capacity, scalar and dispatched schedules, irregular AO/point extents, fully
ordered RKS/UKS inputs, signed zeros, subnormal products, extreme scales,
overflow behavior, and independent dense field contractions. Full-owner tests
retain independent Libxc potential differences at all original three steps and
gates, cache identity/mutation/failure/lifecycle checks, and 792 boundary controls
with capacity fallback and packing both exercised. The wider current-source
cases cover 17/81 AOs, 257 points, 16/256-point tiles, and both spins. One-off
frozen-incumbent checks separately retain all 792 controls for each capacity
mode, with zero classification or bitwise parity failures.

The prototype qualification retains an initial ctypes argument-count mistake;
the corrected direct-kernel and full-owner run passes. Current-source fixture
qualification also retains an initial missing `-p conftest` invocation; its
permanent tests pass and the wider cases are rerun with the fixture loaded.
Neither harness failure is a scientific gate relaxation.

## Remaining complete-endpoint work

The frozen-source `host-nuclear-component-profile-20261113` reconstructs the same
native D without SCF and times complete separate component calls at right
coordinate zero: nuclear XC Fock direction 15.577513367 seconds, nuclear XC
fixed-density curvature column 36.756622290, and ECP fixed-density curvature
column 2.454233028. The first two execute respectively 14647247478 and
33556112408 generated map terms over the entire retained grid. ECP executes
1364508 analytic kernel blocks, 94758 primitive groups, and 630 shell pairs;
there is no radial-grid fallback. Its archive SHA256 is
`83efcef66f1b73c21700ade2a357e9cdbeb85307ab6037b5c9a6dffb5b4fe405`.

Those nuclear calls use an empty density-right leg and do not measure a
stationary response solve, W response, full column, or independent numerical
acceptance. Their totals do not separate AO/partition geometry, generated-map,
and linked scalar costs. The next localized profile should split those nuclear
XC phases before selecting a rewrite, rather than optimizing ECP radial work
or replacing the scalar XC formula. No unchanged 1800-second
molecular rerun is justified by this component change alone. Both public ECP
saddle guards remain closed. Raw complete H/response/modes, required mixed and
stationary cases, practical larger scale, and the requested Ni TS remain
unqualified. The five-issue full-fix goal is not complete.

## Revisit when

Change the schedule only with equivalent ordered-branch, capacity, lifecycle,
independent numerical, and complete-endpoint evidence. A faster BLAS variant
requires model-owned branch/cancellation bounds and a proven original-order
fallback, not a hardcoded runtime density tolerance. Nuclear-map optimization
must respect its compiler ownership and original reduction/branch semantics.

## Terminal retention

Current-source archive SHA256:
`804184eec798615b288a3d6b0e1e3c9610ec3afe04d078a01b708fb22e482d3c`
(113490094 bytes, 715 members). It retains the complete registered source
inventory, matching CPU core/CLI with checked loader selection, validation
sources, component receipt, raw work/timing/cache evidence, and all failed
initial attempts. The retainer exits zero. The final acceptance consists of
223 test/groups without skips (including six documentation subtests), 90
ASan/UBSan cases, 1584 frozen-incumbent boundary controls, four ownership checks,
Sphinx warnings-as-errors, Ruff, and formatting. Combined test invocations that
hit their finite limits are retained, not erased: the nuclear fixture build is
reused, and the remaining unchanged tests complete in bounded file/functional
groups without altering assertions or scientific gates.

The scalar point-lane experiment is sealed as
`121e4030114da3da2c47c898a09d5537747baf5cb1ad8ff6b0cdbecdf0ce6930`
(254327 bytes, 32 members). The first packed, extra-allocation prototype is
sealed as
`217bfde45d8adbf1306f5705af64b8e4451719ba247c64e02f1d3a94b8a9aca9`
(1436012 bytes, 170 members). Neither allocation contract is promoted. Do not
edit or reuse these sealed cohorts as writable test/cache/output directories.
