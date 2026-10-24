# Decision: restore source concurrency in the reference MD-J engine

Status: implemented (opt-in)
Date: 2026-10-19

## Problem

PR #2609 followed GPU4PySCF's architectural decomposition but used a different
execution layout. Each primitive quartet occupied a whole warp; lane zero
computed the radial recurrence, and only 1/4/10 lanes owned the common low-order
outputs. Both directional contractions used per-source shared double atomics,
which compiled to shared CAS loops on sm_120. Upstream instead assigns adjacent
lanes to independent sources and reduces register partials before publication.

This difference dominated the 96-atom water PBE0/def2-SVP regression. A complete
warm energy/force call took 20.800557 s versus 11.388117 s for the incumbent in
the original campaign. CUPTI attributed 9.976203 s to the new uniform J source
kernels versus 0.616629 s for the incumbent. Other kernels took 9.423362 versus
9.395753 s. Hierarchy construction cost only 14.626 microseconds; changing
screening or removing the hierarchy would not address the measured cause.

## Decision

Use 128 threads decomposed into source and component lanes. Choose power-of-two
component groups from upstream's single-density shape targets of 11 output
components and 31 radial states, applied to both independently admitted
directions. Split source lanes into contiguous bra shuffle groups and ket
groups. Low-order threads now own separate sources; high-order components share
interleaved compact radial scratch. The compiler emits strided indexing from
the same scalar IR, with stride one retained for independent math probes.

Accumulate bra outputs in registers across ket batches. Reduce reverse outputs
with bra-group shuffles and one shared-memory writer per ket/component. After
all radial readers finish, reuse that scratch for bra reduction. Only tile
publication uses floating-point atomics. Every physical lane participates in
shuffles and block barriers, including screened sources and ragged tails.

Retain the exact public-AO mask, upward-rounded directional bounds, global
density allowance, original-shell multiplicity, reciprocal inventory, residual
consumer, optional-storage cap, owner stream and bounded fallbacks. This is
still an adaptation: scalar radial lowering and exact screening are deliberate
differences from upstream, not a claim of identical machine instructions.

## Evidence

Source: PR head `ae9f52c5fad596bedd4abfb2178bf9c4d863ea1f` plus this repair.
The original measured head `0d037ce9475aafcf8507dbe7f6285185449cb158`
has identical production sources to that head. Pinned GPU4PySCF source at
`9af748a71ca8f4881c604332c2f0cc1632e033ac` was downloaded and checked
against every registered upstream input hash; no upstream runtime speed claim
is made.

Final native source identity:
`e360218fe0638c62ffc5b026ac886c0758a82e40951377b48722ec79011d0941`.
Library SHA-256:
`26d83c62f8c40ab1ebe10a5037f485262fbe4eaf55764bea91b29091d2a84d4e`.
Release CUDA 12.9.1, sm_120, fast compile OFF, verified ccache 4.5.1;
both arms use the same scalar CPU linear-algebra fallback.

Clean Slurm `n4:1159`, RTX 5090, 8 CPU threads, 96 atoms, 768 spherical AO,
FP64 direct PBE0/def2-SVP, grid 48x16x32 without pruning:

| Complete energy/force phase | Samples per arm | Incumbent (s) | Repaired reference (s) | Fock builds |
| --- | ---: | ---: | ---: | --- |
| Cold, including construction/preparation | 1 | 68.300302 | 68.235496 | 18 / 18 |
| Fixed-density warm median | 5 | 11.418831 | 11.188901 | 1 / 1 |
| Moved geometry, rebuilding and solving | 1 | 57.213238 | 54.987919 | 12 / 12 |
| Moved-geometry warm median | 5 | 11.401163 | 11.208818 | 1 / 1 |

The clean warm reduction is 2.0136% versus the incumbent and 46.2086% versus
the original PR. Cold/moved are single observations, not repeated-process
statistics. This does not qualify default promotion or generalize across GPUs,
bases or grid choices. Independent E/F gates passed all 24 repaired-build PBE0
records: maximum errors 1.01e-11 Eh and 2.94e-11 Eh/bohr, below 1e-8/1e-7 gates.

Separate CUPTI job `n4:1158` found uniform J kernel sums of 0.618999 s
(incumbent) and 0.422586 s (repair), a 31.73% reduction. Other GPU work was
unchanged at about 9.42 s. All traces reported zero dropped records and callback
errors. These diagnostic timings never enter the clean endpoint population.

Separate counted job `n2:3473` confirmed actual reference execution and the
same final-density work as the original PR:

| Semantic work per warm call | Incumbent | Repaired reference |
| --- | ---: | ---: |
| Uniform radial sources | 1,001,504,816 | 964,534,488 |
| Uniform directions | 1,926,600,288 | 1,926,600,288 |
| Uniform contraction summands | 23,506,259,200 | 23,506,259,200 |
| Residual candidates | 209,520,685 | 209,520,685 |
| Residual tasks | 12,903,112 | 12,903,112 |
| Residual radial sources | 93,656,520 | 93,656,520 |

Thus the repair changes execution efficiency, not scientific admission. The
3.69% radial-source reduction versus incumbent already existed in the PR;
the incumbent already had reciprocal reuse in the common low-order classes.

844 host math/contract/native-body/reference tests passed, including runtime
ASan/UBSan variants; 138 additional resource/bridge/endpoint/benchmark tests
passed. Real CUDA job `n2:3471` passed 63 tests, including all 19
graph cases; the pinned-upstream execution test remains deselected because its
built pinned extension is unavailable. Job `n2:3472` passed memcheck, initcheck,
racecheck and synccheck on four representative cases each, including successful
unrestricted graph J/K replay; zero errors, hazards or warnings.

The first sanitizer selection included the graph test's deliberate invalid
capture and reported the three expected CUDA API errors. Preserve that failed
receipt; do not suppress API-error detection. The full ordinary CUDA suite
continues to test failure recovery. Also repair the pre-existing test oracle
mismatch at cutoff 0.02 by comparing the same exact-mask incumbent and keeping
the independent libcint gate. RKS integration must use a retained prepared owner;
singlepoint's non-retained path did not execute MD despite passing energies.

## r2SCAN-3c admission boundary

The 96-atom r2SCAN-3c prescription uses 1088 spherical AO in def2-mTZVPP,
including D4 and gCP, with the same FP64/grid/direct settings. Clean warm
medians were 29.549152 s (incumbent), 29.552259 s (original reference selector)
and 29.545340 s (repaired selector). All independent E/F gates passed.

Actual complete-endpoint work records in `n4:1160` contain incumbent MD work
and no reference-engine rows. Extracted production `MdJHost::prepare` with real
basis shapes accounts for 124,792,120 base bytes plus 27,213,984 hierarchy bytes:
152,006,104 required versus the 134,217,728-byte cap. The incumbent fits; the
reference hierarchy does not. Retain this explicit fallback rather than claim
the new kernel was benchmarked for this basis. A separate standalone probe's
default 256-MiB total plan budget can also reject the entire MD owner; that
probe must not be confused with the full endpoint's incumbent-MD fallback.

## Rejected alternatives and revisit conditions

- Copy upstream float-log screening or decontracted-shell half factors: changes
  the scientific contract without explaining the measured scheduling regression.
- Raise the resident cap to force r2SCAN-3c admission: changes ownership/resource
  policy and does not establish complete-endpoint benefit. A bounded hierarchy
  representation would need its own resource, numerical and performance gates.
- Chase the narrow high-order tile cases immediately: their measured cost is
  small here; preserve the qualified repair and investigate with broader bases.
- Promote from kernel timing alone: forces and other GPU work dominate the
  repaired warm endpoint, limiting the full-call gain to about 2% in this case.

Raw jobs, complete records, SASS, profiler source, upstream hash verification,
build/cache logs and executable analysis are retained locally under
`.artifacts/pr2609/` in the qualification worktree. See
`.artifacts/pr2609/research/report.md`, `repair-summary.json` and
`analyze_repair.py`. Keep the original failed evidence alongside the repair.
The [current architecture](../../../../docs/developer/md_j_engine.md) and
[original integration decision](../architecture/2026-10-19-md-j-reference-pipeline.md)
describe the retained production boundary.

## Subsequent owner-capacity correction

The [owner-capacity decision](2026-10-19-md-j-owner-capacity.md) supersedes the
fixed 128-MiB admission ceiling after the user requested r2SCAN-3c admission.
The older fallback measurements above remain historical evidence. The private
128-MiB reservation, public-ledger protection and bounded fallback are retained;
actual MD admission now uses the capacity already assigned to its direct plan.
