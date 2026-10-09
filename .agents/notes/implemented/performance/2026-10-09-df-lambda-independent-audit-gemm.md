# Decision: lower the original independent Lambda audit through bounded FP64 GEMM

Status: implemented; matched clean and instrumented qualification complete
Date: 2026-10-09

## Problem

Issue #2136's initial parent-consistent 230-AO profile showed the expanded
independent physical-equation audit taking about 86 seconds. Immutable core
retention alone removed about eight seconds from Lambda in a preliminary cold
pair; it cannot remove the fixed audit cost. Those observations belong to the
preserved core-only prototype binary, not the final qualification library.

## Decision

Reuse the compiler's original expanded `independent_transpose` program for the
retained core and original virtual `amplitude_vjp` for the Q contributions. The
existing matrix lowerer changes scheduling, not equations, with bounded Q-axis
lifting and ordered accumulation. Do not consume solver cuts or cached core
values. All arithmetic remains FP64, including cuBLAS storage and accumulation.

The new arena is the maximum of the core and batch-auxiliary requirements.
Charge both full/tail descriptor containers, original arenas and the existing
provider allowance in complete response admission. The provider/context is
shared, never allocated outside its already reserved allowance. Selection,
exclusive bytes and schedule identity are separate from the unchanged semantic
independent-equation hash.

## Failure and independence boundaries

- Resource refusal preserves the original scalar expanded audit; requesting
  matrix audit does not imply it was selected.
- Allocation shortage drops optional core retention first, matrix audit second,
  then ordinary matrix execution and optional Q cuts. Only allocation shortage
  permits retry; finite-check, binding, numerical and driver failures propagate.
- Prepared matrices belong to one immutable owner. Audit coefficients, Q order,
  exact physical final residuals and force publication gates remain unchanged.
- Mathematical independence means different original equation graphs and no
  solver intermediate reuse; it does not require duplicating common GEMM/IR
  lowering primitives. Independent PySCF and force finite differences remain
  necessary to detect common-lowering defects.

## Evidence and preliminary failures

The final small-device run passes 73 tests. Three additional consistently scaled
Hamiltonians exercise small denominators and cancellation from common Fock
shifts. Focused optimized/scalar-audit outputs agree bit for bit on a common
primal with matched Q batches; independent physical residual gates stay strict.
Nonfinite independent audits reject publication in matrix and scalar modes.

The first budget-fallback test accidentally compared Q batch two against an
automatic different batch. Its tiny last-bit differences were not an arithmetic
regression. The test now explicitly matches batch two; no tolerance was relaxed.
The failure is retained in ignored evidence, not counted as a passing run.

The prototype source starts at master
`6b439bb003d50b6dd854e81400977195bb4bf460`, explicitly including then-pending
#2128 at `055dc8b5fe1e8c6fcf4fe5692b69243c9b3d5290`. Master subsequently merged
#2128 and advanced to `957fd60b6fffa267918ab597ceb12fe2b1a8cd81`. Final
qualification rebuilds on that snapshot with only the #2136 patch; it does not
silently relabel or reuse the prototype library. All final clean and
instrumented observations use the same new library, executable, input and GPU.
Original worktree changes remain untouched. No branch, commit, release or
external publication is created by this task.

## Rejected alternatives and revisit conditions

Do not skip the independent audit, replace it with the solver's staged operator,
relax a residual/force tolerance, or add handwritten Lambda equations. Stronger
exact preconditioning and Q-dependent retention remain separate proposals with
their own science and lifetime qualification. Prioritize remaining owners by
removable complete-endpoint parent time rather than microkernel throughput.

## References

- Issue #2136 and dependency #2128.
- [Immutable core retention](2026-10-09-df-lambda-core-invariant-reuse.md).
- `docs/developer/df_ccsdt_gradient.md` describes current controls and fallbacks.
- Raw final evidence: `n2:/data/jzzeng/qc-2136-lambda-master-20261009/` and local
  ignored `.artifacts/issue2136/`; old core-only evidence remains in its separate
  `qc-2136-lambda-20261009` remote root.

## Final complete-endpoint evidence and tradeoffs

The final master-based library passes 76 GPU tests and 15 memcheck cases with
zero errors. Slurm 2749 completes all 16 independently gated cold endpoints on
one frozen library and GPU, then separately instruments new complete endpoints
within the same allocation. Clean samples are never pooled with instrumented
times. Exact source/binary/input identities and the four matched W/cadence pairs
are preserved in the linked core-retention note and reviewed evidence bundle.

For strict W/cadence thirty, the combined candidate reduces Lambda from
196.191 to 120.836 s and complete E+F from 593.991 to 518.652 s, without changing
21 iterations/22 actions or obtaining a gain from RHF/CCSD variation. The stretch
target `<120 s` remains unmet. The standalone fixed-audit optimization must not
be assigned this entire combined saving; parent-consistent profiles distinguish
its fixed audit cost from core-retention action savings.

| Complete Lambda work/capacity | Control | Combined candidate |
| --- | ---: | ---: |
| Actual provider GEMM requests | 42,726 | 45,986 |
| Compiler semantic contraction summands | 40,429,634,645,741 | 38,135,799,179,463 |
| Generated/audit kernel launches | 231,740 | 183,273 |
| Logical packing-output bytes | 1,596,162,795,584 | 1,881,569,130,992 |
| Complete numeric capacity bytes | 6,844,452,897 | 10,081,975,949 |
| Exclusive immutable-core arena bytes | 0 | 1,710,877,464 |
| Exclusive matrix-audit arena bytes | 0 | 1,525,829,744 |

The total packing volume and GEMM count increase: the audit is now packed/GEMM
scheduled instead of executing scalar contractions. Batching shares original
primal-only virtual subexpressions across Q lanes, and core retention shares
primal-only values across seeds. Neither logical bytes nor semantic summands
are DRAM traffic or hardware FLOPs. Capacity is an admission ceiling, not a
measured VRAM peak. Actual whole-endpoint CUDA launches/API/copies are recorded
separately by Nsight Systems; nested action times cannot be added to parents.

The additional capacity is intentional and fully charged. Low budgets/provider
refusal retain original scalar/matrix execution, and numerical failures still
abort. Revisiting audit batch selection or Q-resident reuse requires separate
bounded lifetime/resource proof and a new matched complete-endpoint experiment.

## Final parent-consistent profile

Two new complete strict-W/cadence-thirty endpoints run after, not within, the
clean samples, with the same allocation, library/input hashes and GPU. These
instrumented observations are not substituted into the medians above.

| Immediate Lambda child seconds | Control | Combined candidate |
| --- | ---: | ---: |
| Initialization, including preparation | 1.894 | 2.319 |
| Fresh primal replay | 18.084 | 18.085 |
| RHS | 0.083 | 0.082 |
| GMRES | 83.713 | 74.359 |
| Original independent equation audit | 86.656 | 20.933 |
| Parameter/factor VJP | 5.509 | 5.084 |
| Complete Lambda parent | 195.989 | 120.915 |

The 22 nested physical actions total 81.908 / 72.548 s, already inside GMRES.
Audit lowering removes 65.722 s of fixed work; immutable-core retention reduces
GMRES by 9.354 s without changing actions. Initialization grows by 0.425 s and
final VJP drops by 0.425 s. Do not add nested work to parents.

Nsight Systems observes 546,969 -> 501,762 whole-endpoint kernel launches and
745,144 -> 706,465 runtime API calls. Actual whole-endpoint copies are
4,368,391,064 D2D bytes, 2,386,107,529 H2D bytes, and
1,917,179,701 -> 1,917,179,705 D2H bytes. These are not Lambda-only DRAM traffic.
Launch register/block metadata is retained without achieved-occupancy inference.

The final finite Slurm NCU permission probe returns `ERR_NVGPUCTRPERM`. No
counter permission, visibility or scheduler restriction is bypassed; DRAM
counters and achieved occupancy are unavailable, not invented. The largest
observed whole-endpoint kernel is the existing exact nuclear-force shell-quartet
consumer, two launches totalling about 82 s. That force owner, RHF (~129 s),
remaining Lambda actions and orbital response (~122 s) should be prioritized by
removable complete-parent time under #1401.
