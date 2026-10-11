# Decision: retain immutable FP64 Lambda core values within one solve owner

The later storage-only refinement is documented in
`2026-10-10-lambda-invariant-frontier.md` in this directory. It supersedes the
whole-core retained arena without rewriting this historical qualification or
changing the original dependency proof, adjoint equations and owner epoch.

Status: implemented; matched clean and instrumented qualification complete
Date: 2026-10-09

## Problem and frozen source

Issue #2136 targets physical FP64 Lambda action cost and fixed parameter/audit
work after the cadence experiment in #2128. Cadence cannot remove an Arnoldi
action or the expanded independent audit. The staged matrix core nevertheless
reconstructs primal intermediates and operand layouts with immutable inputs.

The prototype snapshot fetched at task start was
`6b439bb003d50b6dd854e81400977195bb4bf460`, explicitly including then-pending
#2128 at `055dc8b5fe1e8c6fcf4fe5692b69243c9b3d5290`. Before final qualification,
master advanced to `957fd60b6fffa267918ab597ceb12fe2b1a8cd81`, including #2128
and the exact-K scheduling default. Final source is a new detached worktree on
that master with only #2136's patch applied, and a newly built library. Old
screens/profiles remain separately retained and cannot be pooled with final
measurements. Historical #2128 times are not the new baseline. Existing user
changes in the original worktree remain untouched. No branch or commit is
created.

## Decision

Apply `analyze_iteration_reuse` to the existing FP64 staged matrix core. All
`bar_*` inputs remain dynamic; only proven pure, transitively primal-dependent
nodes enter preparation. Reuse `_arena_plan` retained-slot ownership and
`_cuda_program` preparation/dynamic emission, not another Lambda equation set.
The representative graph has 103 invariant and 303 dynamic operations. The two
phases bind distinct GEMM descriptor subsets and share original generated kernels.

First prepare immutable Q-reduced cuts, then restore `q=1`, prepare the optional
core and fence its sticky finite check. Restoration matters: Q preparation can
leave a nonunit tail, which non-batched core bindings correctly refuse. The first
GPU prototype failed that binding before publication; the fixed nonunit/tail
gates pass, and the failed evidence remains local.

Every physical transpose and the final factor-response core use this epoch.
Expanded audits and separate retained-parameter graphs remain independent. Do
not cache seeds or introduce global state: copied device inputs and immutable
cuts establish one owner's epoch, released on destruction. Equal shapes or
pointers across Hamiltonians cannot establish validity.

## Resource and failure invariants

- Retain original scalar, matrix and independent-audit arenas.
- Charge the complete additional exclusive core arena, including dynamic scratch,
  both extra descriptor containers, and the existing provider allowance within
  the complete host/GMRES/detached-publication bound.
- Budget refusal keeps ordinary matrix execution with explicit diagnostics.
- Optional CUDA allocation shortage drops retention before discarding GEMM or
  staged Q reduction. The provider deliberately stays alive and charged.
- Only `cudaErrorMemoryAllocation` permits that retry. Arithmetic, finite-check,
  launch, binding and driver errors propagate without partial publication.
- Restore unit Q at preparation and every core call, including after tails.
- Preserve coefficients, contraction order, strict FP64 actions, exact residuals
  and independent equations. FP32 forward W is a separate experimental control.

## Measurement contracts

Generated queries partition original core summands, logical packing-output bytes
and generated/audit launches by executed phase. Shared context counters count
actual GEMMs/provider calls. Logical bytes are not DRAM traffic, capacity is not
observed peak VRAM, and summands are not hardware FLOPs.

The progress journal separates immediate Lambda initialization, primal replay,
RHS, GMRES, independent audit and parameter/factor VJP. Each GMRES physical action
is tagged initial residual, Arnoldi or exact residual replay with work/transfer/
synchronization increments. Disable this journal for clean timing; do not add
its nested times to parents or pool profiled observations into clean samples.

The bounded driver alternates/reverses repeated controls on one frozen library
and assigned GPU: FP64/FP32 W, cadence 1/30, reuse 0/1. Stale binaries, discarded
attempts, incomplete scopes and failed independent gates reject qualification.
`--screen` is preliminary; an accepted summary appears only after every requested
observation finishes.

## Evidence

- The n2 finite Slurm native gate run passes 62 tests: parameter/factor outputs,
  nonunit tails, equal-shape changed Hamiltonians, resource refusal, exact audits,
  nonfinite/no-publication and small molecular independent PySCF energy/two-step
  force gates. Focused cached/uncached native responses match bit for bit.
- Slurm 2735 passes six reuse/budget tests under memcheck with zero errors.
- 81 focused host compiler/response regressions pass with four conditional skips;
  native GMRES contracts pass. Compiler structure checks 501 modules with zero
  dependency errors. Actual-owner mock tests cover provider destruction, matrix
  refusal and core-only refusal without provider release.
- Fresh PySCF 2.14.0 ethane230 energy executes on n2 under finite CPU Slurm 2733:
  conventional RHF, symmetric metric whitening with relative cutoff `1e-10`,
  total energy `-79.71851664319477 Eh`.
- Retained independent directional energies at `1e-4`/`3e-5` Bohr live in
  `benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json`. Their centered
  geometry is checked against the exact frozen input. These are independent
  reference energies, not historical implementation timing controls.

Raw evidence, failed prototypes, verified ccache stats, binaries and profiles
live at `n2:/data/jzzeng/qc-2136-lambda-20261009/` and ignored local
`.artifacts/issue2136/`. Scientific tests alone establish neither a cold complete
speedup nor `<120 s`. Append endpoint/profile results before closing #2136.

## Rejected alternatives and revisit conditions

The subsequent fixed-audit optimization is recorded separately in
[independent audit GEMM](2026-10-09-df-lambda-independent-audit-gemm.md). It lowers
the original expanded audit graph without consuming this cache; final candidate
qualification explicitly selects both independent optimizations.

Do not hand-code another Lambda implementation, weaken the expanded audit,
increase cadence to pretend an action-cost win, or reuse adjoints across seeds.
Do not retain Q-dependent factors under amplitude identity alone. Shared-Q layout
hoisting is a separate candidate with its own batch/tail lifetime, capacity and
endpoint ablation. Stronger exact right preconditioning/recycling is likewise a
separate numerical change if action cost remains irreducible.

Prioritize the largest removable cold endpoint time revealed by parent-consistent
profiling, including RHF/CCSD/orbital response, not isolated kernel throughput.
No release/distribution or external publication operation is authorized here.

## Final latest-master clean qualification

Slurm 2749 freezes master `957fd60b6fffa267918ab597ceb12fe2b1a8cd81` plus only
#2136's reconstruction patch, library
`50beaeb8f14d767135a9a925a00b9f1485bbd740ce002a6190c86ffe8fe664b6`,
endpoint `d28ddc68f854b3e1f029ce3a4b6544bf5796c6f37b5f777634d98aee5d155c91`,
and input `9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
All 16 fresh-process endpoints pass on PRO 6000 GPU
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, driver 595.91.07, 600 W. The eight
W/cadence/candidate configurations run forward once and reverse once. Candidate
requests both this retention and the independent audit lowering; both are
actually admitted. This is not a cache-only speedup claim.

| W | Cadence | Control E+F median s | Candidate E+F median s | Lambda medians s | Saved endpoint |
| --- | ---: | ---: | ---: | --- | ---: |
| FP64 | 1 | 669.223 | 584.742 | 270.885 / 186.902 | 12.624% |
| FP64 | 30 | 593.991 | 518.652 | 196.191 / 120.836 | 12.684% |
| FP32 | 1 | 660.575 | 576.975 | 270.757 / 186.887 | 12.656% |
| FP32 | 30 | 586.116 | 510.962 | 196.211 / 120.850 | 12.822% |

Every solve uses 21 Arnoldi iterations, with 42 physical actions at cadence one
and 22 at cadence thirty, independent of candidate selection. The strict
cadence-thirty complete saving is 75.339 s, versus 75.355 s within Lambda: RHF,
CCSD and other parents are essentially unchanged. The `<120 s` stretch target
is **not reached**: do not round 120.836 down and claim it.

Maximum independent energy error is `2.637e-11 Eh`, two-step directional force
error `7.589e-9 Eh/bohr`, all-component difference from the common strict baseline
`2.368e-11 Eh/bohr`, and translation defect `1.088e-12 Eh/bohr`. Largest exact
Lambda/Z residuals are `6.117e-13`/`1.357e-13`; stationarity is `7.673e-12`.
The stricter additional `3e-9` common-component sanity gate also passes. Native
and independent physical tolerances are not relaxed.

The final frozen library passes 76 small-GPU tests and 15 focused tests under
memcheck with zero errors. Host checks pass 36 focused regressions (54 genuine
conditional skips), 13 source-ownership checks, and native GMRES restart,
exhaustion/stagnation and exact-residual contracts. Compiler ownership checks
501 modules with no dependency errors.

Reviewed samples, full parent medians/ranges and reconstruction are retained in
`benchmarks/results/df-lambda-cost-2136-20261009/`. Raw final artifacts stay in
`n2:/data/jzzeng/qc-2136-lambda-master-20261009/` and ignored local artifacts.
The clean screen on a different PRO 6000 UUID and the cancelled prototype
experiments are not pooled into these results. Two repeats per cell meet this
issue's ablation requirement, not the shared five-pair performance-promotion
gate; no measured VRAM peak or full solver trajectory is fabricated.

The initial retention check rejected this new bundle because master left only
about 18 KiB under the unchanged 64 MiB aggregate budget. The unrelated older
#206 device-response summary has no live plain-path consumer; it is preserved
byte-for-byte in deterministic gzip, with decoded checksum/size documented in
its README and the shared plain/gzip reader retained. This frees 376,811 bytes
without discarding samples, changing scientific results, raising a limit or
publishing an external archive. The original failed retention check remains in
ignored evidence; it is not relabelled as a pass.

## Qualification snapshot and later master changes

Master `957fd60b6fffa267918ab597ceb12fe2b1a8cd81` was the latest master when the
final qualification source was frozen. During the finite Slurm validation,
master advanced to the subsequently observed
`5cddaa4641797280d24da987d839e2205c49333e`. The newer snapshot changes shared
compiler/tensor IR and other owners, although native Lambda/CC files are
unchanged. The exact decoded measured-source patch passes a cached, dry
application check against that observed tip in a separate temporary index.
This establishes patch applicability only, not scientific or performance
qualification of the newer compiler snapshot. Retain the original source and
binary hashes for every accepted sample; integration onto a later master must
receive its own tests and cannot inherit these timings by relabelling a binary.
