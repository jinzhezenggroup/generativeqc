# Decision: retain scientific owners behind one native endpoint gate

Status: implemented
Date: 2026-10-12

## Problem

The shared compiled-region gate and artifact lease existed, but neither
represented running/terminal-close states. Most built-in method endpoints had
their own transaction/publication bookkeeping. ABI rejection could revoke a
native solve token yet leave the preceding successful transaction or diagnostic
eligible. Shape or generic owner-token identity is not sufficient scientific
binding evidence.

## Decision

Extend `CompiledExecutionRegion` and the existing `PreparedExecutionLease`,
without changing their artifact/scientific schema. Native prepared calculation
and fleet bases enforce host-control transactions before calling the original
HF, KS, MP2, CC or GFN implementation. A separate ABI publication guard rejects
partial argument/copy-out transactions. Completed nonconverged diagnostics are
retained as failed attempts, not successful solves.

Actual scientific binding remains with exact immutable native source inputs,
HF/KS solve generations, snapshot checks and artifact/guard identities. Backend
resource owners still allocate, fence, refresh and destroy their storage. The
gate never owns CUDA resources or creates a second reference/solver registry.
Per-item ragged status and last-good state remain with the original fleet owner.

Stationary CUDA enters the artifact lease only after its real owner has checked
and refreshed its inputs; completion is after the existing endpoint fences.
Generated D3 similarly retains its original topology guards and failure rebuild.
Both refuse destruction/rebind while running and make close terminal.

## Rejected alternatives

- A second prepared schema, context registry or synthetic scientific generation.
  None would authenticate the existing scientific snapshots or resource leases.
- Moving solver/convergence/equation policy into the shared runtime. It would
  duplicate native scientific owners and risk extra per-iteration scheduling.
- Counting every completed fleet as every item succeeding, or resetting good
  neighbors after an isolated ragged failure.
- Relabeling a warm gate as prepared buffer reuse or unobserved resources as zero.
- Claiming #933 complete after wrapping SCF. Inspection and toolchain-denial
  testing found that CPU Direct-KS public forces still have existing JIT
  consumers outside this gate. Completing that AOT/force lifecycle boundary is
  separate scientific-owner work, not a statistics-only fix.

## Invariants

Native algorithms, precision, approximation, tolerances and public defaults do
not change. The native owner/target binding is not a borrowed scientific lease.
The runtime does not import method scientific modules in reverse. Admission and
resource measurements remain distinct; high-water categories cannot be summed.
The added metrics describe native host-control transactions only, not fabricated
allocation, transfer, fence, launch or semantic work counts.

## Evidence

`tests/native/test_prepared_lifecycle.cpp` executes real CPU HF/KS owners;
`tests/python/test_prepared_lifecycle_endpoints.py` executes the actual ABI on
CPU and explicitly allocated CUDA. Existing independent oracles, artifact,
budget, cleanup and toolchain-free tests remain intact.
`benchmarks/prepared_lifecycle.py` records matched separate-process complete
energy/force endpoints and original work/transport diagnostics. Raw local
qualification artifacts live in ignored `build/issue933-evidence/`; device
execution must be identified by its finite Slurm job, not inferred from builds.

### Qualification scope

The comparison baseline is exactly master `a2ba59894217215d68a7885235699249333c77b0`,
not a later advancing master. Release CPU and CUDA/sm_120 builds use the verified
ccache 4.5.1 launcher. CPU qualification passes both native context/lifecycle
tests, 172 Python tests (53 CUDA/capability skips), and the warnings-as-errors
Sphinx build. The documentation/source-link suite passes 19 tests and 60 subtests.

Real RTX 5090 execution uses node1/main and finite Slurm allocations. Job 7647
retains the initial failing receipt: the new test incorrectly assumed every CUDA
owner samples host memory, and the qualification staging omitted the stationary
AOT manifests. The corrected test preserves unknown host usage as `None`, not
zero; original manifest targets supply the missing hash-bound sidecars. Job 7650
passes all 21 lifecycle/artifact-contract cases on actual CPU/CUDA endpoints.

Two existing regression failures also reproduce on the exact baseline: the
def2-SVP open-shell PBE moved/fresh comparison differs by about 1.4e-8 hartree,
and the recomputed CUDA HF resource-estimate fixture differs by 360 bytes. The
candidate's existing regression suite has 193 passes, two failures and three
capability skips. Neither tolerance nor fixture is altered here, and the full
regression gate is not claimed green. These baseline failures and the remaining
public force/AOT/resource scopes prevent treating this slice as #933 closure.

Matched separate-process CPU/CUDA RHF and PBE energy and energy/force comparisons
use five repetitions of H2/He cold, replay, changed input, reset, ragged failure
and recovery. Energies and forces match exactly in this small qualification
matrix; reported Fock/iteration/warm-seed counts and cumulative KS transport
match. Timing includes preparation for cold and the complete requested replay,
but excludes Calculator construction and process/library startup. No speedup or
broad method/default-promotion claim follows from these small endpoints.

Compute Sanitizer memcheck with full leak checking reports zero errors and zero
leaked bytes for both RHF and PBE energy/force workloads. Separate baseline and
candidate Nsight traces use two repetitions. RHF has 610 traced kernels and PBE
51,572; all recorded copies, launch/fence calls and allocation lifetimes match
between the pairs. Peaks are reconstructed from simultaneously live addresses,
not sums across phases: traced device allocations peak at 6,792 bytes for RHF
and 19,458,892 bytes for PBE. Static device allocations and pool backing are
separate categories; host and untraced vendor/driver usage remain unknown.
The compact source/library/hash-bound receipt is
`benchmarks/results/prepared-lifecycle-933-20261012/receipt.json`. Raw JSON,
SQLite traces and original passing/failing logs remain in ignored local evidence
and `/data/jzzeng/qc-933-prepared-20261012/evidence/` on node1. Instrumented trace
elapsed time is not used for endpoint performance claims.

## Revisit when

The remaining public force/AOT scopes and provider/borrow observability have
complete endpoint proof sufficient for #933 closure. Remove backend-specific
bookkeeping only after showing that its scientific/resource duties are truly
superseded; do not delete legitimate native owners merely for uniformity.

## References

- #933, #926, #929, #934, #459, #831, #835
- `docs/developer/prepared_execution.md`
