# Decision: compose the native CPU Hessian cap with the primal plan

Status: implemented
Date: 2026-11-22

## Problem

Linked B3LYP rejected a Python `ResourceBudget`, and linked PBE0 routed such
calls through Python response. The separate native derivative cap did not
account for the still-live primal owner. The dense native owner also let the
frozen integral matrix and independent response panels spend the same residual
aggregate headroom: either fit alone, but the final receipt could exceed the cap.

## Decision

Use the existing native composition engine and the actual prepared KS and
named correction requests. Keep the derivative cap and wrapper wire staging
overlapped with the complete declared primal peak, rather than guessing what
the CPU owner releases at the end of SCF. The frontend knows a small lower bound
from explicit grid, response and output reservations; the bridge retains
authority over the exact source, frame and optional numerical allocations.
This may reject a too-small final derivative cap after a successful SCF, but
before derivative allocation/work that would exceed the cap.

The derivative call consumes the selected plan's priced pageable reservation
and refuses a primal-only batch, a mismatched product or a subsequently failed
native admission. No accepted native method retries the Python scientific
route on a resource refusal. The native dense owner charges both optional
matrix and response assembly panels to a single residual allowance, retaining
bounded column work when the optional panels do not fit. Child grid or
response slack is not a second pool of aggregate capacity.

## Evidence and limits

Corrected PBE0 RKS HeH⁺ and B3LYP UKS H₃ exercise both products with a budget
at the scalar native reservation plus the live primal and marshalling peak,
and one byte below. Exact results match the unrestricted native outputs,
including correction pair counts. A one-byte total budget refuses before SCF;
a one-byte-short scalar derivative budget refuses before derivative assembly.
The installed C++ SDK regression constrains dense execution one byte below
its full optional-panel reservation and requires a bounded fallback and raw
matrix parity. Build-tree and installed SDK/CLI/Python parity pass.

The complete public def2-SVP water endpoints (PBE0-D3(BJ) and B3LYP-D3(BJ),
initial and moved geometries, one BLAS/OpenMP thread) retained exact output
and matching SCF, Fock, nuclear, induced-response XC and correction work across
the old and new unbudgeted binaries and the new 2 GiB budgeted path. Over five
warm complete calls per endpoint and geometry, the new unbudgeted medians
shifted by −1.78% to +1.57% versus the old binary. The budgeted medians shifted
by −0.07% to +3.15% versus the new unbudgeted binary. This does not price the
extra work of the tight, optional-panel-free schedule.

This accounts for declared numeric buffers, not physical RSS, final-frame
export scratch, implicit BLAS/allocator memory or caller inputs. It does not
add CPU provider lease observations or claim GPU execution. It supersedes
the budget-fallback part of the [default inventory decision](../compatibility/2026-11-22-default-cpu-hessian-inventory.md),
not the explicit B3LYP-only package compatibility or unadmitted models.
The #2227 ledger remains pinned to master; this is working-tree R2/R6 evidence.

After integrating master `cae2959e1`, the rebuilt CPU package passed the
seven selected native/CLI CTests and installed SDK/CLI/Python parity. The
installed SDK ran with no Python or toolchain on `PATH`. Five focused PBE0
molecular cases passed, including corrected curvature and independent PySCF
reconverged-gradient comparisons. At the finer `0.0003` Bohr displacement,
the largest raw Hessian error across the two water basis sets and ammonia
was `1.352e-7` Hartree/Bohr². The focused host-Hessian, frontend and contract
suites passed 252 tests and 25 subtests; the broader resource regression passed
700 tests with 20 skips. This was a CPU integration rerun, not a new timing
cohort or real-GPU evidence.

The integrated CUDA 12.9 SM120 build was also checked against the scoped
`#3133` behavior, independently of the CPU resource claim. The loaded native
source identity was `2f0785954c399db5ddbe5b1dcd4c5d5374652c36a0b760c22e0054eabf54ac7c`.
On the RTX 5090, Slurm `1816` passed both force-center lease invalidation and
reborrow cases; `1817` passed all four public Direct Cartesian FP64 LDA/PBE
water-dimer tests and the SDK CUDA probe. The largest HVP error versus the
strict CPU analytic endpoint was `2.513e-12` Hartree/Bohr², the largest finer
reconverged-gradient difference was `7.634e-9` Hartree/Bohr², and the largest
concurrently live device-ledger peak was `77,221,045` bytes. The earlier job
`1815` passed the SDK and ledger tests but could not run the CPU analytic
oracle without `sccache` on `PATH`; it is not an oracle receipt.

Slurm `1818` recorded 12 cold, warm and moved complete endpoints with no
numerical host fallback. Water-dimer HVPs completed one direction with
`516,132` exact primitive components and 18 mixed grid contractions; water
dense Hessians completed nine directions with `396,576` components and 81
mixed contractions. These single-cohort measurements are not a matched
performance comparison or evidence for additional CUDA model domains.
