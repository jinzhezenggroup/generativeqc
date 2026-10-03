# Decision: bounded shared-scalar primary CPU HF target prototype

Status: proposed (implemented in a local qualification worktree only)
Date: 2026-10-03

## Problem

The independent max-pivot reference solver is material in current HF96 endpoint
profiles. Reusing shared scalar cyclic Jacobi avoids private numerical duplication,
but its stopping semantics, acceptance, route scope and resource envelope require
independent qualification. No speedup is inferred from attribution.

## Decision

Only the primary CPU run_cpu_fock_strategy entry opts standard complete exact or
DF RHF/UHF into a fixed EigenOperation, threaded through run_prepared_fock_strategy
and the shared host driver. Existing prepared callers default empty/reference.
Actual backend guards preserve CUDA callbacks. Setup overlap/core/seed validation,
HF preliminary generation, DFT, post-HF physical reference exports, and custom,
mixed or range-separated Fock specs retain the reference route. Normal primary
state/density getters still report the actual primary solve; no recomputation is
substituted. Recursive target retries keep the same callback. No environment or
public Python/provider selector is introduced.

The adapter uses shared congruence, GEMM and explicit scalar/task-parallel/one-thread
Jacobi with a 1e-14 absolute off-diagonal cap. Only the two low-level cap API files
from the qualified PR1753 repair are reused. Shared defaults, LAPACK, CC callers
and independent reference arithmetic remain unchanged. The cap is a stopping rule,
not a rigorous error certificate. Original F/S frame acceptance still requires
absolute residual/metric 1e-8 and scaled residual 1e-12. The cyclic 100-sweep failure
contract differs from the historical max-pivot oracle and remains observable.

## Resource proof

M=8n². Borrowed F/S/X and unchanged caller retained arrays are not counted twice.
The returned C and eigenvalues are included in every applicable live phase.

- Congruence: transformed M + scratch M; scratch dies before the leaf
- Leaf: input M + rotation vectors M + sorted C M + eigenvalues8n + indices sizeof(size_t)*n
- Back transform: old/new C 2M + eigenvalues8n
- Gram allocation: C+SC+transpose(C)+Gram =4M plus eigenvalues8n, FC not yet allocated
- Validation residuals: C+SC+Gram+FC =4M plus eigenvalues8n, transpose already dead

B=max(3M+(8+sizeof(size_t))*n,4M+8n) is no larger than the prior generalized
reference envelope4M+(8+sizeof(size_t))*n. The old coefficient frame retained by
assignment and the earlier alpha frame in UHF are unchanged caller lifetimes.
This is a substitution proof, not a claim of spare room in the 64-matrix inventory.
No global cap, request identity, planner bound or admission threshold changes.
The helper checks dimension arithmetic; it is not a new per-adapter budget gate.

The shared validator allocates FC after the Gram expression's transpose temporary
has died. The independent matrix products have the same arithmetic reductions;
the entire original finite-check, interleaved scalar-reduction, diagnostic and
acceptance body remains unchanged. Product failure precedes norm overflow, then
physical acceptance. Independent product execution/allocation order does change;
identical floating-point trap order or allocation-failure timing is not claimed. There is no private duplicate validator in production.
Only the solver layer gains the exact
shared tensor/cpu_linalg.hpp dependency; reference/initial/gradient boundaries and
unrelated tensor dependencies remain prohibited by structural tests.

Function-pointer callbacks use inline std::function storage in the qualified
libstdc++; successful-path metadata is fixed stack state. Error detail/exception
strings are runtime objects under the existing ResourceRequest exclusions, not
numeric workspace. On the measured GNU LP64 build, physical frame rejection
adds 77 B detail and 101 B exception string requests; complete new/delete peak is 194 B
at n1 and 226 B at n2, exceeding their 40 B/144 B numeric bounds. This is explicitly
excluded runtime metadata, never described as a measured full-process bound.
For the reordered validator, the detail overlaps its complete numeric buffers:
failed-frame peak is max(B+77, M+8n+178), observed as B+77 for n=8,32,96.
Malloc allocator bookkeeping and __cxa exception storage are also outside the
new/delete instrumentation, as documented by the harness.

## Initial evidence

Local sibling evidence directory contains exact commands, source/binary hashes,
full new/delete event lifetimes, reference/adapter strict boundary tests,
independent analytic spectra, and baseline validator equivalence. No publication
or clean timing is part of this prototype. Complete source provenance records the
main-only 8ac08f0a base and exact low-level cap import.

Initial gates include 6,522 baseline/candidate validator comparisons (all output
bits, strings, inputs and storage identity), native analytic/cap suites, actual
reference observer route probes, and exact-cap/one-byte-short CPU resource tests
for direct/DF RHF/UHF with native-context-call spying. These are initial gates,
not complete molecular or default qualification.

## Rejected alternatives and remaining qualification

Do not import the broad bdcbc008 environment policy, use automatic/OpenBLAS,
raise resource caps, silently retry a failed scalar solve through reference,
change overlap/core/seed routes, or call the aggregate allowance spare workspace.
A checked adapter retaining FC during Gram has a 5M peak and is rejected.

The direct adapter test propagates allocation failure without fallback. The
preliminary helper test does not retry bad_alloc and uses the same callback for
its bounded numerical retry. This does not cover ordinary Fleet warm-to-cold
recovery: its unchanged catch path can retry an OOM. Do not claim endpoint-wide
allocation-failure no-retry behavior. The provider remains fixed on every attempt.

Broader molecular state/force/oracle, degeneracy/scale, warm/cold/failure lifecycle,
larger-size/resource and complete endpoint performance qualification must follow
before proposing default readiness. CUDA exclusion is source-proven here only;
no real-device qualification is claimed. The separate exhaustion-injection test
changes a scratch copy's sweep bound and is failure-plumbing evidence, not proof
that natural input reaches 100 sweeps. Existing reference output remains the
independent scientific oracle, not a hidden production fallback.
