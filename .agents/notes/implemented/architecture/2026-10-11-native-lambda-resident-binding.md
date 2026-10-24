# Decision: bind conventional CUDA Lambda to the existing resident controller

Status: implemented (opt-in, not promoted)
Date: 2026-10-11

## Problem

The shared logical Lambda plan did not make its CUDA vectors resident. The
production transpose callback uploaded dense seeds and downloaded dense outputs
on every physical action. The existing resident controller was unpreconditioned,
whereas production Lambda normally uses right diagonal preconditioning.

## Decision

Extend the existing resident controller with a method-owned resident right
action, retaining unpreconditioned candidate residual acceptance. Bind the
resident backend to the same private logical problem token as the host plan;
wrong same-size actions fail before uploads. No second solver or equations are
introduced. Method generation/borrowed-state proofs still belong to method
owners, not this opaque token.

The conventional Lambda adapter stages its existing representative, partner,
sqrt-weight and safe diagonal tables once. Packing/unpacking, preconditioning
and generated transpose actions execute on the immutable terminal owner's
existing stream. Only reductions/error flags and one final packed solution are
downloaded. The original independently generated Lambda action/RHS and scalar
physical gates remain unchanged after the solve, before parameter publication.

Vendor calls belong to the shared tensor vector service, reusing the existing
contraction context's stream, strict-FP64 modes, explicit workspace, versions and
bounded provider preparation. Admission includes the vector arena alongside the
generated arena, borrowed reference allowance and full existing provider
allowance. Workspace is inside that provider allowance, not an extra uncharged
allocation. Numeric arenas and observed provider growth are distinct diagnostics.

Budget/provider-allocation refusal retains the original host-vector solve, with
the same diagonal policy. Other failures propagate. Stream work drains before
any provisional storage or BLAS context is destroyed. This owner is per-call;
the change does not claim prepared reuse or #933 lease coverage.

## Rejected alternatives

- Disabling default diagonal preconditioning to fit the resident interface
  changes iteration/work behavior and would misrepresent production acceptance.
- Submitting cuBLAS directly from the method owner bypasses shared vendor
  ownership and omits the retained provider allowance.
- Promoting on noninteracting microbenchmarks ignores complete endpoint work,
  extra provider memory and the scalar-reduction/fence cost on small problems.
- Treating the logical token as a scientific generation or lease is unsound.

## Evidence and remaining gates

The deterministic backend test covers safe/nonfinite right actions, recovery,
nonsymmetric equations, exact scalar budgets, binding refusal and one-time vector
transfers. Native tests retain independent noninteracting Lambda references,
unsafe/pair-inconsistent denominator fallback and two valid CUDA budgets.
The complete molecular driver pairs the existing native RCCSD E+F schedules,
using pinned PySCF references and a native CUDA reference interaction source
rather than host raw ERIs. Source-matched receipts belong under
`benchmarks/results/response-contract/2026-10-11/`; failed attempts are retained.

Process-warm fresh-state calls are not prepared-plan warm reuse. Tiny H2 tests do
not establish larger-domain performance. The candidate remains non-default;
DF Lambda, frame response and installed CPKS resident migrations and the broader
#932 acceptance matrix remain separate work. The original 13-AO finite cold-JIT
gate is not weakened or counted as passing here.

## References

- The subsequent [DF action binding](2026-10-11-df-lambda-resident-actions.md)
  extends the same controller while retaining both provider allowances.
- #932 and PR #2272.
- [Earlier consumer boundaries](2026-10-11-response-consumer-contracts.md).
- `docs/developer/response.md`.
