# Decision: conventional RCCSD borrows shared prepared contractions

Status: implemented
Date: 2026-10-05

## Problem and decision

The initial #1868 traversal recognized valid directly representable contraction
layouts, but coupled the CC owner to a raw cuBLAS callback and a method option
choosing that implementation. Retain the recognition and original scalar kernel
set while emitting the canonical typed descriptors already used by DF CC.

Prepare every eligible unbatched/leading-batch contraction once through the
shared tensor table. The generated traversal only supplies slot, runtime shape,
borrowed operands and error state. The method owns iteration/amplitudes/DIIS;
the shared provider owns execution and finite-result auditing. No conventional
provider-selector option is added.

## Invariants

- Original equations, arena lifetimes and non-contraction kernels are shared
  between prepared and scalar traversals; no second generated kernel set.
- Independent expanded replay and convergence criteria remain unchanged.
- Descriptor host storage and the shared provider allowance coexist with
  borrowed reference, problem, device arena and detached amplitude storage.
- Dimension/resource admission and optional setup allocation failure retain
  the complete original scalar traversal; execution failures never retry.
- Native descriptors retain scientific/semantic/precision identity and prove
  their matrix recipe against the original modes before replay.

## Validation and scope

Extend native descriptor validation to conventional iteration at nonrepresentative
unit and rectangular dimensions. The injected owner test covers setup failure
unwind and budget fallback without weakening the current DIIS ring behavior.
The supplied-Hamiltonian probe reports conventional binding/work capacities and
compares the prepared solve and exact-budget scalar fallback against independent
determinant-space results. Public molecular energy/force gates remain required.

This removes the direct conventional coupling from #1868 as part of #1886/#1890.
Existing DF implementation-selector cleanup, other response owners, optional
provider selection and complete comparative endpoint performance remain separate
work. Recognition alone is not evidence for a speedup or lower precision.
