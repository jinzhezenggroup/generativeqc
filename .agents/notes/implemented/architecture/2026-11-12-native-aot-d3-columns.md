# Decision: bounded Python-free D3(BJ) columns from canonical scalar Graphs

Status: implemented
Date: 2026-11-12

## Problem

The native D3 production owner exposed first order only. The existing second
order consumer used compiler-generated TensorIR with runtime Python/JIT, which
cannot supply the Python-free CPU stationary endpoint. Deriving another set of
native D3 second derivatives would create a second scientific authority.

## Decision

Four explicit AOT leaves compose the incumbent dispersion scalar Graphs:
coordination, normalized atomic reference weights, pair energy and the reverse
coordination scatter. Graph differentiation supplies every first/right-mixed
term, including both atoms' complete normalized-reference C6 response. The
native consumer owns geometry/elements, numeric admission, traversal and scatter.
The shared physical `D3Parameters` moves to a generated-header-free header so
callers need not include native first-order implementation or generated tables.

The consumer performs three unordered-pair traversals. The first caches one
coordination jet per pair, the second accumulates direct gradients and CN
adjoints/right adjoints, and the third consumes those same cached jets. Seven
reference slots per atom and 49 pair terms are explicit semantic work, not
hidden behind a bounded memory claim. No left-coordinate source replay occurs.

## Numerical boundary

Direct AD of tiny Gaussian normalization can overflow reciprocal powers before
finite normalized weights/derivatives are formed. A 25-atom Ni/I/Cs/H component
control exposed this failure. The generated reference closure subtracts a common
maximum valid Gaussian exponent before exponentiation and differentiates the
same normalized ratio. Common scaling cancels, including at maximum-branch
ties; invalid reference slots remain lazy zero branches. No floor, screening,
parameter change or altered derivative acceptance gate is introduced.

The unscaled normalization remains an explicit positive-domain gate. A dense
40-atom control underflows it completely and is retained as a transactional
negative control, not silently admitted using the first-order fallback. The
positive 40-atom correction control is a different, sparser geometry within the
incumbent qualified domain. Uncut two-body BJ/ATM-off is the same domain as the
existing TensorIR HVP; finite cutoffs and ATM still refuse.

The independent simple-dftd3 oracle explicitly sets all real-space cutoffs to
1e6 bohr, beyond every retained system distance. Its default CN cutoff otherwise
omits pairs beyond 40 bohr in the sparse 40-atom control, producing a different
physical calculation. This oracle alignment does not relax numerical gates.

## Ownership and resources

Every shape-dependent allocation follows numeric/pair/reference cap admission.
The plan includes owned inputs, directions, all simultaneous numeric panels,
cached pair jets, leaf scratch and index permutations; static tables are
reported separately. Callback/object and allocator overhead, caller storage
and total RSS remain outside the numeric cap. Both nuclear outputs publish
together only after final owner validation and actual direction authentication.
Direction/output aliases are safe; output/output aliases and reentry refuse.

## Evidence and scope

`tests/python/test_host_d3_column.py` builds and links the production static
target, compares energy/gradient and multistep HVP/raw-Hessian controls against
independent simple-dftd3, and exercises rigid motion, rotation, owner snapshots,
semantic caps, malformed buffers, final mutation, reentry and recovery. Its
standalone executable runs with no Python/compiler on PATH. Local source-matched
raw evidence is retained under `build/qualification/host-d3-column-20261112`.

These are complete D3-only correction controls. They do not qualify electronic
SCF/CPKS, combined stationary Hessians, installed public endpoints, TS convergence
or complete 25--40-atom electronic costs. The requested correction must be
composed exactly once by the eventual verified stationary owner.

## Rejected alternatives

Production Python/JIT breaks the installed native runtime requirement. Handwritten
Hessian algebra duplicates existing scalar authority. Frozen C6 drops CN response;
per-coordinate source traversals hide work amplification. Dropping normalization
or finite-cutoff gates would promote unqualified domains.

## Revisit when

Independent qualification admits normalized-reference underflow fallbacks, finite
cutoffs or ATM second order, or larger systems justify a different bounded pair
schedule with complete correction-endpoint evidence.
