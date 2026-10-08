# Frozen CPU spectral-resource migration oracle

This fixture contains the complete old CPU GFN2 eigensolver and its local
header/source dependency closure from commit
`5c02b1ba3c69fbee712697565dcb79da6b768606`. Every source listed in `manifest.json`
was copied byte-for-byte using `git show`. The weighted-Gram header was emitted
by the generator and Python compiler extracted from that commit with `git
archive`, never by the current compiler. All bytes are SHA-256 guarded.

The frozen executable sees only these headers and sources, plus the identical
public-API test harness. Its actual shared LP64 provider admits the existing
mock-provider cohort; test callbacks extend that cohort with stack-only
prescribed small-system arithmetic and precise backend failure injection.
Candidate and frozen executables independently run the same ragged restricted
and unrestricted cases. They compare exact storage sizes and offsets, all
published cache/output bytes, caller-owned padding, diagnostics, allocation
counts, provider operation traces, and occupation-body entry counts. Compiler
function instrumentation and `nm` count the unmodified private occupation
routine, including both equal-population restricted solves. The same mechanism
counts complete finite/symmetry scans across the method and shared owner and
records whether they occur before the local BLAS thread scope. Exactly one new
immutable metadata allocation is permitted during setup; steady operations
remain allocation-free.

The harness independently verifies prescribed generalized eigenvalues,
`H C = S C E`, `C^T S C = I`, occupations, density, energy-weighted density,
entropy, band energy and free energy. It tests full-batch atomic publication,
healthy peers around numerical failures, retention of an old failed overlap
factor, repeated/decreasing/nonzero generation stamps, worker-only bindings,
borrowed active spans, alias rejection, guard bytes and zero hot-call allocations.
The plan's implementation-dependent resident metadata byte count is checked
for validity but excluded from the migration byte-identity requirement.

The fixture is test-only and must never enter production source manifests.
Do not refresh it to make a changed implementation pass. An intentional
contract or numerical change needs independent acceptance criteria and review.
