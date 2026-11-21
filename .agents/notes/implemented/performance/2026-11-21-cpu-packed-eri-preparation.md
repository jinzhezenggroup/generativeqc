# Decision: Retain exact packed CPU ERI values under one prepared-batch cap

Status: implemented
Date: 2026-11-21

## Decision

Use eightfold permutation symmetry of full-range real-spherical FP64 ERI
values to store just one double per unordered pair of unordered AO pairs.
The admitted KS batch owns the request and checks the aggregate extent before
allocating the first array. Geometry replay reuses that policy and retires the
old plan before replacing it. This does not pack nuclear derivatives or alter
the orbital, quadrature, response, or electronic-root contracts.

## Rejected alternatives

A process-global environment flag made replay depend on unrelated changes to
the process. Separate per-item caps allowed a fleet to multiply its admission
limit. An automatic dense retry could silently exceed the memory the caller
expected. Keep the default unchanged, and refuse an explicit packed request
when its scientific or storage preconditions are not satisfied.

## Invariants

The packed capacity counts the exact retained value arrays, not other
allocations or RSS. Preserve support for the original size-tagged named KS
descriptor: read only the validated old prefix and initialize the appended
budget to zero. New callers pass the full size. Do not retain any descriptor
storage or string from the caller in prepared owners.

## Verification

The native named KS catalog test exercises the original descriptor through
system creation, single preparation and batch preparation, and checks that
unknown sizes do not publish handles. The spherical ERI projection test
compares packed and ordinary values/Fock execution. A retained-memory result
alone cannot qualify complete Hessian or saddle performance; time full
endpoints and validate derivatives against independent references.
