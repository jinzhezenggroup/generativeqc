# Decision: one native element-template expansion owner

Status: implemented
Date: 2026-10-16

## Problem

The checksum-bound bundled catalog constructor removed one Python atom-expansion
path, but imported/general-contraction bases and scalar ECPs still repeated atom
mapping in Python and the native CLI. A second implementation for custom bases
would preserve the ownership split and invite scientific drift.

## Decision

Expose versioned FP64 element/shell/ECP templates in the additive C ABI and a
matching C++ constructor. The compiled catalog, canonical CLI adapter and Python
adapter all delegate atom mapping to `molecule/basis_template`. Existing
`system_create` and `system_create_ecp` retain normalization, electron/core
bookkeeping and scientific/backend admission. No equations change.

Shared exponents and row-major general contractions expand in stable atom,
shell and row order, preserving explicit zero coefficients. All output dimensions
are checked against uint32 capacity before allocation. Scalar ECP terms retain
their source angular labels until the native owner maps the greatest channel to
local `-1`, independently of input record order. Input storage is borrowed only
for the synchronous call; successful systems own copies.

## Boundaries and rejected alternatives

Do not add a JSON or cryptography dependency to the bare native library merely
to perform this mapping. Format/schema/provenance validation stays in frontend
adapters. Unlike the checksum-bound compiled catalog, caller-supplied numerical
arrays do **not** authenticate canonical JSON or establish full native model
identity. Python resource and model preflight are not retired by this change.

Python marshals only selected unique elements. Unsupported metadata retained for
an unused element must not change the selected molecule's capability. Rejected
native creation must not silently retry the older atom-expanded route. Explicit
`Shell` clients, libraries lacking the additive symbol and the separate
correlation-only auxiliary-through-g adapter retain bounded compatibility paths.

The template owner deliberately preserves the legacy scalar ECP adapter limits:
orbital s/p/d/f, greatest source channel through g and radial powers 0–4. Moving
mapping is not authorization to expand numerical coverage. Revisit these limits
only with backend capability changes and independent force/integral acceptance;
do not infer support from the more permissive retained metadata bounds.

## Validation and revisiting

Executable tests compare raw inventories to the portable adapter over general
rows, zeros, ragged repeated atoms, representations and shuffled ECP channels.
Public ABI tests cover transactional failures, finite/range/core checks, a small
input whose expansion exceeds uint32, and copied lifetimes. Complete prepared
endpoints compare cold/warm/moved, exact budget rejection and failed-member
recovery; independent PySCF references remain separate from production work.

Qualification must compare the exact parent Python **and** native library with
the candidate through construction, resource admission, preparation, execution
and close, retaining numerical/convergence/work signatures and all samples.
Real CUDA qualification is still required before making CUDA equivalence claims.
This slice does not complete the broader #2227 R5/R6 migration.
