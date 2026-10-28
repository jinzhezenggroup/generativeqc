# Decision: share correlated convergence policy with Python planning

Status: implemented
Date: 2026-10-28

## Problem

The native correlated solvers already consume `core::CorrelationConvergence`.
Python CBS planning and standalone CC validation independently repeated its
budget-to-scale and scale-to-ceiling arithmetic. This production call path is a
bounded R4/R6 gap in #2227, independent of HF projection, optional DIIS diagnostics
and pending CUDA provider ownership. Moving research-only refinement ranking
would not address this production duplication.

## Decision

Expose a synchronous public metadata query which invokes the existing native
policy and publishes its scale and eight ceilings atomically. Python CBS obtains
one policy result for its component allowance; standalone Calculator checks its
CC ceilings against its already selected library before creating a context.
No scientific equations, precision, convergence defaults, or solver work change.

Python retains CBS coefficient/error allocation, explicit tighter controls and
the separate reference-density/CC-residual caps. Those are different contracts
from the shared solver's generic ceilings. Moving them into this query would
change its domain or inadvertently loosen an incumbent CBS constraint.

Native-free standalone planning and missing-query libraries retain the previous
metadata arithmetic explicitly. A selected library never queries another
library, and an admitted native failure or malformed receipt never retries the
compatibility implementation. Missing the metadata query does not grant the
separate convergence-transport capability needed to execute relaxed solvers.

## Invariants

- All-zero native allowances select strict convergence; zero RMS means omitted.
- The supported scale bracket, ceilings and scientific owners are unchanged.
- Explicit tighter settings and CBS-specific caps remain binding.
- Errors preserve output, including overlapping input/output storage.
- No solver, context, device, compiler or Python is needed by the SDK query.
- CPU metadata evidence is not CUDA numerical or general native CBS ownership.

## Evidence

The public-header executable checks boundary/invalid values, complete output,
overlap and atomic failure. Compiled-adapter tests compare legacy limits over
600 random budgets and bracket neighbors, exercise concurrent queries, forbid
Python arithmetic on admitted CBS construction, and inject transport failures.
The existing pinned PySCF water CBS/component oracle and complete-endpoint
comparison are retained with source provenance under
`benchmarks/results/correlated-policy-20261028/`.

## Revisit when

CBS composition itself gains a qualified shared native owner, or the supported
correlated stopping policy changes with independent scientific acceptance.
