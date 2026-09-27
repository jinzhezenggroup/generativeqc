# Decision: source-bind stationary capacity qualification

Status: implemented
Date: 2026-09-27

## Problem

DFT-MP-v1 needs exact production-capacity evidence for its frozen semilocal
def2-SVP force rows before GPU qualification. The production admission path did
not expose one shared estimator for every resource gate, while the files owning
those gates were under active development in other pull requests. A detached
estimate could silently become stale and mislabel a case as admitted under a new
source revision.

## Decision

Keep the qualifier read-only and outside the production runtime. Reuse shared
production planners where they exist, compute the remaining deterministic work
bounds from frozen inputs, and accept copied formulas only after verifying their
exact production definitions and gate predicates.

Bind the spherical AO term counts and AO count through the complete native
packing chain. Bind grid point counts through the source-only grid constructor,
public and native GridSpec ABI lowering, generated quadrature layout, native
CUDA materializer, backend selector, and native point-count publication. Bind
the complete public semilocal capability predicate/promotion, CUDA force method,
and stationary packaged-AOT CMake loop. Compute admission for each method/spin
row with its own memory plan, and verify all seven production gates in source
order. Python owner functions use hashes of their exact source spans. Do not
hash `ast.dump()` output: its serialization
changes across supported Python versions even when the source does not. Every
report records the clean Git HEAD, frozen input/basis/grid identities, owner
hashes, gate order, first blocker, and whether the optional packaged AOT is
present and valid.

The writing CLI may regenerate one untracked `.json` report inside the checkout
and exclude exactly that path from the clean-tree check. The public
`build_report()` API has no exemption, and the CLI refuses to replace a tracked
file. No other dirty path is ignored.

Imported planner, basis, grid, and AOT helpers retain their import-time source
path and content identity. A long-lived process fails closed if the same
checkout changes underneath already-imported helper objects.
The tool also pins the checkout HEAD at module import and requires a fresh
interpreter after any commit/checkout transition, covering transitive imported
state that is not represented by a helper's defining file alone.
Each frozen selector is additionally resolved through the public MethodIR/KS
path and must reproduce the audited native ABI ID, spin, semilocal coefficients,
execution domain, native DFT eligibility, batch/energy registry eligibility, and
packaged stationary plan identity.
The public MethodIR must also reproduce the production native functional-family
code that selects grid derivative order and the packaged AOT profile.
The bundled basis-pack and named-record caches are cleared before every report,
so a clean checkout switch cannot pair a new pack hash with stale expanded data.

## Rejected alternatives

- Raising caps or editing production owners was rejected because capacity
  discovery must not preempt active runtime/compiler ownership or change the
  scientific contract.
- Running the public CUDA endpoint was rejected for this phase because a static
  admission failure occurs before useful GPU execution and would not establish
  scientific force accuracy.
- Regex-only checks were rejected because they can retain an error message while
  changing the defining expression or predicate.
- Raw AST serialization was rejected because `ast.dump()` is not stable across
  the repository's supported Python versions.
- Hashing entire owner files was rejected as the only contract because unrelated
  edits would create noise without identifying the resource semantics that moved.

## Invariants

- Source or predicate drift fails closed before any capacity conclusion is
  emitted.
- Row admission uses that row's method/spin memory plan; a conservative
  case-level maximum cannot be propagated to every row.
- The reported first blocker follows the verified complete production gate
  order, including shape, topology, work, device, and host gates.
- AO and grid counts must agree with the verified native production formulas;
  source-only construction is not accepted as an independent proxy.
- Grid memory uses the verified production derivative-order and `plan_tiles`
  inputs before any copied byte bound is accepted.
- Static admission is never reported as scientific qualification.
- The qualifier does not change caps, tolerances, frozen cases, or losing rows.
- A report SHA identifies a clean source revision; only the writing CLI's
  validated untracked JSON output may be excluded from the cleanliness check.
- Changes such as the proposed whole-force to page-local primitive-work contract
  require recomputing the matrix from the merged source, not predicting results.

## Evidence

- `tests/python/test_dft_mp_v1_capacity.py` mutation-tests the production
  definitions, public capability route, packed AO/count chain, native grid count
  chain, clean-source binding, and output-path exception.
- The source-span hashes derive from exact source text rather than interpreter
  serialization; a regression rejects any return to `ast.dump()` fingerprints,
  and repository CI covers Python 3.11 in addition to local Python 3.13.
- `python -m tools.dft_mp_v1.qualify_capacity` emits all frozen required rows
  without loading a native library, initializing CUDA, or compiling science.

## Consequences

The qualifier provides reproducible blocker evidence without taking ownership of
production code. Its deliberate maintenance cost is that an intentional resource
or packing change must update the audited definitions, retained hashes, tests,
and report together.

## Revisit when

- production exposes one public, side-effect-free estimator for every admission
  resource;
- the page-local primitive admission proposed in #1486 lands;
- the packed AO representation or stationary layout owner changes; or
- DFT-MP-v1 changes its frozen basis, grid, products, or required cases.

## References

- #1186
- #1191
- #1462
- `tools/dft_mp_v1/qualify_capacity.py`
- `tests/python/test_dft_mp_v1_capacity.py`
