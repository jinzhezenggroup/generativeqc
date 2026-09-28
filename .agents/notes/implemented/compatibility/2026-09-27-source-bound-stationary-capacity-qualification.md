# Decision: source-bind stationary capacity qualification

Status: implemented
Date: 2026-09-27
Updated: 2026-09-28

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
packing chain. Derive frozen shells through the source-only production
`snapshot_basis(...).shells_for(...)` path and bind the Calculator shell/native
system forwarding methods that connect it to execution. Bind grid point counts
through the source-only grid constructor,
public and native GridSpec ABI lowering, generated quadrature layout, native
CUDA materializer, backend selector, and native point-count publication. Bind
the complete public semilocal capability predicate/promotion, CUDA force method,
and stationary packaged-AOT CMake loop. Compute admission for each method/spin
row with its own memory plan, and verify the complete production gate order.
The 16M primitive budget is page-local: cumulative logical records remain an
exact uint64 coverage metric, while admission checks the maximum single
descriptor and the bounded bulk/scalar page producers. Python owner functions
use hashes of their exact source spans. Do not
hash `ast.dump()` output: its serialization
changes across supported Python versions even when the source does not. Every
report records the clean Git HEAD, frozen input/basis/grid identities, owner
hashes, gate order, first blocker, and whether the optional packaged AOT is
present and valid.

The writing CLI may regenerate one untracked `.json` report inside the checkout
and exclude exactly that path from the clean-tree check. The public
`build_report()` API has no exemption, and the CLI refuses to replace a tracked
file or any multiply linked output. No other dirty path is ignored, and an
output alias cannot truncate consumed AOT evidence through a shared inode or a
symlinked manifest/binary target. Symlinked AOT evidence is rejected outright.

Imported planner, basis, grid, and AOT helpers retain their import-time source
path and content identity. A long-lived process fails closed if the same
checkout changes underneath already-imported helper objects.
Import-time data dependencies are bound separately from their loader modules;
in particular, the public-method registry manifest is fingerprinted before its
generated-method payload can be accepted under a later clean checkout state.
The tool also pins the checkout HEAD at module import and requires a fresh
interpreter after any commit/checkout transition. Its own source is separately
fingerprinted at import, covering a dirty-import/restored-file sequence as well
as transitive state that is not represented by a helper's defining file alone.
All imported repository-local Python modules are fingerprinted for the same
reason, and captured cross-module helpers such as `plan_tiles.jet_indices` keep
an explicit object-identity binding.
The qualifier also rejects repository-local modules loaded before its own
import, because their executed code cannot be reconstructed from a subsequently
restored clean source file. Normal CLI use therefore requires a fresh process.
Each frozen selector is additionally resolved through the public MethodIR/KS
path and must reproduce the audited native ABI ID, spin, semilocal coefficients,
execution domain, native DFT eligibility, batch/energy registry eligibility, and
packaged stationary plan identity.
The public MethodIR must also reproduce the production native functional-family
code that selects grid derivative order and the packaged AOT profile.
The no-runtime-compilation claim is bound to the complete prepared execution
artifact-selection method, including stationary and grid AOT branches.
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
  order, including shape, topology, logical-metric range, grid work, memory, and
  page-local descriptor work.
- AO and grid counts must agree with the verified native production formulas;
  source-only construction is not accepted as an independent proxy.
- Grid memory uses the verified production derivative-order and `plan_tiles`
  inputs before any copied byte bound is accepted.
- Static admission is never reported as scientific qualification.
- The qualifier does not change caps, tolerances, frozen cases, or losing rows.
- A report SHA identifies a clean source revision; only the writing CLI's
  validated untracked JSON output may be excluded from the cleanliness check.
- Git index entries marked assume-unchanged or skip-worktree are rejected before
  the clean-tree check so hidden tracked edits cannot inherit the HEAD identity.
- Report output may not overlap the AOT directory whose manifests/binaries were
  consumed as evidence.
- Page-local primitive admission is source-bound to the merged #1486 bulk,
  scalar, spherical-component producer, component-mode selector, source
  constructor/native-create budget forwarding, flush, executor, and
  endpoint-wiring contracts. The endpoint descriptor-production
  call must remain after the verified host/device admission gates, and its
  nested callback must continue forwarding the same page coordinates and
  nuclear-attraction tasks into `integral_page`. Cumulative logical work is
  evidence, not a whole-force cap.
- The public stationary wrapper is source-bound as a complete function so its
  defaults and every forwarded resource keyword remain identical to the private
  endpoint whose gates are audited.
- Python nuclear-pair routing and native `Owner`, `allocation`,
  `stationary_create`, `stationary_reset`, `stationary_tasks`, and
  `stationary_nuclear` blocks are part of the same contract, binding the exact
  byte formula and shape caps, page-budget retention, reset, validation, and
  cumulative integral and nuclear-pair metric semantics.
- Seal the complete Python stationary source owner and endpoint function plus
  the native stationary header as semantic surfaces, while retaining narrow
  method/function hashes for diagnostic localization. This binds source/rank
  domain enumeration, grid tiling, geometry routing, pair-visit accounting,
  and metric publication without treating a broad digest as the only evidence.

## Evidence

- `tests/python/test_dft_mp_v1_capacity.py` mutation-tests the production
  definitions, public capability route, packed AO/count chain, native grid count
  chain, page-local primitive producers, clean-source binding, and output-path
  exception.
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
- the page-local descriptor or bounded task-executor contract changes;
- the packed AO representation or stationary layout owner changes; or
- DFT-MP-v1 changes its frozen basis, grid, products, or required cases.

## References

- #1186
- #1191
- #1462
- `tools/dft_mp_v1/qualify_capacity.py`
- `tests/python/test_dft_mp_v1_capacity.py`
