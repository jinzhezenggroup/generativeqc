# Decision: reuse row-invariant operands without changing ordered reductions

Status: implemented
Date: 2026-11-14

## Problem

The saved six-atom, 81-AO I–CH3–I request still exceeded its original
1800-second complete-matrix limit. Its carbon-z column separately exposed a
physical response residual above the original absolute `2e-10` gate despite
relative solver convergence. Neither more time nor an isolated kernel win
establishes a qualified Hessian. The previous
[ordered host-grid contraction decision](2026-11-13-ordered-host-grid-contractions.md)
preserved scalar reduction semantics but loaded a row-invariant operand again
for every output row.

A fresh, bounded carbon-z component profile used the original 87,458-point
grid and authenticated saved **native** D′, after revalidating F[D] and the
physical state against the actual loaded core. This was not a new SCF, response
solve, total HVP or Hessian evaluation. Of the approximately 18.05-second XC
source, about 12.07 seconds belonged to generated map execution, 4.12 seconds
to geometry reading and 0.70 seconds to the scalar mixed closure. Rewriting
the scalar closure was therefore not the first measured target.

## Decision

Keep the existing exact-unit, rank-two binary contraction eligibility. When an
input has zero output-row stride and both output extents are at least four,
`src/tensor/ordered_binary_contraction.hpp` shares that input's vector load
across four output rows. Each row has an independent four-column accumulator.
The selection depends only on actual affine strides and extents, not a graph
name, molecular fixture or hard-coded AO count.

Every output retains the same left/right multiplication order, positive-zero
initial accumulator and original serial reduction sequence. No horizontal
reduction, reassociation, FMA, BLAS, packing panel or caller-owned numeric
allocation is introduced. Semantic work and arena queries are unchanged.
Unsupported CPUs retain the scalar lowering. Small extents and contractions
without a row-invariant input keep the previous single-row vector schedule;
row and column tails use the incumbent ordered implementation.

Empty reductions can have null inputs. The remainder-row handoff must not
advance those pointers even when it performs no reads. Tests exercise empty
reductions both inside full row blocks and in the remainder rows, in addition
to enormous empty output shapes that must return without traversal.

## Rejected alternatives

- Changing the scalar XC closure first does not address the measured map
  bottleneck and unnecessarily reopens its domain and near-vacuum semantics.
- BLAS/reassociated sums remain disqualified by the previous vacuum-boundary
  changes; improved component timing does not waive those scientific gates.
- Increasing memory/work caps or packing full panels is unnecessary: this
  change reuses registers while retaining the bounded portable fallback.
- An AVX2-targeted function's local lambda did not inherit its target under the
  tested GCC. Explicit target-scoped vector operations compile without requiring
  global AVX2 flags. The failed private compile is retained as evidence.

## Evidence and provenance

The private instrumented class has a different name and hidden visibility, so
it cannot interpose on the baseline core's ordered kernel. Its outputs were
bitwise equal to the actual baseline owner. The private row-reuse candidate
measured 13.93810 seconds against a same-process 18.04597-second baseline.
These values motivated the production change; they are not production endpoint
timings. The production implementation additionally guards null pointers at
empty-reduction row tails.

Production source identity:
`8ec4fac79711b62daf8da73e21899252ba10ba37c20ed4767cd627412a5eb5e6`.
Core SHA256:
`9d4c8b345d9eb311f368fa79584a2973e19d189817d259985bdfc6197f37ec03`.
The baseline source and core remain separately identified in the retained
physical-state dependency; old physical metadata is not restamped as new.

Raw scripts, logs, numeric panels, identity bindings and test reports belong to
`build/qualification/host-moving-xc-row-block-source-20261114`; the private
diagnoses belong to `host-moving-xc-profile-20261114` and
`host-moving-xc-row-block-candidate-20261114` under the same parent. Their
retention receipts bind the archives and nested original source/state/oracle
dependencies. The production comparison uses the actual `HostGgaNuclearColumn`
from each explicitly bound core, not the renamed private candidate.

Three alternating baseline/production process pairs pass actual library/source
binding checks. All 38 output fields match each other and the original native
carbon-z source **bitwise**. Median complete XC source time is
18.075755910 to 13.971465633 seconds, a 22.706% time reduction. Including fresh
physical-state reconstruction/validation and component setup, median time is
24.732159934 to 20.643294037 seconds. Median peak RSS is 390652 versus
390380 KiB; this is not a material memory-improvement claim. Both variants
execute 33,556,112,408 map terms, 524748 scalar-point visits, 23876628 AO-motion
visits and 342 tiles over the same original grid, with the same 30,557,832-byte
numeric reservation and original 128-MiB cap. The comparison has no newly
solved response column and is not a complete electronic or molecular endpoint.

Permanent affine/grid-map controls pass 222 groups, including direct serial
parity, forced-scalar execution, block/tail shapes, signed/subnormal/cancellation
inputs, unit/nonunit overflow, unchanged work/capacity queries and transactional
refusals. The direct production-header ASan/UBSan probes pass in both native
AVX2 and forced-scalar modes across 1296 control groups each. These probes are
execution controls, not independent molecular derivative oracles.

The original cold five-module consumer test request hit its unchanged
900-second timeout after 83 passing test calls, mostly spending time compiling
the same large native fixture twice. Its exit `124` is retained, not relabeled
as success. A separate replay retains only native temporary build paths; all
fixtures still configure/build normally and every test still runs, with fresh
ordinary test directories and the same 900-second limit. Its JSONL progress
journal retains complete per-phase outcomes even when timeout prevents pytest
from writing a terminal JUnit report.
That replay also hit its finite limit, with 87 fully passed tests and the
three-center unrestricted residual-2 energy-stencil test still in progress.
Only the five uncompleted tests are selected for a separate finite invocation;
no completed setup/call/teardown is replayed or inferred. A collection-derived
inventory and per-phase journals authenticate the merged 92-test report. This
is multiple bounded test invocations, not a claim that the entire test suite
finished within one 900-second budget. Molecular endpoint caps remain unchanged.
The supplement passes all five tests in 391.51 seconds. All 92 consumers are
therefore covered, retaining their independent AO/Libxc and reconverged-energy
controls, nonempty RKS/UKS responses, lifecycle and failure-publication checks.
Six selected CPU CTests, six documentation tests, Sphinx warnings-as-errors,
the compiler-ownership audit, Ruff and C++ formatting also pass. This is not
a claim that all 148 configured CTests, GPU paths or heavy molecular endpoints
have been qualified.

### Newly solved coupled-response follow-up

`build/qualification/host-row-reuse-carbon-response-20261114` subsequently
revalidates the physical state and **newly solves** the same one carbon-z
direction with the new core, rather than injecting the saved component D′.
The original `relative=0`, `absolute=1e-12` diagnostic controls and absolute
physical `2e-10` gate are unchanged. It returns residual
`8.406008203576136e-14`, 14 iterations and 28 actions. D′, W′, integral-density,
moving-XC and residual-ECP panels match the old actual response bitwise; work,
capacity, diagnostics and BJ results are unchanged.

The complete bounded component request takes 242.692852 seconds versus the
retained 249.530575-second baseline, only about 2.74% lower in this single
comparison. First-source preparation remains 168.793351 seconds. The coupled
solve takes 50.721371 seconds, including 41.238827 seconds of actual GMRES
actions and a 9.482544-second remainder. XC column time is 14.003234 seconds;
peak RSS is 437500 KiB. In particular, the GMRES action total is not faster
than the old 40.47935 seconds, so the 22.7% XC component result must not be
extrapolated to a similar response or Hessian speedup. A measured subdivision
of induced-Fock J/K versus cached XC work is a justified next diagnosis;
another unchanged complete-matrix request or a larger cap is not.

## Scope and revisit conditions

This decision qualifies ordered component execution, not a complete SCF,
response, physical raw Hessian, modes or optimizer speedup. Saved native D′ is
only an authenticated test-component input; it is never injected into a
complete endpoint or used to restore a verification token from JSON.

The original full-matrix timeout and absolute-response failure remain open.
Future whole-request qualification must satisfy the original absolute physical
`2e-10` gate for every response column, not merely the relative solver status.
Stricter carbon-z diagnostic controls already demonstrated one passing
direction, not all coordinates. Both public ECP saddle guards remain closed.

Revisit the tile size or ISA only with unchanged-domain, same-graph boundary
controls, independent scientific references, complete-component work/storage
accounting and actual endpoint qualification appropriate to the claim.
