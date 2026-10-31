# Decision: share explicit stationary force-map bindings across compositions

Status: implemented
Date: 2026-10-29

## Problem

Issue #1853 retains cross-functional active-AO numerical, geometry, capability
and resource acceptance. Ordinary forces already use a binding that checks
scientific identity, the grid owner and its generations, derivative layout and
the omission/resource policy. Composite forces separately constructed the same
map types with a domain-only reuse check. Their outer arena identity covered
several of these inputs, but two binding implementations obscured which
lifetime and replacement rules applied to each composition.

Current public force routing has also moved on: an admitted prepared native
item executes `CudaStationaryController`, whose force AO loops currently use
dense selection. Monkeypatching the Python profile registry and observing
`PreparedBatch._public_dft_cuda_force` therefore does not test selected maps on
that public route. A pilot exposed this as an absent work receipt. Do not
disable native capability queries to make an old routing test appear current.

## Decision

Use `_stationary_resident_ao_cache` for both ordinary and composite stationary
owners. Their optional map slot has a shared structural protocol. Closing
either owner clears the map and its binding before the underlying grid dies.
Rebinding drops the old owner before constructing a replacement under the same
allowance. No map predicate, numerical cutoff, profile or production default
changes.

The cross-functional CUDA matrix executes native SCF followed by the explicit
stationary force consumer. Its independent GPU4PySCF reference reconverges both
geometries with matching quadrature and moving-grid response. A separate public
native-force control forbids Python force orchestration and checks the same
independent reference. Dense and selected explicit results must agree with this
native control as well as each other. Receipts identify the explicit route and
retain actual selected/dense work; absence of an observed map cannot pass.
The reference explicitly rebuilds its full Fock. The initial incremental LDA
tetramer reference did not converge at the frozen tolerances; full-Fock replay
converged without weakening the scientific or SCF gates.

The matrix covers LDA, PBE, PBE0, r2SCAN and WB97M-V in RKS/unequal-spin UKS;
native exact-jet bitmasks, sampled composite maps, shared and independent
XC/nonlocal grids; first call, replay, changed geometry, changed-geometry replay
and zero-cache dense fallback. The existing host lifecycle suite now applies
to both owners, including stale-token rejection, generation/policy changes,
replacement failure and cleanup. Empty/noncontiguous maps remain covered by
the resident-map and nonlocal-consumer suites.

## Boundaries and rejected alternatives

- Do not promote a composite profile from small-fixture numerical evidence.
  Public native-controller adoption and complete cold/warm/moved performance
  qualification remain open in #1853. Historical ordinary-force promotion does
  not establish native-controller work or profitability.
- Do not qualify a derivative-order-two force from an SCF order-one map.
  Every map binds the force grid's actual order and full global capacity.
- Do not share maps between independent XC and nonlocal quadratures. The
  explicit split-grid consumer retains a dense nonlocal traversal.
- Do not repair an AOT identity mismatch by rewriting metadata. Build the
  current matching force library and its manifest; the first GPU pilot's stale
  package was correctly rejected before execution.

The test timings include SCF and force publication, with preparation charged
to the first call. They are numerical acceptance observations in a shared
process, not fresh-process cold performance measurements or speedup claims.

## Validation

Host lifecycle, policy, nonlocal composition and resource/schedule checks:
363 passed. The opt-in device matrix passed all 14 cases on node1 under finite
Slurm job 14898, with 140 explicit E+F endpoints and 28 public native controls.
Its energy/force gates are 1e-8 Eh
and 3e-7 Eh/Bohr against GPU4PySCF, with 1e-9 Eh/Bohr selected/dense/native
agreement and physical SCF residual at most 1e-9.

The [retained acceptance](../../../../benchmarks/results/cross-functional-force-ao-20261029/README.md)
contains full numerical/work records and the exact source patch. Stationary AOT
packages were rebuilt from current source with verified sccache; the core
native library was reused and its original source identity is not independently
asserted. No complete current-source native-core performance claim follows.

Reproduce using `GENERATIVEQC_DFT_CUDA_TEST=1 python -m pytest
tests/python/test_cross_functional_force_ao_cuda.py` inside `srun`, with a
qualified native library, both s/p and s/p/d stationary AOT packages, and the
reference dependencies. JUnit `force_map_endpoints` properties preserve each
stage's numerical errors, complete endpoint timing and force-work record.
