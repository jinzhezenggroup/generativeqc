# Decision: compose CUDA KS declared device lifetimes natively

Status: implemented; full endpoint/GPU qualification pending
Date: 2026-10-16

## Problem

CUDA KS allocation leaves already have native owners, but Python and native CLI independently
composed the one-electron setup bound, rebuild excess, retained state/XC/J and
opaque matrix-provider allowances, and stationary minimum/cached alternatives.
Those production decisions could drift from one another despite each leaf
query remaining correct. Moving only a wrapper or renaming the Python loop
would not retire that ownership gap.

## Decision

`CudaKsDeviceFleetResources` owns the complete declared device composition of
those leaves. The CLI directly uses that typed owner for its currently supported
ordinary energy dry runs, without expanding its force/model domain. The additive
private C transport accepts compact topology, existing
native leaf capacities, the explicit XC schedule and both stationary bounds.
It returns component totals, both candidate peaks and diagnostic item phases.
It uses two scalar passes before publication: later invalid items, signed-int64
overflow and input/output aliasing cannot publish a partial plan.

The query is usable in CPU-only builds because composition itself neither
probes CUDA nor discovers providers. It cannot manufacture CUDA execution
support: leaf queries and consumer eligibility remain separate owners. Python
continues to marshal their results and publish the existing resource metadata.

Only absence of the new symbol keeps the old explicit per-item arithmetic.
Selected native refusal or corrupt publication never retries it. The public
candidate reports actual native composition versus legacy leaf compatibility.

## Lifetime invariants

- Value-only one-electron exporter scratch retires before J preparation.
- Quadrature setup overlaps the new Coulomb allocation; ECP scratch is an
  alternative setup phase, not a second persistent owner.
- Rebuilding one item need not retain its retired allocations. Fleet workspace
  charges maximum setup excess above that item's own retained state/XC/J, not
  the sum of setup peaks or a second full allocation of the same owners.
- Every retained matrix-provider allowance remains separate from spendable
  explicit arenas. Stationary minimum/cached consumers remain live during SCF.
- Minimum/cached are independent layout candidates, not an inferred ordering or
  scientific-admission policy. CLI resource-shape errors retain usage exit 2.
- Host-unfused XC suppresses only the device XC retained slot; other setup and
  stationary controls keep their original semantics.
- Declared-capacity composition is not complete physical allocation/RSS
  authority, resource-provider admission, universal force support or HVP.

## Evidence and remaining acceptance

Development checks on master base `e4d00688e3ef3458b1bb4af23aa6eb829926910a`
plus this working patch: the complete 108-test CPU CTest suite and 430 Python resource
tests pass, with nine explicit real-GPU skips. The 42 new Python cases are
included in that total; they use the actual compiled native bridge, varied
ragged/ECP/XC/candidate metadata, old-symbol compatibility, corruption/refusal
injection, public plan equality and exact/one-byte-short declared budgets.
Mocked leaf capacities are not physical CUDA or independent science evidence.

The actual resource API object and ABI/core tests also pass ASan, fail-fast UBSan
and leak checks with unreferenced provider sections discarded at link time; this
is not whole-library/provider sanitizer coverage. Native tests cover signed-int64
boundaries, failed-append rollback, repeated items, malformed later descriptors,
output tails and aliases. No production source-body extraction or generated
function fingerprints are acceptance gates. Installed public-header-only C++20
PBE0 RKS/UKS CPU energy/force/repeat/moved finite-difference controls pass from
an empty working directory with `PATH=/nonexistent`. This uses the existing
caller-supplied composition, not a new public capacity API. Source/API/ownership
checks, changed-file hooks and warnings-as-errors Sphinx also pass.

Independent complete CPU/GPU endpoints,
installed CUDA SDK controls and fixed <=1.05 performance acceptance remain to be
completed before a new PR is declared Ready. No previous SDK/source receipt is
relabeled for this working patch. This advances R2/R6 of #2227, which remains
incomplete; R3 implementation remains deferred.

## References

- `src/scf/ks_device_resources.hpp`
- `src/api/c_api_resources.cpp`
- `python/generativeqc/resources_ks.py`
- `tests/native/test_ks_device_resources.cpp`
- `tests/python/test_ks_device_fleet_resources.py`
