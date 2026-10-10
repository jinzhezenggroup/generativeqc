# Decision: native DFT batch forces reuse the prepared single-system sink

Status: implemented (exact-head qualification outstanding)
Date: 2026-10-11

## Problem

The public native single-system CPU DF-PBE/PBE0 calculation has a
verified-token analytic force implementation, but `KsPreparedBatch::execute`
still rejects *all* native DFT force requests. Python `PreparedBatch`
currently performs post-SCF stationarity/derivative orchestration separately,
leaving C++/CLI and Python capability asymmetry. Copying the complete
force formula into the C++ batch would create another scientific owner.

## Decision

Extract the existing native single-system stationary force block into
`KsPreparedCalculation::add_prepared_native_forces(Result&)`. Both
`KsPreparedCalculation::execute` and `KsPreparedBatch::finish` invoke it
after convergence, before downstream correction application or publication.
The function uses the **same prepared final-state generation token**,
revalidated D/W, bounded stationary H'/Pulay/J'/K' sources, native PBE XC
moving-grid response, and nuclear repulsion as the former single-system path.

The batch entry has a whole-fleet fail-closed admission: every prepared
item must advertise the same existing context-qualified native forces
property. Per-item wrong coordinates, nonconvergence and numerical
execution failures still use the batch's existing item status and output
buffer isolation. `generativeqc_batch_execute` remains the sole public
batch C ABI and publishes forces only from successful native items.
No force-capability bit is added to method-global manifests.

## Rejected alternatives

- Copy single force source equations to a batch-specific implementation:
  permits J/K/XC terms or signs to drift.
- Call Python from the native batch runtime: violates Python-free SDK
  behavior and increases per-item host/device orchestration.
- Enable native CUDA/UKS/Direct/global corrections by method name:
  their source completeness and gradients are separately gated, and a
  successful energy is not a qualified force.
- Switch the public Python calculator immediately: it currently supports
  a broader force domain; cutover requires independent native batch
  evidence and a prepared-context capability query, not a guessed method
  selector or silent fallback.

## Invariants

- Energy/Fock/DF/SCF equations, grid, thresholds, approximation and method
  identities are unchanged.
- Complete force is `-dE/dR`, all four native integral sources plus XC and
  nuclear terms, with no re-solve or CPU oracle substitution.
- A final-state token is checked before/after consumption; malformed input,
  insufficient resources and unqualified contexts cannot publish a force.
- Final state and warm seed of one item are not used as another item's
  provider. A bad item cannot contaminate successful neighbors.
- No claim of production CUDA DFT force residency, universal PBE0 force
  acceptance, or Python API migration is made by this narrow native slice.

## Evidence required before merge

1. Exact-head native CPU and CUDA compilation with no new compiler/runtime
   dependency. Native C++ batch H2/sto-3g CPU DF-PBE E+F compare against
   the existing single-system source, in cold, warm and displaced geometries.
2. The existing reconverged central-difference force gate remains intact;
   independent PBE0 full-force finite differences under #2222 remain required
   before PBE0 acceptance is expanded.
3. Fail-closed CPU Direct and mixed/wrong force-domain tests; invalid
   neighbor must return per-item status with untouched force buffer.
4. Inspect live token ownership, resource budget and no incomplete source
   publication; unchanged Python API tests.
5. No regression beyond the repository's full-endpoint performance gate,
   including allocation/transfer/SCF iteration counts where applicable.

## References

#2151 (native analytic forces); #2222 (PBE0 finite-difference gate);
#926 (shared scientific definition); #933 (lifecycle); #934 (cutover);
#2227 (remaining Python to C++ runtime orchestration).
