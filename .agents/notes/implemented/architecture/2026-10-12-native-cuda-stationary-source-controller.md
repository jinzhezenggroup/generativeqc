# Decision: statically compose the existing CUDA stationary source owner

Status: implemented (private qualification endpoint; no public force promotion)
Date: 2026-10-12

Retained numeric/provider admission is subsequently composed by the
[shared ledger decision](2026-10-12-native-cuda-provider-admission.md); that
follow-up does not promote this private controller to a public force capability.

## Problem

The resident final D/W lease removed dense-frame export from Direct integral
derivatives, but the complete stationary consumer still belonged to Python.
Moving its loop while retaining a host addition of compact sources would not
establish the all-device scientific composition required by #2227.

## Decision

`stationary_native_cuda.py` emits a typed static binding around the unchanged
PBE/PBE0 RKS/UKS wrapper and shared s/p primitive object inventory. Preprocessor
aliases isolate namespaces, C exports, point/pullback types and global Becke
helpers without rewriting mathematical bodies, literals or IR provenance. The
shared external `first_derivative` remains unaliased. Binding identities, source
slots, required mask, exact-exchange fraction, nuclear dispatch and geometry
resource metadata derive from the same compiler plan/programs.

The existing Direct traversal and prepared one-electron pair now have resident
consumer adapters. Host adapters retain their endpoint contracts and submit the
same scientific kernels. One-electron/Pulay share one scratch block; a callback
must enqueue consumption before return. The Direct provider retains two J/K
force channels, with explicit force-to-energy-gradient conversion at transport.
Normal resident delivery performs no D2H, drain or numeric allocation. Capture
is rejected before submission and before exceptional synchronizing guards.
Host result buffers outlive their exceptional drains.

`native_stationary_runtime.cuh` wraps the existing scientific owner rather than
changing the packaged runtime ABI. It retains integral producer streams, drains
before publication/reset/destruction, rejects duplicate or incompatible sources
and validates complete ordered grid coverage. Reset discards failed sources
only after fencing their users. The existing emitted reduction remains the
only source sum; a diagnostic source span never authorizes a complete force.

`CudaStationaryController` binds resident spin density to the existing native
grid producer, consumes its actual generation/paired-feature witnesses, submits
generated XC/AO/Becke and nuclear sources, receives resident integral channels
and publishes atomically under the final token. Its additional device allowance
charges stationary/grid numeric arenas and the existing bounded provider
allowance, not the already-charged SCF/Fock owner. The grid arena formula is
factored into a shared portable capacity helper, without changing alignment,
buffer inventory or the native/Python allocation contract.

The private prepared-KS endpoint is qualification-only. Public force capability,
SDK/Python dispatch, default validation and explicit compatibility routes remain
unchanged. Static bindings are restricted to the qualified NVIDIA device-link
build route; other configurations return unavailable bindings. Provider-free
wheels retain the incumbent final-link import mechanism rather than gaining
strong CUDA/cuBLAS dependencies through the static archive.

## Rejected alternatives and failures

- An unauthenticated dynamic loader would add a competing identity framework.
- A host source sum would retain production scientific composition outside the
  emitted reducer.
- Keeping callback pointers would race subsequent provider scratch reuse.
- Aliasing only namespaces/exports is insufficient: an actual nvlink rejected
  duplicate global `local_ratio_geometry`, `local_ratio_prepared`, `local_norm`,
  `local_ratio`, `local_log` and `local_becke`. Preserve their isolation too.
- Comparing stale rejection against a prior independently repeated floating
  reduction is not an atomic-publication test. Compare against the caller's
  exact current output immediately before the rejected request; separately gate
  repeat numerical equality with the appropriate tolerance.
- Every prepared batch attempt revokes final eligibility, even for malformed
  coordinates rejected before device submission. Rollback preserves the separate
  last-good warm density, not permission to publish against the old final token.
  The ragged tests must distinguish warm recovery from final-state eligibility.

## Evidence

Raw evidence is under `.artifacts/2227/resident-source-gpu/` and
`.artifacts/2227/native-controller-gpu/`. The expanded controller run uses finite
Slurm node1 GPU scheduling and passes three targeted CTests with no skips in
15.45 s. Its library SHA256 is
`f4865bc1f0e46ff16ffa757d091ed58881297100e1d2359e28b4cc6aa3e79b07`.
These measurements precede the final documentation/prototype-include/wheel
dependency edits; final exact-source validation must be retained separately.

Six complete-source cases cover PBE/PBE0 H2 RKS, H3 UKS and empty-beta H UKS.
They compare complete gradients against qualified CPU forces and reconverged
CUDA energy finite differences, genuinely move/restore coordinates, reject
stale tokens, preserve output under resource failure and reuse warm controllers.
The offline PySCF 2.14.0 check accepts all six without skips: maximum gradient
error is 1.72e-11 Hartree/Bohr and maximum energy error is 9.68e-10 Hartree.
Generation tests retain unchanged scientific wrapper text and static identities;
native source tests cover scratch copying/signs, incomplete publication,
duplicate/convention rejection, capture survival and pending-stream destruction.

The prior broad CUDA model-validation failure is still a separately reproduced
baseline issue. A later full build also exposes an incumbent MP2 contract-test
CUDA include dependency; targeted builds are not evidence that the entire CUDA
build/suite passes. Do not loosen model validation or hide those failures.

### Follow-up qualification on 2026-10-12

The later exact-source eight-case run is retained in
`.artifacts/2227/native-controller-final-gpu/`, with its deployed source manifest
in `.artifacts/2227/native-controller-final-files.sha256`. All three targeted
CTests pass without skips in 29.87 s. The library SHA256 is
`b394d580b6ff668bc152024dae09c961b7106bed4b61139aaac5c0518def3e0f`.
The new 18-AO complete-force fixtures use separated H dimers and one primitive
per atom (exponent 0.6, coefficient 1), keeping the independent CPU oracle under
its unchanged work gate. They are not the three-primitive 18-AO lease fixtures.
The eight-case PySCF 2.14.0 report is
`.artifacts/2227/native-controller-final-independent-pyscf.json`: maximum energy
error 9.67224e-10 Hartree and gradient error 1.71540e-11 Hartree/Bohr.

Further test-only expansion adds neutral RKS LiH and charged UKS LiH with
explicit s/p custom primitives and non-axis-aligned coordinates, for both PBE
and PBE0. The twelve-case complete gradient run passes the three targeted CTests
without skips in 34.18 s, retaining the same library hash. Its source checks,
compiler launcher versions and before/after ccache statistics are under
`.artifacts/2227/native-controller-polarized-gpu/`. The corresponding independent
PySCF report, `.artifacts/2227/native-controller-polarized-independent-pyscf.json`,
accepts every case: maximum gradient error 1.94745e-10 Hartree/Bohr and maximum
energy error 9.67224e-10 Hartree. This run precedes the later D/W receipt extension.

All four profiles also exercise ragged native batch owners, same-shape
cross-owner token rejection, per-item resource shortage, malformed and genuinely
singular moved coordinates, successful neighbors, last-good warm rollback and
explicit nonconvergence. Rejected calls preserve the exact prior output.
Compute Sanitizer memcheck of the generated source bindings and common Fock
composition reports zero errors and zero leaked bytes; the raw log is
`.artifacts/2227/native-controller-final-gpu/native-controller-final-memcheck.log`.
Host qualification retains 99 final CPU-oracle/contracts passes and a successful
Sphinx warnings-as-errors build. The missing Sphinx in the build virtualenv was
a tooling prerequisite, resolved by using the already installed documentation
environment rather than weakening the documentation gate.

The subsequent D/W receipt run retains the same library hash and passes all
three targeted CTests without skips in 33.77 s. Its twelve mandatory receipts
are retained in `.artifacts/2227/native-controller-density-gpu/`; the PySCF 2.14.0
report is `.artifacts/2227/native-controller-density-independent-pyscf.json`.
Maximum errors are 9.67224e-10 Hartree for energy, 1.94745e-10 Hartree/Bohr for
the gradient, 3.99765e-9 for D and 7.97460e-9 for W. The unchanged acceptance
limits are 1e-8 for energy, 3e-7 for gradient and 2e-7 for each matrix. These
test-only final-state exports occur after the no-dense-export consumer gate;
the production controller still consumes resident D/W. The offline parser now
fails closed on missing fields as well as incomplete inventories and malformed
or nonfinite matrices. Its focused host contract group passes 83 tests, including
21 receipt rejection cases; compiler ownership checks 526 modules with no
dependency errors and the updated Sphinx warnings-as-errors build passes.

Full Compute Sanitizer memcheck of the twelve-force/ragged/18-AO lease test
binary also completes within its finite ten-minute Slurm allocation, with zero
errors and zero bytes leaked. The log is
`.artifacts/2227/native-controller-polarized-gpu/native-controller-polarized-memcheck.log`.
That binary uses the pre-D/W-receipt test source matched by
`.artifacts/2227/native-controller-polarized-files.sha256`; the scientific library
hash is unchanged. Slurm accounting storage is disabled, so no `sacct` elapsed
receipt is available. This is memory-safety evidence, not complete allocator/RSS
accounting or endpoint performance qualification.

### Integration with the final PR head

PR #2302 merged externally at 2026-10-11 22:59:41 UTC (2026-10-12 06:59:41
Asia/Shanghai), while this private CUDA slice remained uncommitted. Its final
head `4d36d176a500d49237439d727d6700ff7b516525` contains six commits beyond the
earlier `98c52ec045ab4fd1c48604b7c0941a768ed06024` base, including the shared
prepared endpoint lifecycle. The worktree fast-forwards to that head and restores
the local CUDA changes without dropping the concurrent work. The only textual
conflict is an integral-owner test's source-range endpoint; it retains the new
private gradient boundary rather than restoring the old `execute` signature.
This local implementation is not part of the merged PR and does not promote
public CUDA force admission.

Source checks, cached compilation and Slurm qualification against that integrated
tree are retained under `.artifacts/2227/native-controller-integrated-gpu/` and
`.artifacts/2227/native-controller-integrated-files.sha256`. The resulting CUDA
library SHA256 is
`9c5708609fd816ab4b598b604b985bd0f02430d4bb70d326c1ca1d6efa0471dd`.
All four targeted native CTests, including the shared prepared lifecycle, pass
without skips in 36.37 s. The integrated twelve-case PySCF 2.14.0 report preserves
all four gates: maximum energy error 9.67224e-10, gradient error 1.94745e-10,
D error 3.99765e-9 and W error 7.97460e-9. The earlier full memcheck remains
explicitly pre-integration evidence, not an exact integrated-library sanitizer
receipt.

The integrated CPU build initially passes 88/89 CTests: the incremental CLI
catalog retains old automatic Libxc MethodIR identities despite the current
compiler producing new identities. Fresh isolated catalog generation confirms
the mismatch. Explicitly rerunning the existing generator and rebuilding the
CLI makes its unchanged identity assertion pass, then all 89 CTests pass.
Preserve the initial failure in
`.artifacts/2227/native-controller-integrated-host-ctest-initial.log` and the
explicit regeneration receipt; do not describe this as an ordinary untouched
incremental-build pass or relax identity validation. General asset-input closure
for incremental CLI catalogs remains a separate R5 concern.

The integrated focused host contracts pass 89 tests with five explicitly
real-GPU-only lifecycle cases skipped in that host run. Compiler ownership
checks 528 modules without dependency errors, the CUDA ownership ledger checks
358 files and Sphinx warnings-as-errors passes. The documentation cross-reference
now distinguishes native-admitted KS forces from remaining Python stationary
compositions rather than claiming every CPU Direct request still requires JIT.

The public prepared-lifecycle effect tests also run against this integrated
CUDA library through finite Slurm scheduling: all nine tests pass without skips
in 8.74 s, including the five cases explicitly skipped by the host-only run.
The raw log is
`.artifacts/2227/native-controller-integrated-gpu/native-controller-integrated-public-lifecycle.log`.
They exercise real CPU/CUDA energy owners, cold/replay/moved coordinates,
reset/failure/recovery/close, resource rejection and failed-neighbor isolation.
These Python-wrapper effects do not substitute for an installed Python-free
SDK force smoke test or qualify a new public CUDA force route.

## Remaining gates and revisit conditions

This is not completion of #2227 R1–R6. Before public cutover, qualify the wider
scientific domain, installed compiler/Python-free property entry points,
broader ragged failed-neighbor/rollback admission, shared resource/lease authority,
complete endpoint allocation/work/timing and the default regression matrix. Response/HVP and other
runtime policies remain separate unfinished requirements. Do not generalize the
small hydrogen fixtures into universal basis/grid/method admission.

## References

- Supersedes the implemented portion of the [binding proposal](../../proposed/2026-10-12-native-cuda-stationary-controller-binding.md).
- [Resident final-weight lease](2026-10-12-cuda-ks-resident-final-weight-lease.md).
- `src/dft/cuda_stationary_controller.cpp` and `src/dft/native_stationary_runtime.cuh`.
- `tests/native/test_ks_resident_weights.cpp` and `tools/check_native_stationary_cuda_oracles.py`.
