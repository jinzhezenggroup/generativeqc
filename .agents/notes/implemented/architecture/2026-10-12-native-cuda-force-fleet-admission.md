# Decision: native CUDA force capacity belongs to each retained fleet item

Status: implemented, with broader cutover qualification still outstanding
Date: 2026-10-12

## Problem

The private complete stationary controller retains numeric arenas and an opaque
grid-provider allowance per prepared item. The incumbent Python planner instead
reserves one serialized 512 MiB generated-force staging cap. Promoting prepared
CUDA force dispatch without changing that accounting can admit an underfunded
fleet. Allowing early controllers to choose optional cached storage from the
remaining global capacity can also starve a later item's mandatory minimum.

## Decision

Descriptor-plus-shape and authenticated prepared-item capacity queries share the
same native scientific admission rule and nine-field capacity receipt. No CUDA
context, final state, generated runtime compiler, or scientific calculation is
needed to size the additional consumer. Output publication is transactional.
Raw profile sizing remains available but is not scientific property admission.

The native prepared capability admits only compiled all-electron s/p, strict-FP64
Direct PBE/PBE0 RKS/UKS, full-range exact J/K, and three partition iterations.
The prepared public consumer borrows resident final D/W and negates the complete
energy gradient once. Method-global manifests remain unchanged. Missing or mixed
prepared admission retains the incumbent fleet route, not a partial native force.

The global resource planner offers uniform cached and minimum fleet candidates.
Each reserves the sum of complete per-item capacity, not a serialized high water.
Before configuring limits, the Python adapter authenticates every planned receipt
against its actual native prepared item. The selected limits persist across native
geometry replacement, failure, and recovery. Tightening below an existing retained
controller fails without destroying that controller or changing its prior limit.

Native provider reservations stay inside the device ledger admission limit.
They are not marked `runtime_allowance`, which would subtract them a second time.
Existing independently withheld matrix-provider estimates retain that accounting.
Python adds the v3 admission snapshot while preserving numeric v1/v2 allocation,
requested-byte and peak semantics. Reservations do not fabricate numeric events.

The existing CUDA KS base inventory also charges derivative order one: actual
prepared exact KS owners retain first-order J/K leases even for energy-only calls.
The removed staging cap must not hide that already-live provider storage.

The private stationary executor preserves typed device/host allocation failures
as status 2/3 and typed driver failures as status 4. The controller translates
these to public OOM and CUDA_ERROR, respectively, and preserves the grid owner's
public status. Flattening a driver fault into NUMERICAL_FAILURE is unsafe: the
warm fleet scheduler interprets that status as a seed-related failure and retries
with a cold seed. A Slurm-injected resident-density driver fault reproduced a
successful public return and output publication before this error protocol was
fixed. CUDA_ERROR must escape without that retry; failures leave caller output
unchanged and release owned charges. Recoverable injected faults leave the valid
warm reference usable; this is not a recovery guarantee for a poisoned device
context or an unrecoverable hardware/driver failure.
An ordinal rejected by the executor's preflight bounds check is still an invalid
argument, not a driver failure. A host-executed inner-creation double independently
checks that all real nested creation/reset/source/reduction error boundaries
preserve status 2/3/4; the device negative test must not force a new error policy
merely to exercise that boundary.

## Rejected alternatives

- Keep one legacy staging cap for a native fleet: serialized dispatch does not
  retire its per-item controllers.
- Greedily cache while capacity remains: a successful early item can consume a
  later item's mandatory admission.
- Duplicate PBE coefficients, spin, precision and ECP admission in Python: the
  prepared owner and pure descriptor query must use one scientific authority.
- Require zero cumulative cold-path matrix transfers: the default restricted
  MINAO proposal has established setup transfers. Compare matched cold energy
  and force trajectories; require no additional final-frame exports by forces.

## Evidence and remaining gates

Ignored `.artifacts/2227/native-force-dispatch/` retains source and library hashes,
initial failures, compiler/cache provenance and host/device logs. The functional
CUDA library `f3a964adce5953e2aa9ca4d6c5ee39aac3398beb70e4b7c059ed7ad5c5d529fb`
passes four native tests, including twelve independently referenced complete
gradients and fresh public cold/warm minimum consumers, and thirteen public tests.
Public tests independently accept the four explicit-grid PBE/PBE0 spin domains
and actual PBE v2 defaults; minimum fleets cover cold, warm, moved, rollback and
release, and C ABI host/device OOM preserves caller buffers and warm recovery.
The host library passes 89/89 CTest tests and 112 targeted Python contracts, with
seven real-GPU cases intentionally skipped on that host-only invocation.

The first public invocation incorrectly treated legitimate MINAO setup transfers
as force exports; the corrected matched-trajectory check passes without changing
production policies. Preserve the initial log. Compiler/Python-free installed
SDK/CLI smoke, full launch/graph profiling, and exact final-source qualification
remain separate gates. The first endpoint baseline invocation lacked the
incumbent packaged AOT assets; it is not a performance result. Restoring the
original artifact adjacency and building the unchanged declared profile targets
allows an actual incumbent-versus-native comparison in separate processes, with
no backend/model/cache-identity overrides. Five samples for each of four explicit
PBE/PBE0 spin contexts and two public PBE default-grid contexts include construction,
preparation, cold/warm/moved complete energy-plus-force calls and destruction.
Median complete-lifecycle native/incumbent ratios range from 0.1774 to 0.5552;
all six meet the 1.05 non-regression gate. Every paired stage preserves SCF
iteration/Fock counts and warm policy, with maximum energy differences below
`4.3e-14` and force differences below `1.8e-14`. This narrow matrix is not a claim
about larger systems, other basis/functional domains or unobserved kernel counts.

Those library receipts predate the typed driver-status correction. Its initial
failing reproduction and host boundary tests are retained separately; exact-source
device, installed-product and sanitizer qualification must not be inferred from
the earlier successful numerical or performance receipts.

This is an R1/R2 increment, not completion of #2227 R0–R6. Response/HVP lifecycle,
remaining native policies/model resolution and removal of duplicate scientific
Python paths remain outside these receipts. No PR or release is authorized by
this note or by the external closure of the tracking issue.

## Exact-source driver-status qualification

The expanded nested-error correction is device-qualified with CUDA library
`e782033a89c8fe37ba92b4cc875d9dbfb3b749d77166424b2f86a7d6c6e7aca2`.
Ignored `.artifacts/2227/native-force-driver/` retains each source archive,
compiler/cache provenance, the initial incorrect ordinal expectation, the public
driver reproduction, and the corrected receipts. Invalid visible ordinals fail
the existing argument preflight before a driver call, as the corrected negative
device test and unsuppressed sanitizer confirm. The earlier thirteen-pass plus
one-skip public run did not enable the separate DFT CUDA gate and is not the final
fourteen-test acceptance receipt.

The exact compiled source passes 89/89 CPU CTest tests, 87 focused host contracts,
four native Slurm tests and fourteen public Slurm tests. All twelve independent
PySCF 2.14.0 fixtures accept energy, gradient and final spin D/W. Full resident
endpoint and native binding memcheck runs both report zero errors and zero leaked
bytes, without disabling CUDA API error reporting. A fresh CMake installation
passes the four-profile SDK/CLI matrix with Python/compiler tools absent from
PATH; exec traces contain only the environment wrapper and the tested executable.

Five unprofiled samples per context reproduce the complete construction,
preparation, cold/warm/moved energy-plus-force and destruction comparison. The six
native/incumbent lifecycle ratios range from 0.1572 to 0.5181, meeting the 1.05
non-regression gate with paired SCF iteration/Fock counts and warm policy. Maximum
paired energy and force differences are below `4.3e-14` and `1.9e-14`.

Separate Nsight Systems 2025.1.3 traces annotate all six complete lifecycles and
eighteen executions, including setup/close work. They observe 102806 incumbent
versus 111527 native GPU kernels and matching runtime/driver launch API counts:
native launches **increase by 8.483%**, despite the complete endpoint speedup.
No graph launches, capture starts or graph kernel nodes are observed. This is
evidence of endpoint ownership improvement, not a kernel-count or graph win.
Traced simultaneous Device/Array requested-allocation peaks are 20894500 and
19747428 bytes, respectively, with zero dynamic live bytes at process end. Those
are explicitly not physical device RSS or complete opaque context/provider
memory; native numeric/provider admission bounds remain independently enforced
by the minimum, retained-fleet and cleanup tests. The raw reports, SQL exports,
per-stage copy/launch counts, hashes and comparison scripts remain in the ignored
local/Slurm artifact trees. Broader R0–R6 completion is still outstanding.

## References

- [Provider admission](2026-10-12-native-cuda-provider-admission.md).
- [Complete source controller](2026-10-12-native-cuda-stationary-source-controller.md).
- `src/methods/dft_method.cpp`, `src/api/c_api_resources.cpp`,
  `src/api/c_api_batch.cpp`, `python/generativeqc/resources_ks.py`.
- `tests/native/test_ks_resident_weights.cpp`,
  `tests/python/test_native_cuda_force_resources.py`,
  `tests/python/test_public_native_cuda_forces.py`.

## Revisit when

A provider supports safe heterogeneous per-item cache leases selected globally,
or a broader stationary scientific profile is independently qualified. Neither
a smaller measured arena nor a method name changes scientific admission.
