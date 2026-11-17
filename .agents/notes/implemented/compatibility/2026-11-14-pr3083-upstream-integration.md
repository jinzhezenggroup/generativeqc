# Decision: integrate upstream without relabeling native qualification

Status: implemented
Date: 2026-11-14

## Problem

PR #3083 preserved the native Hessian/saddle follow-up at `4675219c8`. Upstream
`master` at `17f9317a2` also contained the squash of earlier PR #3062. Git's older
merge base therefore reported 32 conflicting paths, including many add/add
conflicts that were not independent scientific changes. The user subsequently
requested conflict resolution, not resumption of the stopped chemistry jobs.

## Decision

Compare each conflicting upstream blob with the original branch tip
`53e733d9f`. For 23 paths they are identical: preserve the follow-up version,
including its native source ownership, bounded derivative assembly, public ECP
saddle guards and tests. This is a one-off integration audit, not a required-CI
source fingerprint or a blanket merge strategy.

For the genuinely changed paths, combine the follow-up with upstream's exact
changes: bounded real ECP-weight validation, the typed Gaussian-ECP derivative
repair, the DF spin-admission test, current precision-budget admission and
reservation policy, expanded unsupported-context controls, removal of the
obsolete Ni second-order-capability assertion, and the shared f-shell force
helper used by both Cartesian and spherical test shards. Do not restore the
duplicate parametrized helper or drop either actual shard's numerical coverage.

The automatic merge also moves native B3LYP admission into upstream's separate
`Calculator._initialize_execution` method. Import its canonical-method resolver
in that method's scope; the constructor's local import no longer supplies it.

Regenerate the native Libxc catalog snapshot from the merged tree. Its 222
functional identities change relative to the retained published summary, with
no functional removals or qualified-stage demotions. Acknowledge each exact
transition token in `manifests/libxc/native-qualification.json`; retain the
original previous-summary and campaign bytes. This acknowledges changed
identities, not numerical equivalence or new measurements. The report remains
stale, with zero qualified native endpoint products and no public DFT promotion.
The strict current-source acceptance gate is not weakened.

## Rejected alternatives

- A global `ours` resolution loses upstream's input safety, precision policy and
  shared-shard test fixes. A global `theirs` resolution loses the follow-up's
  native Hessian and molecular-evaluator work.
- Reusing the old native catalog snapshot after merging misstates its live
  scientific identities. Resetting the previous summary or bypassing exact
  transition acknowledgements could hide genuine capability losses.
- Rewriting the earlier endpoint receipts to the merged source would fabricate
  qualification. Preserve those receipts, archive hashes, timeout/cancellation
  outcomes and remaining chemistry gates unchanged.

## Invariants

- Source/build integration is not independent scientific endpoint acceptance.
- Do not relax physical SCF/response/root/mode gates or promote ECP saddles.
- Exact catalog acknowledgements remain revision-bound and reject stale tokens.
- The original source-qualified results in
  `benchmarks/results/host-native-2844-2848-20261114/evidence.json` remain
  historical and are not attributed to the merged binary.
- No stopped Ni/scale calculation, GPU job, release or publishing operation is
  restarted by conflict resolution.

## Evidence

The local integration receipts, one-off blob comparisons, cache statistics,
compiled CPU contracts and regression logs are under
`build/qualification/pr3083-merge-20261114`. A Release CPU build with CUDA
disabled succeeds using verified sccache 0.16.0; the retained compiler command
shows the actual launcher invocation. Its native source identity is
`3ab835997671baf7ea1e0936e837c0ce6b393a29c2991b6d65dd9d0e4551cd3c` and
the rebuilt core SHA-256 is
`037a53a52010183c3aa45a07fe5c654b05944374efe3832b26d32b7681b84469`.
Recomputing the final checkout's native input identity matches this build.

Validation against this integration includes:

- Ten selected native CTest groups pass, including host Hessian, source binding,
  Gaussian ECP, molecular evaluator/saddle and upstream DF spin-admission
  contracts (38.22 seconds).
- The eight bounded closing Python modules pass 183 cases and explicitly skip
  six opt-in heavy cases (8.56 seconds).
- Public linked native CPU Hessian/HVP and molecular-saddle regressions pass
  74 cases (11.56 seconds), exercising RKS/UKS, D3 composition, discrete-grid
  covariance, work/resource limits, failure recovery and owner lifetime.
- Catalog/transition/ECP-input safety tests pass 123 cases; unsupported-context
  controls pass seven; ECP parallel-shard ownership controls pass seven.
- Compiler, native API/runtime, shared SCF and electronic-structure ownership
  checks pass. PR-diff Ruff lint and formatting pass; unrelated unchanged
  upstream benchmark formatting is outside this integration's lint claim.

The completed Python groups total 394 passes and six skips. A separate broader
13-module Python invocation reaches its 300-second cap (exit 124), without a
completed pytest summary; it is not counted as passed or as scientific
acceptance. Its full result remains unqualified. The bounded reruns above do
not replace the uncompleted numerical coverage.

Refresh the generated catalog after the final initializer-scope repair and
build, from the original upstream retained snapshot. The intermediate draft
snapshot is retained locally, not used to reset the published previous-summary
baseline. The final default catalog checker passes with zero qualified native
products and `stale: True`; this is archival consistency, not the stronger
current-source acceptance gate. No historical chemistry receipt is relabeled
for the rebuilt core.

## 2026-11-17 integration of the offline-acceptance baseline

The next integration includes `master@50482bc35`, including cold-CI resource
budgets, stable test-shard ownership and dependency-scoped Libxc acceptance.
Preserve both sets of Hessian/source-identity build inputs in the test conflict;
neither the native Hessian generators nor the upstream stationary/Libxc
generators may disappear from the scientific identity inventory.

For the catalog conflicts, retain upstream's measured campaigns, dependency
inventories, previous-summary baseline and historical generated presentation.
Compute and acknowledge the integrated tree's 222 exact functional identity
transitions against that unchanged upstream baseline. There are no removed
functionals or qualified-stage demotions. Do not carry the older transition
tokens across different baselines, reset the previous summary, or relabel the
original native/Hessian receipts. The explicit
`check_libxc_native_catalog.py --allow-stale-generated-views` check passes with
historical-presentation advisories; this is neither a fresh scientific campaign
nor the stronger current-source acceptance check. The strict generated-view
check remains available and rejects the stale presentation.

The original source/build identities and cancelled/timed-out chemistry evidence
above remain historical. Resolving these conflicts and repairing CI resource
ownership does not establish a practical-scale native HVP, a converged Ni
reaction saddle or raw continuum rotational invariance. Those scientific gates
remain separate from the integration and must not be waived to mark this PR
ready.

## Primitive-free configuration regression

The first integrated CI run `43991950248` exposes a real default-selection
conflict: CUDA resource builds and lightweight native probes explicitly disable
stationary CPU primitives, but the newly default-on host Hessian still forces
its dependent response owner on. Its existing prerequisite check then correctly
refuses configuration. Do not delete that guard or enable the expensive
primitive inventory behind a caller's explicit OFF request.

Default `GENERATIVEQC_ENABLE_HOST_KS_HESSIAN` to the selected
`GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT` value. Ordinary native builds
still enable both by default. Clean primitive-free builds now default the
dependent Hessian off, while an explicit incompatible Hessian ON request still
fails. Three real-project CMake configure tests cover those combinations; the
existing transitive production-codegen CMake regression also passes. No CI
scientific test, physical threshold or explicit capability request is removed.

The separate source-matched 58-module local run exposes the curated CsH(-)
final-projector defect and is stopped after that failure. A later four-worker
run reaches its 1,200-second bound without a completed summary and is not
counted as passed. The curated-state repair is documented in
`../numerics/2026-11-17-curated-cpu-ks-final-projector.md`; bounded focused
qualification does not turn unfinished broader coverage into acceptance.

## Subsequent November 17 force-product integration

Upstream `8f4166f0c` subsequently integrates PR #3090. Combine its public
SDK/CLI/Python force-product changes with this branch rather than replacing
either implementation. The only textual conflicts are the two derived
manifests. Preserve upstream's new measured campaigns, resource receipt,
previous-summary bytes and all archived manifests/snapshots; recompute only
exact transition acknowledgements and the unmeasured CUDA capability audit.

Unlike the earlier `50482bc35` baseline, the newer retained previous-summary
contains measured qualified stages. Upstream already records 20 demotions;
the integrated catalog preserves them and also changes 222 functional
identities. Those retained demotions cover compiled CPU, compiled
CUDA, GPU runtime, molecular SCF and forces for `GGA_X_APBE`, `LDA_C_VWN_4`,
`MGGA_X_GVT4` and `MGGA_X_LTA`. Each transition has an exact revision-bound
acknowledgement recomputed for the new identities. There is no additional
stage loss relative to upstream. No functional is removed, no old receipt is rewritten and no
new measurement is claimed. The current catalog has zero qualified endpoints;
the historical presentation remains advisory and the current-source scientific
acceptance gate remains unfulfilled. Resetting the previous summary or copying
the upstream qualification labels would hide these genuine evidence losses.

The source-matched `845c4cfd4` build previously passed ten native CTest groups,
267 scientific Python tests (six explicit heavy opt-in skips), four CMake
configuration controls and the mixed RSH/DF force regression. These results
remain attached to that source identity, not automatically to this subsequent
integration. Fresh build and regression records are retained separately under
`/data/jzzeng/pr3083-ready-20261117/`. The practical-scale and Ni saddle gates
remain open; this integration does not resume stopped long chemistry jobs.

## CI bootstrap and explicit evidence-storage review

Fresh CI on `9ecff2600` exposes two integration prerequisites independent of
the passing native scientific controls. The CPU benchmark configures the new
default Hessian generators before preparing its NumPy-bearing environment.
Move the existing Python/dependency steps before CMake and explicitly select
that interpreter. Do not disable Hessian generation or drop the benchmark's
physical endpoint coverage to hide the missing build dependency.

The aggregate evidence budget was exactly full on current master. This branch
adds only its original 66,270-byte `evidence.json` and 7,475-byte README. Propose
an explicit, exactly 73,745-byte storage-policy increase in the same PR, with
no unallocated capacity. Preserve both files byte-for-byte; do not delete
scientific evidence or disguise it under another directory to satisfy CI.
The independent per-file, 2-MiB incoming-review and scientific acceptance gates
are unchanged. This is a reviewable storage-policy change, not a claim that
the retained failed or incomplete calculations passed.

The earlier current-master CI also exposes two actual CUDA host-consumer build
failures. Wheel builds intentionally have no `CUDA::cudart` imported target;
reuse their provider-free header/import interface instead of linking a provider
that the wheel contract forbids. SDK and CuMetal consumers retain their normal
runtime interface. The host response also includes the generated split-hybrid
registry through shared composition metadata. Give it the generated include
directory and an explicit dependency on a shared registry-generation owner,
not an incidental core compilation race. CPU-only default builds are unchanged.
Two executable CMake probes compile the actual linked host-response consumer
for native-interface and provider-free wheel modes without executing GPU work.

The transitive-codegen configure regression now explicitly selects its running
Python interpreter, matching the benchmark bootstrap and documented NumPy
build prerequisite. CodeQL also flags the pre-existing narrow unsigned
Cartesian-pair product in the modified ECP owner: route that allocation count
through the existing checked `runtime::size_mul` helper before conversion can
lose range. No Gaussian arithmetic, ordering or physical tolerance changes.

The two host-consumer build probes pass (201.99 seconds), four actual configure
controls pass (13.28 seconds), and 177 benchmark-policy/retention tests pass.
The occupation-stabilization fix passes eleven selected native CTest groups,
including the previously failing UKS state group. A separate newer 13-module
scientific rerun reaches its 900-second limit without a completed summary and
is not passed. Earlier completed scientific receipts remain source-specific.

## Revisit when

A later source-matched qualification completes the remaining scientific gates.
Do not infer practical-scale derivatives or a representative Ni reaction saddle
from this integration or from the earlier small-domain controls.
