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

## Revisit when

A later source-matched qualification completes the remaining scientific gates.
Do not infer practical-scale derivatives or a representative Ni reaction saddle
from this integration or from the earlier small-domain controls.
