# Decision: give CUDA release-resource checks an independent CI budget

Status: implemented
Date: 2026-10-04

## Problem

Production compilation and all release-resource probes shared one 25-minute
job. A cold or partially warm production cache could leave insufficient time
for the required release probes even when compilation and preflights passed.
A timeout is incomplete CI, not successful resource evidence.

Master CI run 37227322593 at cb4924ddc92b103463c87937931495461a6870cb
reproduced this in job 111510892041. It restored an older fallback ccache and
reported 139 hits / 463 calls. Production compilation, the opted-in AO resource
probe, and both Fock/proposal preflights passed. The five f-shell probes finished
at 19:42:12 UTC, generated XC consumers passed, and native grid/XC resource
compilation began at 19:42:39. The job reached its 25-minute limit at 19:44:12,
leaving that probe and stationary/TensorIR qualification incomplete.

The job did save an ordinary master-scoped cache and publish a trusted-cache
artifact before the timeout. Existing consumers accept artifacts only from a
successful complete master CI run, so publication alone did not make the
snapshot eligible. The source-aware refresh improvement remains valuable;
see [its decision](2026-10-04-source-bound-cuda-ci-cache.md). A retry can use the
new ordinary cache, but cold starts still need a complete bounded CI path.

## Decision

Keep production compilation, the opted-in AO resource probe, Fock/proposal
preflights, and all production-cache handling in `cuda-compile`. Move the
existing f-shell, generated XC, native grid/XC, stationary r2SCAN, and TensorIR
resource steps unchanged into `cuda-resources`. Both jobs retain a 25-minute
budget, run after the same merge-queue liveness gate, and are mandatory inputs
to `Pass CI checks`.

Both jobs explicitly check out the event's exact `github.sha`. The resource job
configures the same CMake preset and generates only the existing Ninja file
target `generated/generated_grid_policy.cu`. Its existing CMake edges also
generate the B3LYP, r2SCAN, wB97M-V, and split-hybrid headers. This uses production
source declarations without rebuilding the library or transferring artifacts
between runs. The native resource validator continues to reject source bytes
that differ from compiler emission and compiles with its existing release
flags. Fast production flags cannot supply release-resource evidence.

AO reports are preserved by an always-run upload in the production job under
`ao-radial-release-compile`. XC/native-grid reports and the f-shell report retain
their existing artifact names in the resource job.

## Invariants

- Every existing build, preflight, test, and release-resource command remains
  mandatory, at the same target, workload, flags, and individual timeout.
- `Pass CI checks` depends on both CUDA jobs and all previous required jobs;
  no allowed failure/skip policy is introduced.
- Only confirmed orphaned merge-group refs may skip these jobs. Active runs
  are not cancelled to make room for the split.
- Trusted cache discovery still requires a successful master push of this CI
  workflow. Cache size, keys, compiler identity, refresh rules, fallback, and
  publication permissions remain unchanged.
- No release/distribution workflow, GPU execution campaign, scientific default,
  ownership inventory, compiler generation rule, or production source changes.

## Rejected alternatives

Increasing the shared job timeout hides the serial critical path. Skipping
resource probes or substituting fast-build resource records loses required
coverage. A latest-artifact lookup or accepting a failed overall master run
would change the existing cache trust policy; that is deliberately outside
this split. Duplicating the full production library build wastes work.

## Validation and limits

Regression tests protect both required jobs, exact-source checkout, unchanged
budgets, unconditional single ownership of every existing gate, and native grid
source identity. A host-only CMake fixture executes the real production
generation declarations, generates exactly the grid source and four headers,
and confirms byte equality with compiler emission without compiling the
library or enabling a CUDA compiler. Existing cache-provenance/refresh,
merge-queue, CI ownership, and repository inventory checks remain applicable.

This is a workflow repair, not a measured CUDA speedup or GPU/scientific
qualification. Only CI on the published commit can establish its actual
runner durations and completion of every required gate. Revisit the split if
one individual job still exceeds its budget after identifying its actual
per-stage work and cache behavior.

## References

- [Master CUDA job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37227322593/job/111510892041)
- [Master CI run](https://github.com/jinzhezenggroup/generativeqc/actions/runs/37227322593)
