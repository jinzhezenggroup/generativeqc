# Decision: refresh trusted CUDA CI caches when their inputs change

Status: implemented
Date: 2026-10-04

## Problem

Merge-group CUDA builds restore a ccache artifact from a successful push of
the CI workflow on master. An artifact younger than three hours previously
suppressed publication even after compiler inputs changed. A successful
artifact download also suppresses the ordinary actions/cache fallback, so a
young but obsolete snapshot can repeatedly miss the same expensive objects.

In PR #1792, merge-group jobs 111401471111 and 111407548980 both restored the
snapshot from CI run 37177493331. Both recorded 128 hits and 331 misses.
The second completed the full library build in 18m03s and all five f-shell
release smoke compilations in 5m12s, then reached the unchanged 25-minute job
limit during generated-XC compilation. These were incomplete required gates,
not source/compiler failures or passing CI.

A newer snapshot from successful master CI run 37190789551 became eligible
at 09:43:07 UTC, after both attempts had selected their snapshots. Its artifact
11298568876 was created at 09:12:57 UTC. This changed prerequisite permits a
normal retry without weakening the cache trust boundary. Master 6cbb3634's
later CUDA job recorded 404 hits and 55 misses and completed its production
build in about four minutes. These are existing CI observations, not new
local builds or scientific/performance qualification.

## Decision

Associate the selected live artifact with its owning successful run's
head SHA. On a master push, compare that SHA with the checked-out GITHUB_SHA
before throttling a young snapshot. Refresh when tracked build inputs changed
or the old source identity cannot be compared. Preserve the three-hour refresh
for unchanged inputs and the existing missing-artifact recovery.

The comparison includes every scope in the existing CUDA cache key, with
conservative directory coverage for compiler/runtime source, native tests,
tools, manifests, data and upstream inputs. It also covers .github and project
configuration so CI/compiler flag or dependency changes request a refresh.
Documentation-only changes retain the young-snapshot throttle.

## Invariants

- Consumers still trust only a live artifact belonging to a successful master
  push of this CI workflow, not an artifact name found across the repository.
- A newer run without an artifact cannot donate the source identity for an
  older selected artifact.
- Publication still occurs only on master pushes after the existing build
  steps, with the existing non-gating upload behavior. Token permissions,
  cache size, retention, consumer selection and cold fallback are unchanged.
- Missing source objects request a fresh snapshot without making acceleration
  discovery a required-check failure. No fetch, external diff or textconv
  command is introduced.
- All required checks, compile/resource scopes, scientific tolerances and
  timeouts remain unchanged. A cached compilation is not numerical evidence.

## Validation

Behavioral tests execute the actual embedded Bash with mocked GitHub responses
and tiny real Git histories. They cover young changed/unchanged snapshots,
documentation-only changes, input deletion, expired/missing artifacts,
discovery failure, unavailable/malformed source identities, source-to-artifact
association, trusted run selection and master-only publication. A coverage
check requires the comparison paths to include every existing CUDA cache-key
input. The previous static provenance and fallback tests remain intact.

## Revisit when

If fresh snapshots still cannot keep the complete compile/resource job within
its finite budget, inspect the per-stage work before changing job layout.
Do not skip a required resource compile or infer passing CI from a timeout.
