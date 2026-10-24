# Bounded raw-source Coulomb response

This is a diagnostic qualification, not the README performance campaign.
Native Standard, FP64, default grids and unchanged energy/force gates are used.
Each complete population has one cold and one frozen-native-density warm call.
The 48-atom ablations force dense storage with a 4 GiB DF budget to exercise
source replay; they are not default-resource timings.

`qualification.json.xz` retains every completed phase's independent dense-reference
errors. `summary.json.xz` retains endpoint times, SCF cycles, J timings, response
work and incomplete populations. `measurements.json.xz`, `traces.json.xz` and
`reproduction.json.xz` retain selected numerical records, aggregate operation
timings/work, and exact launchers/patches respectively. Every numerical sample,
setting and build identity is retained. Repeated repository-wide source maps
are replaced by explicit base/patch/hash bindings; detailed per-shell/per-tile
events remain in ignored local artifacts. Original receipt/trace hashes identify
these selections, which do not claim byte identity with their full originals.
Source patches identify both measured candidate revisions; do not combine them
into one publication timing population.

The final binary passes 39 GPU tests and 36 Compute Sanitizer cases (zero
errors). Host response tests pass 25 cases; resource/restart tests pass 76 with
four backend-specific skips. Initial missing-assets and insufficient-budget
attempts remain visible. Generic DF96 warm reaches the 40-minute job limit.
Explicit shell execution and the 32.3 GB host-cache interposer are separate
diagnostics; neither is promoted by this patch. The latter exceeds its job's
default host-memory request and requires proper scheduler/resource admission
before further qualification. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-21-batched-raw-coulomb-response.md)
for the remaining source and derivative bottlenecks.
