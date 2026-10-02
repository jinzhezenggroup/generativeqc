# Decision: separate bounded through-f value availability from default selection

Status: implemented
Date: 2026-10-02

## Problem

PR #1666 made the retained Direct-HF bounded higher-l shell value route automatic
whenever generated/native coverage was incomplete. Numerical equivalence did not
establish that this was a suitable performance default. The matched NVIDIA
receipt on frozen `81ef46baf7771e955969cdbcc1049d97d3d22e0b` switched only the
bounded capability within the same native library and measured a repeated-work
regression, including complete converged physical SCF endpoints.

## Decision

Keep `bounded_value_capability` as truthful owner/resource metadata. Add the
internal per-plan `bounded_value_opt_in`, default false, and select bounded values
only through `direct_jk_bounded_value_enabled()`. Complete generated/native SPD
coverage remains automatic. Through-f production values use the retained
canonical source or existing generic fallback; internal qualification can opt
into the bounded implementation without changing capabilities or public ABI.

The switch must be set before enqueueing work and does not allocate/free storage,
change numerical thresholds, change derivative owners, or change precision
admission. It is not a public/environment policy option. Preparation diagnostics
describe the production default; qualification drivers must record their switch
alongside actual executed work, as the native work census does.

This supersedes the default-route assumption in the earlier
[canonical screening fixture decision](../compatibility/2026-10-01-through-f-canonical-screening-fixture.md).
Its discrete Cartesian-pair-screened oracle and unchanged numerical thresholds
remain valid. The fixture now uses the policy switch rather than changing
retained capability metadata.

## Invariants and dependent joins

- Value availability does not by itself enable the bounded schedule.
- Default and explicit bounded routes remain checked against the same CPU ERIs
  for J/K masks, spins, Cartesian/spherical representations and two geometries.
- Native 58/116-AO work census runs both selections and reports canonical work
  honestly; zero canonical work does not mean zero total bounded-shell work.
- Derivative, mixed-J, range and constrained-memory fallback semantics remain
  separate from full-range value selection.
- Stacked PR #1688's SR/LR predicate uses the same
  `direct_jk_bounded_value_enabled()` gate before opting into bounded range values.
  Its native CPU-ERI comparison retains both default and opt-in SR/LR routes,
  with canonical candidate/radial counts appropriate to each selection.
- Stacked PR #1695's fused RSH value facade must use the same gate, returning its
  existing `NOT_IMPLEMENTED` fallback when disabled, so a direct join cannot
  bypass production value policy. That join is outside this repair.

## Evidence

The frozen Slurm 12038 receipt used RTX 5090, CUDA 12.9, Release/sm_120, with
same-library canonical/bounded selection. At 58/116 public AOs, synthetic warm
host-visible J/K medians were 14.864/45.220 ms canonical versus 269.568/463.386 ms
bounded (18.1/10.2 times slower), with maximum matrix difference 6.9944e-15 under
the unchanged 3e-12 gate.

Physical PBE0 water/two-water cc-pVTZ complete SCF energy endpoints regressed
4.67–5.01/2.21–2.33 times across cold/warm/moved phases. Both routes required
14/1/10 SCF iterations, and maximum matched energy difference was 8.2423e-13 Ha
under the fixed 1e-8 gate. These timings do not include force endpoints and do
not quantify total bounded-shell hardware work. No new timing result is claimed
for the policy repair.

## Rejected alternatives

- Relabeling zero-canonical counters as a speedup: they do not count bounded
  source work and cannot refute complete-endpoint measurements.
- Clearing bounded capability globally: that would conflate qualification with
  availability and discard an explicitly testable implementation.
- Speculative algorithm changes: the receipt establishes the default regression
  but does not isolate its cause; fixing traversal/preparation cost needs separate
  profiling and qualification.

## Revisit when

The bounded route passes matched complete cold/warm/moved endpoints, larger-size,
batch and constrained-memory tests with the exact scientific settings and
actual total-work evidence required by
[`performance_engineering.md`](../../../../docs/maintainer/performance_engineering.md).

## References

- [Frozen numerical/performance receipt](https://github.com/jinzhezenggroup/generativeqc/pull/1666#issuecomment-5944958198)
- [Default-selection review finding](https://github.com/jinzhezenggroup/generativeqc/pull/1666#discussion_r4162579996)
- [Range-value stack](https://github.com/jinzhezenggroup/generativeqc/pull/1688)
- [Fused RSH-value stack](https://github.com/jinzhezenggroup/generativeqc/pull/1695)
