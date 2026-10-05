# Decision: contract order-four force sources through shared weighted roots

Status: implemented; composed endpoint qualification retained separately from profile promotion
Date: 2026-10-05

## Problem

The bounded full-range J/K derivative owner evaluates higher-order primitives
per AO quartet. The diagnostic order-four replay is expensive, while the
shared low-order external-weight consumer already reuses shell-pair geometry
and radial moments across AO components and the two source channels.

## Decision

Generate force-only weighted roots for pppp, dspp, dsds, dpps and ddss using
the existing Weighted IntegralIR, algebra and pressure-aware ordering. Bind
them to the existing canonical task/weight/primitive/scatter adapter. Drain
these whole shell tasks once per scalar lane before the generic warp fallback;
the fallback skips exactly those classes for full-range separate J/K sources.

The scratch bound is two fixed component-weight arrays (at most 81 doubles per
source), scalar geometry and two nine-component gradients per task. No new
resident or quartet-capacity allocation is introduced. Compiler register/local
memory cost may still overwhelm the reuse benefit and must be measured.

## Invariants and scope

- Same shell and AO screening, orbit weights, source coefficients and FP64.
- Independent J/K outputs, including disabled source guards.
- Canonical shell orientation and repeated-atom translation recovery unchanged.
- HF combined force, range sources, f-containing order-four and higher classes
  retain their current consumers.
- No new recurrence implementation or functional-specific dispatch.

## Evidence and consequences

The frozen composed endpoint campaign, Slurm5884, compares control `9ba032c78`
with weighted candidate `b3acd80f7`. This includes #1830, #1833, #1847 and
indexed force scheduling; it is not an unmodified-master benchmark. Complete
48/96 warm medians fall 10.780 to 10.130 s and 32.509 to 29.013 s. Moved-warm
medians fall 10.778 to 10.111 s and 32.554 to 28.811 s. All eight regime medians
are lower, but cold SCF histories differ, so cold savings are not attributed
to this force-only treatment. Moved histories retain 12 native builds in both
arms. The 96-atom stationary derivative component falls about 14.34 to 10.68 s;
grid response remains about 8.8 s.

All 288 independent same-geometry reference pairings pass unchanged energy
and force gates. The independent host Hermite displaced-value oracle covers
1,761,696 coordinates, maximum error 3.36e-10; the weight adapter error is
1.67e-16. Composed native through-f and memcheck/initcheck pass in Slurm5880.
The master transplant, frozen at `dfeabdc0d`, passes 47 focused host tests,
native through-f and both sanitizers in Slurm5907, and builds all six stationary
AOT modules with verified ccache use. The six scientific/dispatch sources are
byte-identical to the composed candidate. Retain both scopes explicitly.

Complete samples, actual SCF histories, numerical verification, source patches,
binary/source identities and transplant receipts are under
`benchmarks/results/pbe0-weighted-order4-20261005/`. The generic evidence envelope
accepts scientific equivalence and reports endpoint observations separately;
the strict interleaved provider-promotion assay with isolated compilation and
allocator peaks was not measured. No provider/profile registry is promoted.

The remaining warm reference ratios are 1.67x/1.88x, and moved-warm ratios
2.52x/2.84x. This does not close #1895 or establish parity. Higher orders and
other schedules require their own whole-endpoint qualification; smaller
static stacks or fewer symbolic operations alone are insufficient. Preserve
the negative Wick-pruning and compact-page results when considering successors.
