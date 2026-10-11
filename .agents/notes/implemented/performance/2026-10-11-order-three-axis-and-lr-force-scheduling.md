# Decision: parallel order-three Hermite axes and bounded LR force CTAs

Status: implemented
Date: 2026-10-11

## Problem

The automatic 32-thread canonical owner retained serial three-axis Hermite
preparation in its leader. Low-order shell domains have little component work
to hide that preparation. The earlier scheduling qualification also left a
physical-force gap: the 12-atom water proxy spent about 35 seconds in forces,
against about 11 seconds for the independent GPU4PySCF full-Fock control.

An intrusive profile of the previously submitted binary separated two dominant
force consumers: 15.352 seconds in the generic LR shell kernel and 10.706 seconds
in the f-containing full-range residual kernel. Both required 255 registers.
This is not evidence that every force source shares the canonical value cache
or that register pressure alone explains the complete gap.

## Decision

For the canonical order-three 32-thread value owner, lanes zero through two
prepare disjoint Cartesian axes with the existing complete Hermite recurrence.
They read immutable primitive-pair metadata directly, not the leader's
unpublished shared copy. The original publication and retirement barriers,
primitive traversal, raw coefficient factors, Coulomb owner, component
screening, and symmetry scatter remain unchanged. Other orders and legacy
packet owners retain their original serial preparation.

The ordinary bounded LR force launch adopts the existing 128-thread force
width. This is a radial-source/default-launch-shape rule, not a functional,
omega, basis, molecule, architecture-name, or environment whitelist. The
producer already advances admission by its actual force width; all candidates
in each page survive. Custom launch shapes and unmeasured short-range sources
retain their previous widths. There is no new storage, mathematical consumer,
precision mode, screening predicate, or production opt-in.

## Rejected alternatives

- Narrowing both generic force kernels to 64 threads regressed the fixed-state
  force diagnostic to 38.414 seconds, versus a subsequent 33.947-second control.
- Blanket LR angular partitioning produced 36.113/36.128-second force replays.
  Re-enumeration and the different retained recurrence frames outweighed its
  specialization. The existing qualification switch is not promoted.
- Partitioning only the f-containing full-range residual, using the already
  emitted fixed-angular consumers and excluding the complete s/p/d mask,
  produced 36.739/37.233-second replays even with the improved LR width.
  The isolated LR-width control was 31.649 seconds and the final retained
  control 34.147 seconds. Numerical agreement is not sufficient for promotion.
- Active-Hermite-only initialization is not revived. The previous
  [small-CTA investigation](2026-10-11-canonical-small-cta-scheduling.md)
  already rejected it without reproducible benefit.

## Evidence and scope

The new frozen base is master `637466c624c4c4c4dd4eb7c511a2c88675744325`
plus the submitted #2236 change. Earlier `f6b9220`/`dec25f592` timings are not
relabeled as measurements of this base.

All fixtures here are water proxies with the complete spherical def2-TZVPD H/O
basis, including diffuse and f shells. They are not OMol25-distribution samples.
Exact Direct WB97M-V/RKS, FP64, the 48-by-16-by-32 per-atom grid, Becke3,
screening/energy/density tolerances, VV10 cutoff, force budgets, and independently
initialized SCF states are retained.

The 12-atom ABBA first-density diagnostic changes its four order-three actions
from 0.489055/0.490369 seconds to 0.421542/0.423628 seconds. The separate
96-atom diagnostic covers all four full/LR order-three bucket products:
174.807447 seconds to 147.513226 seconds. The candidate in these experiments is
an O2 diagnostic interposer, not the final Release binary or a complete endpoint.

Source indices are not a stable angular identity: for this 96-atom producer,
order three occupies indices 4/6/32/34. An initial 4/5/32/33 capture includes
order four and is retained only as the explicitly mixed-order diagnostic.
The accepted angular diagnostic filters the registered order, then verifies the
four source domains. It never publishes an SCF/energy/force result after omitting
other sources.

A separate intrusive force census lends the existing optional native
per-class counters on a fixed final SCF token. Baseline and 128-thread schedules
match exactly for every class. The full-range f residual admits 954,680 shell
quartets; LR admits 7,497,817. Their AO/primitive quantities are *capacity bounds
inside those admitted shells*, not post-component-screening executed work.
LR's capacities are 335,167,456 AO and 767,930,390 primitive quartets. No
executed primitive-force count is invented from them.

The axis consumer passes the native low/high-order numerical/work gates and
memcheck, initcheck, synccheck, and racecheck with zero errors/hazards. Final
Release numerical gates, complete endpoints, source/library identities,
semantic counts, matrix comparisons, and receipt hashes belong in
`benchmarks/results/wb97mv-order3-force-20261011/` rather than this rationale.

Raw artifacts are retained in the ignored local mirror
`.artifacts/canonical-order3-force-20261011/` and on node1 under
`/data/jzzeng/wb97m-order3-force-20261011/`. Compilation uses verified ccache
with checkout-root normalization and preserved caches. Every real-GPU run uses
a finite Slurm allocation and its original device visibility.

## Consequences and revisit conditions

The instrumentation-free matched complete endpoint qualification is retained in
`benchmarks/results/wb97mv-order3-force-20261011/evidence.json.gz`. With two ABBA
samples per arm, the 12-atom complete median is 194.953520 -> 190.796692 seconds
(-2.13%), and physical forces are 36.039981 -> 33.355487 seconds (-7.45%). The
3-atom complete median is 12.118045 -> 11.977786 seconds (-1.16%). Independent
physical E/F gates pass and recorded semantic work/traffic match; these
small-sample measurements do not establish statistical significance.

This narrows, but does not close, the force gap. No complete 96-atom cold E+F or
independent 96-atom physical-force gate is claimed. Diagnostic source windows
and fixed-state force replays are not substitutes for complete cold endpoints.

The next substantial force route should reduce primitive/response recomputation
and private recurrence traffic with a compiler-owned shared derivative consumer,
retaining bounded resource fallbacks and independent radial/atom/orbit gates.
Do not repeat blanket angular splitting merely because its private frame is
smaller or its numerical tests pass.
