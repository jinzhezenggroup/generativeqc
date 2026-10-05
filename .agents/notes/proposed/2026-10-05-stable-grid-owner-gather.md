# Proposal: remove repeated owner-label scans without reordering force sums

Status: proposed; not implemented or qualified
Date: 2026-10-05

An explicit, nondefault implementation is now tracked in
`../implemented/performance/2026-10-05-stable-owner-schedule-experiment.md`.
This proposal remains performance-unqualified; do not interpret the experiment
as production promotion or P0-C completion.

## Evidence

The actual v6 96-atom warm public-force capture in Slurm 6090 spends an aggregate
2.675511 s in `geometry_cooperative_kernel`. This includes XC/point setup, AO
gradient arithmetic, atom gather and owner reduction, not pure gather time.
The source schedule tests membership for every atom and active AO row: its
recorded domain is 37,261,541,376 tests. The separate all-source lane publication
takes only 0.097822 s. Density gather takes 0.014520 s. Optimize the identified
combined consumer before inventing large gains from tiny copy/gather counters.

## Candidate

Build a stable integer head/tail/next grouping from the existing local AO map
inside the cooperative block's already charged shared workspace. Each atom's
chain visits exactly its original ascending local positions, including an
arbitrary/noncontiguous AO-to-atom assignment. Eliminate the repeated all-row
membership scans without reordering a single floating-point accumulation.
This is transient scheduling metadata, not a second global AO coordinate map,
a geometry cache, a retained density tensor, or new scientific screening.

Admit the extra integer scratch only if it fits the existing alias before Becke
overwrites it; otherwise retain the original cooperative scan. Do not grow the
arena, evict another owner, or convert a fitting cooperative tile to the scalar
route merely because optional grouping cannot fit. Preserve empty-map behavior
and the producer's sticky device error handling.

Independently, three owner-coordinate writers can each retain the exact
original AO order for their coordinate instead of one thread interleaving all
three sums. This must preserve per-coordinate sums bitwise; do not replace the
ordered sums with an associative warp reduction or floating-point atomics.

## Acceptance before promotion

- Keep both candidates switchable and compare the same immutable fixed state.
- Bitwise old/new panel and force comparisons protect accumulation order;
  independent RKS/UKS PBE/PBE0 and LDA/GGA/meta-GGA gates protect semantics.
- Verify arbitrary AO maps, noncontiguous atoms, empty/tiny domains, shared
  scratch fallback, lease/error propagation, capture and geometry invalidation.
- Measure the combined kernel and, where possible, separate gather/owner stages.
  Report actual resources, comparisons/additions and logical shared bytes.
- Require clean complete 48/96 E+F improvement with actual cold/moved histories;
  no cache, occupancy or instruction-count-only success claim.

Earlier reordered atom-gather evidence included a cold Fock regression. Do not
revive it under a new name. Stable grouping is a different schedule hypothesis,
but building the chains serially may itself add a barrier tail. Profile rather
than assume that fewer membership tests imply a profitable endpoint.
