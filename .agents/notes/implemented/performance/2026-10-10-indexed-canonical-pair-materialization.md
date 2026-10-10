# Decision: indexed component reuse for canonical order-five J/K

Status: implemented, qualification-only; default disabled
Date: 2026-10-10

Historical qualification record. The default-disabled decision is superseded
by [automatic canonical admission](2026-10-10-automatic-canonical-pair-reuse.md).
The source hashes, opt-in measurement protocol and limits below remain historical
evidence, not a description of the promoted default binary.

## Problem

The WB97M-V water-96 / complete spherical def2-TZVPD cold investigation found
that canonical angular-order-five values repeatedly prepare primitive geometry,
Hermite coefficients and Coulomb states for individual Cartesian components.
The dominant pair-sum `(3,2)` bucket admits 26,062,583,408 Cartesian AO quartets
per radial action. Enabling the existing dense through-f bounded schedule is not
an acceptable substitute: its previously measured endpoint regressions are
documented in the through-f value-policy note.

## Decision

Extend the existing preparation-frozen
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=1` qualification selector to the
canonical DFT value consumer. It remains off by default.

- Borrow the generated owner's already budgeted, immutable primitive-pair cache;
  do not create another geometry owner to obtain that cache.
- Sort shell-pair keys by item and angular sum. Keys are maxima of the original
  canonical AO Schwarz bounds, not alternative scientific or density bounds.
  Prefix only the three order-five shell-pair domains, using the existing CUB
  segmented sort, product comparison and inclusive row scan.
- Persistently traverse indexed shell quartets with 256-lane CTAs. At most 162
  Cartesian components exist at total angular order five, so the complete
  quartet fits one packet. No quartet queue or four-index value tensor is kept.
- Reuse the compiler-owned pair-materialized recurrence and authoritative
  full/SR/LR moments and orbit scatter. The helper can publish independent
  positive J/K planes without changing its existing HF Fock defaults.
- Preserve every original component-level Schwarz comparison. Canonical bounds
  are row-major, whereas the HF helper's gate is column-major. Retain a
  bit-preserving, geometry-time transpose through the existing CUDA matrix
  adapter; do not assume separately computed `(ij|ij)` and `(ji|ji)` bounds are
  bitwise symmetric at a screening boundary. The extra matrix is explicitly
  charged, including the preparation peak.

Canonical census units remain admitted Cartesian AO quartets and contracted
radial evaluations. Shell-prefix entries are a different unit and must not be
reported as AO counts or primitive recurrence counts.

## Boundaries and fallback

This route covers order-five full J/K, J-only, and standalone full/SR/LR K in
strict-FP64 Cartesian sources, including spherical public projections. Other
orders, absent primitive-cache/index leases, unsupported capacities, fixed
screening, compensation planes and resident-source replay retain their original
consumers. A combined full-J/selected-range-K action and the existing paired RSH
consumer also retain the scalar route. Preparation freezes selection; execution
does not reread environment controls or allocate memory.

The optional index and transpose cannot displace an incumbent generated owner.
Budget rejection and allocation failure leave the existing canonical source
usable. No production CPU/PySCF work, threshold relaxation, basis reduction or
grid change is introduced.

## Evidence

Qualification uses a frozen latest-master snapshot
`4444d0376bb133f7981f44a263e2e409bfbfdd27`, with a targeted candidate patch.
The working checkout's unrelated local modifications are not part of that
snapshot. All real-device work uses finite Slurm allocations on permitted nodes.
Compilation uses verified ccache 4.5.1 without clearing shared caches.

The final native provider gates cover independent CPU full/SR/LR matrices,
RHF/UHF, J/K masks, screened diffuse and signed contractions, all three
order-five domains, spherical/Cartesian layouts, two-system batches, displaced
geometries, census agreement, preparation-policy freezing, an exact-budget
fallback and incumbent optional-allocation fallback. The bounds-view test checks
the exact row-to-column correspondence.

The separate real-GPU pair-materialized qualification covers orders 5--12,
tails, repeated pairs, inactive/empty claims, signed contractions, SR/LR moments,
the original HF consumer and derivative consumers. An admitted 162-component
order-five packet with 16 primitive products executes 16 Coulomb preparations,
2,592 component contractions and 162 publications, rather than preparing one
recurrence for every primitive/component combination.

For the final water-96 `(3,2)` source probe, policy 0 and policy 1 use the same
binary and Slurm GPU. Other source buckets are deliberately omitted, and the
observer exits before producing a physical solver result:

| Metric | Scalar canonical | Indexed materialized |
| --- | ---: | ---: |
| Full J/K fenced source seconds | 277.911619 | 156.057308 |
| LR K fenced source seconds | 274.660582 | 159.299363 |
| Combined source seconds | 552.572200 | 315.356671 |
| Indexed outer entries per action | 26,062,583,408 AO quartets | 249,422,096 shell quartets |

This is about 1.752x / 42.93% less **isolated source wall time**, not a complete
energy/force endpoint speedup. Static compiler resource reports show 174
registers / 4,136 stack bytes for the retained RKS scalar order-five kernel and
141 registers / 512 stack bytes / 8,336 shared bytes for the RKS materialized
kernel. These are not achieved-occupancy or measured-spill claims.

Complete endpoint and numerical receipts are retained in
`benchmarks/results/wb97mv-canonical-materialized-20261010/`. The stock
12-atom independent reference failed to converge under the unchanged protocol;
do not weaken its gates or describe a native/native comparison as an independent
reference pass. No complete 96-atom E+F or independent 96-atom accuracy result is
claimed.

The final fresh-owner 3-atom E+F median is 12.740837 -> 12.582382 s over three
samples per policy, all with 15 iterations/15 Focks and independent E/F gates
passing. The final 12-atom native/native pair completes in 582.965492 ->
531.350324 s (8.85% less), both with 21 iterations/21 Focks. Force times are
38.570627/38.533716 s; the reduction is in SCF. Native/native E/F differences
are 5.685e-14 Eh / 5.706e-11 Eh/Bohr. The larger comparison has only one pair
and no converged independent reference; it does not authorize default promotion.

## Rejected alternatives

- Skip exact-zero density weights: the investigated first Cartesian density has
  4,194,304 nonzero entries out of 4,194,304. This does not solve the target.
- Enable the old bounded angular traversal: memory bounds do not remove its
  rejected quartet work or previously measured endpoint regression.
- Replace AO screening with shell screening, assume bitwise bound symmetry, or
  recover raw primitive coefficients by dividing cached weighted coefficients:
  these can change membership or lose zero/underflow scientific inputs.
- Promote a default from a selected kernel or a first-use JIT cache advantage:
  complete, matched endpoints and acceptance gates remain necessary.

## Revisit when

First qualify paired full/LR reuse inside the materialized consumer, keeping
separate source matrices and original masks. Then measure bounded order-six and
order-seven consumers with explicit multi-packet ownership and register/work
gates; simply changing the recurrence order would omit components. Default
promotion still requires complete representative larger cold/warm/moved and
constrained-budget endpoints, appropriate independent accuracy gates, and
semantic work counts.

## References

- [Original investigation](../../proposed/2026-10-10-wb97mv-tzvpd-cold96-source-reuse.md)
- [Compiler recurrence ownership](../architecture/2026-10-06-direct-pair-materialized-recurrence.md)
- [Retained through-f value policy](2026-10-02-through-f-value-policy.md)
- `docs/maintainer/performance_engineering.md`
