# CUDA Fock execution policies

This page documents device-specific scheduling, optional diagnostic
experiments, and final-state validation used by the
[shared Fock construction contract](fock_build.md). It is **not**
the authority for raw J/K spin coefficients or public Fock capability.
For the latter, use the [parent contract](fock_build.md#densities-operators-and-coefficients)
and [method/capability reference](../reference/capabilities.md).

## CUDA MD-J default

Eligible CUDA strict-FP64 exact Coulomb values use the density-contracted
McMurchie–Davidson source by default. Admission requires an s/p/d basis, at least
eight public AOs and sufficient optional resident capacity outside a public
resource ledger. Public `ResourceBudget` plans retain normal J because their
incumbent inventory reserves no optional MD storage, including when the budget
has no explicit cap. This preserves later owners, rebuilds and force storage.
Geometry transforms
are prepared on the owning stream; replay contracts the total density into
Hermite coefficients and projects the Coulomb potential back to public AOs.
Normal generated/canonical K, XC, SCF and finalization owners are unchanged.

The public-AO Schwarz mask remains authoritative. Uniformly accepted shell
quartets use the Hermite contraction, while partially screened quartets retain
per-orientation AO eligibility in bounded source pages. Nonsymmetric and UKS
densities include both input orientations. Additional density screening shares
`min(screening_tolerance, 1e-12)` across the complete shell-pair census for each
output element; K never consumes these density envelopes. Screening zero retains
the unscreened contraction except for exactly zero density work.

Optional descriptors, transforms and scratch are charged within a 128-MiB cap
after retained normal owners. Unsupported angular momentum, small systems,
insufficient optional capacity or optional allocation failure retain normal J.
Mixed J, fixed-mask response and derivative requests bypass MD execution.
Derivative-capable owners may retain the metadata for their zero-order requests.
`direct_schedule` reports admission as `md-j-hermite/retained-k`, not a claim
that every request uses that route.

`GENERATIVEQC_DISABLE_MD_J=1` restores normal J at plan creation for diagnostics;
no enable flag is needed for the default. `GENERATIVEQC_MD_J_COUNTS=1` reports
native execution and geometry-candidate censuses when the owner is released.
Candidate counts are upper bounds on probes, not executed primitive products.

`benchmarks/md_j_normal_cold.py` requires three alternating pairs of fresh
processes, identical native library/input/device identities, actual native
96-atom/768-AO grid counts, independent energy and physical-residual gates, and
MD calls matching complete Fock counts. It explicitly removes the disable flag
for default samples. Run all GPU qualification through finite Slurm allocation;
kernel timings and incomplete cohorts do not qualify a cold advantage.
See the [default decision](../../.agents/notes/implemented/performance/2026-10-08-md-j-default-cold.md)
for error-budget rationale, measured evidence and rejected residual schedules.

## Bounded indexed force schedule

Derivative-capable generated exchange owners use the per-system descending
Schwarz pair order by default. `GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=0` (or
`none`) explicitly restores the triangular schedule for debugging and paired
benchmarking; `1`, `indexed`, and `auto` select the default indexed route.
Invalid nonempty values fail closed to the triangular route. Preparation reuses
the existing geometry-bound readback; no density-dependent index is retained.
Independent full-range J/K force sources use an exclusive prefix over
geometry-live block rows, with 16 independently claimed 64-candidate pages per
admitted block product. Density, exact shell and AO screening and physical
quartet orientation are unchanged. Other consumers retain triangular traversal,
although consumers sharing an indexed derivative owner see its sorted pair order.

The optional device prefix costs `(pair_blocks + 1) * sizeof(uint64_t)` within
the owner's existing budget. Insufficient prefix capacity retains sorted
triangular traversal; inability to admit the owner retains the existing generic
fallback. Sorting and paging reduce candidate amplification and improve load
balance, not the dense worst-case scaling. Complete same-binary qualification
showed warm wins from 3 through 96 atoms; the retained moved-geometry timing
negative remains documented and is not erased by this default promotion. See
the [schedule decision](../../.agents/notes/implemented/performance/2026-10-03-schwarz-indexed-independent-force-domain.md).

`GENERATIVEQC_BOUNDED_ANGULAR_FORCE=1` (or `angular`) separately opts full-range
J/K and omega=0.3 LR force sources into thirteen total-angular-order passes.
It is **off by default** and changes neither the recurrence nor the screening
gates. Full-range passes retain the selected indexed/triangular domain; LR keeps
its existing triangular domain. Cursor/output storage is reused on the owning
stream, while shell enumeration is repeated per order. The purpose is to qualify
the tradeoff between compiled kernel resources and repeated scanning, not to
assume less integral work. The switch is recorded in resource and checkpoint
identity; differing or missing historical policy needs explicit warm admission.
The retained diagnostic has negative endpoint evidence and is not a promotion
candidate; see the [rejected experiment](../../.agents/notes/rejected/2026-10-04-tzvpd-angular-force-schedule.md).

`GENERATIVEQC_DIRECT_COULOMB_REACHABLE=1` (or `reachable`) is a separate,
**off-by-default** Cartesian-source experiment. The compiler passes each AO
component's summed axis powers to the shared Coulomb recurrence, which evaluates
only the exact dependency closure of the consumer's roots. Full/LR moments,
screening, primitive counts and reserved auxiliary storage are unchanged. The
policy is frozen when the native J/K provider is prepared and is included in
resource/checkpoint identity. The public-AO fallback and specialized low-order
workers retain their existing evaluation. Host arithmetic checks do not qualify
CUDA execution or performance; see the [recurrence experiment](../../.agents/notes/proposed/2026-10-04-reachable-coulomb-states.md).

The recurrence control also accepts `values` or `forces` to qualify either
consumer independently; `1`, `reachable` and `all` select both. This selection
is frozen with each source owner. It does not adapt to molecule size or current
SCF iteration.

`GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION=values`, `forces`, or `all` (`1`)
separately selects experimental pair-axis coefficient convolution in generic
strict-FP64 Cartesian contractions of total order five or higher. This replaces
the six pair-index loops with three axis convolutions and a three-axis root
contraction. It retains radial moments, primitive/AO admission and the original
Coulomb workspace; floating-point accumulation order changes. Specialized
low-order/all-center consumers, mixed precision and public-AO fallback keep
their existing contraction. The switch defaults off and participates in
resource/checkpoint identity. It establishes no automatic policy or performance
claim; see the [reassociation experiment](../../.agents/notes/proposed/2026-10-04-hermite-axis-convolution.md).

## CUDA DF final-state validation

The shared final-state selector owns the identity checks, absolute/scaled
acceptance gates and bounded physical-Fock correction loop. Its CUDA algebra
provider evaluates eigen residuals, metric orthogonality, density reconstruction,
DSD idempotency, commutator, electron/spin traces, physical energy and requested
canonicality with FP64 device products. Stable norm reductions detect nonfinite
inputs and overflowing products before acceptance. Matching sizes or a prior
converged flag never authorize reuse.

Ordinary finalization consumes the retained C/epsilon and verifies device
solver status and determinant generation against the full source/model/epoch
identity. Frame validation downloads compact diagnostics. Force selection also
downloads the current physical Fock for the existing ordinary eigenprovider and
checks its occupied projector against the requested density tolerance. A failed
probe is reused by bounded correction. Requested physical-reference C and
force-consumer W remain explicit host outputs. After acceptance, the device
forms W as `D F[D] D / spin_weight` using two counted GEMMs and existing scratch.
Reference export reuses the
selector's stronger canonicality diagnostics. The independent CPU reference
validator remains available for imported references and scientific tests.

One lazy workspace per prepared DF plan holds nine matrices, one spectrum and
bounded reduction storage, serialized across items and spins and included in
tile-budget admission. It never borrows graph scratch. CUDA allocation, library
and execution failures propagate without selecting a CPU fallback.

`GENERATIVEQC_DF_REFERENCE_FINAL_VALIDATION=1` explicitly restores the CPU validation,
projection and W path for independent diagnostics and causal timing comparisons.
`GENERATIVEQC_DF_FORCE_FINAL_REBUILD=1` and `GENERATIVEQC_DF_REFERENCE_FINAL_EIGEN=1` still
perform the actual bounded correction/rebuild path. Component traces report
validation/W GPU intervals, transfers, synchronization and workspace bytes;
host regions and the progress journal retain physical-Fock/eigen/correction
counts. These intrusive diagnostics must be run separately from clean endpoint
timing, with independent oracle checks outside the timed call.

The progress journal separates frame reconstruction from physical fixed-point
defects, and reports fixed-point checks, their eigen solves and rejections.
These extra force checks leave energy-only selection unchanged and must be
included in complete endpoint cost.

See the [device-validation decision](../../.agents/notes/implemented/performance/2026-09-16-device-final-validation.md)
for ownership rationale, resource tradeoffs and qualification evidence.

## Canonical range-separated J/K value experiment

`GENERATIVEQC_CANONICAL_RSH_VALUES=shared` (or `1`) is a default-off canonical
full J/K plus SR/LR K value experiment. Eligible strict-FP64 Cartesian sources
share density preparation and one canonical traversal; generic orders >=5
also share primitive geometry/Hermite preparation. Radial moments and all
three source outputs remain separate. The extra two-spin Cartesian output
matrix costs `2 * batch * cartesian_aos**2 * sizeof(double)` and is admitted
after existing source/force owners within the same budget. Its absence retains
the separate paths. This does not enable bounded through-f values, change
force consumers, or establish an endpoint speedup.

The strict final-state identity and correction rules are defined in
[DF final-state selection](df_final_state.md); an opt-in source
experiment does not change those rules.
