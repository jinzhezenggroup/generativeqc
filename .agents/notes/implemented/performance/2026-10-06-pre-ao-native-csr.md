# Decision: qualify pre-AO envelopes with one native CSR lease owner

Status: implemented (local guarded default; final public-route validation pending)
Date: 2026-10-06

The frozen all-native promotion experiment below is historical. The current
portfolio preserves the subsequently enabled sampled default; see the
[superseding portfolio decision](2026-10-06-pre-ao-portfolio-default.md).

## Problem

Issue #1893's 2026-10-06 code-level plan targets preventing dense AO/jet work,
not another geometry micro-schedule. Sampled-jet force discovery evaluates all
NAO jets, downloads flags, retains Python label maps, then uploads labels on
every selected tile. Warm selection saves contractions, but cold/moved discovery
still pays for the dense AO pass it is supposed to avoid.

The current master baseline for this experiment is
`a85459c13d16741ef5fab37cca6a64fc4a5bca15`. The indexed contract from #1980 is
integrated without its rejected point-parallel geometry experiment. Its native
XC density-launcher binding must use `block.point_start / l.tile_points`, not
the removed `tile` variable.

## Implementation

- Compiler-owned `dft/envelope_cuda.py` lowers the existing absolute-polynomial
  Gaussian region-bound algorithm. Normalized primitive and Cartesian/spherical
  coefficients come from the existing packed basis. All configured jets through
  the requested order participate; AO values at derivative nodes are insufficient.
- Positive arithmetic rounds outward. Gaussian exponents round downward; the
  exponential has the existing positive underflow floor and libm guard. Degree
  overflow/unsupported degrees retain AOs via infinity rather than omitting them.
  `HUGE_VAL` keeps CUDA `nextafter(double, double)` overload resolution in FP64.
- Native `GridPlan` owns CSR offsets and sorted AO labels. Build-time masks are
  warp ballots; integer prefix compaction preserves ascending, unique labels.
  This is storage backing for `AoGridBlockLayout`, not another scientific layout.
- Only scalar offsets have a native host mirror for variable provider shapes.
  Scientific AO labels stay on the device. Warm replay resolves the native span
  and lends the existing AO/density/force/scatter buffers; it neither looks up a
  Python label map nor uploads per-tile AO IDs.
- Density updates invalidate task leases but not the geometry-bound CSR. A
  separate native geometry epoch revokes CSR on center rebinding, even when a
  caller bypasses the Python epoch adapter. Domain tokens additionally include
  basis/grid/geometry identities, point order, device, cutoff and jet capability.
- Ordinary force and compatible same-grid XC/nonlocal consumers use the same
  lease adapter. Automatic inventory sharing with the separate native KS/XC
  owner is **not implemented**. Cross-owner/cross-capability sharing needs an
  explicit identity, stream and lifetime bridge; matching pointers/shapes are
  not such a bridge. Do not claim the entire P0-B sharing goal is complete.

## Resource and lifetime invariants

CSR buffers use `OwnedCudaBuffer` and the device resource ledger. Numeric peak
includes device and host offsets, temporary boxes/bit masks, and retained indices.
Both host and device headroom are reserved after dense scratch admission. Budget
or optional-artifact capability misses stay dense. Native OUT_OF_MEMORY=7 at
the optional map boundary has its own compiler-adapter exception and stays
dense after native RAII discards the incomplete owner; no generic runtime or
Python-memory exception is swallowed. Invalid geometry, point order,
numerical data and device errors propagate instead of silently omitting work.

The source's resident quadrature lease must be token-checked before entering the
owner and remain alive/immutable throughout execution. Pointer equality alone
does not prove scientific identity or allocation lifetime. Native CSR buffers
drain their lifetime stream before the borrowed grid context is destroyed.
Derivative capabilities remain exact-order in this adapter; deriv=1 is never
relabelled as deriv=2.

## Evidence so far

- Focused device-free contract suite: 612 passed, 1 opt-in skip.
- Independent native/host regression suite: 249 passed, 3 opt-in skips.
- Compiler structure: 470 modules, zero dependency errors.
- RTX 5090, n1, finite Slurm job 6176: 25 CUDA gates passed, covering through-order
  jets 0–3, cartesian/spherical and signed contractions, diffuse/tight bases,
  AO nodes, empty tails, overflow retention and invalid-point rejection.
- Job 6176: memcheck zero errors; racecheck zero hazards/errors/warnings.
- An **excluded prewarm**, not promotion evidence, on 48 atoms executes
  205,995,008 force point-AO visits versus dense 452,984,832, and
  169,684,508,672 projection FMA pairs versus dense 695,784,701,952.
  Native pre-AO discovery evaluates zero AO jets and uses about 6.95 MB numeric
  peak. Its warm replay reports zero discovery and zero AO-label transfer.

Raw receipts are retained under `.artifacts/preao/`; job 6178 runs complete
48/96-atom PBE0/RKS A/B/C with five interleaved campaigns. Every projected pair
uses the shared ABBA order, matching geometries, sources, binaries and independent
GPU4PySCF references. Cold/warm/moved/moved-warm E+F and actual Fock histories
remain intact. Intrusive stage profiles are separate from endpoint timings.

## Promotion and deferred work

### Completed 48-atom endpoint comparison

Job 6178 completed all five interleaved A/B/C campaigns (60 complete E+F
records, excluded prewarms omitted). Medians in seconds are:

| Force domain | Cold | Warm | Moved | Moved-warm |
| --- | ---: | ---: | ---: | ---: |
| Dense | 102.509626 | 11.261029 | 57.017906 | 11.208577 |
| Sampled jets | 102.353227 | 9.203358 | 53.431841 | 9.180656 |
| Pre-AO native CSR | 97.421620 | 9.474753 | 52.088667 | 9.451617 |

The candidate reduces executed **force** AO visits/jet entries by 54.525% and
projection FMA pairs by 75.612%. These are not whole-SCF census reductions:
all variants retain master's sampled SCF domain. The raw native SCF diagnostic
contains a one-traversal domain inventory and solve-local `xc_evaluations`.
Keep these semantics and actual Fock histories explicit rather than treating
one inventory as an executed full-solve count.

All independent endpoint gates passed (maximum energy error 8.87e-12 Eh,
maximum force error 2.78e-11 Eh/bohr). Warm/moved/moved-warm improvements over
dense are 15.86% / 8.65% / 15.68%. **The 48-atom profile does not qualify**:
cold improves 4.96%, below the shared robust-noise floor of 6.07%. This is a
failure of the original **all-phase** criterion, not absence of warm benefit. Compared
with sampled jets, the conservative box domain is about 2.95% slower on both
warm phases. No threshold was weakened and no samples were removed.

Endpoint-only machine-readable evidence:
`.artifacts/preao/qualification-48-endpoint-only.json`; separate intrusive
profiles and the complete 96-atom comparison remain pending. Job 6191 is a
finite Slurm continuation dependent on job 6178. It admits only whole,
oracle-qualified campaigns, retains the original scheduler provenance and
reruns missing/incomplete campaigns without overwriting raw receipts.

On 2026-10-06 the user explicitly accepted stable warm improvements as sufficient
to enable a default. This supersedes the all-phase promotion criterion;
it does not turn the cold result into a significant improvement. The local
registry binds ordinary CUDA all-electron non-DF RKS/UKS and deriv=1/2. Following the user's request not to
overfit dispatch to benchmark dimensions, atom/AO/grid counts have no specific
shape whitelist: 48/96 are validation samples, not eligibility identities.
The user additionally requires no GPU-model restriction: architecture/product
names are diagnostic provenance, not a whitelist. Other force compositions or
missing required capabilities remain dense. Producer choice is forwarded through public
dispatch and prepared identities. Mean AO occupancy above 0.8 declines the
entire inventory; it never truncates labels or invents a larger AO cutoff.

The final public-route A/B/C must prove that `auto`, without a producer override,
executes the native CSR and clears both warm robust-noise gates. Cold/moved
timings, memory bounds, independent accuracy and actual work remain mandatory
evidence. Eight full-force independent analytic gates (LDA/PBE/R2SCAN RKS/UKS)
passed on n1 Slurm job 6199. No historical #1833 result or metadata-only change
substitutes for qualification; the complete 96-atom evidence is still pending.

P0-D (restricted-spin projected-panel alias/stride) and P0-E (sparse provider
scheduling) remain deferred until sparse complete-endpoint qualification.
Do not resume stable-owner grouping/schedules 1–3 or point-parallel
`geometry_point_setup`; their retained negative receipts are in the prior
workspaces and #1893/#1980 comments. Do not overwrite those snapshots.

## Revisit when

Final-source A/B/C clears the complete-endpoint and work gates. Then tighten the
production guard, qualify compatible cross-owner sharing, and address remaining
panel traffic/provider bottlenecks using the sparse route's actual profile.
