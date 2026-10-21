# Diagnosis: DF needs earlier contraction and a compact source domain

Status: proposed
Date: 2026-10-22

## Measured difference

Observing stock GPU4PySCF DF J/K after its ordinary calls, without changing
convergence, grids, screening, initial guess or returned operators, identifies
two distinct algorithms. These are diagnostic complete E/F endpoints, not new
README timing populations. Both 96-atom populations and all three frozen warm
repeats pass the unchanged independent README energy/max-force/RMS-force gates.

For r2SCAN-3c, the common basis has 1088 orbital and 6816 auxiliary AOs.
An uncompressed triangular raw source has 32,303,259,648 bytes per complete
pass. This is a logical generated volume, not simultaneous resident memory.
The native streamed Coulomb implementation explicitly generates two such
passes when no occupied-K charge is supplied. The raw-shell prototype reduces
repeated primitive geometry across components but retains this logical domain.
Its recorded `coulomb_exchange_work` is incomplete; do not invent executed
recurrence counts from iteration counts or these size formulas.

The stock reference pure-J path does not construct CDERI or a three-center
tensor. It first transforms D to Hermite coefficients, contracts to the
auxiliary charge, applies the metric to vectors, and contracts back through
Hermite coefficients to J. Its own prepared diagnostic inventory is:

- 928 decontracted orbital shells and 2208 decontracted auxiliary shells;
- 431056 possible triangular orbital shell pairs, of which 62844 remain;
- 338316 Hermite pair coefficients, a 2,706,528-byte pair buffer;
- no resident `_cderi` tensor.

The 14.58% retained-pair fraction is within the reference's decontracted shell
domain. It is not a ratio of native/public AO pairs, a FLOP ratio or a complete
memory-usage ratio. The 2.71 MB buffer is one working intermediate, not its
entire endpoint peak. Observed complete cold/warm times in this diagnostic
population are 25.945 s and 12.507/12.488/12.559 s, with 11/1/1/1 cycles.

PBE0 has a different stock DF path. Its 768 orbital and 3616 auxiliary AOs
would require 8,542,322,688 bytes for the dense triangular raw domain. Its
resident GPU CDERI instead has shape `(3616, 80304)`, or 2,323,034,112 bytes.
This screened pair representation is small enough to retain and reuse. Pure-J
fusion and hybrid compact-factor residency therefore need separate policies.

## What the existing raw-shell prototype establishes

The uncommitted opt-in `GENERATIVEQC_DF_RAW_SHELL_VALUES=1` prototype in the
raw-shell worktree shares generated Gaussian/Boys/moment preparation across
Cartesian components, then applies public/spherical transforms after primitive
contraction. Independent dense/packed, ragged/full, signed-contraction and
orbital-f/auxiliary-g values passed 32 populations with errors below 1e-13.
Six candidate endpoint populations and one matching rollback passed 28 E/F calls.

On r2SCAN-3c/96, same-binary flag-off/on observations are 1222.104/575.751 s
cold and 172.469/100.273 s median warm (2.123x/1.720x), with identical 16 cold
and one warm iteration. This remains far behind the reference. Sharing more
scalar work alone does not remove the algorithmic contraction/materialization
gap. The prototype remains default off; this diagnosis does not promote it.

## Next implementation boundary

For pure J, move the linear density and auxiliary-vector contractions ahead of
Cartesian component expansion, using the existing compiler-owned Hermite/Boys
mathematics. Prepare immutable geometry/topology once and retain bounded
optional working storage with an explicit fallback. Preserve metric rank and
factorization policy, signed contractions, spherical normalization, complete
force response and source/device/stream lifetime ownership. Benchmark complete
endpoints including preparation, projection and force export.

The existing Direct MD lowering has pair angular order at most four and an
8-bit radial-address inventory. Do not silently reuse that boundary as an
orbital-f or auxiliary-g capability claim. Three-center orbital-f pairs plus
auxiliary g require radial order up to ten, with 286 simplex entries; an 8-bit
address cannot represent that full domain. Share/refactor the scientific
lowering, qualify the larger addressing/layout independently, and keep existing
Direct specializations unchanged unless their own tests/evidence require it.

For hybrids, investigate bounded compact pair residency with conservative
native screening and the existing exact occupied/dense fallback contracts.
Any omitted integral work needs an explicit numerical/error contract and
independent force acceptance, not just matching the reference's cutoff value.
Do not replace failed common-grid acceptance by a more permissive gate.

## Retention

Ignored `.artifacts/df-reference-work-v1/` and `df-reference-work-v2/` in
`qc-df-raw-shell-20261022`, and corresponding directories under
`n1:/data/jzzeng/scf-performance-20261021/`, retain the observing scripts,
full receipts, inventory and independent qualification. The first PBE0
observer assumed `_cderi` was a dict; the actual list caused a diagnostic
exception. Its failed receipt remains in v1; the corrected v2 handles both
containers and completed the unmodified reference endpoint. No native failure
or numerical gate was replaced. Reference work uses finite n1 Slurm allocations.

Raw v3 retains matching source manifests, core/AOT packages and all endpoint
receipts. Its core SHA-256 is
`5931c581943ac89b94b7e1d0ad4c6986549d5fbab8406776490918583cac222d`.
The later lightweight header cleanup and 92 passing host dispatch/domain/
lifetime tests are not yet a rebuilt/promoted raw binary. Renew source-bound
qualification before publishing a default policy.

## Completed raw diagnostic follow-up

The frozen v3 r2SCAN-3c/96 trace contains 111 valid records, zero dropped
regions/tiles and 19 actual J operations (16 cold, three warm). Every J produces
74 raw tiles / 64,606,519,296 output bytes; its 37 distinct auxiliary tiles are
produced twice. Cold total is 560.837 s, including 458.116 s in J and 58.411 s
in force response; the warm totals are 96.680/96.703/96.714 s, with ~28.66 s J
and ~58.39 s force response. All four E/F calls pass the independent gates.
The cold J source volume is 1,033,704,308,736 bytes. These are recorded tile/
output-byte counts, not inferred recurrence counts; nested trace intervals
must not be summed. J fusion alone leaves a major force-response bottleneck.

The separate v3 raw sanitizer scope completed all 32 independent populations
with `--target-processes all` and `ERROR SUMMARY: 0 errors`. The first attempt
failed before CUDA because the probe needed a `.so.0` SONAME; a separate
sanitizer-v2 library directory supplies a symlink to the identical frozen core.
Preserve both attempts; no frozen runtime was modified. Evidence lives under
`df-raw-shell-v3/trace-work-summary.json` and `df-raw-shell-sanitizer-v2/` in
the ignored local/remote artifact roots above.

## Qualified native Hermite building blocks (not a production backend)

The separate `qc-df-md-j-20261022` worktree now shares the Direct radial
emitter through order ten and provides native FP64 E preparation, density
precontraction, forward/transpose radial consumers and public-AO projection.
Every unordered shell pair retains all ordered same-shell primitive products;
distinct shell densities fold both orientations. Auxiliary g uses a separate
six-term public expansion stride and the second Hermite recurrence dimension.
The order-ten radial address inventory uses uint16. No three-center public-AO
tensor is produced. The emitted shared math retains its upstream Apache notice.
Direct's original order-eight emitted bytes were checked unchanged once; this
is an opt-in migration check, not a permanent source fingerprint assertion.

A native topology planner preflights the complete optional E/H/descriptor/AO
storage and host peak before upload. Unsupported domains, malformed metadata,
overflow and insufficient budgets have explicit refusal paths. It currently
uses the complete unscreened primitive census and dense public E intervals.
It has **no production owner, default admission, metric integration, screening
policy or force implementation**. The standalone probe's RAII owner is only a
qualification adapter; it is not evidence of production resource lifetime.

Evidence in ignored `.artifacts/df-md-j-device-v3/` and the matching directory
under `n1:/data/jzzeng/scf-performance-20261021/`:

- 178 independent libcint populations, 534 device replays, passed: all 80
  orbital s/p/d/f by s/p/d/f by auxiliary s/p/d/f/g classes in Cartesian and
  spherical form; 18 coincident/diffuse/tight/translated populations; signed
  contractions; non-symmetric densities; zero and changed-input replays.
- Maximum first/replayed charge/J errors are 6.52e-13 / 7.99e-13. These are
  raw charge/projection gates, not molecular energy/force or timing claims.
- 27 compiled host tests pass layout budgets, invalid metadata and orbital-f/
  auxiliary-g admission. 1478 host shared-math/Direct regression checks pass.
- v2's identical E/R kernels passed memcheck on all 178 populations and
  racecheck on 18 geometry populations, with zero errors/hazards. Logs and
  receipts live in `df-md-j-sanitizers-v1/`. v3 changes the host topology
  adapter to the budgeted planner; its fresh 178 device populations also pass.
- v1 compiled successfully but all children failed before CUDA because the
  fixture resolved a library symlink away from its diagnostic SONAME directory.
  v2 preserves the caller-selected library directory; failure receipts remain.

The small probes link the immutable early-final-v3 core only for canonical
basis normalization. Core SHA-256 is
`0228e9e65ef067838b107fa207d8c2a2ea030c06cb8e3e585bca7329c598d8b7`;
new E/R kernels and exact source manifests are compiled separately with verified
sccache 0.16.0 and run under finite n1 Slurm allocations. No reference integrals
enter native preparation. The independent Python oracle supplies comparisons
only. Shared-library linkage does not claim that core contains this DF backend.

An unscreened metadata calculation for the 96-atom r2SCAN-3c orbital basis gives
512 contracted shells, 431952 ordered primitive products in unordered shell
pairs, 1928912 Hermite coefficients, 30,862,592 bytes of density/potential and
109,698,688 bytes of expanded E. This excludes auxiliary, descriptors and all
other runtime storage. Factoring E by axis would reduce its logical storage to
49,020,288 bytes but is not implemented or timed. These are topology/capacity
counts, not measured executed recurrence work. Preserve the distinction from
the reference's screened decontracted-shell census.

## Qualified bounded private owner

The follow-up `df_md_owner.hpp/.cu` provides transactional preparation and
allocation-free charge/projection methods on one borrowed stream. It checks the
combined orbital/auxiliary host and device budgets before GPU allocation,
counts every resident E/H/descriptor/AO buffer, rejects creation during capture,
checks the current device, rejects null or identical input/output pointers,
and drains before freeing partial or completed storage. Full input/output range
disjointness remains the caller's contract. The caller must retire captured graphs
before destroying the owner and keep the borrowed stream alive. Public metric
policy, source identity/stream handoff, SCF admission and force integration are
still absent. No default route or environment selector calls this factory.

`df-md-j-owner-v1/` and `df-md-j-owner-v2/` each pass all 178 libcint populations
and 534 stream/graph replays. Projection runs before charge to prove it does not
depend on stale input or a previous charge call. v2 additionally forces a real
ledger rejection after partial allocation, verifies zero surviving bytes, then
admits at exactly the reported device footprint. Captured changed/zero inputs
allocate no native device storage, and destroying the owner returns ledger live
bytes to zero. v2's maximum first/replay errors remain 6.52e-13 / 7.99e-13.

`df-md-j-owner-sanitizers-v1/` applies memcheck to all 178 v2 populations and
racecheck to all 18 geometry populations: zero errors/hazards, including partial
allocation rollback and graph reuse. These are still charge/projection and
lifecycle gates, not complete molecular energy/force or performance evidence.
The new owner is not yet in the core CMake source registration; the tests compile
its exact translation unit with sccache and link only canonical basis helpers
from the frozen core described above. v2's complete source-input manifest names
base commit `17e393183` plus the recorded owner changes. v1 retained the earlier
campaign base in its metadata; both inventories pin each actual compiled input.

## Force work within the raw v3 diagnostic

Disjoint regions further partition each ~58.4 s force response: the raw Coulomb
charge/potential takes ~24.925 s, and 24 three-center derivative contractions
take ~32.45 s host / ~32.33 s device in total. The trace records one dense raw
source pass with 64,547,192,832 bytes, 24 charge panels and the same number of
three-center derivative weight bytes. See the ignored raw worktree's
`df-raw-shell-v3/force-work-summary.json`. These are children of the response
region and must not be added to their parent. A packed charge consumer or a
properly authenticated final-potential lease may remove redundant charge work;
neither removes the remaining derivative contraction bottleneck. No reuse may
assume that a canonical final density is identical to an earlier proposal.


## Optional source/metric/SCF handoff under qualification

The prepared restricted pure-J adapter now has a default-off
`GENERATIVEQC_DF_MD_COULOMB=1` candidate. The choice participates in prepared
source identity. Core CMake owns the generated DF Hermite header and native
owner translation unit. Installation occurs on the completed value plan's
long-lived stream, after source creation's temporary stream has retired and
before any SCF state or graph exists. It uses the exact normalized systems
that created the source; there is no device-to-host basis reconstruction.

The candidate applies the same retained full-rank eigensystem vector solve as
the raw-panel path between charge and J projection. Resident factors, hybrid
exchange, batches, truncated metrics, angular domains beyond orbital f /
auxiliary g, and unavailable capacity retain the incumbent. The original
source and metric remain available for unchanged force response. Owner
teardown follows SCF graph retirement and precedes destruction of the borrowed
stream. Device/numerical errors propagate; optional allocation refusal does
not authorize a reference-oracle retry.

An initially attempted after-setup-only admission is insufficient at the
96-atom streamed cliff: the recorded raw v3 planner used 14,509,920,430 of
14,536,349,827 value bytes. Its 26 MB slack cannot hold the roughly 200 MB
Hermite owner. The refined adapter shares the native layout census without
materializing metadata to reserve its exact device footprint before sizing
interchangeable fallback panels. It never enlarges the total value cap or
borrows response capacity. If even minimum fallback panels cannot coexist,
the previously feasible ordinary tile plan remains available. All admitted
owner device/host storage is added to the enclosing plan diagnostics.

The standalone source/metric/J test links the actual newly compiled core and
uses independent libcint raw/metric values plus NumPy's spectral solve. It
covers nonsymmetric density, same/zero/changed-input graph replay, two budget
refusals with still-executable incumbent J, exact owner resource accounting,
and duplicate auxiliary shells that must retain the truncated-metric fallback.
Qualification is ongoing; no complete endpoint or default-promotion claim is
implied. Submitted traversal counters explicitly distinguish host submissions
from executed graph replays. The candidate remains unscreened and does not
repair the independent three-center derivative bottleneck.


The integrated `df-md-plan-v2` core (SHA-256
`5fac69b4a90465135a5f72e478def9806320eb85bfaa218ce9775a3945f109e7`)
passes 16 source/metric/J populations: 14 full-rank Cartesian/spherical cases
through orbital f / auxiliary g and two duplicate-auxiliary truncated cases.
Every case also checks two capacity refusals with incumbent replay. Full-rank
cases exercise three captured density replays and exact reservation accounting.
The largest error across the incumbent, native J, and first graph replay against
the independent metric solve is 9.5591e-14. All 16 populations pass memcheck;
four f/g and truncated populations pass racecheck, with zero errors/hazards.
Evidence and source-matched receipts remain in ignored `df-md-plan-v2/` and
`df-md-plan-sanitizers-v1/` locally and under the established n1 evidence root.

The retained v1 failure occurred before candidate execution: the fixture used
full AO/Q extents, which correctly produced a resident value plan and refused
streamed-owner admission in 14 cases. The two truncated fallback populations
passed. v2 uses a partial AO row extent and includes the pre-allocation native
reservation. The failed receipts and frozen v1 source/binary remain available;
no production admission condition was loosened to accommodate the fixture.
Complete endpoint qualification and default promotion remain pending.


The v2 96-atom r2SCAN-3c DF candidate has now completed Standard, native Fast,
and a separate traced Standard population (12 complete E/F calls). All pass
the existing independently stable dense-grid/CPU-spot common gates through
`readme_defaults_qualification.qualify` and the audited reference-reuse manifest.
The maximum errors are about 2.041e-6 Eh/atom, 5.626e-5 Eh/Bohr maximum force,
and 2.477e-5 Eh/Bohr RMS force. Clean Standard measures 180.499 s cold and
107.532 s median warm, with 16 / 1 iterations. The trace has 19 completed
Hermite J operations at about 2.49 s each. Source/metric/J admission is real,
not inferred from the environment selector. The same-binary rollback is still
running; these measurements do not establish a speedup over that rollback or
over stock GPU4PySCF.

Force response still costs about 94.6 s: its raw charge/potential pass alone
costs about 61.4 s and produces 64,547,192,832 raw bytes per call. Its remaining
derivative contractions are a separate bottleneck. The Fast receipt explicitly
reports `strict_standard_fallback` with `fp32_gradient_not_qualified` and
`fast_route_unavailable_using_standard`, explaining its lack of improvement in
this domain. Preserve that executed policy; do not relabel it as an accelerated
Fast solve. The artifacts include `candidate-qualification.json` and
`trace-work-summary.json`. No README redraw or default promotion follows from
this single-method qualification.
