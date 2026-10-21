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
