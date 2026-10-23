# CUDA hybrid KS density factors

Exact-factor warm replay is enabled by default for eligible restricted,
strict-FP64 DF warm seeds. An unset `GENERATIVEQC_CUDA_KS_WARM_FACTOR` or an
explicit value of `1` permits this path; `0` disables it for rollback. Other
values, including an empty string, fail closed. Enabling the policy never bypasses
the ownership, precision, metric-rank or projection-capacity checks below. It
changes neither the DF Hamiltonian nor convergence, physical-residual, or
final-state export gates. First-K matrix auditing remains off by default.

## Native cold density factors

Eligible strict-FP64 restricted DF owners recover an algebraic factor for the
native cold density before the first RI-K call. Automatic admission requires at
least 384 AOs, an occupied rank no greater than half the AO count, a compatible
prepared occupied provider and an available matrix-library binding. The density
and initial-guess algorithm remain unchanged. Imported and transported densities
retain dense first K; they cannot inherit native-cold provenance.

`GENERATIVEQC_CUDA_KS_COLD_FACTOR=0` restores dense first K. Explicit `1` permits
small-system qualification while retaining the rank, capacity and numerical
guards; other values are errors. The owner reuses existing device iteration
scratch and retains at most one `nao`-element host spectrum, included in its host
capacity report. Failure to reserve that optional spectrum leaves dense K active.

The proof uses the same FP64 limits as RHF density-seed exchange: a finite,
ascending spectrum with no eigenvalue below `-1e-13`, discarded Frobenius norm at
most `1e-12`, and reconstruction of every original density entry with maximum
error at most `1e-12` and RMS error at most `1e-13`. The recovered rank must equal
the method's restricted occupation. Both triangles are checked, so a symmetric
eigensolver cannot authorize a factor for a mismatched upper triangle. Numerical
refusal retains dense K; runtime failures drain borrowed buffers and propagate.

The prepared restricted facade applies occupation two. The seed coefficients
therefore reconstruct `D = 2 C C^T`; they carry no canonical generation, warm
orbital frame or stationary force-projection lease. A subsequent canonical SCF
update establishes those identities independently. `ks_cold_factor` traces the
proof and refusal, and `ks_iteration.algebraic_cold_factor` distinguishes its
first K from canonical/warm factors. The existing explicit
`GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT=1` diagnostic also checks this first
occupied K against dense K at the identical D; its historical `warm_seed_audit_*`
counter names remain unchanged. Audit timings are not clean measurements.

Run `generativeqc_ks_cuda_tests --df-cold-factor` through a finite `srun` allocation
for independent CPU endpoint checks, same-density GPU K auditing, ensemble and
external-seed fallback, changed geometry, failed begin and fresh retry.

## Warm representation and ownership

The ordinary last-good warm slot stores a density. Eligible DF solves instead
store the exact AO coefficients that generated that density with the existing
restricted occupied-density kernel. Its storage tag and owner, source binding,
solve epoch, density/factor generation, and occupation metadata remain separate
from transient iteration-frame and provider projection leases.

Publication requires a successful physical SCF endpoint and the existing complete
restricted final fitted-projection lease. That lease requires the full auxiliary
metric rank and sufficient provider projection capacity. No new numeric allocation
is introduced: neither another AO matrix nor another `naux * nao * nocc` buffer.
The warm cache retains the original AO factor directly; a new multiplication by the
overlap orthogonalizer or final-Fock canonicalization cannot substitute a nearby
density. A frame storage tag preserves the incumbent orthonormal-frame/force
handoff unchanged when factor replay is disabled, including across switch changes.

When physical energy confirmation deliberately holds the density unchanged, the
latest eigensolver proposal is not its factor. If the preceding restricted K
consumed a valid occupied AO frame, the owner restores that retained frame before
the next K. Dense K, missing frames, UKS and orthonormal-only frames retain their
ordinary invalidation path. Restoration clears readiness before the asynchronous
copy, so a failed copy cannot publish a factor. No convergence gate changes.

An explicit `GENERATIVEQC_KS_RANGE_RESIDENT_VALUES=1` may separately request an
optional strict range-integral cache for a fitted full-range primary. Automatic
DF range caching remains disabled. The cache has its own exact Direct provider,
complete resource accounting and exact fallback. Warm-energy rebasing preserves
the unchanged fitted primary when its final factor/projection proof is live;
only the independent range contribution needs rebasing. Optional range values
may be retired before a memory-constrained force handoff without retiring the
authenticated fitted projection.

A same-owner replay reconstructs D with the same kernel and occupations, then
borrows C for its first occupied RI-K call through the shared prepared provider.
It does **not** reuse the old mutable fitted projection U. A provider scratch write
can invalidate a final force-projection lease without invalidating the independent
last-good factor. Each new RI-K build establishes a new scratch generation.

Explicit densities, changed geometry/provider, UKS, mixed precision, missing
complete projection capacity and truncated auxiliary metrics do not acquire this
cache. Mismatched replay metadata uses dense K. Disabling the switch after a
factor has been cached still reconstructs D, but selects dense first K. Imported
or mixed densities are never declared canonical by proximity to the cached frame.

Frozen warm updates preserve both representation and generation across subsequent
solves, even if those solves change the current frame or require several Fock
builds. `warm_density()` continues returning D. Factor-backed exports materialize
D in the dead `effective` scratch slot, not final coefficients, stationary weights,
the current density, or force/projection storage. Pending-iteration exports remain
invalid. Clearing the warm state clears both its storage tag and identity.

## Diagnostic ledger

`GENERATIVEQC_DF_TRACE` retains the existing validated component-trace format and
adds these operation roots:

- `ks_begin`: seed origin, warm density/factor generations, restoration decision,
  owner/epoch, retained state/XC/provider bytes, and explicit synchronization count;
- `ks_iteration`: Fock index, consumed density generation, factor generation/epoch,
  dense versus occupied K, and additive `ks_jk`, `ks_xc`, and remaining KS work;
- `ks_finish`: completed generation, Fock count, convergence/full-density audit,
  and the ordinary diagnostic synchronization.

Existing `ri_k`, `ri_k_occupied` and provider projection regions supply the K-side
detail. Their roots overlap `ks_jk`: **do not add them to the enclosing KS totals**.
Likewise, host and GPU intervals are alternative views, not additive costs.
Profiler fences are recorded separately from ordinary synchronizations. Tracing
adds diagnostic synchronization and must be disabled for clean endpoint timing.

For a separate matrix-validation run, add `--audit-first-k` to a traced native
candidate invocation. It compares the actual first occupied J/K with the incumbent
dense route at exactly the same resident D, requiring finite K and full-Fock
differences below `2e-11`. Hcore, XC and any range correction are unchanged, so the
full-Fock difference is `delta_J + exchange_coefficient * delta_K`. Errors are
retained as conservative integer ceilings in units of `1e-16` in the
`warm_seed_audit_*_max_abs_e16_ceil` counters. The diagnostic uses dead iteration
scratch plus temporary host matrices, explicitly counts its transfer/synchronization
and extra K work, and restores a fresh occupied projection before force handoff.
The enclosing `ks_jk` includes its `warm_seed_matrix_audit` child; audit runs must
not be used for either clean timings or the ordinary first-build work ledger.

## Qualification protocol

`benchmarks.ks_df_warm_factor` runs independent DF PySCF/GPU4PySCF references and
native baseline/candidate processes with explicit current-schema spherical orbital
and auxiliary snapshots. It retains cold, five frozen-density warm, moved geometry,
and five moved-warm E+force endpoints, physical Fock histories, binary/source
provenance, and failures. The reference full-density potential wrapper is installed
**after** conversion to the DF owner, so it cannot retain a bound Direct method.
Cold timing includes calculator construction and preparation as well as the full
returned E/force call; source and library hashing remain outside the timer.

For example, connect to an allowed scheduler host (`n1`, `n2`, `n4`, or `n5`),
inspect its `sinfo` and verify the compiler-cache launcher, then run each engine
separately. These machines have separate Slurm controllers: a drained local
`node3` does not imply that the other machines are unavailable. Select compatible
partition/GRES settings on the chosen host; do not schedule on maintenance `n3`.
The following example runs on an RTX 5090 scheduler host such as `n1`:

```bash
sccache --version
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:00:00 \
  env PYTHONPATH=python:. python -m benchmarks.ks_df_warm_factor reference \
  --atoms 96 --functional m06-2x --basis-file "$ORBITAL_BASIS" \
  --auxiliary-file "$AUXILIARY_BASIS" --output .artifacts/2474/reference.json
```

Use `native --variant baseline` and `native --variant candidate` with the same
arguments plus `--reference .artifacts/2474/reference.json`, writing separate output
files. Keep both native variants on the same binary/source and scientific settings.
Repeat independently at 48 atoms and with PBE0/B3LYP. Run separate diagnostic
invocations with fresh `--trace-directory` destinations; do not use their wall times
as performance evidence. Match the original campaign's explicit basis and grid
before attributing its 96-atom branch.

Every native endpoint must pass reference energy `1e-8 Eh` and maximum force
`1e-7 Eh/Bohr` gates. These endpoint gates do not replace a strict first-build
K/Fock comparison. Compare total E+force walls and actual Fock histories; never
normalize by iteration count or subtract a Direct endpoint from a DF endpoint.
Record a negative performance result if measured; missing GPU evidence is not
a measured negative result and does not qualify default enablement.

The retained [source-matched qualification](../../benchmarks/results/ks-df-warm-factor-20261015/README.md)
contains complete cold/frozen-warm/moved populations, a separate first-K matrix
audit, additive diagnostic ledgers and additional same-owner ABBA measurements.
It distinguishes a removed dense K contraction from changed physical convergence
histories. A provider without a complete final occupied-projection lease still
falls back, even if another measured provider layout qualifies. Do not reserve a
new full projection or infer a molecule-size whitelist from that evidence.

The focused native regression entry point is
`generativeqc_ks_cuda_tests --df-warm-factor`; execute it only through a finite
`srun` allocation on an allowed node. It covers explicit `0`, explicit `1` and
the genuinely unset default, including restricted replay and UKS fallback.
Host-only cache and benchmark-control tests
are `tests/python/test_ks_warm_factor.py` and
`tests/python/test_ks_df_warm_factor_benchmark.py`.

See the [default-policy decision](../../.agents/notes/implemented/performance/2026-10-15-ks-df-warm-factor-default.md)
and the [original qualification decision](../../.agents/notes/implemented/performance/2026-10-15-ks-df-warm-factor-candidate.md)
for the implementation boundary, qualification scope and retained fallback policy.
