# Decision: align native WB97M-V defaults toward stock GPU4PySCF

Status: proposed; bounded water12 science/performance pass, water96 unqualified
Date: 2026-11-05

## Problem

Issue #2987 identified unlike stopping contracts in the retained fitted-LR
capsule. Native WB97M-V used E/density `1e-10/1e-8`, physical AO RMS and maximum
`1e-9`, whereas GPU4PySCF 1.9 used E `1e-9` and MO-gradient norm `sqrt(1e-9)`.
Its density change was not a stopping gate. The source-bound offline replay
still passes all 96 sampled E/F comparisons; this does not make the stopping
contracts equivalent or explain all 28 versus 10 cycles (23 native cycles use
provisional semilocal/frozen-VV10 operators).

The user's clarification explicitly chooses **native defaults toward the
reference**, not a stricter reference. The earlier exploratory reference-audit
patch was discarded before submission. No reference defaults are changed.

## Decision

Extend the existing native default-controls query to canonical default-grid
restricted WB97M-V. The CUDA owner independently authenticates its actual
semilocal, full/LR exchange, VV10 and grid composition against the native catalog.
Its Standard SCF candidate requires E change below `1e-9` and physical orbital
gradient below `sqrt(1e-9)`. It does not add a density-change candidate gate.

For D=2 C_occ C_occ^T and X^T S X=I, the Frobenius norm of X^T(FDS-SDF)X is
sqrt(2) times ||2 C_vir^T F C_occ||. Transform the *current physical* residual in
existing scratch and reduce it on device; no CPU/reference work, extra resident
matrix, host synchronization or GPU-name dispatch is introduced. A separate
orthonormal MO evaluation tests this relation in a nonorthogonal AO basis.

Matching candidate thresholds does not replace native publication safeguards.
Fresh full-target energy history, complete canonical proof, electron trace,
idempotency and force-token validation remain required. The final Standard
density/stationarity ceiling remains `1e-5`. Rejected proofs continue SCF.
The nested VV10 confirmation/inner thresholds must use that same Standard
ceiling instead of silently retaining the old `1e-9` oversolve. Explicit tighter
energy or density controls restore the old strict policy, including VV10 inner
thresholds. Fast, CPU, UKS, custom model/grid, chunks and missing resource cases
retain their own bounded policies.

## Rejected alternatives

- Tightening GPU4PySCF: opposite to the user's requested default direction.
- Copying only E/density numbers: WB97M-V was excluded from Standard admission,
  and its physical gates and frozen-inner tolerances would remain strict.
- Equating AO RMS/maximum with MO gradient: wrong representation/normalization.
- Removing final canonical/force proofs or accepting provisional VV10: unsafe.

## Evidence and qualification boundary

- Audit native base: `0a214362fa9d762cf0a3a6be821d7cb5d185f257`.
- Inspected reference: GPU4PySCF 1.9 `2fdc1951670a8e173dce063e697727bbc9f97616`,
  `gpu4pyscf/scf/hf.py`, physical gradient and `_kernel` stopping predicates.
- Frozen replay: six pairs, all 96 independent cross-engine E/F comparisons pass.
- Slurm 15916, physical RTX 5090: the exact changed CUDA diagnostic unit passes
  an independent host-sum versus device MO-gradient reduction, preservation by
  physical diagnostics, and isolation from strict/unsupported diagnostics.
  This standalone unit is not a full production image or an endpoint timing.
- Exact-source CUDA build and targeted native default/explicit-strict E+F
  comparisons remain pending; no new speedup or default qualification claimed.
- An initial host test attempt used an older runtime missing
  `generativeqc_precision_execution_run_v1`: 24 pass/18 fixture errors. The native
  default-query adapter and predicate probe passed; complete current-runtime
  tests must be rerun rather than treating this installation mismatch as a pass.

### Initial complete-image qualification and admission repair

The exact O3 SM80/SM120 build completed with source identity
`da3b9fe66779ceb401d68846dd63b81f64629d9336ec0e453a736a24390331e0`
and core ELF SHA256
`9b594b756ee495567aef1890bb5923a61848d92d1cfff0fc445b11a655ee053a`.
Actual changed-unit compilation was retained after fresh-mtime overlays, not
merely an identity-bearing relink. Slurm 15925 stopped at a host fixture's
missing `ccache`; selecting its supported verified `sccache` launcher in the job
environment corrected that prerequisite without changing unrelated tests.
Slurm 15926 passed all 43 current-runtime host tests and both small water E/F,
own-density warm and displaced-geometry tests. However, its separate admission
audit failed: no physical orbital-gradient checks ran. The job stopped before
the larger endpoint; these scientific passes do not qualify the changed policy.

The cause was not small-matrix admission: 24 AOs exceeds the existing 17-AO
library crossover. The catalog stores SR/LR exchange, while resident Focks use
full/LR-correction operators. Comparing their operator kinds and coefficients
directly rejected the canonical model. Reuse the production RSH lowering to
authenticate `a*SR+b*LR = a*full+(b-a)*LR`; retain all model/resource safeguards.
The real-GPU scientific tests now separately require positive default-path
admission and zero explicit-strict admission, so fallback parity cannot hide
this error again.

The incremental O3 rebuild completed with a retained actual changed-unit compile.
Slurm 15928 authenticated repaired native identity
`be94892b51bb82b61b10f16d88a8fec0bde87e3ab0fba1c2385cf20e5d7cc44d`
and core ELF SHA256
`6bfa709dbc723b520194956e3b80794add5b525c5e4f74c25193fedfcde12781`.
All 43 host tests and both real-5090 small complete E/F tests pass. Each geometry
records positive default-path MO checks and zero explicit-strict checks, with
the independent tight-reference, warm and displacement gates unchanged.
The single bounded clean water96 default/strict/stock-reference population is
in progress; no endpoint win or large-system default qualification is inferred.

CI also exposed the native Libxc catalog's checked-source snapshot changing.
Regeneration with the existing checker changes only current input identities:
the historical measurement remains stale, every capability summary is unchanged,
and no numerical qualification or capability was invented to clear CI.

### Final bounded complete comparison

The water96 Direct default cold endpoint in Slurm 15928 reached its declared
540-second wall cap without publishing any complete E/F sample. The driver
returned 124 and stopped before strict/reference arms. Its raw JSON remains
`running` because termination interrupted the process; the scheduler/driver
receipt establishes that it is incomplete, not a passing measurement. This
does not identify the slow stage or establish a convergence regression. Do not
infer a large-system timing, accuracy result, or zero work from missing samples.
The large case was not repeated.

One final reduced water12 Direct population, Slurm 15930, completed with exit 0
on the same physical 5090 and authenticated source/core. Each arm measures one
complete cold constructor/preparation/SCF/force/export call plus one own-density
frozen warm call, without diagnostic tracing or production oracle imports.

| Arm | Cold seconds | Warm seconds | Cold iterations |
| --- | ---: | ---: | ---: |
| Native default | 6.516086 | 0.845254 | 17 |
| Native explicit strict | 9.139645 | 0.843362 | 30 |
| Stock GPU4PySCF | 15.256367 | 3.969813 | 10 |

Native default cold is 28.705% shorter than explicit strict; warm is +0.224%,
inside the predeclared 1.05 paired guard, not proof of exactly zero regression.
This is one bounded observation, not a statistical/general default certificate.
Cold semantic work changes from 31 J/62 K to 18 J/36 K applications, 25 to 14
provisional Focks, 5 to 3 target Focks, and 4 to 2 VV10 pair sweeps. All native
warm calls use 1 J/2 K, 1 target Fock and zero VV10 sweeps. Comparing 17 native
iterations directly with 10 reference iterations would again conflate operators.

All 12 cold/warm cross-arm E/F comparisons pass the declared common gates.
Default-versus-strict maxima are 9.522e-13 Eh/atom and 1.082e-6 Eh/Bohr force;
default-versus-stock-reference maxima are 3.080e-13 Eh/atom, 1.506e-6 maximum
force and 5.366e-7 force RMS. The independent tighter-reference small/moved
tests remain the separate oracle qualification; stock defaults were not tightened.
The existing Libxc catalog regression additionally passes all 63 host tests.

Receipts: `/data/jzzeng/issue2987-defaults-20261105/endpoint-small-15930/` and
`qualification-15928/` on n1; local authenticated copies and the 12-comparison
audit are under `.artifacts/issue2987/receipts-repair/` and
`.artifacts/issue2987/water12-complete-audit.json`. The audit SHA256 is
`3dfeb3466cc9740902ca77abafcf98800404b4416d914863b2a529f6546e41af`.

The pre-commit bot subsequently added only line wrapping in the RSH lowering
call (commit `9e31525bc381353fae58ef0e2e7a45f8965f8830`, parent `14e7245f5`).
The executed source/core identities above describe the exact pre-format image,
not a fresh build of the differently hashed formatted source. The bot change
has no token/behavior change; its exact existing commit was imported without
rewriting attribution/history, and the checked-source catalog was regenerated
again without promoting historical science. No GPU benchmark was repeated for
formatting. Water96, A100, and the independent 1.576-second bottleneck remain
unqualified/unresolved by this default-alignment fix.

## References

Issues #2976 and #2987. Separate strict fixed-anchor draft #2986 is unchanged.
