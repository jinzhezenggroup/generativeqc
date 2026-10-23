# Decision: prove native cold DF factors and reuse prepared device validation

Status: implemented
Date: 2026-10-23

## Problem

A native restricted KS cold guess is already a low-rank density, but the first
fitted exchange application used the dense-density consumer. The occupied
consumer was reserved for subsequent canonical densities and authenticated
same-owner warm starts. At 96 water atoms the first dense K and host final-state
matrix validation remain material costs in the complete energy/force endpoint.

## Decision

For strict-FP64 native cold fitted RKS, attempt an algebraic factor only when
the occupied provider binding matches the AO count, device and stream, library
matrix products are available, and the requested rank is at most half the AO
count. The default lower bound is 384 AOs. Explicit
`GENERATIVEQC_CUDA_KS_COLD_FACTOR=0` retains dense K; `1` permits smaller
experiments without bypassing any numerical or resource guard.

Diagonalize the unchanged AO density in existing scratch. Share the RHF
spectral and reconstruction proof: finite ascending eigenvalues, negative
limit -1e-13, rank threshold 1e-13, discarded Frobenius norm at most 1e-12,
and complete two-triangle reconstruction maximum/RMS errors at most
1e-12/1e-13. The retained rank must exactly equal the method occupation.
Because the prepared RKS facade supplies occupation two, scale the eigenvalues
by one half before forming C. This proves D = 2 C C^T without changing D.

Only an optional O(n) host spectrum is added and charged during preparation;
allocation failure retains dense K. No additional device storage is allocated.
Numerical refusal falls back to dense K. Runtime failures drain submitted work,
revoke factor readiness and propagate. Imported densities and full-rank
ensembles retain dense first K. The algebraic factor publishes no canonical
generation, warm frame or final stationary projection lease.

Extend the existing resident final-state validation proof to matching fitted
J/K full-range global hybrids at the existing 384-AO threshold. Preserve RKS,
strict-final arithmetic, all-electron, full-range and resource/lifetime guards;
range-separated, nonlocal and incremental paths remain outside this default.
Explicit device-validation opt-out remains available.

The prepared one-electron bridge now honors the existing cooperative default
for Hcore using implicit triangular AO addressing. It still uses component
lanes for overlap-only Pulay and explicit noncooperative controls. No new pair
list, integral equation or storage is introduced.

## Evidence and promotion boundary

The [2026-10-10 Hcore-only failure](2026-10-10-ks-implicit-cooperative-pairs.md)
remains a failed historical population. A new same-library, alternating-order
cohort on base `bbd41a42764e8d22d15c1348f70a633961d4dcf5` uses five processes per
arm and size, each with cold, frozen warm, moved and frozen moved-warm endpoints.
Atom one moves by 0.001 Bohr along z. Each arm owns its independently converged
density. All timings include energy, analytic forces and host export.

| Atoms | Phase | Component lanes (s) | Cooperative (s) | Gain |
| --- | --- | ---: | ---: | ---: |
| 48 | cold | 12.77871 | 12.70160 | 0.603% |
| 48 | warm | 2.632842 | 2.499827 | 5.052% |
| 48 | moved | 7.440834 | 7.287146 | 2.065% |
| 48 | moved-warm | 2.633485 | 2.493125 | 5.330% |
| 96 | cold | 29.60851 | 28.64035 | 3.270% |
| 96 | warm | 7.974777 | 7.004534 | 12.166% |
| 96 | moved | 18.18169 | 17.24901 | 5.130% |
| 96 | moved-warm | 7.971727 | 7.004511 | 12.133% |

The unchanged gain floor is max(2%, twice the sum of relative MADs). The
48-atom cold point does not pass and is not claimed as a speedup. Both sizes'
warm and moved-warm populations pass, as do the moved endpoints. Maximum paired
energy/force differences are 2.27e-12 Eh / 8.96e-13 Eh/Bohr. Moved GPU4PySCF
comparisons have maximum force errors 3.89e-7 (48) and 1.78e-6 (96) Eh/Bohr.
Stationary samples use the retained independent dense-grid reference and
unchanged 1e-5 Eh/atom, 1e-4 maximum-force and 3e-5 RMS-force gates.

The native cold-factor test includes an independent CPU fitted PBE0 provider,
same-density occupied/dense GPU K audit, explicit opt-out, full-rank ensemble,
external determinant, changed geometry, failed begin/retry and unchanged
device storage. Native warm-factor and composition checks also pass.
Forty GPU Python cases cover independent Libcint one-electron derivatives
through f, force finite differences, DF factors and seed admission. The
composition test's private callback type is repaired to match its existing
precision-request argument.

Retained samples and reproduction are in
[`dft-cold-stage1-20261023`](../../../../benchmarks/results/dft-cold-stage1-20261023/README.md).
Final combined-source repeats are recorded separately from mechanism evidence.
Single-sample development results and intrusive profiles do not establish a
combined-source speedup. Provider-suite default MD census failure reproduces
on the untouched base; disabling MD passes that census. An experimental
triangular-Schwarz change and an unqualified raw-J representation are excluded
from this decision and its implementation.

## Consequences and revisit conditions

Cold proof introduces an eigensolve and two small host synchronizations in
exchange-capable cases; the work gate avoids high-rank factors with little
benefit. Keep the dense fallback and test it when provider ownership changes.
Do not extend algebraic factors into stationary response without a separate
canonical provenance proof. Revisit the lower bound with complete repeated
endpoints, including moved geometries and constrained resources.

These changes are an interim improvement. They do not establish that all
96-atom Direct or DF cold endpoints have caught GPU4PySCF, and do not change
grids, auxiliary bases, SCF tolerances or scientific acceptance thresholds.
