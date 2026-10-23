# Bounded raw Coulomb endpoint qualification

Restricted pure-J resident single-packed owners can contract raw three-center
values through the existing spectral metric solve, avoiding tensor whitening.
The force bridge borrows the same authenticated raw values. Rank truncation,
other layouts, exchange and capacity refusal retain the existing paths. Explicit
`GENERATIVEQC_DF_RAW_COULOMB=0` restores whitening; the final source tests the
genuinely unset default. See the [decision and resource boundaries](../../../.agents/notes/implemented/performance/2026-10-23-df-raw-coulomb-headroom.md).

## Integrated default

Measured source: `e8af079a7f51f97e45a035e3c85c3b96af05a219`.
Core SHA-256: `0b285c7493c1fd8402dab1543dd4c58f8e69106d6a69aba5490b0aa7d9c2b6d8`.
The 5,229 source inputs were verified before compilation; all five standalone
stationary AOT libraries and their manifests were also rebuilt. Release sm120,
CUDA 12.9.1, verified sccache 0.16.0, RTX 5090 on n1, finite Slurm allocations
and one numerical host thread are retained in the receipts and provenance.

Five alternating fresh-process off/unset pairs per method execute a complete
cold energy/analytic-force endpoint, then three frozen warm calls. Each process
contributes its median warm time. No grid, basis, screening, precision or
convergence gate changes, and no oracle density enters a production endpoint.

| DF96 method | Phase | Whitening (s) | Raw default (s) | Change |
| --- | --- | ---: | ---: | ---: |
| PBE | cold | 25.604188 | 20.964586 | 18.120% faster |
| r2SCAN | cold | 27.199569 | 22.731225 | 16.428% faster |
| PBE | warm | 6.807359 | 6.925537 | 1.736% slower |
| r2SCAN | warm | 7.721131 | 7.746090 | 0.323% slower |

Both cold gains exceed the unchanged maximum of 2% and twice the sum of
relative median absolute deviations. Neither warm difference exceeds that
floor; this change does not establish a warm speedup. These DF cold endpoints
remain slower than GPU4PySCF.

Another five alternating pairs at each of 48/96 atoms and each method include
cold, warm, atom-one z displacement by 0.001 Bohr, and moved-warm calls. At 96
atoms, moved PBE changes 22.498221 to 17.859850 s (20.617% faster), and r2SCAN
24.304403 to 19.613841 s (19.299% faster). The 48-atom and moved-warm differences
do not establish significant gains. All observations and per-call work counts
remain in the receipts; no population is selected by latency.

`default-matrix.json.xz` adds Standard/Fast × PBE, PBE0, B3LYP, r2SCAN,
r2SCAN-3c and M06-2X at 96 atoms. It is compatibility evidence, not repeated
per-functional speedup evidence. Raw storage is ineligible for hybrid exchange,
and the larger r2SCAN-3c tensor retains its streamed path. WB97M-V response
headroom has separate diagnostic evidence in the decision note; it is outside
raw-J admission.

All 264 native default-source calls pass independent checks. Static samples
reuse the exact audited dense-grid and CPU/GPU reference receipts in
`../readme-final-20261022/`. Moved samples use fresh same-geometry GPU4 DF
references: they establish sampled schedule equivalence, not a moved dense-grid
convergence ranking. Energy, complete analytic force arrays, grids, executed
precision, iterations, Fock counts, source/binary identity and failures are
retained losslessly.

## Earlier opt-in source and failure retention

The separate `stage2-*` cohorts use
`8b68f626c82e7bf0f33bd6a993f91ec64b5c790b`, core
`6339a04c0f2aa6f6ba50de726ac3b35bec7e43ab6182fe2539beb2421d43a936`.
Their five-pair PBE/r2SCAN cold gains are 17.619%/17.404%; warm differences
are insignificant. All 264 earlier native calls pass the same static or moved
gates. They are not pooled with or relabeled as the integrated unset default.

The first integrated assembly retained stale standalone AOT packages after the
core rebuild. Contract identity validation correctly rejected its complete
force calls. `failed-aot-assembly.json.xz` preserves those failed receipts and
`validation-logs.json.xz` preserves their diagnostics. The corrected immutable
assembly rebuilds the five manifest targets and retains its own package hashes;
it does not bypass or rewrite the contract check.

Native independent J/metric/three-center response, changed density/geometry,
rank fallback, both exchange rejection routes, actual external-memory pressure
and post-force replay pass. Compute Sanitizer raw-J memcheck reports zero
errors, full Fock composition passes, and all 55 final GPU Python cases pass.
Focused host, documentation, pre-commit, Sphinx warnings-as-errors and the
native materialization review pass; the latter remains nine unchanged sites.

## Merge-source integration bridge

`merge-bridge.json.xz` retains a separate compatibility run at
`b931f64bae5aa72a2b9c15f817715986f9e7205b`: Standard/Fast × six functionals,
each with cold and warm energy/forces, all 24 calls independently qualified.
These are integration checks, not replacements for the repeated e8af timing
population. The core hash is
`ba836a79daf1d4ac0eb6e3f9cfa4efd6688a50a4fd1ec9110ef2f7d0f26f6afe`;
all five standalone packages and manifests were rebuilt through sccache.
The bundle losslessly reconstructs the verified 5,994-input source inventory
as a delta from `stage2-default-source-inputs.json.xz` and retains package
hashes, runners and compiler-cache statistics.

Independent raw-J/response, external memory pressure/replay, full Fock
composition and Compute Sanitizer pass (zero errors). The first Python run
passed 48 cases and failed seven because the isolated checkout lacked the
retained auxiliary-basis JSON. After copying and authenticating those inputs,
all seven passed; both logs and test-input hashes remain in the bridge.
The complete post-merge integral-parameter AD file passed 12 independent-oracle
cases. CI additionally exposed outdated host test doubles and a response
probe tied to the next workspace variable's name. Their repair passes 46
focused host cases, including raw-J rejection of occupied exchange without
output writes. Production source is unchanged by these test repairs.

`final-merge-bridge.json.xz` adds another separate 24-call matrix at
`e837a0c5c03f3e6e2d340d7988780a4b2d37ac74`, after integrating upstream
range-aware and rank-deficient DF response plus byte-safe generated-source
caching. Every cold and warm call passes the same independent gates. Its core
hash is `8a09d6490ca1e9cee85c822097a5ed3872b0beb2f5172edaeb2ed16acfc0118b`.
All five stationary manifest targets were rebuilt. The authenticated inventory
contains 6,000 source inputs; the bundle retains its exact reconstruction,
runners, package hashes, cache statistics and validation logs. Native raw-J,
rank fallback, current-memory force/replay and full Fock composition pass;
Compute Sanitizer reports zero errors; all 55 GPU Python cases and 217 focused
host integration cases pass. Earlier timing populations retain their original
identities and are not pooled with either merge-source bridge.

## Offline reproduction

```bash
PYTHONPATH=python:. python benchmarks/results/df-raw-coulomb-20261023/reproduce.py \
  > .artifacts/df-raw-coulomb-reproduced.json
```

The script rechecks all 576 native calls and computes every population median
and gain floor from full-precision samples. `provenance.json` binds measured
sources, packages, source manifests, build controls and validation. Original
GPU runners and the moved wrapper are retained in `runners.json.xz`; compiler
cache statistics in `validation-logs.json.xz` belong to the shared server and
can include concurrent compilations. Larger profiler streams and discarded
pilots remain in ignored local and n1 `direct-dft96-cold-20261023` artifacts.
