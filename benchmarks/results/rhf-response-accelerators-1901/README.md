# Exact orbital-response accelerator qualification

Frozen production source `95c71862f`, one binary, Slurm2321 on n2 GPU
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a` (RTX PRO 6000, driver595.91.07).
Inputs are native molecular water7 (o=5,v=2,q=7) and ethane230 (o=9,v=221,q=488),
DIIS8, Q tile8, derived canonical denominators and a 64 GiB budget. Every row
is a complete DF-CCSD(T) force endpoint on an exact conventional RHF reference.
No supplied orbitals/amplitudes enter these comparisons.

Each variant has one observation per molecule. The recycled cold/warm rows are
two calls in one process; their `.wall` covers both and is not assigned to either
individual endpoint. `summary.json` retains each phase, exact action counts,
residuals, forces, capacities and binary/source/input hashes. These timings do
not qualify the subsequently integrated prepared-provider tree.

## Representative large result

| Ethane230 | Exact J/K calls | Z iterations | Orbital/nuclear s | Complete s | RHF s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every-action residual, diagonal | 28 | 12 | 669.660284 | 1333.307518 | 125.191708 |
| Checkpoints, diagonal | 17 | 12 | 472.719821 | 1162.576380 | 144.904430 |
| Checkpoints, DF inverse | 17 | 12 | 468.831745 | 1188.277176 | 177.670906 |
| DF inverse + recycling, cold | 17 | 12 | 469.259052 | 1149.146744 | 138.970030 |
| DF inverse + recycling, warm | 17 | 12 | 476.048724 | 1147.125460 | 126.379129 |

**Checkpoint amortization is the qualified large-system improvement.** The
physical operator and12 Arnoldi iterations are unchanged. Fresh candidate
residual actions fall from12 to1, reducing total exact J/K28→17. The independent
scalar final audit remains. Orbital/nuclear time falls by about197 s, while the
complete endpoint falls by about171 s; unrelated RHF and other phase variation
must not be attributed to this change. This is one matched pair, not a timing
distribution or a cross-GPU calibration.

The stronger inverse does **not** reduce large iteration/action counts. Its
measured setup is0.213 s and conservative response allowance19,609,168 bytes.
No representative large benefit justifies making it default or further tuning
this Coulomb-plus-diagonal-exchange variant on the basis of small-system wins.

Large cold recycling publishes a direction, but the next complete native RHF
frame fails strict identity matching: `z_recycled_guess=0`. The new frame's
solution is published only after its own exact gates. Same geometry or similar
energy is insufficient. The option's response/cache allowance is2,185,912 bytes;
retained cache is also charged in earlier phases. **No large complete-endpoint
recycling gain is demonstrated**, and no identity tolerance is loosened.

## Small result and interpretation

| Water7 | Exact J/K calls | Orbital/nuclear s | Complete s | RHF s | Recycle hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every-action, diagonal | 20 | 0.351033 | 1.676578 | 0.794897 | no |
| Checkpoints, diagonal | 13 | 0.302170 | 1.495946 | 0.660204 | no |
| Checkpoints, DF inverse | 11 | 0.287282 | 1.455096 | 0.637966 | no |
| DF inverse + recycling, cold | 11 | 0.288879 | 1.478287 | 0.657915 | no |
| DF inverse + recycling, warm | 5 | 0.140473 | 0.362281 | 0.019644 | yes |

The small warm solve requires one true Z action and no Arnoldi iteration, but
its complete-time decrease also contains CUDA/RHF context reuse. Only the
recorded action reduction belongs to recycling. The small result does not
establish reuse for large cold frames or changed geometries. Both optional
accelerators remain opt-in; the exact RHF checkpoint policy defaults to its
restart interval30, while general GMRES still defaults to every iteration.

## Correctness and work boundaries

Every accepted solution needs a fresh unpreconditioned physical residual;
projected convergence only requests that check. Restart, breakdown, exhaustion
and the maximum interval also retain checks. An independent scalar CUDA action
and full stationarity remain mandatory. There is no screened physical operator
or explicit orbital Hessian in any row.

All large energy differences from the every-action baseline are at most
8.527e-13 Eh. Maximum all-component force difference across all variants is
4.684e-9 Eh/Bohr, below unchanged atol=rtol=3e-7. Translation is at most
1.061e-11; Lambda, Z residual and stationarity gates pass. Independent small-water
all-coordinate central differences at1e-4 and3e-5 Bohr and failure-publication
tests pass under diagonal and stronger preconditioning. The complete records
also re-audit every large variant against existing independent C0-z/H1-x central
energies at both steps from2288. Accuracy alone is reused, never those timings.
This is not an all-coordinate independent large-force or global stability audit.

For n=ov, the shared compiler map uses the already qualified same-frame factors:
`D=gap-(ii|aa)-(ia|ia)` and `U_Qia=2*B_Qia`. The numerical right inverse is
Woodbury/Cholesky for `D+U^T U`. Its own retained payload is
`8*(qn+q²+2n+q)` bytes, including scratch; the response bound additionally charges
raw D/U and identity while setup copies coexist. Setup is O(q²n+q³), each apply
O(qn+q²). The map's2qn planned contraction summands are **not** total inverse
work. Total inverse and exact J/K FLOPs are unmeasured and remain null.

Positive finite D and Cholesky are admission conditions, not clipping rules.
Unsafe/budget-refused or unsuccessful acceleration falls back to the diagonal
physical solve, with attempted actions retained. Physical/CUDA failures propagate.
Recycling stores one solved direction, independent exact image and projection
scratch; it is not full Arnoldi recycling or geometry extrapolation. Exact
basis/reference/occupation/device/operator/provider identity and a fresh true
residual guard every reuse. A resource-refused retained cache is freed before
retry; unavailable precursor work is explicitly marked, not filled with zero.
No such resource retry occurs in these observations.

The implementations are in `src/response/{native_gmres.cpp,resident_gmres.cpp,
low_rank_preconditioner.hpp}`, `src/hf/{rhf_frame_response.cu,rhf_frame_identity.hpp,
rhf_frame_recycle.hpp}` and compiler `method/rhf_orbital_preconditioner.py`.
Production integral/CC work does not call a CPU/reference oracle. Host numerical
inverse algebra uses the explicitly selected shared scalar linalg provider.

## Validation and reproduction

Build2318 and qualification2319 pass four native suites,15 real response/IR
cases, both independent small-force modes, compiler/SCF/native architecture,
promotion, metadata and ownership checks. Tests independently cover dense
inverse agreement, exact/short budgets, unsafe numerics, stale identities,
same-operator recycling and inconsistent projected convergence. Postprocessing
2330 validates every complete2321 force record and large FD re-audit.

Current parent integration is separate: build2333 and GPU2340 pass the same
response/FD suites; common host2339 passes38 current admission/binding/lifetime
cases. Current shared Lambda/factor and denominator-preconditioner regressions
also pass in2348. Current complete cold/warm force2345 remains separate from
the frozen timing comparison above. Its completed report is retained as
`prepared-provider-summary.json` with the original source-manifest checksum.

### Prepared-provider integration

Production `2a271629d`, complete force job 2345 and report 2347 pass all gates.
This binary predates the later retained-RHF integration. Ethane cold/warm calls
take 1106.050552 / 1183.02 s (see JSON for full precision); both use 17 exact J/K
actions and reject recycled initial guesses. Maximum force differences from the
frozen reference are 2.403e-9 / 1.914e-9 Eh/Bohr. Existing independent large
two-coordinate/two-step FD re-audits pass. No timing ratio is taken against the
old binary: provider changes and RHF variation are part of these endpoints.

The subsequent integration composes parent `c7486bf2e` and retained RHF ownership.
That version has its own source manifest and qualification rather than inheriting
these results. Both branches pass the expanded 77-case admission/lifetime suite
in job 2351; its separate full-device and complete-endpoint results follow.

### Retained-RHF integration

Production `5a805b274084c22b16a1c92ecfbc516dead3b882` composes parent
`c7486bf2e`. Build 2349, host 2351, response/independent small all-coordinate FD
2352, shared Lambda/factor 2354 and complete force 2355 all pass. Report 2359
passes every accuracy and existing independent large FD re-audit. Records are
in `retained-reference-summary.json`, bound by the `v2-sources.sha256` checksum; source,
compiler and policy files were hash-checked against the tested checkout.

| Ethane230 | Exact J/K | Z iterations | Recycle hit | Orbital/nuclear s | Complete s | RHF s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Cold | 17 | 12 | no | 443.300133 | 1120.394947 | 131.935533 |
| Warm | 16 | 11 | no | 424.329886 | 1118.673403 | 151.592693 |

The warm call still rejects strict identity reuse. Its one fewer iteration and
J/K action are **not** a recycling benefit: the new physical RHF frame is
different and `z_recycled_guess=0`. No matched diagonal/stronger ablation exists
on that warm frame, so it also does not establish a stronger-inverse advantage.
Keep both optional choices opt-in. No timing ratio is taken across binaries or
GPU allocations. Small water retains its identical-operator hit (11 → 5 calls).

The maximum large force difference from the frozen reference is 3.615e-9
Eh/Bohr, energy difference 5.685e-13 Eh and translation residual 1.626e-12.
Original residual/stationarity and two-coordinate/two-step independent large FD
gates pass. This remains a limited independent large derivative audit, not an
all-coordinate one. No optional numerical or resource fallback occurs.

Use CUDA12.9.1/sm120 Release, ccache, and finite n2
`main --gres=gpu:pro6000:1` allocations, preserving Slurm device visibility:

```sh
./probe INPUT every-action.json 1 1 1 1 8 8 8 1 1  0 0
./probe INPUT checkpoint.json   1 1 1 1 8 8 8 1 30 0 0
./probe INPUT stronger.json     1 1 1 1 8 8 8 1 30 1 0
./probe INPUT recycled.json     1 1 1 1 8 8 8 1 30 1 1
```

The last command publishes `recycled.json.warm.json` for the second full call.
Raw receipts remain at `n2:/data/jzzeng/cc-1901-20261005/endpoint-2321/`.
Rationale is preserved in the checkpoint and DF-preconditioner/recycling Agent
Notes under `.agents/notes/implemented/performance/2026-10-05-rhf-*.md`.

The complete `current-sources.sha256` and `v2-sources.sha256` lists remain at
`n2:/data/jzzeng/cc-1901-20261005/` and in ignored local qualification artifacts.
Each summary retains their checksums, checked file counts, exact Git revision,
pathspecs and deterministic reconstruction recipe. They can be rebuilt from
existing Git history without retaining another copy of the repository-wide
hash list in the PR. Binary/probe/input hashes and all numerical observations
remain in the tracked summaries.

## Post-merge integration

Production `dc68eb0b2b496de18fd765bda577d836487b2883` composes parent
`33083727b` and master `12d709e46`, including public DF energy registration and
the current triples provider. This is integration acceptance, not another
five-way acceleration comparison. The public `df-rccsd(t)` method remains
energy-only and fails closed for forces; these records exercise the internal
complete force endpoint.

Build 2367 and 104 host admission/lifetime cases in 2369 pass. Response 2370
passes four native suites, 15 GPU/IR cases, both independent small-water
all-coordinate force modes, and architecture/promotion/manifest/ownership
checks. Shared 2372 passes the native Lambda denominator oracle/fallback/budget
suite, 18 Lambda/factor cases and 25 triples cases. Three generated-provider
test-hook cases are explicitly skipped because this Release library lacks test
hooks; those cases are not counted as passing.

Complete job 2373 selects the actual default diagonal/checkpoint response:
stronger preconditioning and recycling are both disabled. Its ethane energy
call takes 282.120948 s: RHF 125.599469 s, source 3.633399 s, CCSD 147.640675 s
and triples 5.247287 s. The force call and source-specific acceptance gates
are retained with this same allocation, separately from earlier timings.

The complete ethane force call takes 1221.661691 s, including RHF 234.671782 s,
CCSD 148.688783 s, triples energy/pullback/Fock response 111.936021 s, Lambda
277.252422 s and orbital/nuclear response 441.628298 s. It retains 17 exact J/K
actions and 12 Z iterations with neither optional accelerator enabled. The
energy and force calls have different RHF timings, so their difference is not
a pure force-overhead measurement. There is no matched every-action ablation
on this source; the frozen comparison remains the evidence for acceleration.

Report 2377 passes all original energy/force, replay, residual, stationarity,
translation and existing independent large two-coordinate/two-step FD gates.
The maximum large force difference from the frozen reference is 2.954e-9
Eh/Bohr, energy difference 4.690e-13 Eh and translation residual 2.077e-12.
This does not extend the independent large audit to all coordinates.
`post-merge-summary.json` preserves every observation, validation-log outcomes
and hashes, the GPU and binary identity, and the reconstructable
`v3-sources.sha256` receipt verified after the complete endpoints. Missing total
FLOPs and unrequested response work remain null. Stronger preconditioning and
recycling remain opt-in under the previously measured selection policy.
