# Exact orbital-response accelerator qualification

Frozen production source `95c71862f`, one binary, Slurm2321 on n2 GPU
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a` (RTX PRO 6000, driver595.91.07).
Inputs are native molecular water7 (o=5,v=2,q=7) and ethane230 (o=9,v=221,q=488),
DIIS8, Q tile8, derived canonical denominators and a64 GiB budget. Every row
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
the frozen timing comparison above until its retained report is complete.

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
