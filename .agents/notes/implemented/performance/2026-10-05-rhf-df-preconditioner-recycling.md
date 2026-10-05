# Decision: exact orbital response with bounded numerical accelerators

Status: implemented; frozen complete endpoint qualified, optional selection retained
Date: 2026-10-05

## Problem and decision

Following [checkpoint amortization](2026-10-05-rhf-exact-response-checkpoints.md),
#1901 also requires a stronger conventional inverse and strict-identity
subspace reuse. Neither may change the physical unscreened exact J/K action.

The independent canonical action is `gap*x + 4(ia|jb)x - (ij|ab)x -(ib|ja)x`.
Compiler TensorIR constructs `D=gap-(ii|aa)-(ia|ia)` and `U_Qia=2 B_Qia` from
the already qualified correlation factors in the same physical RHF frame.
No factor/source is rebuilt. The low-rank inverse retains full Coulomb and
diagonal exchange: with `W=U D^-1/2`, invert `I+WW^T` by Cholesky and use
Woodbury. The optional preconditioner has no scientific authority; every
accepted result still passes the exact physical and independent scalar audits.

Positive D and finite successful Cholesky are admission conditions, without
clipping. Unsafe numerics/capacity refuse the option. A failed accelerated
solve retries diagonal from zero; attempted actions remain in diagnostics.
Only numerical errors inside the optional callback are converted to refusal;
CUDA/exact-operator failures propagate. Setup uses the shared scalar host
linalg provider to keep provider scratch explicit, not a CPU physical oracle.

For n=ov and rank q, inverse retained numeric payload is
`8*(qn+q²+2n+q)`, including conversion and solve scratch. Setup has O(q²n+q³)
numerical work; each application O(qn+q²). The generated factor-map summands
are `2qn`, but that count excludes all inverse setup/application arithmetic.
No counter calls `2qn` the complete preconditioner work. Preparation peak also
charges generated arena, retained D/U, exact identity and still-live CC data.
The inverse bound conservatively includes raw D/U even though they are freed
before the exact owner allocates. All overhead is included in orbital timing.

## Recycling boundary

Retain one solved Krylov direction and its independently computed exact scalar
image, normalized by the image norm. Project a future RHS onto that image and
let ordinary GMRES verify a fresh true residual of the proposed initial guess.
This first slice is rank-one solution-subspace reuse, not retained full Arnoldi
bases or a geometry extrapolator. It needs no extra physical setup action.

The caller owns the non-reentrant cache. Identity includes the full normalized
basis/geometry and bitwise coefficients, energies, density, Fock, overlap,
hcore, weighted density, occupation, energy, device and operator/provider policy.
There is no approximate comparison or pointer-only/hash-only reference match.
One ULP can reject reuse. This deliberately limits usefulness across cold RHF
endpoints whose reductions produce slightly different canonical frames.

The cache charges identity plus three n-vectors and the response retains one
additional n-vector for the audited image until derivative gates pass. It is
also reserved while earlier endpoint phases run. On a resource refusal with a
retained cache, release it and retry once; total time includes that attempt,
and a diagnostic marks its unavailable work so successful-primal counters
cannot be mistaken for total attempted work. No failed response publishes a
new subspace. A stale or short-budget subspace is actually freed.

## Evidence and remaining gates

Job2305 passes 12 independent long-double dense inverse cases, aliasing,
threshold/overflow and exact/short-budget tests. Independent Python IR versus
dense exact-AO physical-action comparisons pass for three shapes. CUDA
build2307 and job2309 pass generated runtime-shape, identity/budget, inverse,
GMRES and compiler/ownership gates. Water complete force with deferred checks
uses 13 exact J/K calls with diagonal and 11 with the DF inverse; this small
result does not establish a representative large endpoint benefit.

Job2310 caught nonfinite image rejection calling the strict norm before its
finite check. Job2311 passes the repaired cache's projection, independent
residual, stale device/hash/frame, and budget tests. The next full build and
real-device same-operator lifecycle/independent FD/large endpoint gates remain
pending. Stronger preconditioning and recycling are opt-in until those results
justify a selection policy. No cross-GPU speedup or all-coordinate large-force
qualification is inferred from small cases.

CUDA build2314 and real-device2315 passed all numerical gates, but the shared
architecture check caught a response-to-posthf capacity dependency. The helper
now owns method-neutral pointer-extent guards; no architecture exemption was
added. Build2318 and full qualification2319 pass: four native suites,15 real
response/IR tests including same-operator recycling, and the independent
all-coordinate small-water FD/publication suite under both diagonal and DF
preconditioning. Compiler, shared-SCF, native dependency, promotion, metadata
and CUDA ownership checks pass. The ledger has no added scientific CUDA lines;
runtime CUDA grows172 lines net relative to #1904. The failed2316 dependency
was cancelled; same-GPU complete comparisons run in2321. Large results remain
pending and no large action-count or speedup claim is made yet.

### Completed frozen qualification and selection

Slurm 2321 and report 2330 qualify frozen source `95c71862f` on one n2 GPU.
Ethane230 checkpoint-diagonal and stronger inverse both use 17 exact J/K calls
and 12 Arnoldi iterations. Orbital time is 472.719821 versus 468.831745 s;
complete time is 1162.576380 versus 1188.277176 s, with separately recorded RHF
variation. Inverse setup takes 0.213 s and its response allowance is 19,609,168
bytes. There is no demonstrated representative large convergence benefit.

The large cold/warm pair also uses 17 calls in both endpoints. The fresh native
RHF frame fails exact identity matching, so `z_recycled_guess=0` on the warm
call. Do not relax identity to obtain a hit. Small water's identical operator
does hit and uses five total exact J/K calls (one true Z action), but its much
shorter complete warm call also contains RHF/context reuse. It does not establish
a large or changed-geometry recycling win.

All five large variants pass the original energy, force, translation, Lambda,
Z and stationarity gates. Maximum force spread is 4.684e-9 Eh/Bohr and energy
spread 8.527e-13 Eh. Independent small all-coordinate FD and large two-coordinate,
two-step FD re-audits pass; no all-coordinate independent large-force claim is
made. See `benchmarks/results/rhf-response-accelerators-1901/` for complete
records, including work counts, capacities and exact binary/source identities.

Keep both accelerators opt-in. Checkpoint amortization is the demonstrated
large-endpoint improvement. Revisit this inverse only with a case that reduces
exact actions enough to repay setup; revisit recycling for genuinely identical
physical operators with different right-hand sides, not approximate frame reuse.
The newer parent integrations have separate complete qualification; the frozen
timings must not be relabeled as those versions.

### Retained-RHF integration

Source `5a805b274` passes build 2349, 77 host lifetime/admission cases in 2351,
response and both independent small-force modes in 2352, shared Lambda/factor
2354, complete cold/warm forces 2355 and report 2359. All force, residual and
limited independent large FD gates pass. Maximum large force difference from
the frozen reference is 3.615e-9 Eh/Bohr.

Ethane cold/warm uses 17 / 16 exact J/K calls and 12 / 11 Arnoldi iterations;
complete times are 1120.394947 / 1118.673403 s. Both have `z_recycled_guess=0`,
so the warm frame's smaller iteration count is not a reuse benefit. Without a
matched diagonal ablation of that frame it cannot establish stronger-inverse
benefit either. Small water still hits (11 → 5 calls). Keep strict identity and
both opt-in policies unchanged. Separate summaries and source manifests in the
benchmark directory preserve the original and both integrated versions.

### Post-merge default-path acceptance

After master `12d709e46`, source `dc68eb0b2` passes build 2367, 104 host cases
in 2369, response and both independent small-force modes in 2370, shared
Lambda/triples 2372, complete default energy/force 2373 and report 2377.
Three triples test-hook cases explicitly skip; the remaining 25 pass. The
large endpoint deliberately disables both optional accelerators and retains
17 exact J/K actions and 12 Z iterations. Maximum force difference is
2.954e-9 Eh/Bohr; all unchanged gates and limited independent large FD pass.

The final merge preserves the public result fields and restores the
caller-owned cache allowance in public numeric-capacity diagnostics at
publication. Optional cache capacity must remain charged even though this
default-path endpoint has no cache. This qualification does not supersede the
frozen stronger/recycling comparison or justify enabling either by default.
See `post-merge-summary.json` in the benchmark directory for the separately
identified source/GPU evidence. Public DF forces still fail closed; only the
internal complete force endpoint is exercised here.
