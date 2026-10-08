# Decision: amortize optional Lambda true-residual replays, not acceptance

Status: implemented internal experiment; omitted-argument default unchanged
Date: 2026-10-09 (local measurement date, Asia/Shanghai)

## Problem

The forward-W precision experiment reduced the complete triples phase by
19.69%, but complete native CCSD(T) E+F time by only 1.08%. Corrected Lambda
still took approximately 274 seconds. Its FP64 GMRES controller used an exact
physical-operator replay after every Arnoldi iteration, yielding 42 counted
actions for 21 iterations. Changing W alone cannot accelerate this work.

## Decision

Expose the existing positive GMRES true-residual interval through the internal
native endpoint and trailing benchmark argument twenty-four. The experimental
configuration uses interval 30 rather than 1. No generic controller, generated
scientific equation, Lambda arithmetic, preconditioner or tolerance changes.

The small Hessenberg residual may request an exact check but cannot accept a
solution. Predicted convergence, restart, breakdown and iteration exhaustion
still force a true physical residual. The separately generated independent
Lambda equation remains an additional publication gate. The existing controller
already implements these rules, including the resident backend.

Both the internal endpoint and CLI reject zero before molecular work: zero is
not a supported "check only at convergence" sentinel. Omitted arguments retain
interval 1. The benchmark reports the requested interval and measured work so
this opt-in remains distinguishable from the historical schedule.

This is **FP64 algorithmic work reduction**, not FP32 Lambda or tensor-core
throughput. It can be combined with the separate FP32 forward-W experiment, but
combined endpoint gains must not be attributed entirely to mixed precision.

## Rejected alternatives

- Relaxing convergence/publication tolerances or trusting the projected
  residual would change scientific acceptance and is not acceptable.
- Skipping the independent physical-equation audit would remove an independent
  check, rather than just amortize redundant intermediate evaluations.
- Extending FP32 to the Lambda operator needs a separate residual-refinement
  contract and difficult-case independent qualification; it is not implemented
  by this change.
- Promoting interval 30 to the default based on this one large paired run would
  exceed the available restart/stagnation and difficult-molecule evidence.

## Invariants

- Exact FP64 true-residual acceptance and independent Lambda audit are mandatory.
- RHF, CCSD, corrected Lambda, reverse triples, Z and nuclear response retain
  their original arithmetic and numerical gates.
- Interval selection changes neither GMRES workspace admission nor the equations.
- Production does not consume the test oracle, retained forces or finite
  differences. Independent references are consumed only by qualification.
- The FP32-W bounded strict fallback and explicit precision provenance remain.
- Report action counts and complete process/endpoint time, not just kernel time.

## Evidence

### Build and provenance

Detached worktree based on master commit
`4385f72751b829883407c01106186917c344317b`; the original checkout is unchanged.
CUDA 12.9.1, Release, sm_120, `portable_cuda` AOT, no optional cuTENSOR and no
fast-compile configuration. Verified ccache 4.5.1 is present in generated CXX
and CUDA launcher commands; the shared cache is retained with checkout-root
`CCACHE_BASEDIR`. Final incremental rebuild cache snapshots are retained (one
hit, three misses, no additional cache errors); they are not whole-build totals.

Library SHA256:
`d0482b1b3acc106fbb89df7c006f8c0f18ac30af16ca9cd9def3dd22540c8895`.
Benchmark SHA256:
`60679707e08c17b8fd9966b873c27ae61a9abbffa619ebd6e6989ac6aafb5dc9`.

### Small-system independent gates

Slurm job 2712: 100 passed, eight optional hook/provider skips. Complete H2,
water and LiH endpoints compare strict, mixed-W and mixed-W/cadenced Lambda.
All pass independent same-Hamiltonian PySCF energy checks (1e-8 Eh) and
directional finite differences at 1e-4 and 3e-5 Bohr (3e-7 Eh/Bohr).
Lambda actions are respectively 2 -> 2, 32 -> 17 and 26 -> 14. Maximum
cadenced-vs-strict force difference is 6.153e-12 Eh/Bohr, and maximum independent
directional finite-difference error is 7.312e-9 Eh/Bohr. Complete optimized-water
compute-sanitizer memcheck reports zero errors.

Three host default-policy tests pass. The existing host/resident GMRES contract
test also passes independently, including unchanged workspace, sparse-check
action savings, predicted-convergence checks and refusal to accept an
inconsistent operator merely because its Hessenberg residual predicts zero.
Ruff, clang-format and whitespace checks pass.

### Large complete E+F pair

Slurm job 2713 on node2, `main`, `gpu:pro6000:1`, finite 30-minute allocation;
Slurm device visibility remains `1`. GPU:
`GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`, RTX PRO 6000 Blackwell, 600 W.
Each mode starts a fresh process with the same executable and GPU. Input:
spherical ethane230, aug-cc-pVTZ / aug-cc-pVTZ-RI, 230 AOs, 488 auxiliaries,
9 occupied / 221 virtual orbitals, 64 GiB complete correlation budget. Input
SHA256: `9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e`.
RHF is conventional exact native RHF; DF is used only for correlation.

| Complete phase or work | FP64 W, interval 1 | FP32 W, interval 30 |
| --- | ---: | ---: |
| Native cold E+F endpoint (s) | 668.881288 | 586.368923 |
| Complete process wall (s) | 669.21 | 586.72 |
| Triples energy + pullback + full-Fock response (s) | 39.227285 | 31.465998 |
| Corrected Lambda (s) | 273.124799 | 197.551919 |
| Lambda iterations | 21 | 21 |
| Lambda counted operator actions | 42 | 22 |
| Lambda GEMM calls | 78,066 | 42,726 |
| Lambda contraction summands | 64,006,668,617,241 | 40,429,634,645,741 |

Observed complete native time decreases by 12.34% (ratio 1.141x) and complete
process wall by 12.33%; Lambda time decreases by 27.67%. Its phase includes
preparation and independent auditing, so halving counted iterative actions
does not halve phase time. Most additional savings are removed FP64 replays,
not mixed-precision arithmetic. This is one fresh-process pair, **not** a
statistically qualified speedup or a guarantee for other molecules/GPUs.

Unchanged work: RHF 19 iterations; CCSD 20 iterations / 38 evaluations / 46,322
GEMMs; Z 12 iterations / 13 actions; JK 17 actions. Triples and Fock-response
summands remain 9,224,865,365,040 and 9,924,048,505,176. Reported complete numeric
capacity stays 7,107,919,137 bytes; it is not sampled total device VRAM.

Candidate energy error against retained independent PySCF is 2.619e-11 Eh.
All 24 forces differ from the matched strict endpoint by at most 2.478e-11
Eh/Bohr and from previously qualified native forces by 1.055e-9 Eh/Bohr.
Retained independent energy finite differences cover C0-z and H1-x at two
steps, not all coordinates; maximum error is 3.099e-8 Eh/Bohr. Translation
residual is 1.200e-12; Lambda residual 6.115e-13, Z residual 1.357e-13 and
stationarity 7.673e-12. All unchanged gates pass with margin.

Ignored local evidence: `.artifacts/lambda-cadence-qualification/`, including
scripts, summaries, raw JSON/timings/traces, accuracy records, compiler-cache
statistics, source/binary identities and sanitizer output. Remote compute root:
`n2:/data/jzzeng/qc-ccsdt-lambda-cadence-20261009-4385f727`.

## Consequences and revisit conditions

The endpoint gains a reproducible internal option without a default-policy
change. Periodic checks can affect stagnation detection timing, even when
mandatory exact acceptance remains identical. Qualify difficult/restarted
Lambda cases and repeated complete endpoints before default promotion. Broader
mixed-precision Lambda requires independent qualification and explicit strict
refinement/fallback, rather than rebranding this work-reduction option.

## References

- [Original forward-W precision decision](../numerics/2026-10-08-df-triples-forward-w-mixed-force.md).
- `src/response/native_gmres.cpp` and `src/response/resident_gmres.cpp`:
  existing mandatory true-residual controller behavior.
- `tests/native/test_native_gmres.cpp`: `amortized_true_residuals` contract.
- `docs/developer/df_ccsdt_gradient.md`: current endpoint arguments and gates.
