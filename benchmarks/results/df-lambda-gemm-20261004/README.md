# DF Lambda matrix adjoints and bounded auxiliary batches

The measured 230-AO ethane case uses conventional FP64 RHF and correlation-only
DF-CCSD(T), aug-cc-pVTZ / aug-cc-pVTZ-RI, o=9, v=221, q=488, and a 64 GiB
complete numeric budget. Neither native endpoint imports oracle orbitals,
factors, amplitudes or response. The matrix and scalar endpoints use one frozen
library, input, Slurm allocation (n2 job 2198), and PRO 6000 GPU UUID
`GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`. CUDA is 12.9.1, driver 595.91.07.
There is one cold sample per configuration, ordered matrix, scalar, energy.

## Complete measured endpoints

| Phase (seconds) | Scalar Lambda force | Matrix Lambda force | Energy only |
| --- | ---: | ---: | ---: |
| RHF | 159.612 | 122.328 | 129.651 |
| DF source | 0.977 | 0.971 | 0.984 |
| CCSD | 171.716 | 171.601 | 172.421 |
| (T), including responses when forces requested | 110.527 | 110.515 | 4.942 |
| Corrected Lambda and parameter VJPs | 679.436 | 274.307 | absent |
| Factor/source nuclear response | 3.694 | 3.685 | absent |
| Exact orbital/nuclear response | 670.037 | 668.622 | absent |
| Complete native endpoint | 1796.022 | 1352.053 | 307.998 |

The matched Lambda speedup is **2.477x**, saving 405.130 s. The raw complete
endpoint ratio is **1.328x**, but RHF independently differs by 37.284 s: that
part of the saved time is not attributed to Lambda. Both use 21 iterations and
42 operator actions. The scalar control includes this change's fused scalar
accumulation. These are not speedups against the historical 1750.506 s endpoint,
which ran on a different GPU UUID.

For the matrix force call, RHF/source/CCSD take 294.900 s; the (T)/Lambda/source/
orbital phases take 1057.129 s. The latter includes the (T) energy as well as its
response. Subtracting independently cold energy/force totals would not measure
an exact incremental force cost. Orbital response remains the largest phase.

## Scientific validation and its boundaries

Both cold force endpoints and the latest-base endpoint return all 24 force
components and pass the independent directional energy-FD gate of 3e-7 Eh/bohr
at h=1e-4 and 3e-5 bohr. Matrix errors are 2.60e-9 and 7.92e-9; latest-base
errors are 4.24e-9 and 6.27e-9. Independent expanded Lambda residuals are about
6.12e-13. `oracle-energy-fd.json` preserves the existing PySCF 2.14 reference
energies and conventions: exact RHF, correlation-only DF, symmetric inverse
metric root with relative cutoff 1e-10. No new oracle calculation is implied.

The extra cold-pair force comparison **failed** its 3e-9 sanity threshold:
maximum component difference is 5.7838889233607915e-9 Eh/bohr. Job 2198 completed
all three endpoints, then exited at that assertion. The failed gate is retained
in `evidence.json`; it was not loosened or relabeled as passed. The scalar/matrix
total-energy difference is 2.84e-14 Eh, which does not replace force validation.

`benchmarks/df_lambda_shared_state.cpp` isolates the changed Lambda arithmetic:
one native RHF/source/CCSD solve and one triples pullback supply *identical*
Problem, amplitudes and energy seeds to both complete Lambda/parameter solves.
It compares every Lambda vector and retained parameter/factor cotangent at
elementwise `abs(matrix-scalar) <= 3e-10 + 3e-10*abs(scalar)`, with independent
residual audits. The first detached publication is reserved in the second
solve's memory budget. Its result is recorded separately from cold endpoints;
it does not replace a full-force endpoint or independently qualify source factors.

Shared-state n2 job 2207 **passes** all 12 arrays (40850094 elements): largest
absolute difference 1.3877787807814457e-17, with independent residuals
6.11479e-13 and 6.11492e-13. Matrix/scalar solves take 274.363 / 679.254 s; both
use 21 iterations and 42 actions. This excludes a significant schedule
difference at the shared physical state; it does not localize the exact cause
of the cold-force discrepancy outside that comparison.

Other completed checks: 9 compiler/reduction tests, 26 prior primal compiler
regressions, 22 initial native Lambda/complete-force tests, and 6 targeted
tail/admission/nonfinite tests under memcheck with zero errors. Rebased n2 job
2201 passes 24 native/complete-force tests. Two stale upstream host test adapters
were updated for current response selectors and cuBLAS ownership; both pass.

The large per-source-factor `atol=rtol=3e-10` qualification remains outstanding.
Global RHF stability remains uncertified. Agreement in Lambda schedules or total
energy/force does not certify either. Orbital J/K screening remains zero.

## Actual work and capacity

| Complete Lambda diagnostic | Scalar | Matrix, Q batch 8 |
| --- | ---: | ---: |
| Auxiliary slices visited | 22448 | 22448 |
| Batch/program visits, including unchanged audits | 22448 | 3660 |
| Semantic contraction summands | 65352182732706 | 64006668617241 |
| Generated/packing/accumulation/audit kernel launches | 1454004 | 372357 |
| GEMM API calls | 0 | 78066 |
| Complete numeric capacity bound, bytes | 5478438027 | 6907244179 |

The kernel count excludes unknown cuBLAS internal kernels. GEMM callbacks
account for 52448827747914 summands. These are not hardware FLOPs. Matrix packing
writes 2942221376440 bytes according to IR output sizes; this is not measured
memory traffic. Matrix H2D/D2H remain about 1.75 GB each because GMRES is still
host-owned. The 6.433 GiB capacity is a combined numeric bound, not observed peak
GPU memory; matrix device-owned capacity is 3032413488 bytes. A charged 96 MiB
provider allowance and zero optional cuBLAS workspace are included.

`work-model.json` contains the exact compiler-derived summand polynomials.
For one transpose, scalar work is `C(o,v) + q*A(o,v) + P(o,v)`; the lifted
program uses `C_m(o,v) + sum_batches A_m(o,v,b) + P_m(o,v)`. At this case the
counts are 1210132747322 and 1178851698575, a 2.585% reduction. The asymptotic
complexity is unchanged: the large speedup comes from GEMM and launch structure.
Only factor-dependent nodes carry Q, with bounded live batch intermediates;
there is no all-Q or dense ovvv/vvvv graph. Batch size shrinks under the complete
budget, then falls back to scalar staged/expanded actions. Arithmetic/driver
errors propagate. Tail, finite-audit and budget gates remain in force.

## Source and reproduction

`library-identities.json` records binary hashes and embedded source identities.
The measured snapshot was reconstructed as commit
`bff1fa4c1` over `6330ba7d8b2ddada2ddfb0f666d33e92462dc8f4`; the exact source diff
is `measured-source.patch.gz`. This preserves the measured inputs after the
working branch was rebased. The latest-base production commit is
`3f9e81da10f1b0952f5d5c13936dc3dfcb8a06ad`, based on
`7fb64d4c466f8685179f2b52e987c60a1f553a51`.
Its complete force qualification is n2 job 2204: 1359.881 s, Lambda 272.876 s.
That run uses a different GPU and is excluded from the matched speedup ratios.

Build with the repository's CUDA configuration and explicit CXX/CUDA ccache
launchers, setting `CCACHE_BASEDIR` to the checkout root. Link
`benchmarks/df_ccsdt_force_endpoint.cpp` and
`benchmarks/df_lambda_shared_state.cpp` against that library with C++20,
`GENERATIVEQC_HAS_CUDA=1`, repository `include`/`src`, and CUDA include paths.
Copy the binaries/library and retained molecular inputs to n2; compile/link host
probes with the system compiler, not the Conda compiler sysroot.

Run on n2 via finite Slurm allocations (`--partition=main --gres=gpu:pro6000:1
--nodes=1 --ntasks=1 --time=01:10:00` for the full pair). Preserve assigned
`CUDA_VISIBLE_DEVICES`. In one allocation and with one immutable library:

```bash
./endpoint ethane230.input matrix-force.json 1 1 1 1 8
./endpoint ethane230.input scalar-force.json 1 1 1 0 8
./endpoint ethane230.input energy.json 1 1 0 1 8
```

The shared-state comparison uses a separate 35-minute allocation and
`./shared-state ethane230.input shared-state.json`; nonzero exit reports a
failed gate. No job should be submitted to node3 for this campaign.
