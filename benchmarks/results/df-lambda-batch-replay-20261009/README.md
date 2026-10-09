# Strict-FP64 Lambda batching and original-graph fresh replay

## Frozen comparison

Complete cold spherical ethane DF-CCSD(T) energy plus analytic forces:
aug-cc-pVTZ / aug-cc-pVTZ-RI, 230 AO, o=9, v=221, q=488, conventional
unscreened RHF, correlation-only DF, symmetric metric inverse square root,
relative cutoff `1e-10`, 64 GiB numeric budget. Production receives molecular
input only. Independent reference data is read after native execution.

Source: pinned master `d5a3173cc89b399ed105b0750124473eb782e2e5`, plus #2156's
dependency diff through `3f79e2a93d28c5ed1d9817871f365473e5710e17`, plus this
follow-up. `measured-source.patch.gz` reconstructs all measured source and driver
changes directly from that master; do not apply the dependency a second time.
Historical evidence, docs and notes are excluded from this source-only patch.
This is not the original #2136 qualification source or a moving latest-master
measurement. Earlier batch screens on old binaries are not pooled here.

All twelve fresh processes use one finite Slurm allocation, job 2767 on
n2/node2, `main`, `gpu:pro6000:1`, assigned visibility `1`, three-hour limit.
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
`GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`, driver 595.91.07, 600 W.
Build: CUDA 12.9.1, Release/sm120/portable-CUDA AOT, fast compile off, verified
ccache 4.5.1. OMP/OpenBLAS/MKL each use two threads; Slurm reserves eight CPUs.

Frozen library SHA-256:
`2df02642da424df6a7c31c2b35b210032cfa0f466d6b93788e843a2360799ebf`.
Frozen endpoint SHA-256:
`1e2e98e0438481a693dfe9f89a45f127c73493474b19803adbfad34359bd41d8`.
Input, both drivers and post-production independent reference hashes are also
retained in `summary.json`; a changed identity prevents a completed summary.

## Clean full-endpoint results

Each cell executes twice, first forward order then reverse, with a fresh process
per sample. W is strictly FP64, Lambda cadence thirty, CCSD batch eight, and both
core retention and matrix audit are enabled throughout. All requested schedules
are actually admitted. Every cell retains 21 Lambda iterations and 22 actions.

Seconds are medians; brackets are the two observed extrema, not confidence
intervals. These clean timings are never pooled with instrumented samples.

| Cell | Lambda Q batch | Fresh matrix replay | Complete E+F seconds | Lambda seconds | E+F saving vs b8 |
| --- | ---: | --- | --- | --- | ---: |
| b8 | 8 | off | 516.922 [516.377, 517.467] | 120.465 [120.453, 120.476] | baseline |
| b32 | 32 | off | 481.469 [481.445, 481.493] | 84.327 [84.298, 84.355] | 6.858% |
| b8_p1 | 8 | on | 507.606 [507.517, 507.695] | 110.613 [110.603, 110.624] | 1.802% |
| b32_p1 | 32 | on | 471.299 [471.214, 471.385] | 74.271 [74.235, 74.308] | 8.826% |

The combined candidate saves 45.623 seconds of the complete endpoint and
46.193 seconds within Lambda, or 38.346% of Lambda time. Larger batches alone
save 36.138 Lambda seconds. Fresh original-graph matrix replay adds
10.055 Lambda seconds / 10.170 complete-endpoint seconds beyond batch 32.
Do not attribute the combined improvement to either mechanism alone or add
nested profiler times to their parents. Outside-Lambda fluctuations remain
visible in the retained full endpoint and phase fields.

The `<120 s` Lambda stretch is reached in this opt-in frozen experiment, not
retroactively in the original #2136 qualification and not as a general default
performance promotion. Ordinary batch eight and fresh replay off remain the
defaults. Two repetitions do not meet the shared five-pair promotion gate;
measured VRAM peak, full compilation cost and full solver trajectories are
unavailable. Scientific gates are unchanged.

## Work and capacity

| Cell | GEMM calls | Modeled contraction summands | Logical packing-output bytes | Q batches | Lambda capacity bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| b8 | 45986 | 38135799179463 | 1881569130992 | 2013 | 10082032819 |
| b32 | 13811 | 37901517693843 | 1608035327072 | 888 | 17873308281 |
| b8_p1 | 47877 | 38135584669791 | 2147065655920 | 1586 | 10643904947 |
| b32_p1 | 14307 | 37901303184171 | 1862138223520 | 416 | 20701823865 |

Capacity increases from 9.390 GiB for b8 to 19.280 GiB for the combined
candidate. This is a complete Lambda numeric/descriptor/provider capacity
reservation, not measured peak VRAM. Logical packing bytes are not physical
DRAM traffic. Fewer small GEMMs explain batching; fresh replay can be faster
despite more GEMMs and packing because it replaces its scalar CUDA lowering.
Iteration/action counts do not decrease, and no FP32 work is introduced.

Fresh matrix replay lowers the original virtual primal graph, with only
`t1`, `t2`, `bov`, `bvv` as inputs, and ordered Q accumulation. It neither
borrows staged solver cuts nor removes mandatory fresh retained-core replay
and physical residual checks. Its arena shares the later independent matrix
audit's disjoint lifetime. Preflight charges their maximum plus descriptors,
retries audit-only capacity on replay refusal, and retains scalar replay if
audit/provider admission is dropped. Nonfinite/binding/numerical/driver failures
still abort response publication.

## Numerical and regression gates

Independent energy uses the existing PySCF 2.14.0 same-Hamiltonian reference,
`../df-lambda-cost-2136-20261009/oracle-energy.json`, tolerance `3e-9 Eh`.
Independent directional forces use the original centered energy differences,
`../df-lambda-gemm-20261004/oracle-energy-fd.json`, at both `1e-4` and `3e-5`
Bohr, tolerance `3e-7 Eh/Bohr`. Centered geometries are checked against the
frozen molecular input. All 24 components additionally match one common
strict native baseline at `3e-7`, with the original stricter `3e-9` sanity
check, translation at `3e-8`, and exact Lambda/Z/stationarity at `1e-9`.

The common native component comparator is not an independent analytic oracle
for every coordinate. Independent force checks remain two energy directions;
global RHF stability is not certified. Exact quantitative errors/tolerances
across the eight clean and four separate instrumented endpoints are retained
in `validation.json`, with a numerical-only accepted publication decision.

Across all twelve endpoints: maximum independent energy error is
`2.0322e-12 Eh`, directional force error `7.5606e-9 Eh/Bohr`, common-baseline
component difference `4.1744e-13`, translation defect `1.0629e-12`, Lambda
residual `6.1164e-13`, Z residual `1.3570e-13` and stationarity `5.3564e-13`.
Full-precision values and separate finite-difference steps remain in the envelope.

- 95 native/library cases pass on the candidate library under Slurm job 2766.
- Ten larger-batch/fresh-replay memchecks pass, with zero sanitizer errors.
- Fresh replay is bit-exact versus scalar replay in small native fixtures,
  including whole/partial Q batches, source/no-source and audit-disabled fallback.
- Three independent NumPy original-primal/packed-graph comparisons pass at the
  original tolerance, and the graph has only its four original primitive inputs.
- 73 targeted host regressions and 39 final-executable early CLI tests pass;
  the final combined host/evidence-retention sweep passes 148 cases.
- Compiler structure: 504 modules / zero errors; CUDA ownership: 338 files;
  provider inventory: 20 files; materialization delta: zero.

The native suite used this same library. A subsequent executable rebuild changed
only usage text, passed the 39 CLI tests and was frozen before the full cohort.
Failed preliminary logging/obsolete-arity attempts retain raw diagnostics only,
not timing observations or completed summaries.

## Profiles and retention

Four separately instrumented fresh endpoints follow the clean sweep inside the
same allocation. `profile-summary.json.gz` keeps mutually exclusive Lambda parent
children and every nested physical-action work record. Their timing is explanatory,
not an additional clean sample. No Nsight/hardware DRAM counters or measured
VRAM peak are claimed.

| Immediate Lambda child, seconds | b8 | b32 | b8_p1 | b32_p1 |
| --- | ---: | ---: | ---: | ---: |
| Initialization | 2.305 | 2.227 | 2.305 | 2.226 |
| Fresh primal replay | 17.996 | 18.000 | 8.261 | 8.085 |
| RHS | 0.083 | 0.081 | 0.083 | 0.081 |
| GMRES parent | 74.085 | 49.282 | 74.043 | 49.243 |
| Original-equation independent audit | 20.881 | 11.137 | 20.883 | 11.139 |
| Parameter/factor VJP | 5.062 | 3.524 | 5.065 | 3.513 |

Batching reduces GMRES, independent audit and VJP time while scalar fresh replay
stays at about 18 seconds. The separate replay flag reduces only that replay to
about eight seconds; it does not improve GMRES or remove independent auditing.
Every child is charged once beneath the same complete Lambda parent.

`samples.json.gz` losslessly shares equal endpoint fields once. After gzip
decompression, reconstruct an
endpoint with `{**shared_endpoint_fields, **observation["endpoint"]}`; all variable
fields, forces, scientific gates, process timings and observation order are
retained. Full raw outputs, failed attempts, compiler products and source archives
remain ignored under `.artifacts/`, not disguised as accepted tracked evidence.
The JSON gzip companions preserve the original decompressed bytes, not rounded
or filtered measurements. The 64 MiB aggregate retention cap and hard 1 MiB/file
limit are unchanged.
There are no external archives, release assets or release tags.

## Reproduction

In a detached worktree at the pinned master, apply the source-only patch. Follow
the normal CUDA build prerequisites; verify `ccache --version` before building,
use checkout-root `CCACHE_BASEDIR`, explicit CXX/CUDA cache launchers,
`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, Release, sm120 and portable-CUDA AOT.
Build `generativeqc` in `build-cuda`, then the native endpoint:

```bash
mkdir -p build
ccache g++ -std=c++20 -O2 -DGENERATIVEQC_HAS_CUDA=1 -Iinclude -Isrc \
  -I"$CUDA_HOME/include" -c benchmarks/df_ccsdt_force_endpoint.cpp \
  -o build/endpoint.o
g++ build/endpoint.o build-cuda/libgenerativeqc.so \
  -Wl,-rpath,"$PWD/build-cuda" -o build/endpoint
export PYTHONPATH="$PWD/python:$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PWD/build-cuda:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=03:00:00 bash -lc '
  python benchmarks/df_lambda_batch_ablation.py \
    --endpoint build/endpoint --library build-cuda/libgenerativeqc.so \
    --input benchmarks/results/rccsd-diis-ring-1900/ethane230.input \
    --oracle benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json \
    --finite-differences benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json \
    --output .artifacts/issue2136-qbatch/reproduction-clean \
    --batch-limits 8 32 --replay-candidates --repetitions 2 &&
  python benchmarks/df_lambda_batch_ablation.py \
    --endpoint build/endpoint --library build-cuda/libgenerativeqc.so \
    --input benchmarks/results/rccsd-diis-ring-1900/ethane230.input \
    --oracle benchmarks/results/df-lambda-cost-2136-20261009/oracle-energy.json \
    --finite-differences benchmarks/results/df-lambda-gemm-20261004/oracle-energy-fd.json \
    --output .artifacts/issue2136-qbatch/reproduction-profile \
    --batch-limits 8 32 --replay-candidates --repetitions 1 --screen --instrumented
'
```

Preserve scheduler-assigned visibility; do not replace it with a physical GPU
index or schedule on node3. Choose equivalent compatible node/resources if n2
is unavailable, but do not pool different GPU/source/input identities into one
matched cohort. The source patch and compact records are sufficient local
reproduction evidence; no expiring external backup is required.
