# Experimental PBE0 Schwarz-indexed force pages

**Review qualification hold:** the records below predate the shared-claim
consumption barrier requested in PR #1767. An empty page or skipped claim could
let the leader overwrite shared state before another warp had read it. Keep
these bytes as historical observations, not acceptance/performance evidence
for the synchronization-corrected binary. Fresh native and endpoint
qualification is required before approval; passing the old energy/force gates
or memcheck/initcheck does not prove shared-memory race freedom.

This is an **opt-in scheduling experiment**, not a new default or a claim that
the large-system GPU4PySCF performance gap is solved. The main README is not
updated. All timings below include complete public energy **and analytic-force**
endpoints, not just the improved integral source.

## Paired complete endpoints

RTX 5090 on n1, finite Slurm jobs 5533 (3–48 atoms) and 5532 (96 atoms).
Each size runs baseline then candidate on the same allocated GPU: cold, five
frozen warm calls, changed geometry, five frozen moved-warm calls. All observations
are retained; no iteration normalization or sample exclusion is applied.

| Atoms | Baseline warm (s) | Indexed pages warm (s) | Warm reduction | Baseline moved-warm (s) | Pages moved-warm (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.346606 | 0.268082 | 22.65% | 0.346468 | 0.268981 |
| 6 | 0.742711 | 0.617528 | 16.85% | 0.743566 | 0.616721 |
| 12 | 1.661318 | 1.393128 | 16.14% | 1.654069 | 1.393836 |
| 24 | 5.017627 | 4.494019 | 10.44% | 5.017894 | 4.506816 |
| 48 | 18.019789 | 15.169900 | 15.82% | 18.090378 | 15.183226 |
| 96 | 78.806055 | 65.513278 | 16.87% | 78.970583 | 65.678894 |

All warm calls take one SCF iteration. The five 96-atom original warm times span
78.760471–78.875741s baseline and 65.338678–65.576368s candidate.

| Atoms | Baseline cold (s) | Pages cold (s) | Baseline moved (s) | Pages moved (s) |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 80.993102 | 81.190376 | 1.938091 | 1.866403 |
| 6 | 69.074905 | 68.870977 | 4.055520 | 3.945070 |
| 12 | 80.318227 | 80.478315 | 10.123085 | 9.878462 |
| 24 | 106.299507 | 106.295726 | 27.040501 | 26.529894 |
| 48 | 188.430139 | 189.359835 | 71.408586 | 68.600860 |
| 96 | 577.476754 | 581.458329 | 263.215809 | 267.119967 |

**Cold and changed-geometry behavior is not uniformly faster.** At 48, cold SCF
iterations are 24→25; at 96, cold iterations are 28→29 and moved iterations 12→13.
Cold includes required JIT/cache setup, without purging caches. These results
are not isolated cold scheduling speedups. All stopping tolerances stay fixed.

All 144 native endpoints pass independent absolute gates of 1e-8 Eh and
1e-7 Eh/Bohr against the unchanged retained GPU4PySCF references. Maximum
all-repeat errors are 1.037e-10 Eh and 3.113e-11 Eh/Bohr. References are reused for
numerical acceptance, **not** represented as a freshly timed comparator here.

The [original protocol](../pbe0-def2-svp-20261003/README.md) supplies water32mer
prefixes, spherical def2-SVP, the offline H/O basis and unpruned moving grid
(48 radial × 16 polar × 32 azimuth points per atom). Native energy/density
tolerances are 1e-12/1e-10, screening 1e-12, maximum 100 SCF iterations. No DF,
mixed precision, changed cutoff, reference density or reference production
consumer is introduced. The only experiment selector is
`GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=1`.

## Work and limits

Sort shell pairs by descending per-system Schwarz bounds, retain one exclusive
prefix over geometry-live block rows, and claim 16 independent 64-candidate pages
per live product. Exact density/shell/AO predicates and physical orientation
remain unchanged. This is output-sensitive traversal, not universal
linear-scaling ERIs; the dense worst case remains dense.

The 96-atom diagnostic has 73,920 pairs and 2,310 blocks. Its identity-order
candidate capacity falls from 852,014,240 to 143,025,952 after sorting. The
complete triangle has 2,669,205 products; the indexed domain has 139,884 products
but **2,238,144 actual page claims**, plus worker termination claims. Additional
device index storage is 18,488 bytes. Tight prefix budgets retain sorted
triangular traversal; failed owner admission retains the existing fallback.
Capacity counts are not executed primitive/root counts.

The separate fixed-density source bracket measures 27.23250→13.59034s at 96,
but that 50.10% source reduction is **not** the 16.87% complete endpoint reduction.
Its 24-atom identity-density counterexample is 0.04329→0.07288s, still slower.
The earlier unpaged candidate concentrated high-order work into 88 blocks rather
than 1,154; paging addresses that load imbalance without a size threshold.
Coarse-candidate summaries remain separate and are never pooled with pages.

The baseline 96 first warm force telemetry still attributes 32.70s to semilocal
geometry response and reports 21,516,784,080 partition grid-pair visits. This
change does not eliminate that distinct structural bottleneck. Unmeasured
component times remain null, not invented from endpoint differences.

## Identity and qualification

Measured clean baseline and dirty candidate both use master
`a20b8801f7f43b83cd7da8001db180716d8df302`. Native records retain full environment,
protocol, force arrays and component/work telemetry; `storage.json` binds their
original decompressed bytes, source/library hashes, outcomes and allocations.
`measured-source.patch.gz` reconstructs the dirty measured tree from that base.

The PR integrates master `e47058a51` and clang-format 23.1.1. The comparison of all
1,349 canonical identity paths finds one production file changed only in
whitespace. Endpoints are **not relabeled as final-binary measurements**:

| Identity | Measured baseline | Measured pages | Formatted PR build |
| --- | --- | --- | --- |
| Source | `f46f35ea7d273ffe1b534e3e93082c28c6ef6508a0a89e6702575f9c75fb4046` | `d062e73d8595e1f753e16d543c8fe951fa7d38fed7c883c47978eb2ec7c2dfbb` | `bf9dbcb826577dc4eaef0891017c98729cc474558723fa239273cee6b4c176d7` |
| Library SHA256 | `bd00a793de8809a136d1f15e692db029308dc25bcd5c8a0d0e1a76a241caa4fb` | `ae00a76b40a2a19e72b4f3bc7d2da63c1b8fc65f400a9ea97ad0f0f2ac8c70c2` | `100be12f9b691ec655b8044a2fa5566e09176e829493647419c0de04c791f180` |

- n1 job 5536 independently checks the final compiled/source identity, runs each
  binary's own baseline/OFF/ON through-f CPU-ERI oracles, and runs candidate
  memcheck/initcheck: both report zero errors.
- Two-system multirow and exact full/full−1/full−prefix budget gates pass;
  maximum batch source error 4.441e-16. Default-OFF, physical domain, tail-page,
  allocation/lifetime and ownership tests: 221 focused host tests pass.
- Builds on n5 use Release CUDA 12.9.1/sm_120, AOT shells, verified ccache 4.5.1,
  explicit C++/CUDA launchers and checkout-root `CCACHE_BASEDIR`. Shared cache
  counters are retained as receipts, not attributed to a single build.
- PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 13.6.0, cuTENSOR 2.2.0, NumPy 2.4.6;
  eight OpenMP/BLAS/MKL threads. Exact settings remain in the native records.

Master subsequently advanced to `53da52eda` (Python lazy XC import and CPU trace
compensation). Those changes are not silently attributed to the timed binaries.

## Recheck and reproduce

From the repository root, no GPU is needed to recheck hashes and all independent
energy/force gates, or to recompute every paired median:

```bash
python benchmarks/results/pbe0-screened-pages-20261003/verify.py
```

Build separate clean and reconstructed measured checkouts with matching source
and library identities. Use verified ccache with explicit
`-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` and
`-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`, checkout-root `CCACHE_BASEDIR`,
`-DCMAKE_BUILD_TYPE=Release`, `-DGENERATIVEQC_ENABLE_CUDA=ON`,
`-DGENERATIVEQC_ENABLE_AOT_SHELLS=ON`,
`-DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120`, and
`-DGENERATIVEQC_AOT_PROFILE=sm_120`. Do not replace identities to reuse caches.

Run `benchmarks/run_pbe0_benchmarks.sh` in each matching checkout, baseline then
candidate **inside the same finite allocation**, with:

```bash
export PBE0_ENGINES=native PBE0_POINT_TIMEOUT=2400
export PBE0_ATOMS="3 6 12 24 48 96"
export PBE0_BASIS_FILE="$PWD/benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json"
export PBE0_BENCHMARK_OUTPUT="$PWD/.artifacts/pbe0-screened-pages-replay"
export PBE0_BENCHMARK_PYTHON="$(command -v python)"
export GENERATIVEQC_LIBRARY=/absolute/path/to/matching/libgenerativeqc.so
export GENERATIVEQC_BOUNDED_SCHWARZ_SCHEDULE=1  # 0 for the clean baseline
for atoms in $PBE0_ATOMS; do
  mkdir -p "$PBE0_BENCHMARK_OUTPUT/$atoms"
  gzip -cd "benchmarks/results/pbe0-def2-svp-20261003/water$atoms-reference.json.gz" \
    > "$PBE0_BENCHMARK_OUTPUT/$atoms/reference.json"
done
```

The allocation must use `srun --partition=main --gres=gpu:5090:1 --nodes=1
--ntasks=1 --time=02:00:00 bash -lc '…'`, preserving Slurm device visibility.
Use the matching CUDA/cuTENSOR environment and ccache compiler wrappers for
required stationary JIT work. Keep each variant's output and provenance separate.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-03-schwarz-indexed-independent-force-domain.md)
records domain invariants, rejected alternatives and the sparse-density caveat.
