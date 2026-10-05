# Native DF CC auxiliary tiles: complete endpoint qualification

The complete energy experiment uses one n2 Slurm allocation (job2284), alternating
baseline/candidate and candidate/baseline order on the same assigned RTX PRO 6000.
Baseline is #1900's DIIS-ring implementation at `bcc92f614`; candidate production
source is `c4f26cade`. This isolates #1903 from the earlier DIIS change.

| Median seconds | Ethane230 baseline | Tile8 | Water7 baseline | Tile7 |
| --- | ---: | ---: | ---: | ---: |
| Process wall | 335.680 | 289.380 | 1.320 | 1.240 |
| Complete native energy | 335.379808 | 289.082502 | 1.042738 | 0.965656 |
| RHF | 154.333406 | 132.856922 | 0.696921 | 0.648906 |
| Source | 3.639904 | 3.643529 | 0.245479 | 0.245502 |
| CCSD | 172.451316 | 147.615177 | 0.088121 | 0.064062 |
| (T) energy | 4.955082 | 4.966769 | 0.012136 | 0.007097 |

CCSD improves **1.168x**, saving 24.836 seconds. The same candidate forced to
one-Q takes 173.445 seconds in CCSD, with 347.160 seconds complete native time.
This supports batching, rather than fused accumulation alone, as the phase win.
The measured complete median difference is 46.297 seconds, but 21.476 seconds
comes from the unmodified RHF phase and is not attributed to this change.
The CCSD saving represents 7.405% of baseline complete native time. Two repeats
per variant and one one-Q observation are retained; no cross-device calibration
or statistical confidence interval is inferred from this small sample.

## Numerical, resource and work gates

Inputs and descriptor settings are unchanged from the adjacent
[`rccsd-diis-ring-1900`](../rccsd-diis-ring-1900/README.md) record: ethane230 has
o=9, v=221, q=488; water7 has o=5, v=2, q=7. DIIS history is eight, budget 64 GiB,
RHF energy/density tolerances 1e-12/1e-11 and CC energy/residual tolerances
1e-12/1e-10. No reference orbitals/amplitudes enter the production endpoints.

All five large energy observations have 20 iterations, 38 evaluations and zero
DIIS restarts. Their maximum total-energy spread is 4.690e-13 Eh; physical replay
residuals remain below 5.317e-13. Water energies are identical across all four
observations; maximum independent replay residual is 7.397e-12.

| Ethane complete CCSD metric | Baseline | Tile8 | Candidate one-Q |
| --- | ---: | ---: | ---: |
| Numeric capacity, bytes | 3,404,763,120 | 4,476,290,544 | 3,404,763,120 |
| Provider calls | 317,528 | 46,322 | 317,528 |
| Semantic contraction summands | 44,063,877,939,352 | 44,063,877,939,352 | 44,063,877,939,352 |
| Packed GEMM summands | 39,019,400,687,592 | 39,166,125,303,240 | 39,019,400,687,592 |
| Logical packing bytes | 7,347,698,382,880 | 5,587,100,796,160 | 7,347,698,382,880 |
| Q slices | unavailable | 19,032 | 19,032 |
| Q tiles | unavailable | 2,806 | 19,032 |
| Generated auxiliary operation calls | unavailable | 179,340 | 1,176,080 |
| Accumulation calls | unavailable | 3,294 | 19,520 |
| Logical accumulation bytes | unavailable | 2,990,275,612,480 | 7,111,761,131,904 |

Summands are not hardware FLOPs, and logical bytes are not measured bus traffic.
Missing baseline counters remain null in `summary.json`; unrequested force
phases are also null. Packed GEMM work increases slightly despite fewer calls.
The tile spends an additional 1,071,527,424 bytes; explicit budget/allocation
fallbacks preserve one-Q matrix and scalar schedules. The Q contraction and
accumulation counts include independent one-Q replay, not only the hot iteration.

For six retained cuts totaling `C=ov+4*o²*v²+v²` elements, a b-row tile's ordered
accumulator has `(b+2)*C*8` logical traffic instead of `3*b*C*8`; it uses one
launch instead of six launches per Q. These are source-derived formulas, not
measured baseline counters. The implementation preserves each original Q
addition and its nonfinite check, including when the initial retained sum is
nonzero.

## Force scope

Job2289 passed small-water complete energy/force behavior and independent PySCF
finite differences for all nuclear coordinates at both 1e-4 and 3e-5 Bohr.
The existing atol=rtol=3e-7 force gate is unchanged. Job2287 completed one full
baseline/candidate force pair per molecule on GPU
`GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`. Its times are a separate measurement
group from job2284; neither GPU is used to calibrate the other.

| Ethane230 full-force phase, seconds | Baseline | Tile8 |
| --- | ---: | ---: |
| Process wall | 1391.54 | 1350.00 |
| Complete native | 1391.235311 | 1349.685283 |
| RHF | 161.765296 | 144.025037 |
| Source | 3.592322 | 3.628883 |
| CCSD | 170.481750 | 146.080942 |
| (T) pullback plus Fock response | 109.930732 | 110.117274 |
| Lambda | 272.789455 | 272.939885 |
| Source response | 3.713752 | 3.724383 |
| Orbital/nuclear response | 668.933803 | 669.137991 |

The force pair saves 24.401 seconds in CCSD; 17.740 seconds of its 41.550-second
complete difference comes from RHF. One pair does not establish statistical
endpoint performance. The force-path `triples_seconds` includes pullback and
Fock response, so it cannot be labeled pure triples energy. Within this force
run RHF/source/CCSD total 293.735 seconds; the remaining 1055.950 seconds include
both (T) energy/pullback and response. Subtracting a standalone energy time from
another GPU would not yield a calibrated energy/force split.

The maximum all-component force difference is 5.620e-9 Eh/Bohr (water: 7.994e-15).
Candidate Lambda residual is 6.116e-13, Z residual 1.360e-13, maximum stationarity
7.155e-12, and translational residual 6.109e-12. Both variants use 21 Lambda
iterations/42 actions and 28 exact J/K actions. Complete force capacity stays
7,170,696,275 bytes because another phase dominates the peak; the primal phase's
capacity increases by the separately reported tile storage.

Job2288 evaluates independent central energy differences for two nonzero
components using the same candidate source and original geometry/bases, with
no oracle orbitals/amplitudes. At 1e-4/3e-5 Bohr, C0-z errors are
1.929e-9/1.655e-8 Eh/Bohr; H1-x errors are 5.301e-10/3.062e-8. Every unchanged
atol=rtol=3e-7 gate passes (maximum allowance ratio 0.102). All energies,
analytical values and gate ratios are retained in `summary.json`. These two
components are not an all-coordinate independent large-force audit and do not
establish benzene264 or global reference stability.

## Reproduction and retained artifacts

GPU UUID for job2284: `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, driver595.91.07;
CUDA12.9.1/sm120. Use the #1900 build settings, ccache and assigned Slurm device
visibility, and compile `benchmarks/df_ccsdt_force_endpoint.cpp` against each
variant's own headers/library. Run all energy comparisons in one finite n2
`main --gres=gpu:pro6000:1` allocation with OMP/OpenBLAS/MKL threads two:

```sh
./probe INPUT OUTPUT.json 1 1 0 1 8 8
# Candidate-only one-Q comparison; final argument is residual Q limit.
./candidate INPUT one-q.json 1 1 0 1 8 8 1
```

Frozen inputs are `../rccsd-diis-ring-1900/{ethane230,water7}.input`. Full source,
build/test logs, toolchain/device records, binaries and hashes remain at
`n2:/data/jzzeng/cc-1903-20261005/`, with endpoint records in `endpoint-2284/`.
Local ignored copies are in `.artifacts/1903/{endpoint-2284,force-2287,fd-2288}/`.
Every changed production/compiler/test file was hash-compared with the tested
remote source. The nine normalized energy observations, four force observations,
paired medians, spreads and large FD comparisons are in `summary.json`.
No external archive, release or release tag was published.
