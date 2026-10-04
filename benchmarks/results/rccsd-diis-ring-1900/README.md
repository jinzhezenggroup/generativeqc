# Native CUDA DIIS ring qualification (#1900)

The ring reduces exact bookkeeping work, but this ethane230 endpoint is not
DIIS-bound. No complete-endpoint speedup is established. The change is retained
for eliminating history movement and redundant dots, including before packed
state work, rather than as a remedy for the dominant CCSD/response cost.

All observations below are measured on one n2 RTX PRO 6000 Blackwell, Slurm
job2280, GPU UUID `GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`. Baseline/candidate
orders alternate; every sample is retained in [summary.json](summary.json).
Both variants reconstruct native RHF, the physical DF source, CCSD and triples
in a fresh process. Device/driver caches are not reset. There are two large and
five tiny observations per variant, not enough to infer a sub-millisecond
complete-endpoint advantage. Concurrent CPU build/test allocations and the
unmodified RHF phase can contribute to host/endpoint variation.

| Measured median, seconds | ethane230 baseline | ring | water7 baseline | ring |
| --- | ---: | ---: | ---: | ---: |
| Complete process wall, including startup/teardown | 310.495 | 317.025 | 1.270 | 1.270 |
| Native complete endpoint | 310.186 | 316.712 | 0.999556 | 1.000185 |
| Native RHF | 131.098 | 137.634 | 0.651124 | 0.652206 |
| CCSD | 170.570 | 170.575 | 0.090046 | 0.090297 |
| DIIS, excluding timed trial residual | 0.122595 | 0.062368 | 0.001205 | 0.001174 |

The first large pair's apparent 11.8-second endpoint win is explained by RHF
variation; reversing the order removes that interpretation. DIIS itself is
about 1.97x faster, saving 0.0602 seconds. Eliminating all baseline DIIS would
save at most **0.0395%** of this native complete endpoint (Amdahl bound from the
measured baseline phase fraction). The actual measured DIIS saving is about
0.0194% of baseline total. The five tiny DIIS observations all improve; there
is no observed winning tiny legacy DIIS domain requiring a bounded fallback.
Tiny complete endpoint differences overlap run variation and are not promoted.

## Numerical and work gates

- Ethane: `o=9`, `v=221`, `q=488`, aug-cc-pVTZ/aug-cc-pVTZ-RI, conventional
  native RHF with correlation-only DF. Water: `o=5`, `v=2`, `q=7`, the committed
  water STO-3G orbital basis reused as the auxiliary basis; this is a tiny
  solver endpoint, not a claim about auxiliary-basis accuracy.
- History eight, 64 GiB complete correlation allowance; RHF energy/density
  tolerances `1e-12/1e-11`, CC energy/residual tolerances `1e-12/1e-10`.
- Every large sample uses 20 CC iterations, 38 evaluations, no DIIS restart,
  and complete independent replay. Maximum all-pair total-energy difference
  is `8.952838470577262e-13 Eh`; all replay residual maxima are below
  `5.317e-13`. Tiny samples use 14 iterations and identical published energy.
  Solver capacity is unchanged (`3,404,763,120` bytes for ethane).
- Candidate ethane counters are measured: `1,203,265,440` history insertion
  destination bytes, zero history-shift bytes, `490,805,640` residual dot
  summands, 229 Gram scalar writes, `486,847,530` successful combine summands.
- Missing baseline counters are **null** in the report. Separately, source-loop
  derivation from the observed 18 successful Gram/solve calls and zero restarts
  gives 19 insertions with live counts `1..8, 8 x 11`. For
  `N=ov+(ov)^2=3,958,110`, baseline triangle work is `515N=2,038,426,650`
  summands, compared with measured candidate `124N`. The corresponding modeled
  baseline history-shift destination bytes are `11*2*7*N*8=4,876,391,520`,
  and modeled full-Gram writes are 907. These are derived work counts,
  not measured bus traffic, hardware FLOPs, or GPU time estimates.

CPU/CUDA determinant-space and pinned molecular tests independently qualify
the solver. Seventy-five direct real-GPU ring cases additionally check exact
reduction-tree preservation, chronological coefficient/combine behavior,
wrap/retirement/zero/singular cases, and a sentinel in an old-old Gram entry.
Resident-JIT H2/H2O, repeated solve and live-state compatibility checks pass.
Full build/validation records remain at
`n2:/data/jzzeng/cc-1900-20261005/`; raw endpoint records are in its
`endpoint-2280/` directory. Local retained copies are under
`.artifacts/1900/endpoint-2280/` in the qualification worktree. No external
archive was published.

## Reproduction and provenance

Baseline: `ecfbfd959862da50c64cc740a6060f04bb73fcb9`.
Candidate production source: `e27a16d56f974444da39e3db6d119989e16fedaf`.
All changed production/compiler/package files were hash-compared with the
tested candidate archive. Binary hashes and device identity are in the report.

Build each checkout with CUDA 12.9.1, Release/Ninja, g++, ccache CXX/CUDA
launchers, checkout-root `CCACHE_BASEDIR`, architecture 120, CUDA=ON,
AOT_SHELLS=ON, AOT_PROFILE=portable_cuda, CLI=OFF,
STATIONARY_FORCE_AOT=OFF, STATIONARY_CPU_FORCE_AOT=OFF and CUDA_COMPILE_JOBS=4.
These are the corresponding `GENERATIVEQC_` CMake options; build target
`generativeqc`. Verify ccache before configuring and retain statistics.

Compile `benchmarks/df_ccsdt_force_endpoint.cpp` from the candidate against each
variant's own headers/library (`-std=c++20 -O2`, includes `src`, `include`, CUDA
`include`). For the baseline probe only, omit the five field lines using
`diis_history_insert_bytes`, `diis_history_shift_bytes`,
`diis_residual_dot_terms`, `diis_gram_updates`, `diis_combine_terms`; these
diagnostics do not exist in that baseline. Do not substitute zero counters.
Individual object compilation must use ccache.

Use one finite n2 allocation to run all paired commands (for example,
`srun -p main --gres=gpu:pro6000:1 -N1 -n1 --time=01:00:00 --pty bash`).
Inside that allocated shell, run each probe as:

```sh
/usr/bin/time -f '%e' -o result.wall \
  ./probe ethane230.input result.json 1 1 0 1 8 8
```

Use [ethane230.input](ethane230.input) and [water7.input](water7.input) without
alteration. SHA-256:

```text
9428f2b1d1db38ffa374387705099e8d57fde98e0e068faed2861b04604a1c6e  ethane230.input
817b6788f2196adaadcc7cd73a0d5104cb508f309543a805be9b9864cf649196  water7.input
```

Keep the allocation's `CUDA_VISIBLE_DEVICES`, record the assigned UUID, use
OMP/OpenBLAS/MKL thread counts two, and alternate both orders. No force,
Lambda, orbital-response or benzene264 qualification is supplied by this record.
