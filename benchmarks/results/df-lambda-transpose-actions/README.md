# DF Lambda transpose-aware operands: bounded phase benefits

This record compares frozen master `4385f72751b829883407c01106186917c344317b`
with transpose-aware DF matrix operands. The production-code difference is only
`python/generativeqc_compiler/cc/df_gemm.py`; a checksum comparison verified the
baseline's remaining production/compiler/build sources against the same tree.
Both executables use the same committed complete-force benchmark frontend.

**CCSD phase benefits are measured on both eligible cases, and a small Lambda
phase benefit is measured on methane. No complete energy/force speedup, water
Lambda speedup, or large-molecule speedup is established.** This change removes
unnecessary materialization rather than addressing the dominant nuclear branch.

## Complete endpoint observations

Slurm job 2706 on n2/node2, `main`, `gpu:pro6000:1`: RTX PRO 6000 Blackwell,
GPU `GPU-4b4be14f-ec84-6736-a7d8-968d62900c72`, driver 595.91.07, 600 W limit.
Slurm assigned `CUDA_VISIBLE_DEVICES=1`, retained unchanged. Each variant has six
fresh-process observations per case, alternating baseline/candidate and
candidate/baseline order. All 24 measured observations and four excluded
qualification/cache-prime observations are retained by workload in
[water24.samples.json](water24.samples.json) and
[methane34.samples.json](methane34.samples.json).

Conventional unscreened RHF supplies Fock/orbitals; only correlation is fitted.
Both bases are spherical: cc-pVDZ / cc-pVDZ-RI. Water has `(o,v,q)=(5,19,84)`;
methane has `(5,29,112)`. Inputs contain only atoms, primitive shells and a
64-GiB admission limit, never oracle state. Both builds are Release / sm_120,
CUDA 12.9.1 / GCC 11, portable CUDA AOT, identical scalar CPU-linalg fallback,
with OMP/OpenBLAS/MKL threads fixed to two. No fast math, screening, precision
change, tolerance relaxation or reference/orbital recycling is introduced.

| Median seconds | Water baseline | Candidate | Methane baseline | Candidate |
| --- | ---: | ---: | ---: | ---: |
| Complete native E+force | 12.737517 | 12.718372 | 29.676612 | 29.678232 |
| CCSD | 0.249294 | 0.240549 | 0.372004 | 0.368531 |
| Complete Lambda response | 0.660757 | 0.651148 | 1.100662 | 1.093416 |

Native timing includes RHF/source/CCSD/(T)/corrected Lambda/source response and
orbital/nuclear response. Process startup/teardown time is retained separately.
Source compilation is excluded; persistent compiler/JIT caches are retained.
The excluded runs prime optional compilation/provider artifacts, not density,
orbitals, amplitudes or response state for a later process.

CCSD median time falls **3.51% / 0.93%**; all six pairs improve in each case.
Methane Lambda falls **0.66%**, improving in all six pairs. Paired median-change
percentile-bootstrap intervals are respectively `[1.18,4.08]`, `[0.53,1.06]` and
`[0.54,1.20]` percent. Water Lambda's 1.45% point estimate has interval
`[-0.30,2.81]`; it is not a qualified win. Complete native intervals are
`[-0.34,0.76]` and `[-0.087,0.129]` percent, encompassing zero. These descriptive
intervals from six pairs do not certify other devices or large/asymptotic domains.

Unchanged orbital response dominates: approximately 7.34 / 24.82 seconds,
including 6.07 / 23.71 seconds in two-electron nuclear derivatives. Neither its
timing fluctuations nor RHF variation are attributed to the DF layout change.

## Scientific and semantic-work gates

Every completed native observation passes pinned PySCF 2.14.0 same-Hamiltonian
energy checks and two independent nuclear directions at both `1e-4` and `3e-5`
Bohr. Explicit symmetric metric whitening preserves the `1e-10` relative cutoff;
the oracle does not substitute DF-RHF Fock/orbitals. The native timing includes
none of that independent oracle work.

- Maximum independent energy error: `1.57e-13 Eh`, gate `3e-9`.
- Maximum independent force-direction error: `2.52e-9 Eh/Bohr`, unchanged
  `atol=rtol=3e-7` gate.
- Maximum matched baseline/candidate force difference: `2.98e-14 Eh/Bohr`;
  energies agree within `1e-10`.
- Fresh expanded Lambda, Z/stationarity and translation-closure gates pass;
  no discarded/retried primal or resident-J/K attempt occurs.
- CCSD/Lambda semantic contraction work and iteration/action counts are unchanged.

| Measured Lambda work | Water baseline | Candidate | Methane baseline | Candidate |
| --- | ---: | ---: | ---: | ---: |
| Semantic contraction summands | 4,017,985,537 | 4,017,985,537 | 14,389,708,243 | 14,389,708,243 |
| Operator actions | 36 | 36 | 34 | 34 |
| Generated kernel calls | 73,859 | 69,804 | 86,433 | 81,836 |
| Logical packing output bytes | 1,544,025,224 | 1,501,364,656 | 3,940,153,464 | 3,837,204,336 |

Summands are not hardware FLOPs; logical bytes are not measured DRAM traffic or
whole-process peak VRAM. Native capacity and transfer diagnostics remain in the
raw samples; no claim about whole-process memory peaks is made.

Full candidate-library qualification in finite GPU Slurm job 2705:
**109 passed, 10 conditionally skipped**, covering native DF solves/Lambda,
transactional failure behavior, complete H2/H2O/LiH force finite differences,
high-angular/fixed-rank edge cases and typed contraction bindings. Compiler
structure, electronic boundaries/production paths, 338-file CUDA ownership
inventory, focused Ruff and diff checks pass. There is no handwritten scientific
or runtime CUDA growth. Existing scalar/expanded paths remain audits/fallbacks.

## Ineligible large case

The same protocol also attempted ethane cc-pVDZ/cc-pVDZ-RI, `(o,v,q)=(9,49,196)`.
Its **baseline** qualification/cache-prime process exceeded the finite 300-second
subprocess limit before publishing a result. No candidate or measured ethane
sample was attempted. Its input and independent oracle are retained, and missing
times are null in [summary.json](summary.json). This is an eligibility limit,
not evidence of a candidate regression or speedup. The complete original job
therefore ends with a timeout; the two completed six-pair groups are reported
without dropping or reclassifying any measured observation.

## Reproduction and provenance

Exact binary hashes, GPU identity, selectors, raw observations, independent
energy/finite-difference records and input hashes are committed here. The tested
candidate compiler source SHA256 is
`c3f1aba50fc65e3dfbd65eeaa326b44e6f1a1b43045be5d02bed20d4bfcca671`.
Full build/test/measurement scripts, logs and unabridged files are retained at
`/data/jzzeng/qc-df-lambda-transpose-20261008-qualification` on n2 and under
`.artifacts/df-lambda-transpose-qualification` in the development worktree.

Build each frozen source tree with the same flags, verified ccache launchers and
checkout-root `CCACHE_BASEDIR`; do not clear the cache. The candidate build's
shared before/after cache statistics increase by 467 hits and six misses, not an
exclusive per-build cache attribution. Native CMake commands actually use ccache.

```bash
ccache --version
export CCACHE_BASEDIR="$PWD"
cmake -S . -B build-cuda -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGENERATIVEQC_BUILD_TESTS=OFF -DGENERATIVEQC_BUILD_CLI=OFF \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=ON -DGENERATIVEQC_AOT_PROFILE=portable_cuda \
  -DGENERATIVEQC_BUNDLE_XTB_OPENBLAS=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
cmake --build build-cuda --target generativeqc -j8
```

Compile the unchanged `benchmarks/df_ccsdt_force_endpoint.cpp` with the matching
system C++ compiler, source/include/CUDA headers, and link one executable against
each variant's library with a corresponding runtime search path. The established
benchmark selectors are not an assertion about every public API default.

Run the two executables inside one explicit finite Slurm GPU allocation. For
each eligible input, first retain one excluded qualification run per variant,
then retain six pairs alternating B/C and C/B. Rebuild every scientific owner
in a fresh process; do not supply reference state. Configure CUDA library paths
and OMP/OpenBLAS/MKL threads as above. A single observation is:

```bash
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:10:00 \
  ./endpoint water24.input new-output.json \
  1 1 1 1 8 8 8 0 0 2 1 30 0 0 1 auto 1 0 auto 0
```

Preserve Slurm's device visibility. Use n1/n2/n4/n5 and a compatible GPU request;
do not use n3 while it is under maintenance. Use the committed oracle coordinates, basis
definitions and FD directions with pinned PySCF 2.14.0 for independent replay.
Only explicitly whitened correlation factors enter that oracle's CCSD/T solver;
no oracle numeric state enters a native endpoint.
