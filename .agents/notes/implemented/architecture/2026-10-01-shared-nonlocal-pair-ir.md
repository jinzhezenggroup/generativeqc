# Decision: share native VV10/rVV10 pair algebra through TensorIR

Status: implemented
Date: 2026-10-01

## Problem

CPU and CUDA shared the MethodIR nonlocal-correlation specification, but each
maintained its own handwritten pair energy and analytic derivative formulas.
Output-demand improvements could therefore land independently. The preceding
CPU zero-weight and radial-output changes (#1645, #1646) exposed that ownership
gap; they did not make the backend screening contracts equivalent.

## Decision

`method.nonlocal_pair` now constructs the pair expressions using the existing
scalar TensorIR operations. `tools/generate_nonlocal_pair_native.py` emits one
host/device header consumed by both native runtimes. No second algebra, AD
engine, runtime compiler, functional-name dispatch, or HF/DFT solver fork is
introduced. Formula ownership moves to the compiler; native runtimes retain
local-scale construction, allocation, residency, ordering, reduction, screening,
and numerical-failure publication.

The generated family has twelve live-output closures: four feature/geometry
combinations for VV10, raw rVV10, and preconditioned rVV10. The preconditioned
VV10 template modes reuse the same four VV10 closures. All retain FP64,
left-associated source products/sums, direct division, and the same analytic
expressions. Each generated closure carries its TensorIR logical hash.
The scientific `NonlocalCorrelationSpec` identity and public ABI are unchanged;
compiler/native source identities include the new generator and expressions.

### Exceptional arithmetic is part of the contract

The generic scalar emitter's default remains checked, transactional publication.
Two explicit options support migration of existing native arithmetic:

- `caller_owned_checks`: emit native exceptional values without new input,
  intermediate, domain, or output guards. Do not simplify zero times infinity or
  zero divided by zero. The enclosing runtime owns validation/publication.
- `ordered_native_sums`: initialize a sum from its first ordered term, spelling
  unit coefficients as identity/unary minus. This preserves runtime signed zero
  instead of introducing TensorIR's usual initial positive-zero accumulator.

These options do not request reassociation or FMA. Ordinary production
optimization still runs, and constant-only folding retains its existing
TensorIR semantics. The migrated signed-zero-sensitive expressions depend on
runtime inputs. Other emitter consumers use the unchanged defaults.

### Keep the two rVV10 representations explicit

CPU consumes raw omega/kappa and evaluates `(kappa_i*kappa_j)^1.5` per pair.
CUDA already retains omega/kappa and kappa^1.5 point invariants and supplies a
row inverse raw kappa for requested feature derivatives. Replacing one with the
other would change rounding and exceptional-arithmetic domains. They are
therefore explicit input/lowering representations of the shared expressions,
not a hidden cross-backend conversion.

### Output use and failure observation are different

CPU still computes/checks every legacy pair field outside #1646's existing
bounded weighted-potential-only VV10 admission, even for physically unused
outputs. Inside that envelope only its already-elided radial root is absent.
CUDA keeps its existing Features/Geometry demand and row-output failure checks.
Generated liveness selects roots only after those runtime contracts decide
which outputs must remain observable.

## Invariants

- CPU exact-zero integration weights are not CUDA density-screened negative-zero
  markers. Neither predicate is unified or broadened by this change
- Pair domain, traversal/reduction order, signed-weight treatment, thresholds,
  SCF policy, grid construction and geometry/force assembly stay unchanged
- Preserve out-of-envelope failures, including unused radial overflow on CPU
- Keep raw rVV10 and CUDA preconditioned rVV10 input meanings distinct
- No new workspace, allocation, transfer, per-pair policy branch or CUDA launch
- Independent Python/PySCF references stay independent of production generation
- Generation works from an uninstalled checkout without NumPy, native runtime,
  GPU, or oracle imports and is deterministic across CWD/Python hash seeds

## Rejected alternatives

- A shared handwritten native header would remove duplication but leave the
  scientific pair expressions outside the existing compiler IR
- IntegralIR scalar expressions lower division to multiply-by-reciprocal;
  using that path unchanged would alter rounding and overflow/underflow domains
- Default checked TensorIR emission would reject finite results reached through
  infinite denominators and could alter CUDA's existing output-demand domain
- Regenerating all derivatives with a different AD expansion would mix an
  ownership migration with a numerical-expression change; the stable analytic
  expressions are retained and checked independently by finite differences
- Extending the CPU finite envelope or changing GPU row masking in the same
  slice would combine distinct scientific admission and scheduling decisions

## Evidence

The new pair-codegen gate host-compiles the generated header and frozen legacy
CPU/CUDA scalar formulas with contraction disabled. It checks 966 deterministic
ordinary/adversarial input cases across all sixteen template modes, requiring
exact finite values, infinity signs and signed zeros, and matching NaN
classification. It separately compares derivatives to an 80-digit Decimal
finite-difference reference, checks output-root liveness and backend adoption,
and blocks runtime/oracle imports during standalone generation.

The updated signed-weight test still host-executes the actual CUDA local-scale,
ordered-row and force-seed kernels, now consuming the generated pair header.
These are scalar/source checks, not CUDA compiler or actual GPU qualification.

## Consequences

This is an ownership and duplication improvement. It adds explicit IR,
generation and qualification code while removing the two independent native
pair-formula implementations. It does not claim reduced scientific work or a
speedup merely because output closures are shared. Complete CPU endpoint
measurements must separate the incremental effect from #1645/#1646, and no
real-device GPU performance claim is possible on this CPU-only runner.

## Revisit when

A shared local-scale IR or proof-carrying screening/output-admission contract
can preserve the existing backend domains, or independent qualification permits
changing rVV10 representation/order. Future performance transformations should
be applied to the shared expressions and qualified separately for each target.

## Qualification results and reproduction

Baseline is merged #1646/master
`076bdf1e882901668395a5d919bc15e1506dac46`, tree
`06428b6465809d713d8a982abc7c5af24d583ab5`. Its copied final native library has
SHA256 `82b71748c4e32c78707c755eb538f208bad0b622b373ae83a8720bd07d4e71ff`.
Measured candidate code is local commit `893b20b820c8c3cb48bad56fff665802ba065300`;
subsequent edits only append this qualification record. Candidate library SHA256
is `46cc71f740038d1a29719ef253706e0e51d60cdbe62be7e2f12ba81c935db8a7`.
Rebasing from the identical local #1646 tree onto the actual merged baseline
preserved the complete candidate tree and native binary byte-for-byte.

Builds use GCC 14.2, C++20, RelWithDebInfo (`-O2`), the same existing AOT,
CPU-only/OpenBLAS configuration, and numerical-library threads=1. The shared
runner reports Intel Xeon Platinum 8573C, nine visible CPUs and about 10 GB RAM.
Three alternating fresh-process baseline/candidate pairs run for each case;
each process records cold, two process-warm one-shot calls and a 1% changed
geometry. Imports/build/loading are excluded; preparation and complete SCF are
included. No other builds/tests/oracles/profilers ran during endpoint timing.

All sixty corresponding energies and full KS diagnostic records match exactly.
All timing samples are retained. The medians below are **observations, not an
established speedup**; each cell is baseline / candidate seconds (six warm and
three cold/changed samples per build).

| Case | Cold | Warm | Changed geometry |
| --- | --- | --- | --- |
| Water/STO-3G 12/4/8 | 0.29443 / 0.27956 | 0.27974 / 0.27017 | 0.31057 / 0.28534 |
| Water/STO-3G 16/6/12 | 1.00596 / 1.01007 | 1.01181 / 1.00041 | 1.02458 / 1.01344 |
| Water/STO-3G 24/8/16 | 6.01463 / 5.47090 | 5.53390 / 5.24937 | 5.27577 / 5.30295 |
| Water/def2-SVP 16/6/12 | 4.37561 / 4.21166 | 5.88691 / 4.40189 | 4.79812 / 4.15868 |
| OH UKS/STO-3G 16/6/12 | 0.78363 / 0.74687 | 0.76130 / 0.76146 | 0.73103 / 0.75410 |

Shared-runner variability is substantial. For example, def2-SVP warm samples
span 4.41017–6.60466 s on baseline and 4.21758–4.94380 s on candidate. Its apparent
25.2% median improvement cannot reliably be attributed to the pair migration: a separate
instrumented run measures essentially identical VV10 execution (0.84349 /
0.84397 s). OH warm medians are effectively equal, and its changed-geometry
median is 3.2% slower. Therefore this slice makes **no new CPU speedup claim**.
Its justification is eliminating separately maintained scientific pair formulas
and moving output closure selection into the shared compiler.

Separate instrumented endpoint runs verify identical semantic work and exact
energies/diagnostics; instrumented timings are excluded from the table.

| Case | VV10 calls | Executed pairs | AO calls | AO-point pairs |
| --- | ---: | ---: | ---: | ---: |
| Water/STO-3G 12/4/8 | 11 | 8504480 | 165 | 266112 |
| Water/STO-3G 16/6/12 | 11 | 76107148 | 462 | 798336 |
| Water/STO-3G 24/8/16 | 11 | 545150324 | 1188 | 2128896 |
| Water/def2-SVP 16/6/12 | 15 | 106452024 | 630 | 3888000 |
| OH UKS/STO-3G 16/6/12 | 17 | 51863225 | 459 | 705024 |

Validation on the exact candidate binary:

- Native CTest: 57/57 passed
- Focused nonlocal/WB97M-V Python gates: 215 passed, 6 CUDA skips, including
  complete force/reconverged finite-difference coverage
- Scalar/GFN2 codegen gates: 55 passed; all eight existing generated GFN2 headers
  match the baseline byte-for-byte
- Pair-codegen gate: 64 passed; signed-weight host kernels: 21 passed (both are
  included in the 215 focused Python count, not additional independent totals)
- Five independently reconverged PySCF 2.14.0 / Libxc 7.0.0 RKS/UKS fixtures:
  energy-at-density error <=3.1e-14 Eh, full Fock error <=4.6e-14, density error
  <=8.6e-12
- Same-grid water/STO-3G and def2-SVP independent reconvergence: energy errors
  <=1.6e-13 and 2.8e-13 Eh
- Independent review found no blocker; its additional FMA-enabled host probe
  matched all 61,824 scalar outputs bitwise or by NaN classification
- Compiler/SCF/electronic-structure boundaries, native complexity, source
  identity, CUDA ownership, default-promotion inventory, method manifest,
  evidence, Ruff and clang-format gates passed. Full compiler type checking
  passed with existing allowlisted warnings; changed compiler files are clean

Reproduce with two identically configured builds, changing only the source:

```sh
cmake -S . -B build/cpu-dev -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DGENERATIVEQC_ENABLE_CUDA=OFF -DGENERATIVEQC_BUILD_TESTS=ON \
  -DGENERATIVEQC_BUILD_CLI=ON -DGENERATIVEQC_PYTHON_WHEEL=ON
cmake --build build/cpu-dev --parallel 3
ctest --test-dir build/cpu-dev --output-on-failure --parallel 2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/python"
export GENERATIVEQC_LIBRARY="$PWD/build/cpu-dev/libgenerativeqc.so"
python -m pytest -q tests/python/test_vv10*.py tests/python/test_nonlocal*.py \
  tests/python/test_wb97mv_complete.py tests/python/test_wb97mv_production_composition.py
build/cpu-dev/generativeqc_wb97mv_scf_tests native-scf.jsonl
python tools/verify_wb97mv_scf.py native-scf.jsonl
python benchmarks/cpu_wb97mv_endpoint.py --case water --basis sto-3g \
  --grid 16 6 12 --repeats 2
```

Repeat the endpoint command with each case/grid in the table, in alternating
fresh baseline/candidate processes. Ensure imports select that worktree rather
than another editable installation. Compare every corresponding energy and
complete diagnostic record before interpreting timing. Raw qualification
artifacts remain local; no archive/release was published.

This is controlled-grid, small-system, single-thread CPU qualification. It does
not establish default-grid convergence, large-molecule/multithread or force
performance. At this initial qualification stage, no local CUDA toolkit/GPU or full
Python-suite execution was available. The follow-up below adds an isolated
release compiler comparison; actual device execution remains unavailable.

## Follow-up: performance validation exposed a scheduling issue

The architecture-only change was subsequently subjected to a stronger
performance nonregression check. All raw measurements are retained, including
those that are inconclusive or unfavorable; they are not pooled to select a
preferred result.

### CPU observations and limits

A predeclared second wall-time campaign used fixed binaries, one pinned visible
CPU, balanced baseline/candidate process order, twelve process pairs per water
case and twenty for OH. Each process performed cold, two warm and changed-
geometry endpoints. All **272 corresponding energies and full diagnostics**
matched exactly. Warm calls were collapsed to one process median before paired
log-ratio bootstrap analysis, avoiding treating them as independent processes.

The shared host remained too variable to establish the requested strict
one-sided 95% upper bound of 3% for every case/phase. Every two-sided interval
included zero, but several upper bounds were wide. In particular:

- OH changed geometry: paired median +0.219% time; one-sided upper bound +2.060%
- OH warm: +0.754%; upper bound +5.476%, so the strict bound is inconclusive
- Water/STO-3G medium warm: -0.531%; upper bound +5.476%, also inconclusive
- Water/STO-3G large cold: +5.550%; upper bound +18.378%, not a performance pass

A separate instrumented campaign covered four process pairs for each STO-3G
water grid, eight for def2-SVP and twelve for OH, with cold/warm/changed phases.
All **96 paired endpoint outputs and semantic work counters** matched exactly.
It recorded both component wall and thread CPU time. Variability also affected
unchanged AO work; for example, the four medium-water changed-geometry pairs
had a +31.3% paired AO CPU-time median. Such small diagnostic samples do not
establish causal regressions or precise confidence bounds. OH changed-geometry
total CPU time was +0.237% with one-sided bootstrap upper bound +2.436%.
**These data do not establish a universal <3% endpoint nonregression bound.**

Independent compiled-code inspection provides a narrower, stronger fact: the
bounded CPU weighted-VV10 E/V hot loop has exactly the same normalized
69-instruction sequence in baseline, initial shared emitter and output-ordered
emitter. A fixed register bijection, stack-slot relocation and corresponding
branch labels explain all differences. All have three FP divides, 25 reads,
five conditional branches, no calls and no FP spills. Encoded loop lengths are
346 / 347 / 336 bytes. This establishes no added hot-loop work or dependencies,
not a complete-endpoint wall-time guarantee.

CodSpeed for initial published head `3e18722` versus `076bdf1` classified all
three measured benchmarks as untouched: WB97M-V 2.5/2.5 s, RHF 474.9/475 ms,
and PBE 6.7/6.7 s (rounded displays), with 5% reporting thresholds. These are
**CPU Simulation**, despite the benchmark function's walltime name. The page
warned of different runtime CPUs (EPYC 7763 versus EPYC 9V74); eight skipped
benchmarks reused their baselines. This is supporting evidence only.

### Release CUDA compilation and the actual fix

An evidence-local NVIDIA CUDA 12.9.1 redistributable prefix supplied NVCC
12.9.86, matching CI, plus cudart/CCCL headers and binary inspection tools.
Every official archive's published length and SHA256 were verified against
`https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json`.
No system/repository headers, compiler defaults, driver or workflow changed.

The complete native CUDA translation unit was compiled at the same source
filename/include roots for baseline and candidate, using C++20, `-O3`,
`--Ofast-compile=0`, `-arch=sm_120`, `--fmad=true`, `--ftz=false`,
`--prec-div=true` and `--prec-sqrt=true`. Local GCC 14.2 differs from CI GCC 13.3.
A documented evidence-local copy of glibc's `bits/mathcalls.h` suppressed only
its new sinpi/cospi declaration macros while under NVCC, addressing the known
CUDA12.9/glibc2.41 noexcept conflict (NVIDIA/cuda-samples#378). Toolkit headers
and all used arithmetic stayed unchanged; neither PTX referenced these names.
The same overlay was applied to both revisions and original/patched hashes
were retained.

The initial depth/hash emission order **did** produce a static performance risk:
unmasked rVV10 geometry-only increased from 68 to 74 registers. NVIDIA's
occupancy model at the existing 128-thread launch predicted seven to six
blocks/SM (58.3% to 50% occupancy upper bound). Compilation success alone would
not have detected this. The initial candidate is therefore not the selected
final emission schedule.

The fix adds an opt-in `output_dependency_order` to the existing scalar emitter.
It reuses Program's iterative topological traversal, starting from the explicit
phi, feature-omega, feature-kappa, radial output order. Shared definitions emit
once; operands, mathematical serialization, CSE and failure/publication
semantics remain unchanged. Defaults remain byte-identical, checked across
632 old/new emission comparisons. This is a generic compiler scheduling option,
not a backend-specific copy of the pair equations.

The implementation checkpoint is `32a32a37c00583ab29355b4722f320e2a4cba402`;
`262500c20ed1d06b1e3b9caec79737d8449a3075` only adds tests. For this schedule:

- All 27 compiled kernels retain identical register, stack, spill, shared/local
  and constant-memory usage; the 68-register/seven-block bracket is restored
- All eight VV10 pair specializations and eleven non-pair kernels have identical
  machine instruction bytes (19 of 27 entries)
- The eight rVV10 kernels retain machine-code differences with the same static
  arithmetic counts/resources. Geometry-only changes are six commutative DMUL
  source-operand swaps per kernel; other modes change scheduling/register
  allocation and some MOV/NOP instructions
- All 27 constant banks and parameter ABI records are unchanged; instruction-
  offset metadata moves where corresponding EXIT instructions move
- All 22 native code-relocation sections are empty/unchanged. Mercury
  relocations retain symbol/type/count while code offsets/addends move; six
  opaque Mercury code-finalization sections remain different and are not
  certified as harmless metadata or globally equivalent

This removes the identified register/occupancy regression and establishes
byte-identical VV10 device kernels for the qualified compiler/architecture.
It does **not** establish whole-cubin equality, rVV10 machine-code identity,
real-device numerical execution or GPU endpoint timing. CuMetal PR probes do
not execute VV10, and the existing release grid/XC resource report excludes
its pair kernel, so those CI jobs are not substitutes for these missing gates.

The final ordered CPU library is
`77043c8d5ad1af4693e728e3bca87cc1357b5867317275ea58d0c67317424823`.
The earlier timing campaigns used `46cc71f7...`; the compiled-loop comparison
covers both binaries. Final validation on that ordered library passed 57/57 native tests, 303 focused
Python tests (6 CUDA skips), and the five matched-grid PySCF fixtures with the
same error bounds above. An updated host FMA-enabled comparison passed all
61,824 outputs. Final endpoint replays refer to this library explicitly.
No CPU/GPU speedup is claimed for this change.
