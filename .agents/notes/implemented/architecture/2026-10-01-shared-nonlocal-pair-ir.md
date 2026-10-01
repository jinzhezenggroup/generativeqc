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
performance. Local CUDA toolkit/GPU and full Python-suite execution are absent;
CUDA compiler/device qualification remains a separate CI/real-device gate.
