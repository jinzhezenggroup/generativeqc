# Decision: share resident spatial XC point work across functionals and AO pairs

Status: implemented
Date: 2026-11-01

## Problem

Prepared spatial CUDA XC admitted only canonical LDA/PBE by name in several
layers. Other compositions downloaded features/jets for CPU contraction, even
when the scalar compiler already represented their expressions. The CUDA local
potential kernel also reevaluated the same nonlinear point expression for every
spin/AO pair, in addition to its energy evaluation. Bounded memory concealed
`points * (1 + 2 * active_ao**2)` scalar evaluations.

Native r²SCAN is not a drop-in substitute: its production tail continuation is
different from the prepared compiler's strict `interior-v1` contract. The native
stationary geometry consumer also borrows r²SCAN feature-only grid views without
using the grid's point evaluator.

## Decision

The XC compiler lowers an optional exact `FunctionalSpec` through the existing
backend-neutral semilocal scalar emitter. It supplies a typed `GridXcKernel` to
DFT; DFT does not import XC mathematics. The artifact exports its full scientific
identity, and the prepared XC owner resolves that identity before lending grid
buffers. Scheduling accepts the resulting capability instead of maintaining
another name whitelist.

The initial extension covers linear compositions of the nine audited
`PUBLIC_COMPONENTS` (LDA/PW, PBE, SCAN and r²SCAN), which share the implemented
interior checks. Other families retain their existing route. Restricted inputs
use the polarized evaluator only after checking equal spin density, gradient
and, for meta-GGA, tau. The ordinary canonical LDA/PBE entries preserve their
existing numerical implementation.

Evaluate energy and coefficients once per point into eleven FP64 rows. Energy
reduction and every AO pair then read the same bounded tile panel. Tau assembly
uses one half of the scalar tau derivative. Numeric allocation and schedule
liveness account for `88 * tile_points` extra bytes, including the minimum
one-AO/one-point allocation; no full molecular AO grid is retained.

The grid view ABI remains v1. XC evaluation keeps its v2 call layout and adds
an artifact-bound evaluator key alongside its two existing keys. The current
ingredient-driven feature leases still serve independently admitted stationary
geometry consumers, including r²SCAN. They do not admit name-only point
evaluation. This distinction avoids coupling production geometry to an optional
interior evaluator. The point panel is charged in the shared native
`cuda_grid_storage` authority as well as the Python tile and liveness models.

## Rejected alternatives

- Expanding name lists alone would select missing evaluators or the wrong
  expression for modified coefficients.
- Reusing the native r²SCAN production evaluator would silently change the
  prepared numerical domain.
- Putting XC imports into DFT would reverse the compiler dependency boundary.
- Reusing the contracted AO work panel for all point coefficients would overrun
  the minimum single-AO allocation; dedicated charged storage is bounded and
  leaves feature/jet leases intact.
- Claiming SCF or force speedups from this fixed-density endpoint would exceed
  the measured consumer contract.

## Invariants

- Source generation loads no GPU, public runtime, PySCF or Libxc oracle.
- Exact composition and domain identity govern admission; mismatches retain the
  existing explicit host route when its ingredient contract is complete.
- An explicit incompatible device schedule fails at preparation. Runtime GPU
  failures propagate and do not become host fallback timings.
- Preserve signed weights, spin conventions, both gradient legs, tau's one-half
  factor, deterministic point reduction, FP64, and disabled FMA.
- Preserve density/orbital lease lifetimes, invalid-input rejection and recovery.
- This adds no SCF, force, response, exact-exchange or nonlocal capability and
  does not change native KS point scheduling or mixed-precision admission.

## Evidence

The host adapter executes the exact emitted point source against independent
PySCF/Libxc coefficients, including all nine components and six representative
compositions. Real CUDA tests execute the complete prepared energy/potential
consumer against independent analytic Gaussian jets and Libxc, with CPU
scientific entry points disabled during device execution. They cover both spin
layouts, density matrices and orbital factors, one-AO/one-point and ragged
tiles, noncontiguous local maps, density directional differences, wrong artifact
identity, missing tau, expired leases and invalid-density recovery. The normal
GPU coefficient/potential gate is `atol=rtol=5e-10`; density directional checks
use two steps and `3e-8` absolute error.

The independent oracle is PySCF 2.14.0 with Libxc 7.0.0. After porting to
master `28d540736f772e305e7fb31f61cf505741de9443`, validation passes:

- 105 host/codegen/admission tests, with two installed-native-AO tests excluded;
- 529 runtime publication/creation cases and 13 lifecycle/precision cases;
- 120 real-GPU endpoint cases on node1, Slurm job 15065;
- five compute-sanitizer memcheck cases on node1, with zero errors; and
- the complete pre-commit hook set, including compiler dependencies, native
  ownership, CUDA inventory, evidence retention, Ruff and clang-format.

A final liveness correction retains density features through electron-total
reduction; all 21 schedule cases pass again. Runtime JIT compilation used verified
sccache 0.16.0, with before/after statistics retained. The first allocated attempt
could not load libcublas because the toolkit library directory was absent from
the validation environment. Setting its loader path and reusing compiled caches
produced the complete passing run; no device-visibility override was used.

### Fixed-density endpoint comparison

Baseline is `28d540736f772e305e7fb31f61cf505741de9443`. Slurm job 15066 on node1,
RTX 5090, CUDA 12.9.86 compares the old and new complete prepared E/V calls for
32 analytic Gaussian AOs, 4096 points, two spin densities, the full collocation
feature mask, no screening and 128-point tiles. Each warm median has five
executions after one cold call;
timing includes source handling, collocation, XC, assembly, transfers and
statistics. Independent reference work is outside timing. Compilation,
preparation and cold execution are recorded separately in the retained JSON.

| Functional | Previous warm median | New warm median | Previous/new scalar evaluations |
| --- | ---: | ---: | ---: |
| LDA_XC_PW | 598.431 ms | 13.384 ms | 8,392,704 / 4,096 |
| PBE | 920.597 ms | 17.247 ms | 8,392,704 / 4,096 |
| r²SCAN | 66.979 ms | 26.000 ms | 4,096 / 4,096 |

The r²SCAN baseline is its existing host XC route; LDA/PBE compare two CUDA
routes. LDA/PBE output errors against the independent reference are unchanged;
all three new energy errors are at most `5.83e-11` Eh and maximum potential
errors at most `1.55e-11`. Device allocation rises from 4,674,816 to 4,686,080
bytes, exactly the 11,264-byte point panel. These are synthetic fixed-density
results, not complete SCF or molecular-force performance claims.

Ignored matched-baseline evidence lives in `.artifacts/spatial-xc-pr-evidence/`,
including the six JSON measurements, CUDA JUnit report, sanitizer log,
`pr_spatial_xc_benchmark.py`, `pr_spatial_bench_inputs.py`, and finite Slurm
benchmark/sanitizer runners. Baseline runtime SHA256 is
`8032364671bd403258fe374e89b1450d9d500242ee81e189b8145efc2ee6f0a6`;
new built-in runtime SHA256 is
`fdab2ce06795b86ef4c760b204d052fc15933ae70e6fa4f80714c766114191c2`;
new r²SCAN runtime SHA256 is
`01f8fc02eec1590da8590525d9e465e7957dc07947fa115d6040d9cf0237cfca`.
Earlier evidence against the original pre-port baseline remains separately under
`.artifacts/spatial-xc-evidence/` and is not the source of this table.

## Revisit when

Additional functional families need device interior-domain checks, or a caller
requires response/geometry outputs from this prepared consumer. Qualify those
contracts independently before extending admission. Thread/tile tuning and
family-specific coefficient compression are separate optimizations; retain
complete endpoint timings and semantic scalar-work counts when comparing them.

## Subsequent extension

[Libxc spatial point programs](2026-11-01-libxc-spatial-point-programs.md) extends
the admission boundary to existing source-derived work-domain lowerers without
changing this interior contract or the common point cache/assembly.

## References

- `docs/developer/xc_contractions.md`
- `docs/maintainer/performance_engineering.md`
- `tests/python/test_spatial_xc_codegen.py`
- `tests/python/test_spatial_xc_cuda.py`
- `.agents/notes/implemented/architecture/2026-09-23-dft-shared-semilocal-cpu-cuda.md`
