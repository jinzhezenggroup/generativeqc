# Proposal: gradient-first application-budget Fast

Status: proposed; source review only, not a qualified runtime change
Date: 2026-10-13

## Why revisit the implementation

The user supplied `z5476t4508/gpu4pyscf-mixedprec` as an example of useful
mixed-precision acceleration. Its important distinction from our current Fast
is **which work is narrowed**, not merely a larger benchmark molecule. Our
current PBE0 route narrows provisional XC work but leaves J/K and every analytic
force in FP64. The external project narrows DF contractions and first-derivative
integral/contraction work as well. The earlier negative scaling result only
characterizes our XC-only route; it is not evidence against this broader design.

This proposal complements, rather than replaces the evidence in
[the size-only expansion rejection](../rejected/2026-10-13-fast-size-only-expansion.md).
Production admission and the current Fast behavior are unchanged by this note.

## Upstream snapshot and evidence limits

Reviewed default branch `mixed-precision-hf` at
`0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862` on 2026-10-13. The repository is
Apache-2.0; this investigation copies no implementation into production.
Sources were read from commit-pinned GitHub API responses, retained under ignored
`.artifacts/gpu4pyscf-mixedprec-study-20261013/`. The following are **upstream
reported measurements**, not benchmarks reproduced by GenerativeQC:

| Upstream experiment | Reported result | Scope |
| --- | --- | --- |
| Tamoxifen, DF-r2SCAN/def2-TZVPP, 1274 AOs | Gradient 8.01 to 1.18 s, 6.80x; max error about `1.1e-5 Eh/bohr` | Gradient only |
| Same representative system | SCF + gradient 33.72 to 15.24 s, 2.21x | Sum of their SCF/gradient timings, not our construction-to-release endpoint |
| DF-r2SCAN/def2-TZVP, 1877 AOs | Gradient 7.1x | Larger pure-functional case, not PBE0 |
| DF-B3LYP table, medium/large systems | Gradient approximately 2.3–3.4x | Hybrid-functional comparison; still not measured PBE0 |
| Healthy SCF cases | Typically 1.2–1.8x | FP32 early work, FP64 tail |
| Azadirachtin/TZVP, HF/B3LYP SCF | 0.32x / 0.42x | Divergent FP32 work followed by a successful but costly strict fallback |

The benchmark report describes DF with `def2-universal-jkfit`, default XC grids,
and `conv_tol=1e-10`. Our recent PBE0 campaign instead uses exact non-DF J/K,
an explicit unpruned moving grid and full grid-response forces. Neither DF nor
omitting a response contribution is an arithmetic-precision optimization.
Do not silently change our Hamiltonian, auxiliary basis or force contract to
reproduce their timing. Their main matrix does not contain PBE0.

The report's gradient-error summary is not a universal bound: its detailed
table includes values up to `3.3e-5 Eh/bohr`, some larger CPU reference cells
are absent, and a small force discrepancy does not prove the same optimizer
trajectory, transition state or long-time MD behavior. The full raw benchmark
blob is 55,634,041 bytes (`15eee85ea13253230cbbd82162a453020eabe98c`); it was
not downloaded/recomputed in this source review. Do not imply its arrays were
independently revalidated.

One reproduction caveat requires attention before quoting new timings: the
report says SCF cold-start sampling was corrected, but the current `gpu_eg`
function in `mixedprec/benchmark/run_benchmark.py` reuses one `mf` object across
`mf.kernel()` calls and takes their median. Historical records may have been
repaired separately; this script alone does not establish that every timing is
cold. Use fresh-owner, synchronized endpoints for our comparison instead of
assuming its prose and current driver have identical timing semantics.

## What the code actually does

1. **DF J/K:** `gpu4pyscf/df/df_jk.py:get_jk` narrows occupied-density factors,
   streamed CDERI blocks, intermediate contractions and J/K buffers to float32.
   It converts returned matrices to float64. A float64 return value is therefore
   not evidence of float64 inner arithmetic. `_cast_cderi` converts a contiguous
   parent before transposing rather than paying for a badly strided cast.
2. **Gradient integrals:** `lib/gvhf-rys/ejk_int3c2e_ip1_f32.cu` uses float
   recurrence/shared buffers, density operands, per-thread derivative sums and
   block reductions; final atom/auxiliary-atom outputs use double atomics.
   Its tabulated-root Clenshaw recurrence remains double, despite an older
   header comment suggesting all interpolation is float. Read the executable
   expressions, not the file name or header, to establish precision.
3. **Sensitive DF preparation:** `df/grad/rhf.py` deliberately retains metric
   solves and derivative-density precontractions in float64. Both J-only and
   J/K consumers dispatch the narrowed first-derivative kernel. Narrowing only
   one branch would miss pure or hybrid consumers.
4. **XC:** `dft/numint.py:_eval_rho2` forms density through occupied MO products
   and a sum of squares, using float AO/MO contractions but double density
   outputs for XC evaluation. `grad/rks.py` narrows large AO contractions while
   accumulating gradient outputs in double. This is not an all-FP32 AO/libxc
   pipeline. The sum-of-squares density representation is worth investigating
   separately from our dense AO–density contraction: do not assert it fixes
   our 384-AO retry until the actual failing operator/inputs are identified.
5. **SCF versus gradient policy:** `scf/hf.py` runs the auto lane provisionally
   in FP32 and then strictly; it has a divergence-triggered handoff. The
   gradient owner's `auto` resolves independently to FP32, rather than inheriting
   the strict mode that SCF used at exit. Current wrappers restore the global
   mode with `try/finally`; an older STATUS warning about missing restoration
   is stale. We should retain our per-owner policy, not adopt a process-global
   precision switch.

The most useful reported ablation is gradient 8.01 s FP64, 4.04 s with mixed XC,
2.42 s after adding the J derivative kernel, and 1.18 s after removing double
inner sums that were immediately narrowed into float shared storage. It supports
profiling the real derivative arithmetic and reductions, not just changing
inputs or a mode label. It does not justify indiscriminately narrowing every sum.

## Proposed implementation order in GenerativeQC

### 1. Qualify low-precision analytic gradients independently of SCF

Start from the same FP64-converged density/orbitals to separate gradient-kernel
error from SCF trajectory changes. Narrow the expensive first-derivative integral
and XC contractions where they buy measured time; retain sensitive root/metric
work and global accumulation as justified by cancellation and endpoint tests.
Connect admission to the requested force budget, not to a preset name alone.
Retain strict fallback and honest per-operator compute/accumulate/storage counts.

Use existing compiler algebra and precision planning, not a second hand-copied
Rys implementation. Exact four-center force consumers live under
`python/generativeqc_compiler/integral/lowering/force_*`; explicit DF consumers
have `integral/df_derivatives_cuda.py` and
`src/scf/cuda/df_force_response.cpp`. Their scientific models and qualification
records stay distinct. A successful DF implementation cannot silently replace
exact J/K, and a three-center speed ratio cannot predict a four-center ratio.

### 2. Extend provisional heavy contractions, not just XC point arithmetic

For callers explicitly choosing DF, investigate FP32 J/K contractions with a
strict final SCF and bounded retries. Keep reusable high-precision CDERI data
for the strict tail; avoid repeat casts on noncontiguous views. Independently
evaluate a generated mixed exact-J/K route for non-DF callers. Preserve the
same density, geometry, screening and source-lifetime identities across reuse.

Do not promote the external screening-only FP32 CDERI-build option to default
Fast: the source itself separates that option, and reports substantially larger
energy/gap errors. Similarly, FP32 SCF divergence is a real endpoint cost, not
something to omit from the timing or fix with molecule-name exclusions.

### 3. Add optimizer-aware effort only under an explicit application contract

An optimization endpoint may eventually relax early SCF work and tighten near
the final geometry. Keep this separate from arithmetic-only benchmarks and
from explicitly requested fixed SCF tolerances. Upstream's gradient-error
coefficient `0.07*sqrt(conv_tol)` is a calibration on one molecule, not a
transferable mathematical bound. Its actual scheduler can return a larger
tolerance than the configured floor; do not repeat its comment that this
always preserves the caller's original strictness.

Use final strict validation and any requested audits without two unconditional
full SCF/force solves on every supported Fast endpoint. Empirical admission
still must not be presented as a measured per-call error certificate.

## Acceptance before promotion

- Compare FP64 and mixed derivatives at an identical converged state; then
  compare independent full energy/force endpoints and actual optimization work.
- Test occupied orbital representations, cancellation-prone/diffuse cases,
  J-only and hybrid paths, coordinate derivatives and basis/auxiliary response.
  Include force/energy consistency and translation tests; MD needs its own gate.
- Use explicit application-budget force gates. Our prior `1e-6 Eh/bohr` gate
  qualified strict final gradients; it is not evidence that an FP32 gradient
  with an allowed `1e-4` budget must also deliver `1e-6` accuracy. Never weaken a
  caller's explicitly requested accuracy to get a passing sample.
- Report all first calls, fallbacks, Fock/force counts, conversion work, memory
  peaks and complete endpoint times. Use chemically diverse cases beyond water
  replicas, including the 208/537/1042/1877-AO size regime where practical.
- Diagnose our 768-AO generated-force OOM through allocation/liveness evidence;
  do not call it an inherent FP32 or size limit. Upstream's corrected Hessian
  buffer budgeting is useful precedent, not a diagnosis of our different path.
- Run all real-device work through finite Slurm allocations. No GPU reproduction
  or new production precision path was executed as part of this source review.

## Pinned references

- [Upstream benchmark report](https://github.com/z5476t4508/gpu4pyscf-mixedprec/blob/0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862/mixedprec/benchmark/REPORT.md)
- [Gradient ablation and SCF-plus-gradient example](https://github.com/z5476t4508/gpu4pyscf-mixedprec/blob/0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862/mixedprec/STATUS.md#L969)
- [Derivative kernel arithmetic](https://github.com/z5476t4508/gpu4pyscf-mixedprec/blob/0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862/gpu4pyscf/lib/gvhf-rys/ejk_int3c2e_ip1_f32.cu#L298)
- [DF J/K compute and conversion](https://github.com/z5476t4508/gpu4pyscf-mixedprec/blob/0a4cf33f0db146ef3e04ac97f2ce1f2deeaeb862/gpu4pyscf/df/df_jk.py#L509)
- [Our current XC-only arithmetic](../implemented/numerics/2026-10-13-budget-selected-fp32-xc.md)
