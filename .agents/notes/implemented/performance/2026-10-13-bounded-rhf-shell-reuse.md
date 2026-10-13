# Decision: preserve whole-shell recurrence reuse in bounded GPU RHF

Status: implemented
Date: 2026-10-13

## Problem

The 434-spherical-AO QM7 C7H16/cc-pVTZ reference preparation spent essentially
all its GPU time in the generic bounded Fock consumer. The original diagnostic
completed 14 Fock kernels totaling 1312.588 GPU seconds before a 25-minute Slurm
timeout; it did **not** finish RHF. SCF convergence was not established by that
trace and MP2 never ran.

Two routing facts explain the work rather than an eigensolver/DIIS bottleneck:

- This particular `portable_cuda` build has zero generated Fock-class coverage.
  Do not describe its measurements as tuned-AOT performance.
- The optional canonical ERI source requires 55,559,675,520 bytes for this
  geometry, exceeding its 8-GiB allowance. Refusing it is correct. However, the
  bounded fallback then prepares Hermite/Boys/Coulomb recurrence separately for
  Cartesian AO components. Whole-shell reuse already exists in the compiler
  and in fixed-topology/independent J/K consumers, but was not connected here.

The bounded route is required for large reference topology. Removing its
resource guard or just increasing the quartic cache limit does not fix it.

## Decision

For uncovered full-range FP64 classes of total angular order 3--12, stream the
existing class-major shell-pair topology directly into the compiler-owned
`contract_materialized_direct_pair_fock` consumer. One CTA owns a whole shell
quartet, sharing preparation across every live component. A 64-bit grid-stride
domain avoids a topology-sized task arena and a 32-bit persistent-counter
overflow. The existing component-lane/slot bounds cover *all* s/p/d/f
compositions, including fddd and ffff tails.

Generated classes keep their current owners. Each newly consumed class is
excluded from the generic fallback, which still handles remaining classes.
The existing `GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=0` control restores
that fallback for A/B qualification. Missing primitive-pair storage, mixed
precision and incompatible recurrence modes also retain the original path.

## Invariants

- No change to exact RHF versus DF-RHF, screening tolerances, convergence
  tolerances, public basis representation, primitive coefficient order or force
  equations. No production CPU/oracle or reference-density seed is introduced.
- Borrow the existing geometry-bound primitive cache; no new global numeric
  allocation, four-index tensor or extra admission budget. Shared recurrence
  remains bounded by the compiler's existing 48-KiB static limit.
- Preserve canonical pair orientation, per-shell density/Schwarz screening,
  component-level screening, RHF/UHF scatter, and compensated reference output.
- Inactive and screened lanes participate in shared publication/retirement.
  Warm geometry updates use the existing cache rebuild and graph ownership.
- Class timers count accepted shell quartets, not primitive recurrence work or
  AO-component counts. Complete endpoint counters remain SCF/Fock actions.

The existing optional progress journal also records completed split-provider
iterations at their already-required stream fence. It adds no GPU transfer or
synchronization. Failed submission/solver/fence observations are not completed
iterations; the callback's iteration index is already one based. Fully
device-tail iteration retains its existing final completion counters.

## Qualification

The input is QM7 index 5604 (zero based), not an arbitrary molecule sharing
the formula C7H16. Promote the dataset's float32 Bohr coordinates unchanged to
float64. There are 23 atoms, 58 electrons and 434 spherical/485 Cartesian AOs.

- Dataset SHA256:
  `fd6c26c1086887310518e6a6da778253bc5d3ceb8166007a679a740fff9ee6ea`.
- Geometry/basis input SHA256:
  `696b783d661883745a71f15cb529e83802773d7ace6cbaf38f3e69f2e6cc16d0`.
- Baseline source: `c67a913dfeb7297725aad3dc42fc4c727da71d92`.
- Baseline library SHA256:
  `36966c4d7338b1b7e281dc239aa7d0e6281fe108683577adf6c369dba2d7b0c9`.
- Independent PySCF 2.14.0 conventional unscreened RHF energy:
  `-274.496819172566 Eh`; its retained density and orbital spectrum are oracles,
  not native inputs.

`tests/python/test_rhf_bounded_materialized.py` checks full E+F against PySCF
with spdf shells and signed contractions, RHF/UHF, Cartesian/spherical bases, two
screening tolerances, both scheduling choices, batched warm replay and a changed
geometry. The public HF API requires positive screening; the C7 physical
reference qualification separately exercises exactly zero screening.

The clean baseline completed on node4/RTX 5090 in Slurm job 805: 2719.5223 s
including native setup and physical-reference publication, 29 SCF iterations,
29 exact Fock builds, zero extra final Fock builds and no execution retries.
The automatic provisional DF guess was refused by its independent 512-MiB
budget, so no DF iterations or externally supplied density entered this run.
Absolute errors versus the independent oracle were 5.69e-14 Eh in energy,
1.74e-11 in density and 6.88e-11 Eh in the orbital spectrum.

The portable candidate completes on the same physical RTX 5090 in job 807:
1348.6332 s, again 29 iterations/29 exact Focks, no extra final Fock and no
execution retry. Energy/density/orbital-energy errors are 5.12e-13 Eh,
2.69e-11 and 5.65e-11 Eh. The 2.0165x endpoint improvement removes redundant
generic preparation, but approximately 46.5 s per iteration remains too slow;
this is not evidence that tuned production RHF is optimized.

On node5, two interleaved ethane/cc-pVTZ (144 spherical AOs) controls take
4.00471/3.88644 s before and 3.89636/3.89780 s after. Each retains 13 native
provisional DF iterations and 12 exact Focks; the natural resident-source
route is unchanged. All four results have the same independent maximum
energy/density/orbital-spectrum errors, 6.97e-13 Eh/1.44e-11/4.18e-11 Eh.
These two samples establish no meaningful small-case speedup.

Portable numerical qualification passes 16 independent E+F/DF-CBS tests on
node5 in job 1498, including the new bounded RHF/UHF cases and the existing
DF-MP2/DF-CBS force/reuse gates. Filtered new-kernel memcheck passes with zero
errors (node2 job 2941). The full E+F racecheck reaches its 20-minute limit and
is not a pass; a focused two-iteration spdf probe passes racecheck with zero
hazards (node2 job 2943). That probe is not a converged endpoint.

Acceptance gates for the same C7 input are 2e-9 Eh in energy, 3e-8 in maximum
density error and 3e-8 Eh in maximum orbital-energy error, plus the native
physical/canonical residual checks and valid complete Fock-action counters.
The small public E+F gates are 2e-9 Eh and 2e-8 Eh/Bohr, respectively.

The pinned input can be reproduced without a reference SCF:

```sh
PYTHONPATH=python:. python -m benchmarks.df_hf_preconvergence prepare-case \
  --case qm7-c7h16-tz --qm7 /path/to/qm7.mat \
  --output .artifacts/rhf-bounded/c7-tz.input
```

Compile `benchmarks/df_hf_preconvergence.cpp` against the qualified CUDA library
and run `df-hf-preconvergence INPUT - OUTPUT auto-direct hf` inside a finite
`srun --partition=main --gres=gpu:5090:1` allocation. In this mode the unused JK
file argument supplies no data. Keep `GENERATIVEQC_PROFILE=off`, native FP64,
zero screening, E/density tolerances 1e-12/1e-11 and the exported 64-GiB numeric
budget unchanged. The input exporter reproduced the measured bytes exactly.

Raw profiles, receipts and binaries stay in ignored `.artifacts/` and the
qualification workspace rather than increasing the tracked evidence budget.
All device execution, including profilers and sanitizers, uses finite Slurm
allocations. Complete endpoint timing must be distinguished from the bounded
two-iteration, nonconverged profiling probe.

## Revisit

Profile actual generated-class gaps on tuned builds before changing their
dispatch. Further optimizations should target measured remaining classes or
source reuse without replacing exact reference semantics with density fitting.
