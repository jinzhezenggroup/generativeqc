# Decision: bind one UCCSD iteration policy to spin-aware providers and executors

Status: implemented
Date: 2026-10-12

## Problem

The shared UCCSD graph in PR #2316 supplies fixed-amplitude equations but not a
converged correlated endpoint. Restricted provider shorthands and pair-metric
packing cannot be reused for unrestricted coordinates. Separate CPU/CUDA
iteration algorithms would also duplicate convergence and failure policies.

## Decision

Reuse `UHFReferenceSnapshot`, `SpinMOBlock`, `ConventionalProvider`, CC
`SolverOptions`, the coordinate-independent CC DIIS algebra and both native
TensorIR executors. One owner converts spin-labelled Coulomb blocks to physical
antisymmetrized inputs and iterates raw packed excitation representatives.
Canonicalize only chemists' eightfold integral permutations, retaining slot
spin ownership in every key. Never substitute restricted pair symmetries.

CUDA is an explicit staged, host-controlled contract: four-center transforms and
CC contractions run on CUDA; Fock transforms, immutable feed/checkpoint storage,
denominators and excitation-state DIIS remain on the host. The reference and AO
source are explicit producers. No CPU equation or PySCF fallback is permitted.

Admit the complete provider transition under its existing lock before any AO
read. Mirror actual retention through one `BlockPlan.retained_bytes` property.
Download/copy every requested MO block before an LRU eviction can close its
device owner. Prepared feeds then survive provider/source closure independently.

## Investigation and rejected alternatives

The first GPU qualification found that counting CUDA array/workspace allocation
alone admitted a pin set that failed only after AO reads: actual provider
retention also charged each block's 96 MiB library allowance. This must not be
fixed by increasing the qualification budget or ignoring the allowance. The
shared retention property and explicit bounded LRU path fix the admission
definition. A host-only no-device gate now detects the same mismatch.

Also rejected:

- Reuse the restricted `_provider_blocks` cache-key dry run: it omits spin keys.
- Apply restricted Frobenius weights to unrestricted DIIS: packed determinant
  excitation coordinates have their own unweighted metric.
- Project arbitrary supplied full amplitudes to repair their antisymmetry:
  initial-state admission remains exact; residual representative selection
  only accommodates arithmetic roundoff in physical equations.
- Treat proposal norms or shifted residuals as scientific convergence: full
  physical residuals, energy change and a fresh selected-backend replay gate
  acceptance.
- Hide a CPU fallback behind a CUDA executor or mutate a borrowed provider's
  cache policy automatically when pin admission fails.

## Invariants

- Independent alpha/beta populations and signed same-spin representatives.
- Same physical graph and iteration policy across CPU/CUDA.
- Denominator/complete-cache/storage admission before AO reads.
- Provider and solver budgets remain explicit; no process-RSS claim.
- Immutable last-finite outputs and diagnosed nonconvergence.
- Shared UHF response ownership remains intact for later Lambda/forces.

## Evidence and consequences

`tests/python/test_uccsd_solver.py` qualifies complete molecular convergence,
restricted-limit agreement with the existing RCCSD solver, independent
determinant projections with equal populations/distinct spin bases, admission,
level-shift/damping invariance, bounded LRU and resource lifetimes. Its real-GPU
molecular tier runs through finite Slurm allocations on n1 and exercises CUDA
MO transforms as well as iterative CC contractions. Raw qualification output
stays under ignored `.artifacts/`, not tracked benchmark/publication assets.

Source-matched CPU qualification also builds `libgenerativeqc` and exercises
native UHF export plus streamed native shell tiles for H2, Li and OH. All three
complete energies pass independent PySCF gates. The oracle shares only exact
shell primitives: Li's separately rounded named STO-3G table initially differed
by 2.43e-8 hartree at HF. Comparing the same supplied basis fixes the Hamiltonian
identity without relaxing the 2e-9 HF or 3e-9 correlated-energy gates. The broader
host equation/layout/solver/provider/TensorIR suite passes 212 tests with seven
explicit optional/CUDA skips.

The final source-matched CUDA solver run passes three complete Li/NH2/broken-H2
gates on n1 in Slurm job 7681 (106.94 seconds for the pytest run, not a solver
benchmark). Job 7678 selected a system pytest interpreter without the oracle
dependencies and skipped all gates; it is not qualification evidence. Select
the qualification environment explicitly with `.venv/bin/python -m pytest`.

This is an internal conventional UCCSD energy/solver owner, not native
Calculator registration or a resident-performance claim. #1822 still requires
tiled unrestricted triples and #1824/#1825 require solved Lambda, explicit
excitation-dual pullbacks, shared UHF response and complete forces.

## Revisit when

A provider shares one CUDA handle/library allowance across multiple blocks, or
resident/tiled UCCSD evaluation changes storage ownership. Update the one
retention authority and executable endpoint gates, rather than introducing a
second cache arithmetic or silently changing the staged contract.

## References

- #1821, #1822, #1824, #1825; PR #2316.
- [Shared equation decision](2026-10-12-uccsd-spin-summed-tensorir.md).
- `docs/developer/uccsd_solver.md`.
