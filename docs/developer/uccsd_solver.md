---
orphan: true
---

# Canonical UCCSD solver and provider boundary

`tools.generativeqc_cc.PreparedUCCSD` supplies an internal conventional,
real-FP64 UCCSD energy/iteration owner. CPU and ordinary-stream CUDA use the
same [spin-resolved energy/residual TensorIR](uccsd_equations.md), the same
packed iteration policy, and the existing conventional integral provider.
This API does not register a native Calculator method, implement DF, or supply
unrestricted triples, Lambda or analytic forces.

## Reference and execution admission

Supply a validated `UHFReferenceSnapshot` and an identity-matched
`ConventionalProvider`. The existing UHF snapshot/response owner remains the
authority for canonical orbitals, real alpha/beta populations, precision,
overlap and SCF validation. RHF snapshots and UKS references are rejected;
frozen/truncated spaces, complex/spinor references and screened/DF Hamiltonians
are not silently relabelled as supported conventional UHF inputs.

`SolverOptions` is shared with the existing CC solver. The default tensor
executor is `NativeCCTensorExecutor` and requires a CPU provider. CUDA requires
an explicit `CudaCCTensorExecutor`, a CUDA provider with an explicitly compiled
transform artifact, and matching visible device ordinals. Mismatched execution
backends fail closed. Changing a borrowed executor's admission configuration
requires constructing a fresh prepared owner.

```python
from tools.generativeqc_cc import PreparedUCCSD, SolverOptions

with PreparedUCCSD(snapshot, provider, options=SolverOptions()) as prepared:
    result = prepared.solve()
    if not result.converged:
        raise RuntimeError(result.reason)
    energy = result.total_energy
```

CUDA is an explicit **host-controlled, staged** contract: four-center MO
integral transforms and CC contractions run on CUDA. Fock matrix transforms,
denominators, amplitude checkpoints and DIIS remain on the host. The supplied
AO source and UHF reference have their own explicit producer provenance.
No PySCF import, CPU CC equation fallback, resident iteration or all-device
complete molecular endpoint is hidden behind a CUDA label.

## Integral conversion and ownership

The owner requests `SpinMOBlock` chemists' Coulomb tensors, attaching the
appropriate alpha/beta coefficient matrix to every slot. It constructs the
antisymmetrized physicists' inputs requested by TensorIR through explicit
direct/exchange permutations and signs. Only the ordinary eightfold Coulomb
permutation symmetry canonicalizes requests; spin owners participate in every
key, including when populations and MO index tuples are identical.

`ConventionalProvider.admit` dry-runs the complete pin/LRU cache transition
without changing the real cache or reading AO integrals. UCCSD holds the
provider's reentrant lock across admission and conversion. Provider planning
and actual cache retention share `BlockPlan.retained_bytes`. In particular,
each retained CUDA block charges its library allowance, not merely the
explicitly allocated array/workspace arena.

Prepared MO feeds own immutable host storage. Downloaded CUDA blocks are copied
before an LRU eviction can invalidate their device owners. Closing/clearing a
borrowed provider or its source after successful preparation does not invalidate
these owned feeds. Closing the prepared owner releases its feeds/denominators
and rejects later evaluation; it never closes borrowed providers/executors.

## Coordinates, proposals and acceptance

Amplitude state uses the exact [five-block packed layout](uccsd_layout.md).
Same-spin representatives satisfy `i<j,a<b`; mixed-spin coordinates are
unrestricted within their declared alpha/beta axes. Supplied full amplitudes
must pass exact layout admission, without projection or multiplicity scaling.
Native residuals may have floating-point antisymmetry roundoff; iteration reads
their physical excitation representatives rather than projecting an invalid
supplied amplitude. DIIS uses the unweighted determinant-excitation metric,
not the restricted pair metric or the five-full-block Frobenius metric.

Canonical occupied-minus-virtual denominators are checked before AO reads,
including actual same-spin pair extrema, mixed-spin/singles gaps and finite
level-shifted proposal extrema. Near-zero, non-Aufbau and nonfinite cases fail
closed; no regularization changes the Hamiltonian. Damping and level shifts
modify proposals only. Each stored DIIS error belongs to the actual trial
amplitudes used to evaluate it.

Convergence requires the configured energy-change gate and the **full-array
physical maxima of all five residual blocks**, followed by a fresh evaluation
on the selected backend. Neither shifted/preconditioned residuals, DIIS error
norms, finite energies nor small amplitude steps imply convergence. Empty
excitation domains have zero correlation energy. Exhaustion and nonfinite
proposals return explicit failure status and the last finite evaluated state.
Returned amplitudes are owned and irreversibly read-only.

## Resources and evidence

`SolverOptions.max_bytes` bounds logical owned numeric preparation/iteration
buffers plus the planned native executor peak. Provider storage has its
separate explicit `budget_bytes`, including source/reference storage and cached
outputs. These are not process-RSS or allocator/code-stack limits. Both complete
admissions occur before AO reads. A pin set exceeding the provider budget fails
before conversion; explicitly choose `cache_policy="lru"` for bounded cache
retention. This is particularly important for CUDA's per-block library allowance.
There is no silent CPU spill or unbounded pin-cache promotion.

Provenance records reference/source/integral/equation identities, precision and
backend roles, denominator minima, numeric bounds, physical evaluations, DIIS
restarts and this preparation's provider work counters. Preparation and solve
timings cover this supplied-reference API, not upstream SCF/source/artifact
creation. They are diagnostics, not complete Calculator benchmark claims.

Host gates:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_uccsd_solver.py \
  tests/python/test_uccsd_equations.py tests/python/test_uccsd_layout.py \
  tests/python/test_uccsd_native.py -q
```

The solver converges open-shell and broken-symmetry molecules against independent
PySCF UCCSD and agrees with the existing RCCSD solver in the restricted limit.
Equal-population, distinct-spin coefficient matrices are additionally checked
against a determinant-space oracle. Proposal controls, strict admission,
pin/LRU limits, library-allowance accounting, immutable publication, last-finite
failure state and provider/prepared lifetimes have executable gates.

The native producer gate requires a source-matched `libgenerativeqc` build
(`GENERATIVEQC_LIBRARY` may select it). It runs native direct-UHF export,
streamed native shell integrals and the complete CPU UCCSD solver for H2, Li
and OH. The independent PySCF integral/SCF/UCCSD oracle receives the exact
shell primitives, not its separately rounded named basis table. Only basis
definitions are shared; no production calculation invokes the oracle.

The real-CUDA molecular gates must run on an available permitted node through
a finite Slurm allocation, preserving assigned device visibility. Set
`GENERATIVEQC_TENSOR_CUDA_TEST=1`, `GENERATIVEQC_NVCC`, and a compatible
`GENERATIVEQC_TENSOR_ARCH`, then select `-k complete_cuda` from the solver tests.
They exercise actual CUDA MO transformations and iterative CC contractions,
comparing molecular energies and all five amplitude blocks with native CPU and
the independent PySCF energy. A skipped host CUDA gate is not GPU evidence.
