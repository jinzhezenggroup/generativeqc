---
orphan: true
---

# UCCSD energy and residual TensorIR

`generativeqc_compiler.cc.uccsd_equations.build_program(oa, ob, va, vb)`
builds a real-FP64 fixed-amplitude UCCSD graph. It returns
`correlation_energy`, `r1a`, `r1b`, `r2aa`, `r2ab`, and `r2bb`.
The five amplitude inputs use the [UCCSD layout contract](uccsd_layout.md).
Alpha and beta occupied/virtual populations are independent; empty domains are
legal mathematical blocks. This internal graph does **not** register a public
UCCSD method, admit UHF references, supply integral providers or denominators,
or implement an iterative solver, Lambda, triples, or nuclear forces.

The separate [canonical UCCSD solver](uccsd_solver.md) binds this graph to
validated UHF references, conventional spin providers and packed CPU/CUDA
iteration state. Its reference/resource policies do not belong to this builder.

## Hamiltonian and residual convention

Integral inputs are real antisymmetrized physicists' integrals
`g[p,q,r,s] = <pq||rs>`, not chemists' Coulomb blocks. In a common spatial basis,
their definition is

```text
<pσ qτ || rυ sω> = δσυ δτω (pr|qs) - δσω δτυ (ps|qr).
```

The graph requires the six domain patterns `oooo`, `ooov`, `oovv`, `ovov`,
`ovvv`, and `vvvv`. Names combine the `g` prefix, domain pattern, an underscore,
and four ordered spin labels: for example, `goovv_abab`. Two-index Fock inputs
are `foo_aa`, `foo_bb`, `fov_aa`, `fov_bb`, `fvv_aa`, and `fvv_bb`.
All input axes use the declared `oa`, `ob`, `va`, or `vb` spaces. Only
spin-conserving four-index assignments are requested. The live input nodes of
the returned program are authoritative for the exact names and dimensions.
Callers must supply mutually consistent integral permutation symmetries and
real symmetric Fock matrices; arbitrary independently edited integral blocks
are not an admitted physical Hamiltonian.

Residuals are the projections of `exp(-T) H exp(T)|Phi>` on singles and doubles
excited determinants, in the layout's excitation order. The occupied and virtual
Fock diagonals remain in the graph. For canonical references, a Jacobi proposal
`t_new` from a denominator-divided implementation therefore satisfies
`r = D * (t_new - t)`, with occupied-minus-virtual denominators. Neither
denominator division nor a level shift belongs to these physical residuals.

The correlation energy has the full-spin-orbital factors

```text
E_corr = sum_ia f_ia t_ia
       + (1/4) sum_ijab <ij||ab> t_ijab
       + (1/2) sum_ijab <ij||ab> t_ia t_jb.
```

These factors act before spin summation. The four signed mixed-spin
permutations of `t2ab` are resolved to that one stored block. Same-spin occupied
and virtual permutations are independent; no restricted pair symmetry or
closed-shell multiplicity is substituted.

## Shared execution and derivatives

The builder sums allowed contracted spin assignments into typed TensorIR
`einsum`, `add`, and `transpose` operations. It never constructs a combined
spin-orbital amplitude or integral tensor. Intermediate Fock, tau, and W blocks
also retain explicit spin axes. Rational coefficients and program identities
are shared by the interpreter, `NativeCCTensorExecutor`, and
`CudaCCTensorExecutor`; there is no separate CUDA equation inventory or CPU
fallback in the CUDA executor.

Both native executors use their existing explicit storage admission and finite
compiler lifecycle; the CPU owner also enforces its scalar-work limit.
This conventional graph still has four-index
`vvvv` inputs and intermediates. It is not a tiled integral provider, a
molecular memory planner, a resident iterative solver, or a performance claim.

`linearize` and `transpose_program` differentiate the same graph with respect
to all five amplitude blocks and the requested Fock/integral inputs. Their dual
coordinates are **full-array TensorIR coordinates**. A packed Lambda solver must
compose the amplitude-storage pullback and determinant excitation normalization;
using a full-array Frobenius dual as a packed excitation dual is not justified.
Fixed-amplitude derivative identities alone do not qualify orbital response or
analytic nuclear forces.

## Validation

Run the host and independent molecular gates with the pinned reference-test
dependencies:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_uccsd_equations.py \
  tests/python/test_uccsd_layout.py -q
```

The tests compare random-amplitude energy and all five residual blocks with a
complete fixed-particle determinant calculation using fermionic bit actions
and finite nilpotent exponentials, rather than another CC intermediate
implementation. Unequal spin populations, simultaneous nonzero same-spin
sectors, one-spin references, empty blocks, and the vacuum are covered.
Independent PySCF UCCSD random-amplitude and converged molecular gates include
open-shell and broken-symmetry references; random RCCSD amplitudes test the
collapsed-spin restricted limit against both PySCF and the existing RCCSD
TensorIR. JVP/VJP transpose identities and centered
finite differences cover amplitude and Hamiltonian inputs. Admission tests
check invalid dimensions, shapes, precision, nonfinite feeds and exhausted
interpreter budgets without modifying caller storage.

The native CUDA determinant-oracle gates are opt-in and must run inside a
finite Slurm GPU allocation, preserving its assigned device visibility:

```sh
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 env GENERATIVEQC_TENSOR_CUDA_TEST=1 \
  GENERATIVEQC_NVCC=/path/to/nvcc GENERATIVEQC_TENSOR_ARCH=sm_120 \
  PYTHONPATH=python:. python -m pytest \
  tests/python/test_uccsd_equations.py -k real_cuda -q
```

Choose a currently available permitted node and its compatible partition/GPU
type; do not schedule a maintenance node. An unallocated host run skips the
CUDA-only cases and is not evidence of CUDA correctness.

## Shared unrestricted triples tiles

The [unrestricted triples tile graph](uccsd_triples.md) reuses these amplitudes,
spin-labelled feed conventions and native TensorIR backends for canonical (T)
channels. Its fixed-amplitude qualification does not imply a complete UCCSD(T)
energy or nuclear-force owner.
