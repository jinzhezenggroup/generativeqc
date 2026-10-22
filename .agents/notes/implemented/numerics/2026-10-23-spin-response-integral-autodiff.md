# Decision: spin-resolved integral AD with coupled response and CUDA execution

Status: implemented
Date: 2026-10-23

## Problem

The [first integral AD endpoint](2026-10-23-rhf-integral-parameter-autodiff.md)
differentiated conventional CPU RHF once. UHF needs independent spin densities;
second derivatives need the response of the converged determinant and overlap
multiplier. A differentiated frozen-density energy is not an energy Hessian.
CUDA must consume the differentiated integral values rather than silently
recompute another Hamiltonian through an ordinary integral provider.

## Decision

Generalize the owner to `IntegralHF`, retaining `IntegralRHF` and adding
`IntegralUHF` specializations. The compiler transposes a shared spin Lagrangian
with unit-occupation alpha/beta densities and physical weighted densities:

    L = (Da + Db):h - (Wa + Wb):S
        + g_abcd ((Da + Db)_ab (Da + Db)_cd / 2
                  - Da_ac Da_bd / 2 - Db_ac Db_bd / 2).

Its TensorIR linearization supplies moving integral cotangents. Native shared
HF solvers consume the generated S/h/ERI/Enuc pack. The physical final Fock
commutator, canonical-density consistency and true response residual gate
publication. The shared spin orbital response action includes explicit integral
motion, metric transport and induced J/K; RHF solves its symmetric spin sector.
Differentiate `W = D F D` in this unit-occupation convention, including all
three product terms. Neither DIIS history nor an arbitrary eigenvector gauge
defines the derivative. Degeneracy or an unconverged response fails explicitly.

For second order and for CUDA first order, serialize the canonical scalar primal graph as an immutable
instruction tape. Graph AD generates each local value/partial and directional
partial rule. The native runtime only traverses the tape and owns scratch.
Boys leaves reconnect to their original argument with Fm, Fm+1 and Fm+2;
the shared stable Boys implementation now admits F18. N-ary arithmetic retains
C-family left association. Normalization and zero/signed raw coefficients
remain in the differentiated chain. Tape consumers do not first construct
expanded partial graphs they would discard.

CUDA uses the same expressions/tapes with 32 independent source-task lanes and
deterministic reduction of private parameter cotangents. It keeps explicit
dense ERIs for a generated J/K contraction and uses the shared CUDA tensor
matrix product and ordinary eigensolver owners. SCF/DIIS and Krylov control,
small response matrix algebra, and stationary source weights are explicitly
host operations. There is no CPU integral, Fock or eigen retry. This execution
split is part of the public contract, not a claim of a fully device-resident
solver. The external-integral prepared plan never matches a system-only cache.

The Array API supports `jvp(grad(loss), ...)` and
`jvp(value_and_grad(loss), ...)` through a quadratic molecular source jet.
Both TensorIR transforms precede snapshot binding. The source Hessian includes
all requested geometry/exponent/coefficient blocks, so nonlinear outer
coordinates and logarithmic exponents receive both chain-rule terms.
Arbitrary nested gradients, orders above two, DF, KS and post-HF are not
implemented by this decision and must continue to reject explicitly.

## Bounds and rejected alternatives

The explicit dense-primal endpoint remains subject to numeric payload,
primitive work, symbolic work and finite compilation budgets. Tape scratch is
charged before compilation, including every CUDA lane. Full Hessians admit
the entire output and all column work before execution, retain unsymmetrized
columns and publish aggregate diagnostics only after complete success.

Expanded second partials produced 1,647,758 C++ lines for STO-3G water and
exceeded a 300-second native compilation budget. Tapes eliminate that expansion.
However, specializing the interpreter by each primitive's tape length still
made CUDA optimize hundreds of equivalent traversals and exceed its 300-second
budget. The expanded first-order CUDA water source also exceeded 300 seconds,
so CUDA first order uses the compact tape with first-order scalar rules only.
Instruction/input counts are now runtime data; CUDA uses a non-inlined
interpreter shared by Boys order and scalar type. This is a compilation-size
choice, not a new integral recurrence. Preserve finite compiler limits rather
than hiding this failure by increasing timeouts indefinitely.

Rejected alternatives include numerical energy/gradient differences in
production, reusing force APIs as integral pullbacks, frozen orbital Hessians,
dense response-matrix inversion, direct cuBLAS ownership outside the shared
tensor boundary, and silently switching backends after a device failure.

## Evidence and acceptance gates

Tests use independent PySCF/libcint values, gradients and analytic nuclear
Hessians. Mixed exponent/coefficient/geometry second derivatives use
reconverged four-point independent energies at two step sizes, with a final
absolute mixed-direction error below 2e-7. RHF/UHF HVP symmetry, translation
null directions, `H c = -grad_c` for coefficient scale invariance, arbitrary
nonsymmetric integral cotangents and immutable replay are covered.

STO-3G water exercises nontrivial p shells and orbital relaxation, accepting
the independent analytic nuclear Hessian action within 2e-8. Both engines
receive the same raw basis definition: their built-in STO-3G catalogs retain
different coefficient digits, which otherwise shift total energies by about
2.4e-8 Hartree and obscure a tight gate. CUDA RHF/UHF and water acceptance runs through
finite Slurm allocations on node n1 with an RTX 5090. Generated integral
values/JVP/VJP, energies, gradients and joint HVPs are compared to the CPU
chain and independent references. All six real-device tests pass, including
default first order, the one-electron empty response space and public Array API
joint HVP composition. Compute Sanitizer memcheck/leak qualification of the
RHF/UHF complete integral/JVP/VJP/response endpoints reports zero errors and
zero leaked bytes. Six shared native Fock/RHF/UHF/GMRES regressions also pass.
No large-basis throughput claim follows
from these bounded endpoint tests.

Raw compilation, compiler-cache and test logs are kept under ignored
`.artifacts/autodiff/`; remote qualification uses
`n1:/data/jzzeng/integral-ad-expanded-20261023`. The independent test files are
`test_integral_parameter_ad.py`, `test_integral_autodiff_second.py`,
`test_integral_autodiff_cuda.py` and `test_array_api_integral_ad.py`.

## Revisit when

Additional methods expose independently qualified stationary source cotangents
and their response, or a tiled primal schedule can retain all parameter
derivatives with bounded endpoint work. Supporting DF requires differentiation
of the actual three-center/metric factorization; supporting KS also requires
AO/grid/XC parameter motion. Neither may be relabeled exact HF.
