---
orphan: true
---

# Shared unrestricted triples tile graph

`generativeqc_compiler.cc.uccsd_triples.build_tile_program` lowers one canonical
real-FP64 UCCSD(T) spin-channel tile to the existing TensorIR. The same scalar
graph supports native CPU/CUDA execution and the existing JVP/VJP transformations.
It has no runtime, PySCF or device-probing dependency during source generation.

This is the fixed-amplitude tile definition. The separate
[complete UCCSD(T) energy owner](uccsd_t_owner.md) binds a converged
[UCCSD state](uccsd_solver.md), admits denominator and storage/work limits,
enumerates the tile coverage, accumulates the four channels and publishes their
provenance. Tile cotangents are not yet unrestricted analytic nuclear forces.

## Scientific definition

Use real physicists' antisymmetrized integrals and the
[five independent UCCSD blocks](uccsd_layout.md). The general spin-orbital
connected and disconnected seeds are

```text
W0[i,j,k,a,b,c] = sum_e t2[j,k,a,e] g[b,c,e,i]
               - sum_m t2[i,m,b,c] g[m,a,j,k]
V0[i,j,k,a,b,c] = t1[i,a] g[b,c,j,k] + fov[i,a] t2[j,k,b,c]
```

Apply the three cyclic occupied permutations and, independently, the three
cyclic virtual permutations to both seeds. Their built-in j/k and b/c
antisymmetry makes this the nine-term antisymmetrizer, not a restricted pair
symmetry. Denominators are sums of three occupied energies minus three virtual
energies, without shifts or regularization in the physical correction.

For the full ordered spin-orbital sum the energy is `W * (W+V) / (36 D)`.
The canonical `aaa`, `aab`, `abb`, `bbb` channel patterns fix both the occupied
and virtual spin order. Each has `choose(3,nalpha)^2` equivalent ordered spin
patterns: the exact channel factors are consequently `1/36`, `1/4`, `1/4`,
`1/36`. No RCCSD(T) projection or closed-shell prefactor enters this graph.
Forbidden contracted spin assignments disappear at graph construction. Every
mixed-spin doubles permutation retains its fermionic sign relative to `t2ab`.

The audited upstream generic formula and independent unrestricted oracle are
PySCF 2.14.0 `gccsd_t.py` and `uccsd_t_slow.py`; exact file identities and the
existing license notice reside in
`tools/generativeqc_cc/uccsd_triples_source_manifest.json`
and `tools/generativeqc_cc/NOTICE`.

## Tile bindings and resource boundary

`channel_shape` returns the six full domain populations. `channel_possible`
requires enough distinct same-spin occupied and virtual orbitals for a physical
triple. The builder rejects unknown channels and zero, non-integer or oversized
tile axes. Tile dimensions are independent across all six coordinates.

`TriplesTileProgram.operands` specifies each live input's canonical source,
transpose axes and coordinate labels. A caller transposes the source and
selects its external i/j/k/a/b/c coordinates; contracted e/m axes remain whole.
Source signs live in the graph, never in an undocumented feed scaling.

The existing UCCSD `ovvv`, `ooov`, `oovv` feeds supply the seeds via real-integral
symmetries. There is no extra integral producer or full-spin-orbital embedding.
The same binding can scatter a tile input's cotangent back into its canonical
source, accumulating repeated/shared coordinates explicitly. That scatter and
the solved-state response composition remain responsibilities of the complete
owner, not an implicit compiler operation.

Physical offsets do not enter the graph. Equal-shaped tiles share a scientific
artifact and compiled kernel even when they cover different orbital regions.
No molecular T3 input/output is required; native executor planning bounds the
selected tile's temporary moments. Whole conventional four-index feeds and
the complete tile enumeration must still receive separate owner admission.
Tiling is a memory policy, not a claim of reduced algorithmic work or resident
GPU execution.

## Complete Fixed-Amplitude Response

`build_tile_response_program` applies the shared generic TensorIR transpose to
the actual tile energy. Its outputs include that energy and every gathered
operand's full-Frobenius cotangent. The scalar seed `bar_triples_energy` is one;
the orbital-energy operands are differentiated along with amplitudes, Fock and
integrals. There is no separate handwritten triples derivative inventory.

`PreparedUCCSDTriplesResponse` borrows the live converged UCCSD parent and its
matching native CPU/CUDA executor. It reuses `PreparedUCCSDT`'s denominator
admission, full/edge template planning, deterministic tile fallback, fresh
physical CCSD replay and complete ordered grid. Admission counts the derivative
graph's actual scalar-loop work and backend execution storage, as well as full
canonical source bars, FP64 compensated accumulation, immutable publication
and bounded scatter scratch. The bound is logical numeric storage, not RSS,
compiler-cache storage or arbitrary caller-owned provider storage. Smaller tiles
do not bypass the complete tile/work caps.

Each binding is reversed by its explicit transpose and coordinate scatter;
overlapping contributions from independent tiles, permutations and all four
channels are compensated and added, without mixed-spin symmetry folding.
The returned `UCCSDTriplesResponseResult` owns immutable bars for every parent
source, the five amplitude blocks, and `eps_o_a/eps_v_a/eps_o_b/eps_v_b`. Unused
sources have exact zero bars. Its packed amplitude cotangent is the transpose
of the physical amplitude expansion (`U*`), not representative packing or a
same-spin division by four. Calling `hamiltonian_cotangents()` explicitly adds
each occupied/virtual eps bar to the corresponding spin Fock diagonal and
returns immutable Fock/integral weights without amplitude/eps inputs. This
canonical coordinate pullback is not orbital relaxation. State, template and
response identities retain
the exact physical/execution binding. Late native failures, malformed outputs
or nonfinite tile/scatter results publish neither partial energies nor bars.

These are **fixed-amplitude (T) cotangents**, not corrected Lambda, relaxed
densities or nuclear forces. A physical UCCSD(T) consumer must solve and freshly
qualify the **corrected** amplitude stationarity equations, retain a distinct
corrected-state identity, explicitly chain orbital-energy bars to each spin's
Fock diagonal, and bind all Hamiltonian/orbital/nuclear derivative chains.
Corrected multipliers must not be passed through standard UCCSD Lambda
qualification as if they were standard multipliers. See the
[standard physical-response contract](uccsd_physical_response.md).

## Corrected Lambda

`PreparedUCCSDTLambda` borrows identity-matched standard adjoint and bounded
triples-response owners. It solves
`J* lambda = -U* grad(E_CCSD + E_T)` with the standard native J* action, physical
dual lift and checked GMRES. The independent alpha/beta amplitude blocks and
same-spin residual seeds retain their original conventions. The triples
amplitude bars receive **no extra one-quarter factor** when pulled back with U*.

`UCCSDTLambdaResult` is a distinct record, not a standard Lambda subtype.
Both solving and qualification execute a fresh complete triples replay and
actual corrected physical amplitude stationarity. Qualification verifies the
parent, every correction bar, per-channel/total energies, equation/template,
backend and state binding rather than trusting cached diagnostics or requiring
bit-identical floating results across valid native replays. Aggregate work and
logical numeric storage cover both borrowed consumers, retained corrections,
Krylov/maps and publication. Parameter requests receive additional admission.
Closing a corrected owner closes none of its borrowed owners/resources.

`qualify_with_correction()` returns the immutable packed dual, freshly qualified
triples response and diagnostics together, so a future physical consumer need
not trust an earlier detached correction. `parameter_cotangent()` adds the
corrected CC Lagrangian VJP to the explicit canonical-input triples bar. Its
same-space Fock blocks currently contain the **diagonal denominator chain**,
not complete covariant oo/vv response.

Complete orbital/nuclear response additionally requires same-space off-diagonal
Fock resolvent moments from `d D^-1 = -D^-1 (dD) D^-1`. A diagonal eps pullback
alone does not supply these moments; dividing an occupied/virtual internal
orbital gap is not a valid replacement at degeneracies. The shared restricted
implementation describes this contract in
`python/generativeqc_compiler/cc/triples_fock_response.py`. The unrestricted
full-Fock owner below supplies this missing denominator chain separately;
corrected Lambda parameter requests retain their fixed-canonical-input scope.
Complete corrected Hamiltonian/UHF/nuclear composition is not implemented.
The standard physical-response owner intentionally still rejects corrected
owners/results. Its standard Lambda qualification is not disabled or relabelled.

## Complete Same-Space Fock Moments

`PreparedUCCSDTriplesFockResponse` borrows a live bounded triples-response owner.
`response()` first freshly replays the parent and all source bars, then computes
the complete same-space denominator response on the same native CPU/CUDA
executor. The shared nine-term compiler exposes bounded `X=C/D` and
`Y=(C+V)/D` tiles; it does not introduce another triples equation inventory.
Each of the six occupied/virtual slots contributes a paired contraction over
five identical physical coordinates. Occupied moments have negative signs,
virtual moments positive signs, and the original channel factors remain
`1/36` for aaa/bbb and `1/4` for aab/abb.

Triangular open-axis page pairs are scattered to both off-diagonal matrix
blocks exactly once. At most two X/Y pages are retained; right pages are
recomputed instead of retaining a complete open-axis grid. For `n` open-axis
pages and `m` common-coordinate tiles, both vector-page evaluations and moment
calls number `m*n*(n+1)/2`. All axes/channels, response replay, recomputation
work, gather/native staging, compensated accumulation and immutable publication
are admitted before execution. Numeric fallback may shrink pages but cannot
evade total pair or semantic-work gates. Logical numeric bounds are not RSS,
opaque compiler/cache/context allocation or performance guarantees.

`UCCSDTriplesFockResponseResult.hamiltonian_cotangents()` **replaces** the old
diagonal-only denominator pullback by these full moments; it does not add both
and double-count the diagonal. Every completed response independently checks
its diagonal against freshly differentiated eps bars at the bounded configured
tolerance. No occupied/occupied or virtual/virtual energy gap is divided, so
exact internal degeneracies remain supported. Late vector, moment, output or
diagonal-gate failures expose neither partial moments nor prefix correction
energies/source bars. Closing this consumer leaves borrowed owners open.

This is covariant canonical-reference triples Hamiltonian response, not a
normal-ordering, corrected orbital response or nuclear-force endpoint. A future
physical consumer must explicitly combine fresh full moments with freshly
qualified corrected Lambda and retain the complete all-rotation gate.

## Qualification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_uccsd_triples.py \
  tests/python/test_uccsd_triples_response.py tests/python/test_uccsd_t_lambda.py \
  tests/python/test_uccsd_triples_fock_response.py -q
```

Random-amplitude Li/OH/H6 and broken-symmetry H4 cases compare every channel with the
independent PySCF slow oracle and its optimized total. Occupied/virtual tile
partitions preserve the channel sums. H6 supplies nonzero coverage of all four
channels. A collapsed-spin reference agrees with existing RCCSD(T).
Deliberately perturbed normalization and occupied-pair exchange signs fail
the independent channel gate. All four tile channels have native CPU and
JVP/VJP transpose/finite-difference tests.

The real-CUDA tier requires explicit `GENERATIVEQC_TENSOR_CUDA_TEST=1`, a finite
Slurm allocation and the configured compiler/architecture. Select
`-k real_cuda` to execute every channel against the independent oracle. Host
skips are not GPU qualification, and the tile tier is not complete molecular
UCCSD(T) convergence/forces evidence.
