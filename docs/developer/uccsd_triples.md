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

## Qualification

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_uccsd_triples.py -q
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
