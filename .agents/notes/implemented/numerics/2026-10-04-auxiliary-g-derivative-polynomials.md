# Decision: explicit auxiliary-g derivative polynomial lowering

Status: implemented (compiler variant; native consumer integration remains gated)
Date: 2026-10-04

## Problem

The native DF energy source covers orbital s/p/d/f and auxiliary s/p/d/f/g.
Large CC molecular forces also need derivatives of three-center f/f/g and
metric g/g integrals. The default derivative polynomial domain ends at F_10
and does not admit auxiliary g. Reusing it by removing a native capability
check would produce unsupported internal powers or undersized work arrays.

## Decision

Add an explicit `auxiliary_g_derivative` domain to the shared axis-moment DAG
and an opt-in `auxiliary_g=True` variant of the existing CPU/CUDA derivative
emitter. The same Gaussian moment, analytic raising/lowering and Boys algebra
are used. No recurrence or scientific expression is copied into native code.

For f/f/g, one internally raised orbital factor needs F_11 and twelve
coefficient slots. For g/g metric, only the first real center is differentiated
independently: its power may rise to five with the absent B power fixed at zero.
The other auxiliary center follows translation invariance. Public orbital g,
both orbital powers raised together and auxiliary powers above g remain
outside this variant's runtime contract.

The explicit variant has its own generated namespace and header guard. The
default s/p/d/f derivative source is byte-identical to its parent, SHA-256
`71430c3d8929868c10156d36578f719bbd56d595b35c4ae56131930395f4ce04`.
Existing auxiliary-g value admission is unchanged; it still cannot silently
select derivative raising.

## Independent evidence and retained failure

The new module covers all nine metric signatures containing g and all sixteen
orbital s/p/d/f pairs with auxiliary g, for asymmetric and coincident centers.
CPU and Slurm RTX 5090 CUDA execute the same generated arithmetic. Every
contracted Cartesian block, spherical projection and mathematical center is
compared with independently normalized libcint ip1/ip2 derivatives at
atol=rtol=8e-11. Translation sums are checked at absolute 2e-12 and asymmetric
auxiliary motion must be nonzero. All 101 cases passed CUDA memcheck with zero
errors in 14.96 s. CPU-only execution passes 51 and skips 50 CUDA cases.
The combined pre-existing derivative/value and new response host regression
passes 197 cases, with 107 CUDA cases skipped outside an allocation, in 16.80 s.

The first candidate failed six pure-axis f/f/g derivative entries, with maximum
error 6.38e-6. This was a generator storage rewrite bug, not a changed accuracy
gate: globally replacing `[11]` by `[12]` also rewrote the valid coefficient
assignment `out[11]` to `out[12]`. Resizing only the explicit array declarations
preserves that highest coefficient and fixes the full-block tests. Future
changes must distinguish a declaration's capacity from an element's index.

## Native integration boundary

This compiler variant does not remove the native source's g-derivative
rejection and does not enable public CC forces. Raw coordinate derivative
replay and weighted nuclear contraction still need independent native tests.
The raw source already keeps up to six public-to-Cartesian expansion terms,
whereas the generic weighted derivative bridge and
`cuda_gaussian_products::contract` use the existing three-term AO stride.
Simply allowing g in `pack()` would truncate or misaddress spherical g metadata.

For the weighted consumer, preserve the current three-term specialization and
add an explicitly admitted six-term variant (with matching metadata, work and
budget accounting), or use an equivalently qualified source-owned public-basis
projection. Contract all center derivatives with final weights during one
source traversal. Replaying all A derivatives separately for each nuclear
coordinate would introduce avoidable atom-count work amplification even if
the temporary memory remains bounded. Source-response callback outputs are
provisional and must remain on the owning stream until successful drain.

## References

- `.agents/notes/implemented/numerics/2026-10-04-df-occupied-fock-resolvent.md`
- `python/generativeqc_compiler/integral/df_derivatives_cuda.py`
- `src/scf/cuda/df_source.cpp`, `src/scf/cuda/df_gradient_bridge.cu`
- Ignored `.artifacts/aux-g-response/` test and sanitizer logs.
