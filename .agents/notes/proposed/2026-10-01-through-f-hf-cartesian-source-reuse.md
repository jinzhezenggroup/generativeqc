# Proposal: reuse HF Cartesian sources for through-f RSH consumers

Status: implemented candidate; native numerics qualified, complete endpoint qualification pending
Date: 2026-10-01

## Evidence motivating the next stage

The automatic screened public-AO source on master
`737f3481fcc5a82039bd59b676f14d5e963a517d` passes independent full/SR/LR value,
screening-boundary, budget fallback and spin/representation derivative gates.
Slurm 11957's complete 58-AO endpoint passes all twelve original/moved calls:
warm median 13.3336 s versus GPU4PySCF 16.4248 s, with maximum energy/force
errors 8.09e-12 Eh / 5.98e-11 Eh/Bohr. Its binary is
`3e3dec619a6bd6222e54c23c3f52e2c945da0b0831c910d4ef99a1e1ef5f2070`.
The last force assembly still spends 7.615 s in native integral derivatives.
The following 116-AO cold endpoint does not finish within its explicit 330 s
process limit. There is no completed six-atom native timing, and no 100-atom
performance evidence. Do not rerun the same slow large matrix without changing
the source work or establishing a narrower profiling question first.

Removing rejected pair products fixes traversal amplification, but cannot
remove the primitive/component work inside each admitted public-AO source.
HF's prepared one-electron metadata/kernel reuse is implemented separately;
it removes hot metadata uploads, not the dominant two-electron source cost.

## Bounded implementation candidate

Reuse HF's existing public-to-Cartesian density projections, Cartesian source
contraction emitted by `direct_source_contraction_cuda.py`, primitive-pair
contracts, and Cartesian-to-public J/K projections. Do not reproduce recurrence
mathematics or pretend the generated SPD shell inventory contains f kernels.

Retain a small prepared through-f source owner with explicitly charged transform,
packed Cartesian metadata, full-range Cartesian Schwarz keys, interleaved-spin
density/output matrices, projection scratch and optional screened rows. Direct
AO angular buckets specialize existing compiler evaluators. Reuse the same
transformed final densities for source-major RSH force weights, with existing
eight-permutation and unique-atom/translational reconstruction semantics.

The Cartesian dimension differs from the public spherical AO dimension. Never
reuse public matrix/pair offsets without changing their strides. Match component
normalizations to the HF transform ABI; never apply normalization twice. The
native physical AO order and final output shapes remain public/spherical.

Prefer automatic capacity/work admission, retaining the already-qualified
public-AO canonical path if the additional transform/source storage cannot fit.
Do not introduce a production environment setting. Geometry-dependent source
state must be rebuilt on geometry changes; unchanged SCF/force calls borrow it
on the prepared owner stream.

## Qualification before promotion

- Independent CPU full/SR/LR matrices through f, Cartesian and spherical output,
  restricted/unrestricted and nonsymmetric value densities, all output masks.
- Fixed-density CPU displaced-energy RSH derivatives, both spins and geometries;
  complete reconverged RKS/UKS forces and directional finite differences.
- Explicit source-work counters and budgeted public-AO fallback; avoid confusing
  Cartesian component evaluations with symmetry-unique public-AO work counts.
- Screening is basis dependent: Cartesian Schwarz admission must be qualified
  against the original physical endpoint gates, not asserted bitwise identical
  to public-AO screening. Keep full FP64 and the unchanged requested tolerances.
- Full cold, five original warm, moved, and five moved warm host-force endpoints
  at 58 and 116 AOs before retrying larger sizes; preserve failed journals.
- Separately validate the resource-bounded stationary-force domain beyond
  1024 AOs before publishing a 96-atom/1856-AO endpoint.

## Candidate implementation and traps found during qualification

The native owner now automatically admits the Cartesian source when its full
metadata, bounds, projection, spin scratch and optional row inventory fit the
existing budget. A public-AO canonical owner remains the smaller-storage
fallback. This candidate does not extend the generated SPD shell-vector mask
and does not yet reuse a primitive-pair cache for all high-angular contractions.

The shared HF projections accept optional exact shell-local transform spans.
Density and Fock contractions then scan at most the matching shell's public or
Cartesian components, rather than every AO. The resident span inventory is
linear in both AO dimensions and charged to the same optional budget. Null
spans retain the original dense HF projection. Independent public-AO matrix
gates exercise both implementations, including nonsymmetric densities,
interleaved spin storage and two geometries. Item-local force projection uses
an active mask so other, uninitialized batch densities are never read.

Two otherwise subtle HF source contracts surfaced in the first short test:

- The order-two value shortcut needs `shell_direct_ao_offsets`, even for a
  Schwarz diagonal. The old packer also omits spherical transforms below the
  persistent-ERI threshold and omits identity transforms for Cartesian bases.
  Uploading an empty transform with a nonzero rectangular byte count surfaced
  as a generic metadata-upload error. Slurm 11960 and 11962 are failed
  candidates, not evidence for promotion. All Cartesian metadata is now
  explicitly retained and charged; the owner requests tiny spherical transforms
  and bypasses projection entirely when public/source ordering is identical.
- Its closed-form output is a vector of `double` values. Casting that output to
  `Dual3` silently discards the full-range derivative seed. The compiler emitter
  now restricts the shortcut to value scalars; derivative consumers retain the
  existing seeded scalar recurrence. A spectator-f test contracts only compact
  s+p or s+d densities and compares the fused Cartesian J/SR/LR derivatives
  against independent displaced CPU energies. This specifically exercises the
  order-two classes that a pure s+f force fixture cannot cover.

Both build launchers use ccache and all incremental builds use `-j40`. Neither
successful compilation nor the host code-generation tests qualifies the
candidate endpoint. Keep the measured public-AO baseline until the independent
native and complete RKS/UKS force gates pass and a complete endpoint improves.

Slurm 11964 passes independent screened full/SR/LR public matrices, dense and
shell-local HF projections, Cartesian order-two derivative seeds, through-f
RSH displaced-energy gates and prepared one-electron CPU-gradient/zero-H2D
checks. The work census distinguishes 58/116 public AOs from 63/121 Cartesian
source AOs in its synthetic fixtures. Its value timings are not molecular
energy/force endpoints. The compiler seed fix has a separate
[numerical decision](../implemented/numerics/2026-10-01-cartesian-order-two-derivative-seeds.md).

## References

- [Automatic screened default](../implemented/performance/2026-10-01-default-screened-through-f.md)
- `src/scf/cuda/direct_coulomb.cpp`: retained HF density/output projections
- `python/generativeqc_compiler/integral/direct_source_contraction_cuda.py`
- `src/scf/cuda/direct_bounded_fallback.cu`: bounded shell consumer contracts
