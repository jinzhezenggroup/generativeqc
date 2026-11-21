# Decision: Tile dense fitted Hessian contractions and reuse temporary storage

Status: implemented
Date: 2026-11-21

## Problem

The full-rank PBE0 RKS water dimer with spherical cc-pVDZ/full
def2-universal-JKFIT (48 AO and 226 auxiliary functions) exceeded the
scalar-work bound in the fixed-geometry induced Fock owner (`1769`). After
lowering its dense products to tiles, the materialize-everything schedule
still exceeded the public 64 MiB fitted-owner allocation (`1770`). These
are distinct limits; raising either without changing its execution would
not address redundant data movement or intermediate lifetime.

## Decision

The compiler maps binary dense contractions (optionally with scalar
operands) into batch, free-row, free-column and contracted labels without
reordering reduction labels. A 16×16 shared-memory tile reuses each input
panel across outputs. Generated addressing preserves the original physical
axis order, including spin batching, transposes, flattened multiple indices,
and irregular tails. Small or nonmatrix contractions retain scalar execution.
This lowering needs no library workspace, device allocation, JIT or host
numerical provider.

Charge scalar visits and padded matrix-product slots separately per reference
and per action. Keep the 100-million scalar bound and impose a one-billion
slot bound on the tiled path. This is an explicit admission of a different
schedule, not a claim that padding, memory passes or all multiply-adds are
free. Keep finite validation and publication gates. Reuse a same-sized scratch
slot only after the last read on the same CUDA stream, with outputs live through
publication. This supports runtime shapes without speculative in-place writes,
and retains the 64 MiB owner cap.

## Evidence and limits

For the water dimer, the emitted RKS nuclear response map charges 18,400,411
scalar visits and 523,100,160 padded product slots per direction. The fixed
geometry Fock and assembly maps charge 2,645,448 scalar visits and 99,975,168
slots per direction, plus the one-time whitened panel preparation. These
counts exclude raw integral derivatives, SCF, XC and response iterations and
are not FLOP or endpoint speedup claims.

RTX 5090 Slurm `1772` completed the public PBE0 RKS dimer HVP in 50.48 s
under the default 64 MiB fitted-owner budget, with 28 induced Fock actions,
one DF value pass and one DF contraction pass. The maximum error against
the independent PySCF/Libxc reference was `4.23e-10 Ha/Bohr²`. The reference
uses Richardson extrapolation of reconverged gradients at steps `8e-4` and
`4e-4` Bohr (maximum step difference `6.20e-9 Ha/Bohr²`). Translation passed;
the native receipt shows no host numerical fallback, intermediate download,
or displaced SCF. Source-matched checksums, inputs and logs are retained in
`build/hessian-gpu-sm120/r2scan-orbital-f/` (`tiled-reuse-binaries.sha256`,
`probe-pbe0-water-dimer.py`, `pbe-water-dimer-oracle/pbe0-hvp.npy`, and
`pbe0-water-dimer-tiled-reuse-n4.log`).

The new independent expanded-ERI cases use 17 AO and 19 auxiliary functions
to exercise tiled transpose and tail indexing in both spin conventions,
including exchange-only. Host-only admission checks reject a tight tiled cap
and an oversized shape, and admit the dimer within the public workspace
budget after charging its raw panel. Slurm `1773` on the reuse build passed
68 native/algebra/oracle tests and 17 public frontend regressions; 16 angular
frontend cases were not enabled for this replay. Larger hybrid spectral actions still have an
independent generic divided-difference work bound; the rank-one Coulomb
specialization does not relax it.
