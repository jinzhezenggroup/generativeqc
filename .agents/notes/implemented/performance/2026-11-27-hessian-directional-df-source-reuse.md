# Decision: Reuse bounded all-s DF nuclear source jets

Status: implemented
Date: 2026-11-27

## Problem

In the 96-H FP64 r²SCAN Hessian, the raw fitted source evaluated identical
primitive components before and after the orbital solve. Then one lane per
coordinate serially folded a window's atom incidences. Bare kernel time is
not a speed claim: on the same RTX 5090 (Slurm 20812, first paired sample),
the pre-change Direct endpoint took 299.87 s versus 102.27 s in GPU4PySCF;
the fitted endpoint took 589.43 s versus 94.03 s. Each endpoint includes
fresh SCF and all 288 Cartesian Hessian columns on the same unfurled
82,944-point moving grid. The auxiliary basis is Cartesian STO-3G in both
fitted arms, with full auxiliary response. The native selected-card peaks
were 7,895 and 6,127 MiB out of 32,607 MiB; the maximum raw errors were
3.37e-9 and 4.88e-9 Ha/Bohr², respectively.

Separate 18-second CUDA trace windows (20810 and 20811) locate the early
Direct cost in the frozen J/K compact-owner and launch sequence. In the
DF column loop, the factorized integral, Becke row, and DF scatter kernels
account for 39.6%, 33.3%, and 14.8% of sampled GPU kernel time. This is
neither a whole-endpoint attribution nor a cross-device speed comparison.
The later Direct column-loop window (20819, 12 seconds) attributes 81.4%
of GPU kernel time to the same Becke row kernel. The early frozen integral
bottleneck alone cannot explain or close the complete Direct gap.

## Decision

- Admit one reusable all-s raw source tape only inside the existing dense
  Hessian allocation limits. It stores the primal, first center derivatives,
  and one direction's second jets per group, never a coordinate-major
  derivative tensor. The values and contraction passes keep the original
  linked AD and both count even when the contraction has no primitive replay.
- Reuse the second jets only on the original stream, after a successful
  matching values pass and a device bitwise comparison of the direction.
  Consume this capability once. A different direction, non-batched shell
  closure, or refused reservation retains the ordinary streamed source;
  one allocation retry without the tape handles optional device OOM.
- The primal and first jets are independent of the right direction. After
  seeding them, prove a unit Cartesian direction on device before reusing
  them for groups not incident to the moving atom. The geometry-motion term
  of those groups is exactly zero; the moving-weight term is still contracted.
  Empty primitive slots in a partially active batch write only to scratch,
  not to the tape; entirely inactive windows launch no primitive kernel.
  Non-unit values passes rebuild the tape. A failed values pass retires it.
- Fold the single-component DF incidences with a fixed warp tree in FP64.
  Packing the metric bit with the local center removes a group-table gather.
  The primitive accumulation order and higher-angular fallback are unchanged.

The wider globally tiled Cartesian Becke candidate changes scratch capacity
only. It keeps the per-point prefix/suffix algebra, the exact same point and
pair work, and the separate bounded repair queue. The old 32 MiB cap fits
only 101 blocks at 96 atoms; let a larger explicit geometry lease cover more
concurrent points, bounded by the actual remaining bytes and the point tile.
Measure this independently of any claim to have reduced Becke work.

## Validation and revisit

The standalone raw DF tests use libcint/PySCF and displaced weighted
gradients as independent oracles. The cache regression overwrites the same
device direction pointer and compares successive unit directions with
primitive-streamed contracts, retaining revocation and alias checks.
The isolated compiled DF and grid probes passed 108 cases in Slurm 20818
(eight optional-probe skips). The build-only geometry generator test requires
the checkout's `tools/` tree and passed separately on the host.
The full endpoint gate is raw matrix error at most 2e-8 Ha/Bohr², 288
completed columns, original asymmetry/translation checks, and no numerical
host fallback, with peak selected-card memory and physical response work
recorded. Slurm 20824 passed both complete endpoints on one RTX 5090:

| Provider | Native SCF + Hessian | GPU4PySCF SCF + Hessian | Max raw error (Ha/Bohr²) | Native peak |
| --- | ---: | ---: | ---: | ---: |
| Direct | 194.16 s | 98.57 s | 3.37e-9 | 8,221 MiB |
| DF | 211.01 s | 93.68 s | 4.88e-9 | 6,583 MiB |

Direct still evaluated 1,470,176,973 exact primitive components. Both
providers retained 82,944 mixed grid contractions, and DF retained 288 value
and 288 contraction passes. The native peaks are well below the card's
32,607 MiB. A sampled profiler time is not part of an endpoint timing, and
neither full endpoint yet beats its reference.

Common Becke replay and Direct frozen J/K remain distinct optimization
targets. Do not extrapolate an all-s tape result to higher angular momentum,
other auxiliary bases, or the ordinary energy/force CUDA package.
