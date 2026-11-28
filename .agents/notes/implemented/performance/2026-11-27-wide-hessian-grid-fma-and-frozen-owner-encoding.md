# Decision: Reduce wide Becke arithmetic and frozen-owner gathers

Status: implemented
Date: 2026-11-27

Wide dense Hessians still spend substantial GPU time replaying moving Becke
columns. Enable ordinary FP64 contraction of multiply-add operations only for
the Hessian geometry translation unit. This does not enable fast math, change
the quadrature, or change the SCF/force grid or integral compiler policy.
FMA changes rounding, so the separate geometry oracle and full raw Hessian
gates apply rather than byte identity across builds.

The source-major compact frozen J/K sweep also reread a group descriptor for
every candidate atom. When the window already admits packed offsets and at
most 256 atom IDs, encode the four IDs and projected offset in each owner
record once after upload. Compact the encoded records in their original
order and use the same FP64 reduction. No additional arena or change to the
projected sources is required. Noncompact and larger windows keep the
original records. Encoding atoms in the group table alone does not remove
the indirect load and was not retained.

Slurm 20837 compared Becke FMA alone against the uncontracted build on one
RTX 5090: Direct fresh SCF plus Hessian improved from 183.10 to 174.23 s.
Slurm 20842 isolated owner encoding (182.59 to 176.16 s); the two raw
matrices differ by 4.06e-15 Ha/Bohr² and have identical primitive counts.
The combined build passed the complete same-card comparison in Slurm 20845:

| Provider | Native | GPU4PySCF | Max raw error (Ha/Bohr²) | Peak native MiB |
| --- | ---: | ---: | ---: | ---: |
| Direct | 167.13 s | 94.25 s | 3.37e-9 | 8,139 |
| DF | 164.02 s | 93.51 s | 4.88e-9 | 6,545 |

All 288 columns, 82,944 grid mixed contractions, and 2,880 physical induced
Fock actions remain in each arm. Direct still evaluates 1,470,176,973 exact
primitive components. No numerical host fallback or matrix symmetrization is
used. FMA geometry qualification passed 61 tests (four optional skips).
Five frozen/window J/K probes passed; the narrow-budget Coulomb case reports
`operation not supported` both with and without encoding (20847 and 20848).
These changes do not yet beat GPU4PySCF. The next algorithmic target is to
contract the common fixed-point partition derivatives across columns without
repeating the complete point-pair sweep for each right direction.
