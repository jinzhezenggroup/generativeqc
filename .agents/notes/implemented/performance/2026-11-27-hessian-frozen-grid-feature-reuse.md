# Decision: Reuse frozen-motion XC features in the dense CUDA Hessian

Status: implemented
Date: 2026-11-27

The Cartesian column first computes XC feature tangents from the moving AOs
at fixed density for the orbital-response RHS. The mixed contractions need
the same tangents plus the features of the induced density at fixed AOs.
Replaying the full moving-feature contraction after CPKS computes the frozen
contribution twice, and copies AO, weight and point motion twice.

When an authenticated all-row Cartesian column is available, add the
fixed-AO density features to the resident frozen-motion features on the same
stream. Check the original density tangent is zero on device and require the
same private transaction. Keep ordinary signed HVPs and non-batched columns
on the original path. The explicit wide lease may spend 32 rather than 16
MiB on fixed XC response coefficients, still within the existing aggregate
device budget and with the original replay path available.

Slurm 20835 compared the two complete native endpoints and GPU4PySCF on one
RTX 5090, using FP64 r²SCAN and the same 82,944-point moving quadrature as
20824 and 20832. Results are fresh SCF plus all 288 Cartesian columns:

| Provider | Native | GPU4PySCF | Max raw error (Ha/Bohr²) | Native peak |
| --- | ---: | ---: | ---: | ---: |
| Direct | 183.14 s | 94.23 s | 3.37e-9 | 8,237 MiB |
| DF | 173.82 s | 94.92 s | 4.88e-9 | 6,639 MiB |

Both pass the 2e-8 unsymmetrized matrix gate, without numerical host
fallback. Both still perform 82,944 mixed grid contractions; Direct still
evaluates 1,470,176,973 exact primitive components and both providers report
2,880 physical induced-Fock actions. The Cartesian versus signed-HVP check
and the fitted cached versus physical solve test passed in Slurm 20836.
All six public CUDA Cartesian tests, including the independent reconverged
CPU gradient checks for the water dimer, passed in Slurm 20843.

Placing switch jets in block-local storage restricted CTA concurrency and
regressed the same-card Direct endpoint from 193.97 to 228.29 s (20834).
Expanding both switch orientations for coalesced reads also did not lower
Becke time per column in the DF column-window profile (20841). Neither
storage experiment is retained. The remaining Becke cost requires reducing
repeated mathematical work, not just changing the switch storage layout.
