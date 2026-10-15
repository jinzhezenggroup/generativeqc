# Experiment: confirm an already stationary initialized density

Status: default-off initializer follow-up; qualification pending
Date: 2026-10-14

The inexact-inner 96-atom cold trial (`7369698bc`, Slurm 8302) completed in
1629.527 seconds, slower than the original 1562.126-second frozen initializer.
It reduced total Focks from 40 to 38 but increased actual VV10 pair sweeps from
6 to 7. After the fourth physical anchor, density and residual changes were
already tiny; subsequent compensated energy differences of 1.534e-12 and
1.101e-12 still exceeded the original 1e-12 gate before the final 6.154e-14
change passed. All these observations and the whole-point failures are retained.

When an initialized, complete physical target Fock passes the original density,
RMS residual and maximum residual gates, hold its input density for a further
complete target evaluation instead of applying another sub-tolerance proposal.
The unchanged energy gate, consecutive-target requirement and total solver
budget still decide acceptance. Every J/K, semilocal, AO, diagonalization and
physical validation step still executes; the existing exact VV10 ingredient
cache may avoid repeating pair work for the identical density. A failed energy
confirmation remains a failed/unconverged solve; no equality is fabricated.

The held density cannot borrow the proposed orbital frame or its occupied
factor, so that optional identity is invalidated. Provisional inner operators
never enter this path. Ordinary uninitialized solves remain unchanged. This
is a solver initialization experiment, not default promotion or a #2426 fix
until independent complete energy/force gates and full populations pass.
