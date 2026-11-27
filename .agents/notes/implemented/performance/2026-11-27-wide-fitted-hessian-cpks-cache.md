# Decision: Admit the FP64 CPKS cache for wide fitted semilocal Hessians

Status: implemented
Date: 2026-11-27

The ordinary fitted dense path caps the optional orbital-response matrix at
512 KiB. For the 96-H r²SCAN case this repeats the physical induced-Fock
operator 4,608 times. The raw DF tape eliminates a separate integral replay,
not the orbital-response iterations.

Use the established wide Direct cache gate for fitted restricted semilocal
Hessians only when the caller explicitly admits the wide workspace profile.
The fitted arm must have no exchange, ECP, correction, or nonlocal grid. The
rotation-to-RHS amortization, actual response and aggregate memory budgets,
allocation fallback, and independent physical residual for each column
continue to apply. The directional and ordinary fitted paths are unchanged.

Slurm 20832 compared both native builds and GPU4PySCF on the same RTX 5090.
Fresh SCF plus all 288 FP64 Cartesian columns took 211.41 s without the
cache, 183.99 s with it, and 93.47 s in GPU4PySCF. Both native results use
the same 82,944-point moving quadrature and STO-3G auxiliary basis, with no
symmetrization or numerical host fallback. The cache stored 42,467,328 bytes,
reduced physical induced-Fock actions to 2,880, and checked all 288 physical
residuals. Peak selected-card use rose from 6,581 to 6,625 MiB, and the
maximum raw error against the independent matrix was 4.89e-9 Ha/Bohr²
against a 2e-8 gate. Neither route changes the 82,944 grid mixed contractions.

The public 36-H PBE regression compares the wide cached matrix with the
ordinary physical solve and requires every column's physical residual check.
The cache is a measured improvement, not a GPU4PySCF speed win; repeated
moving-grid weight work still requires a separate optimization.
