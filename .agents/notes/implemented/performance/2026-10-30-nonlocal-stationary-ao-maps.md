# Decision: bind nonlocal force AO reuse to the exact quadrature

Status: implemented
Date: 2026-10-30

The previous stationary map admission excluded every nonlocal functional, even
though the existing derivative-jet map can serve its semilocal force work.
This extends the policy in [the earlier capacity note](2026-10-30-stationary-ao-capacity-cliff.md)
to nonlocal profiles, superseding that note's nonlocal exclusion only.

Keep the original work threshold, 16-MiB optional cap, observed occupancy gate
and complete derivative-jet discovery at cutoff 1e-16. ECP remains excluded.
The nonlocal force loop may reuse the map only when its point pointer and full
span equal the frozen semilocal quadrature. Distinct VV10 grids stay dense in
that loop even when counts match. Minimum-capacity execution, immutable geometry
ownership and allocation-failure fallback remain unchanged.

The 48-atom PRO 6000 WB97M-V test compares shared and distinct Treutler grids
against the prior dense binary, with cold, warm and displaced geometry. Maximum
force differences are 1.01e-12 and 4.37e-13 Eh/Bohr respectively; energy
differences are below 7e-13 Eh. Trace receipts prove map discovery on cold and
geometry replacement, zero rediscovery on warm replay, and indexed execution
only in the candidate. Complete 3/12/48/96-atom matrices and independent oracle
qualification are retained with the DF projection evidence.

This change does not fit long-range exchange or alter either quadrature.
Native WB97M-V DF still uses fitted full-range plus exact long-range exchange;
GPU4PySCF fits both. Keep that approximation difference visible in comparisons.
