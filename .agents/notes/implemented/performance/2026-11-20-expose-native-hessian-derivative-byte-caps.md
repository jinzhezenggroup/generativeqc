# Decision: Expose native Hessian derivative byte caps

Status: implemented
Date: 2026-11-20

The linked CPU owner can reuse coordinate first-integral panels and frozen
integral curvature across Cartesian columns. The 25-atom, 205-AO all-electron
UKS diagnostic passed explicit work admission, but the fixed 128 MiB response
reservation could not hold its optional coordinate panel alongside the live
response state and solver workspace. The bounded column fallback planned
209,194,620,300 first and the same number of curvature ERI primitive visits.
Giving that fallback a larger *work* cap does not make it reuse integrals.

The Python Hessian and HVP endpoints let callers explicitly select the already
existing native aggregate and response byte reservations. Both remain positive
signed-int64 byte counts, with the response reservation contained in the
aggregate. Keep the defaults (512 MiB aggregate, 128 MiB response), integral and
grid byte caps, GMRES workspace limit, exact source identities, response gates,
and default `2**40` map-work cap independent. Other owners reject these native
keywords. An unavailable optional panel still falls back to the bounded column
plan, subject to a fresh full work admission; no scientific owner changes.

Positive and negative tests exercised the native public ABI with actual UKS
matrix/HVP agreement, an insufficient aggregate reservation, and malformed
budgets. A 768/256 MiB explicit scale trial is separate from the default arm;
it must demonstrate its actual admitted integral schedule and complete endpoint
work, raw derivative comparison and peak RSS before any scale claim. Neither
cap is a primal SCF or process RSS limit.

Related: #2844. Reproducers and traces are under `installed-ae-scale-matrix*`
in the November 20 qualification evidence root.
