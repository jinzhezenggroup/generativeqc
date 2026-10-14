# Experiment: one outer VV10 secant proposal

Status: proposed
Date: 2026-10-14

The 96-atom original-protocol trial at `68f4c5f70` needs four changing-density
VV10 sweeps and 31 provisional Focks. Its 1018.641-second cold endpoint still
cannot complete the 12-call population within 1800 seconds. Reusing or rebasing
inner DIIS histories did not remove this outer fixed-point work.

`GENERATIVEQC_VV10_OUTER_SECANT=1` tries one two-step acceleration after two
converged frozen-inner solves. For density increments r0 and r1 it computes
`alpha = -dot(r1, r1-r0) / dot(r1-r0, r1-r0)`. It requires at least tenfold
norm contraction, a finite denominator above the noise floor and |alpha| ≤ 0.02.
A refusal uses the incumbent proposal. No clipping of an inadmissible coefficient
is used. The once-per-solve attempt is reset on every begin, including failure,
geometry changes and warm replay.

The extrapolation acts on the two converged inner Focks. Their Hermitian linear
combination is diagonalized with the existing resident eigensolver, and its
occupied orbitals form a positive density with the original electron count.
Direct density extrapolation was avoided because even a tiny extrapolation can
introduce negative eigenvalues and invalid dilute-grid densities. A rejected
optional eigensolve leaves the original inner proposal untouched. The scratch
orbital frame is invalidated on either outcome.

Three AO matrices and one scalar are charged in the KS arena and dry-run bound.
A device allocation refusal retries the incumbent smaller arena; host metadata
failures and other runtime errors still propagate. External-ledger execution,
UKS, lower precision, incremental J/K, frozen range exchange and inner-history
continuation do not select this experiment. Only scalar reductions cross to the
host, with explicit synchronization and accounting.

The extrapolated state is a proposal, never a physical observation. It starts a
fresh target energy epoch, and all original physical maximum-residual, density,
energy, electron and final-state gates remain. The original 100-iteration budget
is retained, with the existing bounded ordinary-target tail. The expected saving
is one changing-density pair sweep; a negative benchmark must remain retained.
This experiment is default-off and does not close #2426.
