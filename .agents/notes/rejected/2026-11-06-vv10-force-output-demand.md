# Rejection: borrowed SCF potentials in the VV10 force sweep

Status: rejected; implementation removed before the SCF feature change
Date: 2026-11-06

## Candidate and guard

Publish strict final SCF vrho/vsigma alongside the existing resident nonlocal
lease. Validate the complete seven-ingredient key, final-state owner/epoch,
grid pointers/weights, parameters and schedule before borrowing those outputs
for stationary forces. A matching force packet accumulates phi and xyz (four
fields) instead of recalculating six fields. Mismatch and unsupported producers
retain the complete six-field route. There is no pair omission or new storage.
Phi remains explicit for signed/positive-zero quadrature-weight response.

## Evidence and decision

The actual immutable source is
`d37bc98c962a88c9ce097596ae04f95ec4259004fe00f7929c1941da091ad45c`,
ELF `55eed564d6bbdc972c7435f344e376ff5a58ba4ea02f2d3012f18cf60991e53b`,
over master `b681a1fc3`, before #3020 and without unmerged #3021.

The320000-point RTX5090 component improves about11.53%, but the complete
96-atom WB97M-V/full-Direct/Standard cold+three-frozen-warm endpoints do not
justify promotion:

| Actual job | Cold rollback -> candidate/s | Warm median rollback -> candidate/s |
| --- | ---: | ---: |
| A100 LSF226041 | 111.895463 ->112.448947 |15.005145 ->15.015738 |
| RTX5090 Slurm16592 |193.435453 ->191.009830 |14.290041 ->14.313847 |

Both jobs exit0; cold/warm retain18/1 Focks and unchanged scientific controls.
Each card passes48 independent packet-array cases. Eleven native molecular
cases pass on5090. A separately forced-uncached diagnostic confirms four rather
than six accumulation fields, but still one force pair sweep and identical AO
work. That diagnostic is not an ordinary cached warm timing.

Large retained-reference qualification was not completed. Supplemental UKS
jobs16593/16594 stop on missing packaged `wb97mv_uks` Python AOT artifacts;
they establish neither a numerical failure nor a pass. The sanitizer stage
following that test did not run. A result-tar absolute pytest temporary symlink
is safely excluded during extraction, and the owned transfer server is stopped.

The A100 cold observation is0.49% slower and the warm changes are noise-scale;
a component win is insufficient reason to add lifetime/dispatch complexity.
The prototype is removed from production, not left as an opt-in. Raw endpoints,
the exact rejected source overlay, receipts and limitations are retained in the
`rejected_force` section of the SCF feature benchmark evidence bundle.

## Revisit when

Reopen only with a new endpoint-level reason or an implementation that removes
an actual sweep, not a repeat of the four-versus-six-field component comparison.
Do not mix these old-master timings with later #3020 or #3021 populations.

Reference: issue #2976; `benchmarks/results/vv10-scf-features-20261106/`.
