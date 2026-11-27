# Decision: Admit Hessian one-electron jobs by bytes and work

Status: implemented
Date: 2026-11-26

## Problem

One-electron admission rejected the 96-H response at 100,000 ordered jobs
with a bare `invalid argument`. This was not an allocation failure: the full
903,168-job census fits if the integral budget exceeds 145 MiB.

## Decision

Count the ordered jobs by angular class, exact linked program multiplicity,
and attraction-center count. Bound descriptor and scalar arena bytes before
traversing the shell pairs for primitive/projection work. Scalar workspace
queries do not expand job records; optional parallel plans and real preparation
retain the same ordered schedule and scalar fallback. The Hessian drops its
default job-count ceiling but honors explicit ceilings and existing byte and
work budgets. This changes neither the derivatives nor their summation order.

## Evidence and limits

- The host planner test exercises the exact 96-H census, the byte boundary
  and the primitive-work boundary. It also retains optional-allocation
  failure coverage.
- Slurm 2666 (node5 RTX 5090) reproduces the 64 MiB integral refusal with
  the required 152,059,904 workspace bytes plus 768 bytes for charges.
  The one-electron planner passes at 512 MiB without overriding `max_jobs`.
  The real-device one-electron suite passes 23 tests (four molecular-probe
  tests need a separate binary and were skipped).
- Slurm 2667 passes 17 public endpoint regressions including Direct and DF
  Hessian/HVP cases. Slurm 2668 completes the matched 38-H Direct r²SCAN
  case in 7.66 s including SCF (6.48 s for Hessian); its 114 raw columns
  differ from the previous native result by at most `4.72e-15 Ha/Bohr²`
  and from the matched GPU4PySCF result by `3.99e-9 Ha/Bohr²`.
- This does not qualify a 96-H Hessian, and the eligibility change is not
  an endpoint speedup claim. Direct induced Fock and raw DF refuse separately.
  See the [remaining capacity investigation](../../proposed/2026-11-25-direct-and-fitted-hessian-capacity.md).
