# Candidate: exclude unused MD-J capacity from lazy LR force admission

Status: proposed; independent complete-endpoint qualification pending
Date: 2026-10-24

## Failure and diagnosis

Stage29 water108 WB97M-V DF reaches force admission but fails with `stationary
CUDA byte budget is not representable`. No endpoint was published. Its exact
record and independent dense-grid oracle remain separate from the qualified
five water96 timing pairs. A dry query of the loaded Stage29 library finds that
the derivative-only LR provider still charges the optional 128 MiB MD-J value
engine. This lease consumes exact exchange derivatives, not Coulomb values.

For 864 public / 900 Cartesian AOs, its quoted capacity is 363,898,409 bytes;
without optional MD-J it is 229,680,681 bytes. The unchanged composite 1 GiB
cap then admits a 256-point tile at 1,072,926,009 bytes in the dry seven-source
inventory. Native integral and nonlocal storage remain charged. The same query
for water96 preserves 256-point admission. These dry results are capacity
evidence, not a successful force or performance measurement.

## Decision and invariants

Pass `reserve_optional_md=false` only for `KsProviderPurpose::derivative_only`.
Retain the existing SCF allowance, Cartesian shape bounds, first-order integral
owners, explicit budget rejection and allocation rollback. The derivative
constructor receives the same bounded core capacity quoted by the dry query.
No user cap, screening, precision, recurrence, convergence rule or grid changes.
The optional J implementation already has bounded admission if capacity remains;
this patch removes its unconditional quote from an exchange-only force lease.

An executable host policy probe must distinguish primary SCF from the lazy
force purpose. Re-run independent mixed WB force, cold/warm, water108 and
water96 composition gates against the rebuilt library before claiming success.
Revisit if a lazy derivative lease gains an actual Coulomb-value consumer; its
required owner must then appear explicitly in the resource contract.
