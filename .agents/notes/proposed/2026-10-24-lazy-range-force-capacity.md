# Candidate: exclude unused MD-J capacity from lazy LR force admission

Status: implemented on the candidate branch; independent endpoints qualified
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

## Current-source qualification

Measured source `ebe88719525b70aaf00b9c62267e165778c0adeb`, core SHA-256
`5f31db0ea769f548accc73d860f9a457cdf7459339b76c2c92e45b7c1ce94a11`:
all 14 new water96/water108 cold/warm, Fast and displaced calls pass independent
energy/full-force gates. Water108 now completes at 685.370 / 55.518 s versus
GPU4 501.614 / 173.594 s; cold performance remains behind at that larger size.
Water96 Standard 312.862 / 38.328 s and Fast 312.893 / 38.377 s compare with
GPU4 379.866 / 130.868 s. These single bridge observations must not be pooled
with the historical Stage29 five-pair population. The original failed record
remains unchanged, including its absent completed binary receipt.

Slurm jobs 13858/13860/13877 own the larger/bridge/moved endpoints; 13859 repeats
24 Direct/high-angular, 36 range, six fitted-WB Fast, one failure-injection and
native independent composition/factor cases, all passing. Verified sccache
0.16.0 supplies the build (13857); 38 host checks pass. The original two-grid
water108 oracle is reused under a retained explicit contract after verifying
unchanged drivers, geometry, scientific inputs and package versions.

The [retained campaign](../../../benchmarks/results/fitted-range-cold-20261024/README.md)
authenticates complete raw endpoints, unchanged historical failure, manifests,
binary/build/cache/allocation/test receipts and both production-equivalence
bridges. Its offline reproducer independently checks all 50 complete calls.

## Current-parent bridge

Source `b69d38fb8bf6c6c369f63917ef8f6032708a1a15`, core SHA-256
`2093ae0501d80e3b885897606b083ff5fde0ba4ae25762963664534bcb9326e2`, integrates
the newer ordered/affine DIIS, XC and automatic range policies without changing
this derivative-only capacity repair. All 66 new complete calls independently
qualify. Water108 still completes under its unchanged cap at 673.204 / 54.855 s
versus GPU4 492.145 / 171.437 s; the larger cold gap remains open. Water96 WB DF
Standard is 317.736 / 39.039 s versus 385.779 / 132.806 s. These are single
observations, not a replacement five-pair performance population.

The retained bridge includes seven-method Direct/DF, Fast and displaced results,
plus native DIIS/composition/factor checks and 172 passing GPU Python tests.
Its semilocal screen accidentally retained spatial-order value 2 from a separate
local candidate absent from this narrower PR: that setting skips spatial SCF
admission here. Keep those receipts and controls visible and do not label them
current-default timings. The separate supported-spatial screen and restoration
of the DF-charge/spatial composition are ongoing performance work, not changes
to this capacity claim. `reproduce_integration.py` rechecks the complete new
population separately from the historical 50 calls.
