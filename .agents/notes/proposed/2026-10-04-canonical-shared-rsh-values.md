# Experiment: join canonical full J/K and range K preparation

Status: proposed (default-off implementation; qualification pending)
Date: 2026-10-04

## Scope and mechanism

The full-TZVPD endpoint uses canonical Cartesian values. Its existing fused
RSH facade admitted only the rejected-through-f bounded value experiment,
so the normal path transformed density and traversed canonical candidates
separately for full J/K and LR K. The [structural priority](2026-10-04-tzvpd-shared-full-lr-work.md)
requires removing actual repeated work while preserving scientific sources.

`GENERATIVEQC_CANONICAL_RSH_VALUES=shared` (alias `1`) opts into a joint
canonical value enqueue when canonical Cartesian storage is retained and a
complete generated full-range value owner does not already have priority.
It never enables the bounded-value switch. Missing policy, source capacity,
extra budget or optional allocation keeps the separate existing sources.
Admission is frozen by the retained optional buffer; execution does not read
the environment or allocate memory. Checkpoint/resource policy includes the
new selector, so changing it requires explicit warm-restart admission.

The joint enqueue projects density once, enumerates each canonical candidate
once, and emits independent J_full, K_full and K_range matrices. The existing
exact product predicate, sorted/prefix or dense domain, orbit scatter and
spin normalization are reused. Each output retains its own final projection.
For total orders >=5 the compiler contracts both radial values within one
primitive loop, sharing product centers and Hermite coefficients. Full and
requested-range Coulomb moments and their contractions execute separately;
no SR integral is formed by subtracting full and LR values. Low-order scalar
specializations are retained. Mixed integral arithmetic is not admitted.

This step does not implement cooperative reuse across AO components, a new
force-class queue, or fused derivatives. Generic shell-class templates already
exist; no claim to invent them is made. Both force consumers remain unchanged.
PR #1841 independently owns the single-screen compact force-page experiment
(audited at `6d7f6f02a`); it currently changes full-range derivative scheduling,
not LR scheduling. Reuse that owner after qualification instead of creating a
second force classifier here. Its device performance is not yet established.
The preparatory reachable/convolution experiments remain independent controls;
initial device/endpoint comparisons must keep them off.

## Work and storage

For N admitted AO quartets, each with P_q primitive products, and common
preparation H_q, the changed categories are two candidate traversals to one,
two density preparations to one, and high-order 2*sum(P_q*H_q) toward
sum(P_q*H_q). Both radial evaluations and all three matrix scatters remain.
Actual candidate/radial counts are checked by the native fixture; molecular
primitive counts and FLOPs remain null until an actual measured census exists.
No speedup is inferred from these formulas.

The additional retained output is 2*B*M_cart^2*sizeof(double), reserving both
spin blocks for a batch B. At M_cart=2048 and B=1 this is 64 MiB. It is admitted
only after all existing canonical metadata and generated value/force owners,
and is charged to the same explicit device budget. Optional-allocation rollback
fences and preserves earlier owners. No new quartet list is stored. Primitive
radial calls consume their auxiliary sequentially, but linked stack/register
resources and actual traffic still require measurement; source lifetimes alone
do not establish reduced GPU storage or occupancy.

## Independent qualification

The added host test compiles the emitted pair consumer and separate consumers
with verified ccache. Values and xyz jets at orders 5--12 are compared with an
independent Gaussian Laplace/Wick oracle for Full/LR/SR, diffuse exponents,
repeated/distinct centers and common translations. It also compares each paired
result with the corresponding separate result bitwise in its no-FMA host build.
This is not CUDA execution or a performance result.

The n2 CPU-only qualification passed **481 tests**, including 70 paired
primitive/policy cases and 16 source-executing admission/fault cases. Admission
tests cover both spins' reserved capacity, batches, exact/insufficient budgets,
prior SPD versus later through-f owner accounting, complete-owner priority,
host/device allocation failures and propagation of non-allocation errors.
The existing optional-rollback fixture now also uses the verified ccache helper.
Receipts are retained under `.artifacts/canonical-shared-rsh-20261004/`.

`--shared-rsh-values-only` requires explicit admission and actual nonempty joins,
both spin modes, Cartesian/projected bases, SR/LR, and screened/dense rows.
Independent CPU integrals supply all three matrix oracles. It compares actual
candidate and radial counts with the two separate canonical calls; it also
retains alias and numerical-status gates, disables the selector after preparation
to exercise frozen admission, and denies the final allocation through the real
device resource ledger to verify earlier owners and their charges survive.
Merely setting the
selector without reaching the candidate fails coverage. Device execution,
sanitisers, frozen policy checks and complete original/displaced cold plus
five warm E/F calls remain required, then larger full-TZVPD endpoints.
Keep the 1e-8 Eh / 1e-7 Eh/Bohr endpoint gates and record cold iteration changes.
