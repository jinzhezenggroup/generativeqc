# Decision: bound expanded second-source compilation by component partitions

Status: implemented
Date: 2026-11-04

## Problem

The isolated Ni one-electron second-source probe exceeded the existing finite
300-second C++ compiler deadline for a unit-charge nuclear-attraction d/f tile:
60 AO components and six coordinate outputs were differentiated into one
expanded scalar/CSE program. Bounded native numeric storage does not bound
optimizer complexity. The earlier complete HVP attempt also terminated with
signal11 in this phase, but the isolated GDB probe did not reproduce that
signal; its cause is not established.

## Decision

Put validated lazy AO-component partitions in the common compiler
`second_order_layout` owner. Expanded CPU second-source lowering selects at
most eight components per artifact, independently of the existing coordinate
partitions. Sum all disjoint subsets with the same fixed weights and physical
atom mapping. Remove the runtime-local partition helper and migrate both
consumers directly. Shared factorized host ERIs explicitly retain their
64-component pages; existing CUDA lowering retains its 64-component schedule.

This changes program granularity, not the S/T/V or ERI recurrence, scalar AD,
normalization, output ABI, physical variational space, arithmetic precision,
weights or derivative semantics. All selected components/outputs are already
bound into program/artifact identity, so no cache identity override or
compatibility alias is needed. The native numeric budgets, finite 300-second
compiler deadline and explicit low-budget ERI fallback remain in force.
The scalar-lowering component limit is exposed alongside execution, primitive,
factorized-component and canonical/ordered shell work counts.

## Evidence and limits

The initial local eight-component d/f attraction probe compiled in 21.77 s and
matched independent Libcint analytic second derivatives to 3.71e-14 absolute.
It exercised the same angular and unit-charge class that timed out above, but
only one component/output subset. This is not complete source, molecule,
installed/public endpoint qualification or an endpoint speedup claim.

Permanent executable coverage checks complete disjoint partitions (including
the 10,000-component ffff domain), eager invalid/overflow rejection, and full
60-component d/f S/T/V weighted HVPs against independent analytic Libcint
Hessians. Signed weights, two-record accumulation, all output coordinates and
common-translation recovery are checked at fixed 2e-10 absolute / 5e-11 relative
and 2e-11 translation gates. There is deliberately no wall-clock assertion,
production-source fingerprint, or exact internal invocation-count requirement.
The new native/partition tests and existing coordinate-layout contracts pass
30 cases in 530.40 s. The compiler structure check covers 586 modules with zero
dependency errors. These source-checkout results are not a complete Ni endpoint.
The complete small-molecule global-hybrid and WB97M-V direct/full-rank-DF
RKS/UKS files pass 52 cases in 625.72 s with eight JUnit warnings, using an
explicit retained warm JIT cache without overriding identities. Existing
compiled native/input, orbit, response and orbital-gap guards pass 98 cases
in 321.35 s; 28 CUDA cases are intentionally unexecuted without a Slurm job.
All applicable staged hooks pass.

The new complete Ni attempt uses the production orbital-gap factory with the
original default restart30/max200 and unchanged true-residual gates. Its
response converges in 17 iterations / 34 actions at 8.110805132056192e-12 in
29.49 s; later second-source assembly remains separate. The first-source phase
took 257.56 s in this cold/invalidation trial, so no endpoint speedup is claimed.
Independent moving-grid PySCF gradient differences at three steps are now
available from fresh atomic reference guesses at the original convergence
controls. Earlier minao/seeded reference nonconvergence remains retained; no
native density is used as a reference guess and no tolerance is loosened.
Generating an oracle does not qualify a native HVP or raw Hessian by itself.

Local receipts are under `/tmp/gqc-2843-2849/`: `ni-second-gdb.log`,
`df-second-tile-eight.log`, and `second-components.{log,xml}`. The original
timeouts and signal failure remain retained. A complete Ni HVP/raw Hessian,
independent moving-grid multi-step checks and default-grid/public qualification
must be evaluated separately; this scheduling change alone completes no issue.

## Rejected alternatives and revisit

Do not raise every compile timeout or memory cap, change chemistry/basis, add a
second one-electron recurrence, or assume the ERI factorized family implements
S/T/V. Smaller programs trade additional native traversals and artifact count
for bounded optimizer inputs. Revisit grouping/factorized one-electron execution
only with complete endpoint timing, semantic work and independent acceptance.
