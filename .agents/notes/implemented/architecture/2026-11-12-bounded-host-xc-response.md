# Decision: consume linked XC derivatives through a bounded native AO action

Status: implemented (fixed-grid semilocal action only; no endpoint promotion)
Date: 2026-11-12

## Problem

The [build-time scalar package](2026-11-11-host-hessian-scalar-aot-leaves.md)
does not contract a density direction into an AO Fock response. The existing
native CUDA response bridge and LDA/PBE host point selector cannot be relabeled
as a CPU physical B3LYP second-response owner. A Python-free stationary owner
needs an actual native contraction, not another wrapper around Python response.

## Decision

Add `HostGgaXcResponse` to the optional, CPU-only static target. It consumes
the existing canonical XC feature Hessian and coefficient-response AD programs
by exact registry identity. It does not duplicate functional or sigma-potential
derivative formulas. Only elementary real-AO density/gradient contractions,
feature JVPs, a feature-Hessian action and AO accumulation belong here. The
current linked inventory is physical Gaussian/VWN-RPA B3LYP, not a new
named-functional solver or a method-selection fallback.

Reference density, grid and real AO representation remain owned by the prepared
stationary consumer. A consecutive-point tile reader supplies AO values/first
derivatives and weights. A live-state validator runs around tile reads and
before publication; it must authenticate geometry, method, spin, representation,
grid and state identity. RKS uses a total-density matrix; UKS uses two
occupation-one density matrices and retains cross-spin response.

The scratch query checks shape and arithmetic overflow before allocation.
A single exact-size numeric allocation contains tile panels and transactional
AO output staging. Storage scales as spin AO matrices plus tile points times
AO count and a fixed feature program closure. It does not scale with the full
grid. Borrowed state, callback objects, provider workspace and process RSS are
explicitly outside this query, as are compiled scalar stack and allocator
overhead; they must be charged by a complete endpoint owner.
Every action still visits the entire grid and performs work proportional to
point count times squared AO count. Diagnostic action/point/tile visits are
semantic work, not performance evidence.

## Safety and domain

The initial positive-density cohort below is preserved historically. The
[native coupled-response follow-up](2026-11-12-native-host-coupled-response.md)
extends the consumer to the unchanged production graph's lazy vacuum branch;
it does not retroactively qualify that extension with this earlier evidence.

Every sampled spin density must be strictly positive with representable
coefficients. Vacuum/spin nodes, invalid densities and nonfinite intermediates
fail closed; no floor, derivative clipping or alternate formula is introduced.
This limited point-domain admission must be extended and independently qualified
before claiming a complete molecular response domain. Signed finite quadrature
weights are supported. Every reader output element must be initialized.

Failures publish no partial AO output. A failed reader, owner invalidation,
nonfinite late tile or scalar failure releases the invocation guard and permits
recovery. Output may alias the direction because publication is transactional;
it may not overlap the borrowed reference density. Concurrent/reentrant actions
on one consumer are refused. No compiler, Python callback adapter or reference
engine is invoked by the linked native implementation.

## Rejected alternatives

- Do not retain an unbudgeted full-grid AO or point-Hessian cache to make a
  bounded scratch query appear cheaper. A future optional resident path needs
  identity/lifetime checks and this streaming fallback.
- Do not fill invented coefficient inputs. The response Graph eliminates primal
  density-potential inputs: it consumes `dv0`/`dv1`, not unused `v0`/`v1` in UKS.
  The consumer binds the actual reachable labels and refuses a different ABI.
- Do not bypass an invalid state token, soften point-domain gates, reuse a CUDA
  method selector for B3LYP, or replace missing native work with Libxc/PySCF.
- Do not enable a native TS CLI or public Hessian capability from an AO XC action.

## Qualification

`tests/python/test_host_xc_response.py` compiles and executes the real linked
scalar programs plus C++ consumer. The independent oracle differentiates
Libxc first-potential AO matrices at three declared fourth-order steps
(`1e-3`, `3e-4`, `1e-4`), with fixed `3e-9` absolute / `3e-8` relative gates.
It never calls the production response algebra to construct reference values.
Controls cover both spins, nontrivial cross-spin response, spin exchange,
restricted/equal-spin equivalence, tiling, bilinear symmetry, aliasing,
overflow/budget refusal, invalid identities, undefined point domains, lifetime,
partial/failed readers, late nonfinite failure and reentrancy recovery.

The initial compile probe failed a signed-size warning under `-Werror`; the
next probe correctly exposed an assumed unreachable-input ABI. Both logs remain
retained, and neither failure changes a numerical tolerance. An initial broader
test command omitted the native library selection, producing missing-protocol
setup errors and native-only skips; it is not a scientific response failure.
The corrected command explicitly selects the existing immutable CPU library.

Evidence is retained under `build/qualification/host-xc-response-20261111/`.
The completed selected leaf/consumer tests report **39 passed in 29.01 seconds**.
The corrected broader run actually exits zero with **252 passed, two skipped
in 44.36 seconds**. The skips are explicitly opt-in expensive molecular UKS
saddle oracles, not successful TS controls. The separate real SiH3 run remains
unaccepted until its original complete gates and process exit are observed.
Changed-file pre-commit and strict Sphinx warnings-as-errors both exit zero.
The CPU-only root target actually builds through verified sccache 0.16.0;
verbose commands are retained. The static consumer's numeric scratch is not an
installed endpoint or peak-memory receipt. No frozen IH2/SiH3 source or shared
library is rebuilt, and no real GPU test is run.

The earlier scalar-only archive is retained separately at
`build/qualification/host-hessian-aot-20261111/frozen-scalar-only/` with its
original SHA256
`7e11479e56a915ded1fdba855b1448da2f474544b6642d201331a989afdc8506`.
Removing the new consumer object from a separate copy of the deterministic
archive and regenerating its symbol index reproduces that exact hash; the
earlier qualification receipt is not rewritten to describe the expanded target.

## Remaining ownership

The target is still not installed or linked into the public core or CLI. Direct
J/K action, coupled occupied/virtual solves and nuclear RHS, metric/Pulay and
moving-grid/integral assembly, ECP derivatives, stationary state/resource
ownership and complete endpoint qualification are not supplied by this class.
The requested Ni saddle, final frequencies and installed Python-free TS CLI
remain separate gates. Revisit resident caching only with measured complete
endpoint work and an explicit owner-bound storage allowance.

## Completed final qualification and local recovery

The final repeat selects sccache for all compiler-fixture builds and actually
exits zero: **252 passed, two opt-in saddle skips in 52.00 seconds**. The final
root static build, changed-file pre-commit and incremental strict Sphinx build
also actually exit zero. The expanded static artifact is 240,326 bytes, SHA256
`3136469387d7c7042b9636c66bcda5875a51e496aaece9c384e684a42bf83a98`.
`build/qualification/host-xc-response-20261111/receipt.json` authenticates the
uncommitted source cohort explicitly; it does not claim a clean published
checkout or a complete endpoint. Its independent gates are unchanged.

The separately completed recovery
`build/qualification/host-xc-response-recovery-20261112/host-fixed-grid-xc-response.tar.gz`
contains 2,617 members and 5,021,599 bytes, SHA256
`068e1bdc321fd2f63f070459f0e71a6468c105e8a23ec1126229258e68c4cf86`.
Its manifest SHA256 is
`8b4670a105b55b42665de2cf7b9f49265cf51ae9e36c3f6d247cc10e819a40e4`.
The generator actually exits zero and rehashes every member and original. It
retains compiler sources, native source, executable tests and final test-built
assets, static artifacts, all completed logs and qualification receipts; it
excludes live scientific outputs. Complete native CPU Hessian, native TS CLI
and public-admission qualification flags remain false. No release or remote
asset is created.
