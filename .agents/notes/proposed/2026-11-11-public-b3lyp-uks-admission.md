# Policy candidate: public direct CPU B3LYP unrestricted curvature

Status: proposed; installed light-element tests pass, public Ni qualification pending
Date: 2026-11-11

## Evidence and decision boundary

The [accepted complete native Ni UKS matrix](../implemented/numerics/2026-11-11-ni-production-full-matrix-acceptance.md)
and existing independently checked coupled light-element UKS/D3 consumers make
the public CPU B3LYP UKS boundary the next separately testable slice. This
working-tree candidate changes semantic admission and public coverage, not
SCF/response mathematics or the nuclear source inventory. The existing
`CompositeRKSResponse.from_native` already binds the genuine unrestricted
owner for polarized states; the public wrapper must preserve that ownership.

Match B3LYP/VWN-RPA semilocal mathematics with the same physical spin, independent
of the graph's user-controlled identifier. Preserve the supplied charge and
multiplicity, same-spin exchange, cross-spin XC, overlap/Pulay response and
all existing convergence/state/resource/lifetime checks. Keep direct CPU,
all-electron FP64 and through-f Cartesian/spherical boundaries. Uncut two-body
D3(BJ), ATM off, uses the existing independently checked additive owner, with
no additional electronic response. This is not ECP, DF, CUDA, a TS adapter or
production-scale admission.

## Rejected broader promotion

An initial candidate also admitted PBE0 UKS. Its real public regression fails
at native unrestricted response-point evaluation with numerical error 5 before
reaching the deliberately insufficient GMRES workspace test. The isolated
reproducer actually exits one in 22.98 seconds. Preserve
`/tmp/gqc-public-uks-pbe0-failure.log`,
`/tmp/gqc-public-uks-pbe0-failure-time.txt` and the initial patch
`/tmp/gqc-public-uks-initial-wide-admission.patch` (SHA
`1c467468523d1dfac153fa2581aea145b3f5d0882f362bacd00a9702a395e84d`).
The full initial batch finished with one failure and 39 passes in 1299.99 seconds,
not an accepted narrower run. Do not weaken the expected error or numeric gates to
hide this failure. PBE0 RKS remains unchanged; PBE0 UKS retains its original
closed public boundary until its distinct point-provider domain is qualified.
This does not select a different chemistry to fulfill the requested B3LYP path.

## Required next acceptance

The narrowed semantic/resource guards pass thirteen executable cases. Real
public B3LYP UKS matrix/H@v/recovery and exactly-once D3 tests separately pass
two cases in 252.16 seconds. Public results now report actual density, basis/grid,
occupation and physical-residual provenance without retaining native owners;
the corrected fingerprint/public-endpoint suite passes nineteen cases in
650.90 seconds. The full pre-commit suite passes. No final
installed/public Ni acceptance or issue
closure follows from the older low-level matrix or this admission edit.
Record a new source-matched installed artifact, actual state/root diagnostics,
independent complete matrices, work/Fock counts, wall time and peak memory
before treating the public Ni endpoint as qualified. Retain all original failed
broader tests and numerical gates; move this policy note to implemented only
when the candidate's required acceptance is complete.

The first complete narrowed run passed 40 cases but failed a newly added
cross-call grid-identity equality assertion (1288.79 seconds). This was not a
curvature failure: native `ExplicitGrid` provenance intentionally includes its
owner address, so numerically identical independently owned grids have distinct
identities. Preserve the owner-bound identity for lease safety; add a separately
framed, chunk-hashed quadrature-content fingerprint for cross-call qualification.
The regression computes this fingerprint independently from the actual private
fixture buffers. Do not remove owner identity from the stationary/response model
or weaken numerical comparisons to make independent snapshots appear identical.

The narrowed full run's forty other cases retain their independent full raw
UKS/D3 gradient-difference gates; the subsequent nineteen-case run exercises
the modified public metadata, raw matrix/H@v, owner cleanup, budget recovery,
geometry freshness and negative admission boundaries. Preserve both actual
process results instead of restamping the failed complete batch as a pass.

## Fresh installed cohort and complete small-molecule suite

The clean implementation source is
`e5c60110175b658459a81c0a7e83dcfd589877d0`. Its CPU wheel builds in 7.70 seconds
through verified sccache 0.16.0: one actual compiler invocation, one miss, no
cache-hit claim. The initial auditwheel repair cannot locate the linked BLAS
library and remains a failed attempt; a distinct normal repair supplies that
library directory to auditwheel without manually patching RPATHs. Installed
imports and dependency resolution are verified outside the checkout.

Repaired wheel SHA256 is
`73e430f9a1c0c6012ffd0fca533fa00865a91a30ce2b22788f392d78e7c9d9df`;
native library SHA256 is
`7ff0721c476b1eddacb4038973b5805cb2d46df784c910ebfa61bcaaf172d7bc`.
The actual installed four-file public/UKS/D3 suite finishes zero:
**41 passed in 1286.66 seconds**, complete measured process 1288.02 seconds.
This includes the independent raw UKS/D3 matrix gates, public matrix/H@v and
exactly-once composition, state/provenance, cleanup and negative boundaries.
These are qualification-process measurements, not an isolated endpoint speedup.

The member/hash-verified immutable build recovery is
`build/qualification/public-ni-uks-20261111/build-recovery.tar.gz`,
1,559 members, 130,258,382 bytes, SHA
`cce8c0a39692997bdb0b7428290a427c755756fcba9c1c1399bcc24b2eec8567`.
It retains the installed files/headers/libraries, wheels, actual compiler/cache
and repair evidence, source results and first failed Ni preflight. It explicitly
excludes mutable live endpoint/test outputs and does not declare Ni acceptance.

## Ni preparation and still-live actual public curvature

For the requested B3LYP/VWN-RPA, spherical external def2-SVP, default-grid,
neutral all-electron NiH3 doublet, the new installed central state matches an
independently converged reference: density error `7.873e-12`, total-energy error
`2.728e-12` Eh and native physical residual `1.261e-13`. Occupations are `(16,15)`
with 50,064 quadrature points. A separate actual native force evaluation agrees
with independent electronic plus D3 gradients to `4.175e-12` Eh/Bohr, with
identical native density. Each preparation SCF takes 123 iterations / 131 Fock
builds; the complete successful preparation work takes 899.93 seconds. All 72
original independent gradient columns are reauthenticated by input/hash/root
gates; minimum occupied-root cross-overlap is `0.9998716754`, above `0.99`.

Three harness attempts remain visibly failed: commutative XC-string ordering;
an incorrect expected exception type at the output-budget guard; and comparing
JSON-loaded list payloads with the corresponding in-memory tuples. None is a
failed numerical gate or a successful public matrix run. Physical XC comparison
uses resolved LibXC terms, preserving VWN/VWN-RPA and exchange-fraction negative
controls. Exact JSON normalization handles the serialization boundary. The
actual insufficient-output contract is a `ValueError` naming
`output_budget_bytes`; no production exception behavior is changed.

The final distinct run reauthenticates the completed preparation, including its
exact old driver/source/input hashes, central state and all reference columns;
it independently reconstructs the three electronic+D3 reference matrices.
It does not reuse native owners, seed a production density, replay a production
matrix or replace production callables. Preparation wall time and work remain
separate retained measurements. The actual public full Hessian has now started;
it performs its own fresh SCF and must report the same actual state fingerprint.
Two further real public HVPs and all original raw-matrix/residual/invariance
gates remain required. The installed artifact stays attributed to `e5c601101`,
not subsequent documentation-only commits.

All three failed drivers/receipts and the completed installed test results are
separately member/hash-verified in
`build/qualification/public-ni-uks-20261111/preflight-recovery.tar.gz`,
25 members, 1,729,999 bytes, SHA
`045803eb3e43e9e69a36e7799ff9cbb0b57e6a5030a34d1a27625ce12b1d1b81`.
The older accepted low-level Ni matrix remains its own source cohort; neither
it nor successful preparation is relabeled as the still-pending public result.
