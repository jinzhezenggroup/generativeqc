# Policy candidate: public direct CPU B3LYP unrestricted curvature

Status: proposed; installed/public numerical qualification pending
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
