# Decision: retain DIIS history only across an exact frozen VV10 operator change

Status: implemented, default off; linked-device and complete endpoint qualification pending
Date: 2026-11-07

Superseded on 2026-11-09 by
[retirement of the rejected experiment](../../rejected/2026-11-09-retire-vv10-history-rebase.md).
The implementation described below is historical, not a current runtime option.

The outer VV10 solver clears provisional history before every fresh target. A
later frozen problem differs from the preceding frozen problem by one isolated
additive AO potential. The opt-in `GENERATIVEQC_KS_VV10_REBASE_HISTORY=1` banks the
newest H−1 actual live triples before the clear, then waits for the unchanged
fresh target to choose another frozen solve. It transfers F by DeltaV and R by
(DeltaV D)S−(S D)DeltaV, retaining the existing FP64 matrix owner. It transforms
no energy and cannot certify a physical target or remove a pair sweep.

The prepared owner freezes the request. Density is recorded at the actual
physical insertion head, before DIIS can retire dependent entries. Bank selection
reads the surviving native count/head and tags. Immutable owner/solve identity
binds the basis, overlap, both grids, coefficients, spin and precision; isolated
potential generation binds the operator. Semilocal seed Va is exactly zero.
Imported, warm, mixed, non-Direct, non-Standard, non-ordinary, alternate-metric or
alternate-reducer owners retain incumbent behavior.

One checked lease holds exactly (4H−2)N² doubles plus explicit metadata, capped
at256MiB. The state-capacity query reserves this conservative optional bound;
lazy cold admission consumes only the current owner's admitted promise after
mandatory preparation, or refuses. A retained lease stays charged through warm
and force overlap until normal drained retirement. Default off has no new
numeric allocation/copy; warm has no density-ring writes or bridge.

Snapshot precedes the incumbent count/head clear and preserves Va before XC
can overwrite it. Rebase runs only after finish has copied the original proposal
and tmp1/tmp2 orbital frame. Both scratch matrices are then dead on the serialized
full-Direct nonlocal owner; occupied-XC and fitted-factor consumers are excluded.
The snapshot crosses exactly the scheduled target, even though its frozen flag
is temporarily false. A tail, convergence, refinement, failure or new epoch
invalidates it. The target occupies slot0 and remains untouched. Restored older
slots H−b..H−1 become live only after a full ordered physical Gram rebuild and
all finite/identity checks; count/head publication is last. Numerical refusal
keeps count1/head1, so the next insertion initializes its prior diagonal normally.
CUDA errors retain endpoint failure semantics.

Rejected alternatives are unrebased nonlinear history continuation, transpose
shortcuts in the residual correction, guessed host ring windows, preserved
history in the target solve, and borrowing scratch before its warm-frame copy.
No source-level algebra or local test establishes device parity or an endpoint
gain. Qualification must retain independent small-matrix residual/Gram controls,
original WB E/F gates and each of three warm limits before promotion.
