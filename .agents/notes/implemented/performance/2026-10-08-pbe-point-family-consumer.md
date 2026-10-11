# Decision: Keep PBE point-family lowering opt-in

Status: implemented (qualification pending)
Date: 2026-10-08

## Problem

The admitted PBE point launch still forwards a runtime semilocal-family value
into its canonical device consumer. The older #1113 PBE specialization reduced
register allocation from 255 to 148 without materially improving its measured
same-work complete endpoint. A smaller static footprint therefore cannot justify
promotion by itself.

## Decision

Resolve the prepared PBE physical or admitted signed-response entry to either
generic or family-specialized canonical point lowering. The explicit
`GENERATIVEQC_CUDA_XC_PBE_POINT_SPECIALIZATION=1` selects specialized lowering;
unset or `0` retains generic. Other families retain generic lowering. The
prepared physical entry also determines the batched point entry, so the two
cannot select different family identities. PBE0's semilocal exchange scaling
remains in the canonical consumer; exact exchange remains in J/K.

## Invariants

- No new functional formula, precision, response capability, point tile, AO
  map, reduction order, or batch admission is introduced by the switch.
- A changed geometry prepares a new owner and applies the same selected policy.
- Unsupported pairs and invalid switch values on PBE owners fail during preparation.
- Generic remains the default until a source-matched real-device comparison
  passes independent numerical and complete E+F endpoint gates.

## Evidence and limits

The initial source commit `af5e192225d6498de07b6e0d6034f9d293c21a00`
passed host generation and selector checks, including 24 remote C++ tests.
Those tests cannot establish device resource use, dynamic traffic, numerical
acceptance on CUDA, or endpoint profitability. #1113's negative complete
endpoint result belongs to its older binary and is historical context only.
The fixed-work `--point-specialization` PBE0 pair protocol records both arm
identities and requires equal tile and batch requests, independent reference
vectors, actual SCF work, and retained warm/moved preparation outside replay
timing. This note makes no speedup claim.

The source-matched H100 `sm_90` build at `e07873007310620306f62dad36a52d7f67c38e2c`
passed the native semilocal E/V and point-batch gates with the switch both off
and on. For the physical four-feature unbatched PBE entry, `cuobjdump` reported
142,752 versus 9,232 SASS rows, 255 versus 162 registers per thread, and 576
versus 96 bytes of stack for generic versus specialized. These are static
properties, not dynamic local traffic or endpoint benefit. The retained logs
are under `/inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/vibeqc-2072-point-family/runs/issue-2072-pbe-point-h100-e0787300-20261009/`.

The H100 complete E+F attempt did not qualify. Its `portable_cuda` production
shell profile has no kernels, so enabling shell AOT and building the PBE0
stationary force manifests still leaves Direct J/K without a retained shell
derivative lease. The uninstrumented 384-AO setup also failed at its first
cuSOLVER `Xsyevd` launch; a diagnostic interposer synchronizing only that
first launch let SCF continue, but does not establish a production fix or a
valid timing run. Nsight Compute returned `ERR_NVGPUCTRPERM`, so dynamic local
traffic remains unmeasured until an authorized counter-enabled environment is
confirmed. Keep these independent blockers distinct from the point-consumer
correctness gates and do not promote the specialization on static resource
counts alone.

A separate 12-atom feasibility probe on the same H100 build failed during
`batch_prepare` in `CUDA KS matrix product` with the CUDA 12.9 runtime. Loading
the installed CUDA 12.8 runtime libraries let both owners prepare, but this
mixed build/runtime diagnostic did not complete E+F and does not establish the
cause of the original failure. Its first force setup spent sustained host time
in `_stationary_cuda.py` expanding four-center AO derivative task pages; a
two-minute Python stack sample identified that path. The bounded probe was
ended before completion because this smaller-domain fallback cannot replace
the required 48/96-atom production shell derivative and endpoint gates. The
retained diagnostic is under the H100 run's `point-family-12-12.8-bg/` directory;
`complete.txt` is absent and `exit.txt` is 99.

A separate source-pinned H100 48-atom cold **E-only diagnostic** completed with
the CUDA 12.9-built binary loading system CUDA 12.8 runtime libraries. Generic
and specialized owners both converged in 19 iterations and 19 Fock builds;
their independent reference energy errors were `2.96e-12` and `2.73e-12` Eh.
Each recorded 1,179,648 grid points, 4,608 tiles of 256 points, and identical
AO selection/visit counts. Their single cold energy timings were 55.94 and
56.02 seconds. This is neither a repeat-based speedup comparison nor a
complete E+F result, and the mixed runtime is diagnostic only. The complete
JSON and input/library hashes are retained under the H100 run's
`eonly-48-12.8/` directory; `exit.txt` is 0 and `complete.txt` is present.

## Revisit when

The point-family evidence verifier distinguishes an executed negative timing
result from default promotion. It independently recomputes the full paired
timing assessment and all numerical/work gates; default evidence auditing can
retain a within-noise or slower result, while `require_improvement=True` still
requires significant improvement in both replay phases. The older scheduling
publication verifier retains its improvement requirement. Synthetic fixtures
exercise negative, invalid and positive cases and are not device measurements.

A source-matched CUDA build shows selected entry identity, generated and device
footprint, registers/stack and dynamic local traffic, plus generic versus
specialized point GPU time, full XC time, and cold/warm/moved complete E+F with
the required independent numerical gates. A static reduction alone is
insufficient.

## Integration with the shared automatic CUDA point traversal

The merged #2261 bridge owns a single bounded `evaluate_point_tiles` traversal
for curated and automatic point providers. Curated specialization carries
`static_family` in `CuratedPointEvaluator<static_family>`: both `physical_point`
and `response_point` forward it into the existing canonical algebra, while the
global `evaluate_points` entry retains its fourth family parameter. Physical
batches bind the same prepared family identity. Automatic providers retain
their separate evaluator and their existing closed batch/response admission.

Keeping a second private traversal would duplicate tile/spin layout ownership;
dropping the family at the shared adapter would lose specialization for both
physical and response consumers. The composition test executes the emitted
adapter/traversal with host stubs over both spin layouts and ragged batches.
That test verifies template/layout composition only. Its compilation requires
the native compiler/cache fixture, and it supplies no GPU numerical, dynamic
traffic or complete-endpoint evidence.

## References

- Issue #2072; scheduling-only sibling #2073; historical #1113 and #1125.
- `docs/developer/xc_native_cuda.md` and
  `benchmarks/pbe0_xc_tile_pairs.py`.
