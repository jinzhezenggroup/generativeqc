# Decision: bounded exact response XC point-partial reuse

Status: implemented; complete endpoint qualification is tracked separately
Date: 2026-11-13

## Problem

Source-local integral curvature reduces integral work but leaves repeated
fixed-geometry electronic response unchanged. Each induced Fock action rereads
the AO/grid tiles, contracts the same stationary density, then reevaluates the
same linked nonlinear XC derivative program at every point. The preceding real
Cs matrix receipts retain 164/152 induced Fock actions and more than five million
response point visits. Removing nuclear-coordinate integral repetition alone
does not establish practical complete-electronic scale qualification.

## Decision

Retain exact linked XC inputs and output partials for a whole-tile point prefix
fitting a separately admitted byte allowance. The prefix may cover the whole
grid but never implicitly allocates it. No retained AO matrices, response tensor,
new functional algebra, finite difference or derivative floor is introduced.
Unretained points execute the original scalar program. Numeric storage is
bounded by the extra allowance, including all stored inputs and outputs.

Admission allocates only and happens before source/Fock work. First successful
response prepares the bank using the incumbent compiled evaluator. A failed
first invocation leaves the bank unprepared even if some prefix tiles completed.
Later invocations recompute the primal point inputs in the original order and
authenticate every input bit before copying the exact outputs. A changed finite
input fails; it does not trigger an implicit refresh or numerical fallback.
An already prepared bank remains bound across failed invocations, permitting
recovery only when the original point inputs and live owner are restored.

AO readers, density/domain checks, grid/state validators, tangent contractions,
coefficient-response IR and output assembly remain active on every invocation.
Zero-density lazy branches and nonvacuum singularities remain the linked graph's
responsibility. The optimization removes repeated scalar XC evaluation only;
it does not remove all O(points * AO^2) response work or change the coupled
solver, induced Fock count, physical residual or electronic state.

## Composed resource invariant

Reserve actual GMRES and original source/response panels first. The optional
prefix fits both the response owner's cap and the nuclear composition's remaining
allowance; neither cap grows. Every owning layer accounts for newly allocated
bytes immediately without an intervening owner callback that could throw and
leave the parent accounting stale. Perturbation resource queries now derive
response array bytes from the live child rather than a construction-time copy;
solve capacity uses that same current bound. Existing full-matrix integral
schedule priority and all original streamed children/caps remain intact.

Full matrix and HVP C API consumers attempt this allocation after selected
whole-request integral work admission and before numerical preparation. Their
public ABI is unchanged. Counters distinguish point/tile visits, scalar slots
submitted to the linked evaluator and exact replayed slots. A failed scalar
batch may terminate early, so submitted work is not falsely labeled completed.

## Rejected alternatives

- Cache AO/grid tensors or a dense response matrix: different storage/work
  contracts and unnecessary for exact scalar partial reuse.
- Trust a density pointer, epoch or tolerance alone: cached inputs are compared
  bitwise in addition to retaining all original live-owner checks.
- Fill a full-grid bank regardless of remaining response/GMRES capacity: breaks
  the existing composed reservation and loses the bounded streamed fallback.
- Catch numerical or owner errors and stream instead: hides invalid state and
  would turn a failed computation into a success-like endpoint.
- Treat a source-only or small Cs result as complete scale/TS acceptance: those
  are different requirements with unchanged independent scientific gates.

## Evidence and remaining scope

Current source-matched evidence belongs in
`build/qualification/host-response-xc-bank-20261113`; the source-local curvature
predecessor remains sealed. Unit tests compare all prefix schedules to streamed
actions and three independent Libxc finite-difference steps, both spins,
tiling/aliasing, lazy vacuum, exact-field mismatch, late preparation failure,
live provider/lifetime/reentry failures and checked huge-grid planning. Composed
controls verify no numerical work during admission, original response+GMRES caps,
partial/full reuse and exact native D'/W' parity.

Neither a passed small Cs endpoint nor the partial bank opens public ECP saddle
guards. I/Cs molecular saddle-consumer state/raw-H/mode qualification, stationary
mixed Ni/ECP modes, practical complete-electronic 25--40-atom work/wall/memory
and representative requested-level Ni TS remain separate full-goal obligations.

### Terminal evidence

The final producer is
`a915658cfbb47eeb9b8bc8d6b4684ed161be47c90064bec7337ea055fc675a3b`,
with frozen core
`8f7d6b473dde4dcc3a509c04fb627352c6ff91eb4aec8cd8ec6a3297a3011fc3`.
Final runs pass 44 XC, 183 integral-source, 16 composed-response and 123 targeted
Python cases (five other cases deselected), seven native/API/CLI contract groups,
and both real public Cs RKS/UKS raw-H/two-HVP cases. Formatting, lint, compiler
ownership, source identity and Sphinx warnings-as-errors pass. A local one-off
comparison finds both raw matrices and both HVPs per spin array-equal to the
sealed source-local-curvature predecessor, with maximum difference zero. This
comparison is retained evidence, not a permanent CI implementation fingerprint.

Cs RKS/UKS raw-H endpoint times are 104.25/185.06 seconds; complete native H plus
two HVPs takes 152.52/262.21 seconds. Additional retained XC storage is
3,217,536/10,993,248 bytes for the same 33,516-point grids. Independent finest-step
matrix errors remain 9.0146214e-9/5.2988553e-9; minimum occupied overlaps exceed
0.9999967. Fock counts (164/152), response point visits (5,496,624/5,094,432),
response iterations (76/70), and first/curvature integral work remain unchanged.
Measurements run concurrently and do not establish a controlled general speedup.
Inclusive native-plus-independent-oracle driver VmHWM is 572,788 KiB, not isolated
endpoint RSS; no endpoint memory reduction follows from that number. Compiler
cache statistics record 90 requests, one hit and 89 misses, not a warm-cache run.

The predecessor's separate six-case stationary-energy timeout remains exit 124,
not a pass. Original IHH remains terminal at its original 7200-second limit and
nonstationary accepted attempt four. Neither calculation is restarted or relabeled
by this evidence. All four full-goal obligations above remain outstanding.

## References

- [Source-local integral matrix](2026-11-13-source-local-frozen-integral-matrix.md).
- [Current fixed-grid response contract](../../../../docs/developer/dft_hessian.md).
