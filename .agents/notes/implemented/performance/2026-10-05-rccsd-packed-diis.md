# Decision: restrict pair packing to DIIS storage first

Status: implemented, opt-in; frozen and retained-RHF complete endpoints qualified
Date: 2026-10-05

## Representation and scientific boundary

#1902's largest straightforward capacity target is two DIIS histories, not
the generated CC equations. Let m=ov, F=m+m², P=m(m+1)/2 and C=m+P. The
restricted involution `(i,j,a,b) -> (j,i,b,a)` has m fixed points and
(m²-m)/2 two-element orbits. Pair labels x=iv+a,y=jv+b give a triangular
storage map with weights one/two. The compiler owns the mapping and FP64
rounding-projection expression; generic CUDA history consumers own execution.
This uses the same physical orbit as Lambda's packed coordinates, without
changing Lambda's ordering or independently qualified solver.

For spin-summed excitations E_ai, the commutator with E_bj is
`delta_ib E_aj - delta_ja E_bi = 0`, since occupied and virtual domains are
disjoint. Thus the two simultaneous-pair orderings are the same excitation;
the restricted symmetric T2 coordinates and physical projected R2 have this
involution. A linear DIIS amplitude combination preserves it. The full
Frobenius error metric contains each non-fixed orbit twice, proving the one/two
weights for history errors; the same storage orbit and direct expansion serve
amplitude histories. This statement assumes a physical restricted Hamiltonian,
not arbitrary supplied blocks or exact bitwise symmetry of lowered arithmetic.
Those separate input and rounding domains are audited below.

For symmetric tensors the weighted packed dot is exactly the full metric in
real arithmetic. The changed FP64 summation order needs independent numerical
gates. Initial supplied amplitudes require bitwise symmetry: an arbitrary
supplied tensor is never projected merely because the method is restricted.
Generated iteration tensors admit only asymmetry at most
`64*eps*(1+max(abs(first),abs(second)))`. A mean then selects the symmetric
coordinate. Equal members and fixed points bypass the mean so subnormals and
ordinary exact values survive bitwise. This is an explicit rounding policy,
not permission to alter the physical residual tolerance or factor precision.

The independent expanded physical replay remains mandatory; packing does not
change any CC residual/Jacobi/triples/response equation. All native/force gates
must be qualified, including early trajectories and independent determinant
residuals. Total energy agreement alone cannot validate the new representation.

## Budget and fallback

Amplitude/error histories use a separate allocation so a refused representation
can be freed before replacement. On excessive iteration asymmetry, retain the
current full amplitudes/residual, discard the packed history and restart full
DIIS. If full history cannot fit or its allocation alone fails, continue with
bounded Jacobi updates and unchanged physical acceptance. Real CUDA/arithmetic
errors propagate. Construction and every injected setup failure must release
both allocations, stream, events and provider state.

The source-derived history payload changes from `16*h*F` bytes to
`16*h*C+C` bytes, before alignment and scalar audit payloads. The C term is the
one-byte orbit-weight vector, not an omitted conversion scratch. At ethane230,
m=1989 and h=8, the nominal saving is251,083,404 bytes. Frozen endpoint records
measure251,083,520 bytes after actual alignment; keep model and observation
distinct.

Initial host symmetry admission makes O(m²) bitwise comparisons. Each packed
history insertion reads both full amplitude/error tensors and writes the two
packed rows plus weights: `16*F+16*C+C` logical bytes. The generic map has no
O(m²) index table, inverse-square-root lookup or unpacked-history temporary.
The audit uses a block reduction and one atomic maximum per block, not one
global atomic per pair. Audit fences and transfers are included in actual
endpoint time and counters.

If L is the sum of live row counts over accepted history insertions, Gram
summands change `L*F -> L*C` with `L*C` extra weight multiplications. Old-old
Gram dots remain reused by the #1900 chronological ring. Extrapolation still
performs chronological sums for F full output coordinates and writes ordinary
T1/T2 directly; there is no claimed halving of CC or extrapolation FLOPs.
Packing/refusal work stays in diagnostics. Logical bytes are not measured
device bandwidth, and kernel/task counts are not FLOPs.

## Scope and rejected expansion

Current/last/residual arrays and public snapshots remain full. They feed
full-layout contractions; packing them now would immediately add conversion
and complicate independent replay for less storage benefit than histories.
No UCCSD representation or precision/provider policy is changed. Revisit
further packing only when an actual consumer can operate on the packed layout
and complete endpoint evidence supports the added complexity.

## Evidence and remaining gates

Job2320 passes orbit/metric/roundtrip/emitted-host, resource-unwind and ring
metadata tests; an unrelated stale canonical-admission test harness failed
and was repaired in parent #1916. Job2323 passes all10 host cases (one explicit
GPU skip), compiler/SCF/native architecture, promotion, metadata and CUDA
ownership checks. The owner tests include packed/full setup failures and
full-history/Jacobi resource refusal. Initial build2317 caught a non-rational
TensorIR coefficient; use exact Fraction(1,2) before rebuilding in2325.
Build2325 passes. Job2326 passes29 real-device solver/generated-consumer cases,
two Gram/ring cases, and both independent small-water all-coordinate force FD
and failure-publication cases. The complete-force probe requires packing to
remain active, so these gates cannot silently qualify a full-layout fallback.
All changed production/compiler/test sources match the frozen n2 checkout by
SHA-256. Full energy/force/budget-admission evidence is still required.

No performance result or default promotion is inferred from these host gates.

## Frozen endpoint result and selection

Jobs2327–2329 pass complete energy, full force and constrained-budget admission;
postprocessing2331 passes all-repeat energy/replay, all-component force and the
independent two-coordinate/two-step large FD re-audit. The complete records,
identities and work accounting are retained in
`benchmarks/results/cc-packed-diis-1902/`. Small-water all-coordinate independent
FD is separate; the large check does not qualify all nuclear coordinates.

Ethane device/complete-energy capacity falls251,083,520 bytes. Complete force
peak remains7,107,400,123 bytes because later response phases dominate it. Full
history is refused at a2,735,193,944-byte complete scalar budget, while packed
history converges there. Unpublished rejected-run work stays null.

There is no complete wall-time benefit: large CCSD medians145.885→145.959 s;
the tiny-water CCSD/DIIS paths regress slightly. Large DIIS alone saves about
11 ms. Keep packing opt-in as a capacity tool, preserving full history as the
ordinary latency choice. Do not extend packing into full contraction state
without a consumer and new complete-endpoint evidence.

## Current prepared-provider integration

The parent subsequently incorporated master's shared prepared contraction
owner. Retain that owner, its full/tail descriptor charges and precision guards;
do not restore the older method-local GEMM callback when resolving the packed
history merge. Both numeric allocations coexist with descriptor storage, and
the history-refusal bound includes the retained provider/descriptor capacity.
The host ownership/lifetime fixtures now audit both allocations. Job2339 passes
all 38 current admission, binding and lifetime cases on each integrated branch,
plus compiler, SCF and native architecture checks. New real-GPU qualification
is separate from the frozen2327–2329 endpoint receipts; those receipts keep
their original binary and source identity.

The prepared-provider version `1b63282e1` subsequently passes build 2334,
real-device and independent small FD 2341, Lambda/factor 2348, complete
energy/force/budget 2342–2344 and report 2346. The 251,083,520-byte device/energy
saving is unchanged, with the same successful bounded endpoint and no
complete-force capacity reduction. Complete timing differences are dominated by
unmodified RHF variation; the opt-in capacity policy is unchanged. Its separate
records and source manifest are retained beside the frozen results.

Retained-RHF ownership from parent `c7486bf2e` is integrated as `d3548f1fd`.
Preserve the parent's optional reference-plan parameter alongside the packing
selector, without reinstating obsolete method-local providers. The expanded
77-case host admission/lifetime suite passes in 2351. Build 2350, real-device
solver/independent small all-coordinate FD 2353, shared Lambda/factor 2354,
complete energy/force/budget 2356–2358 and report 2360 subsequently pass.

On this latest version, large energy CCSD medians are 145.958333 / 145.959855 s
(full / packed). Complete medians 354.075731 / 298.333061 s reflect RHF medians
199.615963 / 143.875171 s, not a packing speedup. The separate complete-force
pair is 1336.988423 / 1351.807057 s. Memory savings remain 251,083,520 bytes in
CCSD/energy, with no complete-force peak reduction. At the unchanged
2,735,193,944-byte scalar budget, full history refuses while packed converges.
Maximum force difference is 3.365e-9 Eh/Bohr and all original gates, including
limited independent large FD re-audits, pass. Preserve the capacity-only opt-in
selection; do not extend packing to full contraction state without new evidence.
