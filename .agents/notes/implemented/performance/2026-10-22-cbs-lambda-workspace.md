# Decision: budget live GMRES storage before selecting DF Lambda tiles

Status: implemented
Date: 2026-10-22

## Problem

CBS Fast's C7 Lambda solve selected Q4 under a 32 GiB complete numeric budget,
while C6 selected Q32. The host GMRES capacity polynomial reserved several
unused Krylov bases even when diagnostics were disabled. At restart 30 its
leading term was 171 dimension-sized vectors; the controller actually holds
71 at its largest overlap. This artificial reservation constrained the GPU
planner even though those extra host arrays were never allocated.

## Decision

Bound the controller's simultaneous numeric payload: `(2 * restart + 12) * n`
doubles for its two bases, iteration vectors and one vector of publication
headroom; `restart**2 + 5 * restart + 1` for Hessenberg/Givens/backsolve storage.
Add `restart * n` only when retaining a basis, and `max_iterations + 2` only
when retaining history. The iteration algorithm, restart length, preconditioner,
true-residual checks, convergence tolerances and FP32/FP64 handoff are unchanged.

The ten non-basis vectors are x/image/residual, work/candidate/candidate image/
candidate residual, best x/residual, and the cycle's base x. All large vectors
reserve their capacities before append/assign. Returned arrays move their
tracked ownership; publication does not copy another live Krylov workspace.
Input spans, callback-owned storage and allocator metadata retain their separate
ownership/budgets. No caller may treat this as a whole-process memory bound.

The Python host facade now builds identical diagnostic flags for capacity queries
and execution. It always requests history/timing, and includes a basis when the
caller or host recycling needs one. The old excess reservation had concealed
this mismatch. Exact-budget and one-byte-short tests cover both public basis
choices and recycled/sequence calls. The sequence-prefix fixture also leaves
room for the preceding independent RHS-rank diagnostic; its storage can dominate
a very small Krylov solve after this correction.

Existing DF matrix, audit, cache, provider and allocation refusals remain bounded
fallbacks. This is a shared host-controller capacity correction, so Standard and
other native consumers also receive the corrected bound; it is not a new Fast
precision or stopping policy.

## Endpoint evidence

The matched baseline is master `976f756b391bc65edb79f76df1a8c3fe807518a7`.
The [retained campaign](../../../../benchmarks/results/cbs-fast-memory-20261021/README.md)
contains complete energy-plus-force observations, raw outputs, reconstruction
patch, independent checks, work counts and reproduction commands.

On the first same-GPU C7 pair, prior Fast takes 1225.765 s and the candidate
1093.350 s, a 10.80% time reduction. Lambda takes 645.109 -> 513.805 s. Q4 -> Q16
reduces actual FP32 GEMMs from 69,660 to 17,496, while both variants retain
12 mixed and 8 strict shared transpose actions. No energy changes in this pair;
maximum composite force difference is 1.68e-13 Eh/Bohr. These controls are not an
independent full-vector C7 oracle.

A reverse-order pair on a second GPU confirms 1224.092 -> 1094.678 s,
a 10.57% reduction. Both comparisons match GPU identity within the pair;
different-GPU observations are not pooled into a median or statistical claim.
The refreshed candidate Standard/Fast C7 curve is 1447.813/1093.350 s (1.324x).

The released host reservation is about 6.9 GiB for this shape. The planner uses
it for larger GPU tiles: Lambda owned device capacity grows from 7.33 to
14.64 GiB, and its conservative numeric capacity from 30.75 to 31.16 GiB.
Neither number is measured peak VRAM. This change reduces unused reservation,
not actual peak device memory.

C6's same-GPU observation is 500.025 -> 488.979 s, with Lambda
219.382 -> 208.153 s and Q32 in both variants. Three alternating fresh-process
C4 pairs are effectively unchanged (medians 68.067 -> 68.054 s). The benefit
is resource dependent; there is no uniform or other-device speedup claim.

Separate complete C6 diagnostic calls explain the smaller gain: the existing
invariant-core cache is refused at baseline and admitted once in the candidate.
Closed Lambda trace subtrees record 10,177 -> 9,550 FP64 GEMMs, with unchanged
1,270 Q batches. The counter deltas exclude overlapping parent GMRES totals;
semantic summands and logical packing bytes are not hardware FLOPs/DRAM traffic.
Trace-enabled calls are separate from the performance cohorts.

Independent qualification retains C2 energies and two-step directional forces
only after full protocol equality, plus newly reconverged full-vector water
finite differences for exact and RI-JK references, original/displaced geometry,
process-cold/replay and 256 MiB constrained-memory calls. Composite allowances
remain 1e-5 Eh, 1e-4 Eh/Bohr maximum force and 3e-5 Eh/Bohr RMS, with constituent
allowances divided by the combined coefficient L1 norm. Larger alkanes also
pass full-vector controls against Standard.

Tracked native ownership tests exercise default/small restarts, exhaustion,
optional diagnostics, late nonfinite returns and pre-allocation refusal. Native
ASan/UBSan, Python response/CBS tests, independent full-DF GPU tests, optional
Lambda paths and Compute Sanitizer complete Fast endpoints pass. The latter
reports zero errors. Every real-GPU execution uses finite Slurm allocations on
allowed nodes; compiler caches and source/binary identities are retained.

## Revisit when

Measure remaining FP64 core, independent replay/audit and triples costs before
changing equations or precision. In particular, a larger main Q tile does not
prove that a larger independent primal-replay tile also fits. Separate bounded
phase schedules may be useful but need their own complete endpoint qualification.
Do not remove independent audits or repeat the tighter FP32 warmup handoff that
was slower in the [preceding Lambda study](2026-10-21-cbs-fast-lambda.md).
