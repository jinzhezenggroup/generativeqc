# Decision: do not promote per-sweep paired DF packing reuse yet

Status: rejected for default/performance promotion
Date: 2026-10-10

## Problem and candidate

After #2191 the paired batched Q action still repeats four full dense
transposes of current T2 and projected tau at every Q tile. These operands are
immutable within one primary/trial evaluation but change between evaluations.
The packed singleton action has two invariant tau transposes. Reuse may remove
data movement without changing contraction trees, precision, or polynomial
degree, unlike the rejected two-sided dressing alternative.

The compiler's existing `analyze_iteration_reuse` proves dependencies from
`t1`, `t2`, and `df_tau_occupied_pairs`. This candidate permits only transpose
nodes, gives them exclusive retained arena slots, and leaves the original
scientific Program and equation hashes unchanged. Dynamic copy-roundtrip
elision must use this same retained coloring and cannot elide retained nodes.
All eleven original nonpaired generated artifacts remain byte-identical.

The native owner admits the extra storage after ordinary pairs, replay and
the primal tile. Both budget refusal and numeric allocation failure retain the
original paired materialized schedule before sacrificing another option.
There must be at least two actions using the same retained coloring; a single
batched tile followed by a singleton tail has no reuse benefit and is excluded.
No pointer, shape or previous solver-state identity establishes validity.

## Epoch and singleton tails

After each successful per-state projection, prepare copies from the current
amplitudes and projected tau before the Q loop. Do not clear sticky arithmetic
state in preparation or in the dynamic action. A projection refusal selects the
original expanded Q loop; a later admitted evaluation always re-prepares.

Batched and singleton programs use different retained slot colorings. A
singleton tail must re-prepare its two tau copies after the final batched action.
It may overwrite the old batched retention only because no batched reader remains
in that sweep. Q tails larger than one share the batched retained prefix, whose
dimensions contain no Q symbol. Ordinary DIIS, ascending individual-Q
accumulation, expanded replay, and FP64 (T) remain untouched.

## Qualification and proposed measurement gate

- Thirty-three focused host copy/generation checks pass, plus four new dependency,
  emission and exclusive-slot checks.
- Two native-materialized interpreter bit replays pass across six Q-factor
  changes and two amplitude epochs per variant. The test adapter invokes the
  existing interpreter primitives, not another tensor algebra.
- Slurm 2817 builds the CUDA library and compares eleven unchanged artifacts.
  Initial qualification exposes unnecessary retained storage for zero-reuse
  tile/tail cases. Slurm 2819 performs the targeted incremental rebuild after
  adding the minimum-repeat admission gate: nine solver tests and ten action
  tests pass, including reused and ordinary actions, uneven tails, short budgets,
  canaries and preseeded sticky state. Reused-action memcheck has zero errors.
- The ordinary Python interpreter keeps transpose views; detaching a prepare
  output makes it C-contiguous. NumPy einsum can then differ by an ulp despite
  optimize=False. Do not claim a bitwise materialized-schedule proof from those
  strided views. The strict bit test explicitly materializes every primitive,
  matching the native emitter's dense-copy contract. Preserve the fixture-failure
  logs; this is not permission to relax a GPU or physical replay gate.

For o9/v221/Q488/Q8 and 38 admitted evaluations, include four prepare copies per
evaluation in executed work: expected net removal is 9,120 operations and
448,993,359,360 logical packing read-plus-write bytes. These are not measured
DRAM transactions. Contraction/GEMM work and all admission/projection scans
must remain identical. Extra arena payload is 49,231,728 bytes before surrounding
alignment. Account for this cost rather than describing reuse as memory-free.

Slurm 2820 completes a feasibility-only pair of fresh energy endpoints against
the checksum-pinned #2191 baseline on node2/PRO6000. Every independent total
energy 1e-8, (T) energy 1e-10 and expanded physical replay 1e-10 gate passes.
Energy/(T) bits and physical replay maxima are identical. Iterations,
evaluations, contractions/GEMMs, Q/projection/setup work remain identical;
operation and packing deltas match the counts above exactly.

Complete wall is 156.9750084511 -> 157.0248215031 s (+0.0317%); complete CCSD is
61.082937863 -> 60.888837228 s (-0.3178%). Numerical capacity increases
4,576,754,470 -> 4,625,986,086 bytes: 49,231,616 more bytes after alignment.
This single pair neither proves a regression nor demonstrates an endpoint win.
The substantial logical data-movement reduction does not establish that packing
dominates the endpoint. Do not pay more memory or launch full ABBA merely to
turn this tiny CCSD change into a performance PR. Production remains unchanged;
the prototype stays isolated. Force/Lambda/response promotion is not established.

Revisit with a demonstrated packing bottleneck, a storage-neutral schedule, a
different hardware domain, or a broader optimization that amortizes the retained
copies. Keep the exact copy, epoch, tail, admission and independent scientific
gates rather than weakening them to chase a timing observation.

## Source and evidence boundaries

Experiments remain under ignored `.artifacts/df-cc-q-invariant-packing-20261010/`
and `/data/jzzeng/qc-cc-q-invariant-packing-20261010/`. The reconstruction uses
the already measured frozen parent and its baseline overlays, not a different
SCF/CLI tree. Observed master 15bc69700 has no delta in the relevant CC/compiler
files; unrelated DFT/MP2/CI commits do not justify rerunning this campaign or
relabeling it latest-master endpoint qualification.

The aggregate retained benchmark budget has only 134 bytes free. Do not increase
the cap, drop accepted samples, delete unrelated evidence, or publish an archive.
If promoted, a compact new bundle can replace only this loop's immediately
preceding #2191 evidence bundle, with an explicit recovery manifest pinned to
its existing merged Git commit and per-file checksums. Historical evidence must
remain recoverable from that Git history; no new external backup is authorized.

## References

- PR #2191; local source snapshot 0cf1d8d55e635b33790c253db185b0acb96f05c1.
- `../rejected/2026-10-10-unconditional-two-sided-df-ladder.md`.
- `../implemented/performance/2026-10-10-df-cc-native-copy-roundtrips.md`.
