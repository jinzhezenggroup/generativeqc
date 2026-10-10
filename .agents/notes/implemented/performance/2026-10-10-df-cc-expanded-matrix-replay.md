# Decision: independently pack the original DF CCSD physical replay

Status: implemented
Date: 2026-10-10

## Problem

The accepted-amplitude physical audit must not reuse the primary solver's
auxiliary-reduced cuts. Its original expanded one-Q virtual graph remained on
scalar CUDA contractions even when the primary iteration already owned an
admitted FP64 matrix provider. Existing solver timers measured approximately
18.1 s of replay inside a 99.8-s CCSD solve for ethane230.

Retaining primary amplitude layouts was investigated first and rejected:
logical copy traffic decreased, but its complete-endpoint gain was smaller
than observed noise. See the rejected note
`../../rejected/2026-10-10-df-cc-amplitude-layout-epoch.md`. Do not restore that
policy merely because this different replay optimization succeeds.

## Decision

Apply the existing compiler packing transform to the original expanded virtual
graph, independently of primal hoisting. Prepare its contraction table once
using the already admitted provider context. Every replay still consumes the
original Bov/Bvv factors and accepted T1/T2 amplitudes, traverses Q in ascending
order, and runs the original expanded core replay. No previous replay or primal
cut is an audit input. Storage, compute and accumulation remain FP64.

Admit the optional descriptor storage and maximum of primal/replay scratch only
after selecting the primary tile. Insufficient dimensions/budget, numeric
allocation failure or host descriptor allocation failure drops replay packing
before reducing an already admitted primary tile. Retain the original scalar
audit. Execution/arithmetic failures propagate; they cannot trigger a retry of
partially evaluated physical work.

The compiler's packing optimizer deduplicates one shared contraction. It saves
exactly `nocc*nvir*nvir` scalar summands per replay Q slice; it does not hoist
the audit's physical Q-dependent work. Count all actual GEMMs, packing traffic,
operations and summands. Explicitly order native aggregate outputs rather than
relying on canonicalized TensorIR output-map order.

The internal ablation/selection field `df_replay_matrix_gemm` is inventoried
under the existing #1890 provider-selection migration debt. The scientific
owner does not introduce a vendor callback or another provider context.

## Evidence

Frozen parent `b315d4056` is identified by its full SHA in the retained
provenance. n2 Slurm 2797 qualifies 33 standalone native solver checks and
17 CUDA action checks. Slurm 2798 qualifies six frozen-library admission,
ordered-accumulation and sticky-error checks, followed by all four preserved
fresh-process endpoints in baseline/candidate/candidate/baseline order.
Host graph tests pass 28 cases; native CPU solver checks pass 22 cases.

For ethane230, 9 occupied / 221 virtual / 488 auxiliary functions, DIIS 8,
Q tile limit 8, 64-GiB budget and one PRO 6000, complete process-wall medians
change from 195.912753 to 187.624613 s, a 4.23% reduction. Complete CCSD changes
from 99.831593 to 91.389140 s; its existing replay timer changes from 18.127537
to 9.678092 s. Primary iteration medians remain 81.555446 / 81.560398 s.
This is source-matched conditional evidence, not a universal performance claim.
Two observations per selection do not satisfy the shared formal promotion
envelope's five-pair/history/compile-cost requirements; do not mark that gate
passed or manufacture missing iteration histories.

Every endpoint passes an independent pinned PySCF total-energy gate of `1e-8`
Eh, separate-(T) gate of `1e-10` Eh and physical residual maxima of `1e-10`.
Observed maximum energy / (T) errors are `2.203e-12` / `8.448e-14` Eh. Primary
iteration/evaluation counts, physical Q scope and accumulation work match.
All work changes equal the independently derived replay-only deltas.

The tradeoff is explicit: logical packing bytes increase from
5,578,493,904,992 to 6,263,154,422,624, and numeric capacity increases 49,566
bytes. Packing bytes are logical reads/writes, not measured DRAM traffic.
All 488 original replay Q slices remain; 13,176 prepared GEMM executions are
added, and 214,509,672 duplicate scalar summands disappear.

While qualifying, master advanced through #2163 and #2153. The former changes
Lambda/force policy after the energy-only return; the latter changes SCF, not
this CC lowering. Integrate their source without repeating the full GPU matrix.
The relevant CC generator bytes remain unchanged, and the endpoint and CUDA
solver compile with the latest master headers. The published timing parent
remains frozen: these are not latest-master RHF timing claims.

## Consequences and revisit conditions

The remaining approximately 81.6-s primary iteration cost dominates the CCSD
endpoint. Prioritize a measured primary-work bottleneck instead of attributing
unchanged RHF/(T) time to this optimization. Revisit replay tiling/layout reuse
only with complete endpoint evidence and independent expanded audit inputs.
Do not remove scalar replay or sacrifice a primary tile to admit audit storage.

## Retained records

`benchmarks/results/df-cc-expanded-replay-20261010/` retains the four complete
observations, exact reconstruction patch, oracle/input/binary identities,
driver, build procedure and scoped numerical validation envelope. Full build
trees, binaries, compiler-cache receipts and raw logs remain ignored artifacts
under `.artifacts/df-cc-replay-20261010/` and
`n2:/data/jzzeng/qc-cc-replay-20261010/`. No Release, release asset or external
archive is created.
