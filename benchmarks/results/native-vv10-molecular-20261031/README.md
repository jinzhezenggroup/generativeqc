# Native molecular VV10 schedule integration

The stationary controller now requests the existing molecular VV10 pair layout,
matching its density-domain producer and capacity planner. Previously it used
the raw-row default, preventing the qualified masked/admitted/parallel schedules.
Selection depends on mathematical admission and workload, without a GPU-model
allowlist. FP64 arithmetic, quadratures, density thresholds, memory limits and
independent acceptance gates are unchanged. Positive-zero and negative active
quadrature weights keep their potential and weight derivatives.

## Complete endpoint results

These are stock-grid WB97M-V energy-plus-force endpoints: one cold call plus
three warm calls per fresh owner, one math thread, and the frozen engine-local
cold density. All 24 populations (DF/Direct, 3/12/48/96 atoms, three devices)
pass the unchanged independent README oracle gates. The 48/96-atom def2-SVP
cases contain 384/768 orbital functions. `report.py` prints all populations,
cold timings and warm medians; `observations.json.xz` also keeps every repeat,
energy, iteration/Fock count, available grid/nonlocal work, per-repeat oracle
errors and original receipt identities.

Warm medians in seconds:

| Mode / atoms | A100 | GPU4PySCF A100 | RTX 5090 | GPU4PySCF 5090 | RTX PRO 6000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DF / 3 | 0.185076 | 0.685909 | 0.307147 | 0.730209 | 0.304180 |
| DF / 12 | 0.833886 | 1.393277 | 1.153329 | 2.760538 | 1.136679 |
| DF / 48 | 8.673187 | 8.313316 | 13.098544 | 30.797052 | 12.708160 |
| DF / 96 | 25.938773 | 29.231100 | 60.705224 | 125.896358 | 45.233481 |
| Direct / 3 | 0.178600 | 1.499125 | 0.329927 | 1.626125 | 0.328207 |
| Direct / 12 | 0.979120 | 2.232114 | 1.292394 | 3.971172 | 1.277757 |
| Direct / 48 | 10.951818 | 8.979256 | 14.298812 | 32.387447 | 13.874878 |
| Direct / 96 | 30.672998 | 28.504040 | 47.112860 | 119.103250 | 46.180419 |

The preceding DF96 production baselines were 31.930048 s (A100), 94.581586 s
(5090) and 73.360937 s (PRO 6000). No paired DF matrix population regresses
more than 5%. Paired matrix energy and force differences are below 1.9e-12 Eh
and 7.6e-13 Eh/Bohr respectively. Direct populations have independent oracle
qualification but no paired Direct baseline in this campaign.

Cold 96-atom results remain separate:

| Device / mode | Native | GPU4PySCF |
| --- | ---: | ---: |
| A100 / DF | 149.648838 | 87.394808 |
| A100 / Direct | 174.092282 | 88.495928 |
| RTX 5090 / DF | 236.889982 | 376.172867 |
| RTX 5090 / Direct | 248.137469 | 369.145460 |
| RTX PRO 6000 / DF | 226.104688 | unmeasured |
| RTX PRO 6000 / Direct | 240.095434 | unmeasured |

Native and GPU4PySCF WB97M-V DF have different long-range fitting contracts.
The retained 5090 reference is on n4 and current native runs are on n5. PRO
6000 has no GPU4PySCF reference. These static default-grid observations are
not a universal accuracy-qualified ranking. A100 DF48, Direct48 and Direct96
still trail; the other seven README methods' A100 DF96 gaps from the preceding
[portable DF campaign](../ecnu-portable-df-20261031/README.md) remain open.

## Validation and controls

Each device passes 99 VV10 tests covering independent extended-precision pair
oracles, signed/zero weights, screening boundaries, reuse, failure recovery and
automatic schedule crossovers. Each also passes two moved-geometry RKS/UKS
tests with independent energy differences and constrained-capacity fallback.
The latter test now passes the exact compiler MethodIR instead of incorrectly
turning profile IDs into public aliases. CUDA probe compilation uses the
existing verified compiler-cache runner. Tests recognize actual Slurm or LSF
allocations. No new sanitizer campaign is claimed here.

Host validation passes 132 signed-weight/pair-codegen tests and 413 phased
resource tests; 577 compiler modules have zero dependency errors. Source-bound
Libxc metadata is refreshed without changing capability identities or stages.
Historical native evidence remains stale; these WB97M-V results do not confer
general Libxc catalog qualification.

The separate ABBA controls use baseline/candidate/candidate/baseline fresh
owners and retain each population's independent gate, all cold/warm samples,
sample variability and all cross-arm paired errors in `abba-analysis.json`
inside the compressed bundle. They are distinct from the matrix above.

All 12 accepted control populations independently qualify. Each arm has six
warm samples from two owners; medians and full observed ranges are:

| Device | Baseline median [min, max] | Candidate median [min, max] |
| --- | ---: | ---: |
| A100 | 31.964742 [31.899559, 31.978400] | 25.945740 [25.906252, 25.965247] |
| RTX 5090 | 94.588939 [94.564661, 94.680663] | 60.678209 [60.658815, 60.734158] |
| RTX PRO 6000 | 73.386316 [73.374403, 73.392916] | 45.254062 [45.234884, 45.260178] |

Slurm job 1904 times out during the final 5090 baseline. The receipt finishes
at 06:32:14 +0800, after the scheduler's 06:31:17 allocation end: its complete
payload stays retained but is **excluded** from accepted controls. A separate
15-minute allocation, job 1905, starts at 06:33:01 after that process has
completed and successfully reruns only the final baseline. Scheduler completion
alone and JSON `status=complete` alone are each insufficient to establish acceptance.

## Rejected batching experiment

A same-binary diagnostic compares 256 and 2048 force-grid points with a 64-MiB
geometry cap, one cold plus **one warm** call each. It is not a three-repeat
ranking population. Larger grids reduce semilocal work from 2.914 to 1.474 s
and nonlocal grid work from 2.308 to 1.407 s. However, retained controller
storage grows from 388,545,756 to 889,391,164 bytes and warm free headroom falls
from 1,457,848,320 to 954,531,840 bytes. No optional SCF cache is retired.

At 256 points the DF response retains 740,556,800 bytes and recycles
2,415,919,104 bytes of final-projection scratch. At 2048 points the occupied
storage admission fails; the bounded fallback executes 4,489 fitted-panel
GEMMs over 67 auxiliary panels. Its phase grows from 0.924 to 52.846 s and the
complete warm endpoint from 25.964 to 75.572 s. VV10 pair work stays near
8.69 s; SCF remains one iteration and one Fock. The compact diagnostic keeps
source/trace hashes, actual work counters and synchronized phase timings.
Production retains its original tiles and caps. Future automatic batching
must jointly account for response headroom or share scratch across dead phases.

## Provenance and retained failures

Measurements start from `000517f82dc62130459a6fd92a27518a158c49cc` with the two
controller/capacity source changes. The accepted core libraries are:

- A100: `bdbcffd7e425d9bd25815513a90f8f6958af716b11a4723e7d3cf4b342d402c4`.
- Both Blackwell devices: `76fc01df557e89d095ee4876be457d34638c2fed75fd0c573ead230b9fb9daaa`.

Actual scheduler/device visibility, source identities, changed-file hashes,
build/module hashes and post-measurement receipts are retained. A100 uses
finite LSF job 223903 on gpu22; Blackwell uses finite Slurm allocations on n5
and n2. The A100 shell ultimately reports EXIT(1): its already-expired HTTP
helper was unsuccessfully killed immediately before shell exit. Its completed
measurement script reports zero and all four ABBA populations are complete;
the scheduler state is not rewritten as DONE.

The initial A100 overlay preserved old source mtimes and left the baseline
binary unchanged. Its receipts are explicitly rejected. After verified
controller recompilation, every accepted test and population was rerun. Initial
trace instrumentation queried the grid before production; its failed log stays
retained. Blackwell's initial source archive omitted manifests, and the first
n2 probe used a host compiler missing `cc1plus`. Original failed logs and
allocation receipts remain visible; manifests were supplied and n2 reran with
`NVCC_CCBIN=/usr/bin/g++`. No failed setup is relabelled as successful execution.

Full raw vectors, build logs, traces and experimental sources remain under
ignored `.artifacts/vv10-native-mask-20261031/`; the compact bundle cannot
reconstruct omitted vectors. Verified transfer archive hashes bind those local
artifacts. See the [integration decision](../../../.agents/notes/implemented/performance/2026-10-31-native-vv10-molecular-schedule.md)
and [batching investigation](../../../.agents/notes/proposed/2026-10-31-stationary-grid-batching.md).
