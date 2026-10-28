# Shared XC paired-batch regression qualification

This repairs the four TH/BKL RKS batched regressions retained in
[the structural discovery campaign](../xc-structural-sharing-20261028/README.md).
After the host proves RKS, the batched adapter supplies two explicitly identical
physical input channels to the existing restricted scalar body. Ordinary/general
adapters, launch parameter storage, 128-thread scheduling and compact output
layouts retain their existing contracts. No functional-name exception is added.

The final selected package still covers 73 registrations in 12 groups. It has
**48 point kernels versus 292 literal kernels**, and **7,972,672 versus 41,532,592
object bytes (80.80% smaller)**. This is the selected XC package, not the complete
native library. Automatic AOT and broad discovery remain opt-in; the qualified
policy continues to select KTBM without enrolling new groups.

## Point timing and remaining boundary

NVCC 12.9.86 (CUDA 12.9.1 installation), GCC 11.4 and SM120 run on a Slurm PRO6000
allocation on node2. Compilation uses verified sccache 0.16.0, preserving caches
and before/after statistics. Cache-aware build times are retained, not presented
as controlled cache-miss speedups. All real-device commands use finite `srun`
allocations and preserve scheduler device visibility.

The final campaign compares the frozen literal/shared baselines and the accepted
source-matched package. It covers 120 cases: representatives of every shared
group, all four original failures and KTBM controls; both spins; 16,384/262,144
points; ordinary/tile-127 traversal. Three randomized rounds retain 50 CUDA-event
samples after 10 warm launches per arm: **1,080 process measurements**. Case ratios
divide medians of process medians; no slow sample is removed.

For the four original failures (`spins=1`, `points=16384`, `tile=127`):

| Functional | Before / literal | Accepted / literal |
| --- | ---: | ---: |
| TH_FCO | 1.030572 | 1.000784 |
| TH_FCFO | 1.023618 | 0.993252 |
| BKL1 | 1.021982 | 0.999069 |
| BKL2 | 1.021999 | 0.999627 |

All four pass the existing per-case 1.01 boundary. Across all 120 cases, no
accepted/before ratio exceeds 1.01; its maximum is 1.004961. The accepted/literal
aggregate median is 0.999723. An unchanged ordinary UKS `GGA_X_APBE` case remains
at 1.015387 versus 1.015741 before the fix. This negative control is retained and
**does not support blanket discovery promotion**.

Static restricted-batched resources change from 148 to 146 registers for TH and
112 to 105 for BKL. Static global-load instructions decrease from 11 to 8 in both
examples; multiplication/addition counts also decrease. STACK:96 and LOCAL:0
remain unchanged. These are disassembly/resource observations, not measured
dynamic instructions or an occupancy claim. Nsight Compute was denied hardware
counter access; its failure receipt is retained.

## Numerical and complete-endpoint gates

The 274 host/compiler/CMake checks include a compiled execution of the generated
host launchers, adapters and traversal against independent Libxc. It covers both
selectors, multiple parameter rows, real U1/P2 layouts, invalid inputs and partial
16/127-point tiles. All **657 real CUDA point/Fock checks** pass across the 73
registrations, including replay, error and directional-derivative oracles.

The independent public PySCF KS campaign has **23 passes** and retains the earlier MVSB UKS convergence
limitation as an explicit skip, with the original literal/shared failures linked
above. It is not a passing cohort or a convergence fix. A supplemental device
oracle adds **16 passes** for the four repaired registrations with 128 points and
tile 127, including the final one-point tile, both spins and both selector settings.

Complete endpoint pairs cover the four repaired registrations, KTBM_0/GAP and
Wigner in both spin layouts, plus a four-water APBE RKS control: 15 cohorts,
three fresh-process pairs each. Each process measures cold preparation, three
frozen-seed warm calls and displaced geometry. The initial 450 calls preserve all
iteration/Fock counts and precision/work ledgers. One TH_FCFO RKS cold median is
1.020619, slightly above the 1.02 endpoint boundary. Before further measurement,
five additional paired rounds were fixed for that cohort and the TH_FCO RKS
control; every original sample remains included.

Those independent confirmation phase ratios span 0.988805–1.004964. Across all
**550 complete calls**, pooled per-cohort phase ratios span **0.985944–1.014979**,
with no work mismatch or failed process. Maximum paired energy difference is
**1.137e-13 Eh**, below the 3e-9 gate. The reducer reports initial, confirmation
and pooled populations separately so the original borderline result stays visible.

CPU and CUDA registries are generated from the same frozen producer snapshot.
Both are rebuilt and relinked against the same remaining native endpoint objects
from the preceding campaign. Those reused owners' original metadata is not
relabeled as a new full native build. Generated sources, replacement-object links,
compiler inventory and exact measured library hashes authenticate the experiment.

## Retained experiments and reproduction

All six pilot matrices remain available, including unsuccessful block-size,
fixed-spin, coefficient-sign, alignment, parameter-order, grid-constant and
ordinary-input-pairing experiments. The broad candidate's complete 120-case
point run is retained: applying `__grid_constant__` to general kernels introduced
an APBE UKS batch regression. That attempt was stopped during numerical validation;
its partial output is retained without claiming a completed numerical gate.
The first supplemental tail probe requested 263 points, exceeding the native
test driver's explicit 128-point cap; all 16 calls were rejected before launch.
That setup failure and its source are retained. The corrected 128-point probe
respects the existing cap and tests the same one-point tail as 16,384/tile-127;
the host execution separately covers 263 points. No production check was relaxed.

The accepted implementation needs only the existing RKS batched execution
boundary. See the [decision note](../../../.agents/notes/implemented/performance/2026-10-28-xc-shared-point-abi.md)
for the rejected alternatives and proof obligations.

- `receipts.json.xz` consolidates the preceding `structural_discovery` and current
  `paired_batch` records. Every numerical value, paired sample, negative result,
  work ledger, dirty producer source, build/cache/binary record and driver remains
  intact. Sharing their compression context avoids duplicating provenance.
- Generated source bodies and raw disassembly are transient products in ignored
  run storage. Their identities, generation recipes and static instruction/resource
  summaries remain in the receipt. The original discovery archive identity is
  retained as provenance, not as a dependency on unpublished Git history.
- `reproduce_sources.py` reconstructs producers from published ancestor
  `cbeca43a0` plus exact retained overlays. It regenerates CPU/CUDA sources and
  checks every file against the measured hashes, without compiling or using a GPU.
  Accepted, structural, publication and rejected producer snapshots are separate.

The reducer checks recorded source/object/library bindings and reports every
population, including failed timing gates. It uses Python and the project's
`defusedxml` test dependency; it does not require Git history or generate sources:

```bash
python benchmarks/results/xc-shared-point-abi-20261028/reduce.py
```

Optional source reconstruction needs compiler Python dependencies and the local
published Git ancestor. It refuses existing output and never fetches history:

```bash
python benchmarks/results/xc-shared-point-abi-20261028/reproduce_sources.py \
  --snapshot accepted --output .artifacts/xc-paired-batch-source-audit
```

No other architecture, full catalog default, or complete-library size reduction
is qualified by this evidence.

## Subsequent qualification

The [structural default qualification](../xc-structural-default-20261028/README.md)
addresses the remaining ordinary UKS regression and supersedes this snapshot's
opt-in decision. The paired-batch fix and the negative results recorded here
remain part of its evidence; the later storage change applies to a different ABI.

The subsequent campaign losslessly interns this receipt's exact timing values in
a sorted table with integer offsets. Both historical reducers restore and
authenticate every original population before analysis; their complete outputs
remain byte-identical to the pre-compaction reports. No observation, failure,
source reconstruction input or numerical gate was removed.
