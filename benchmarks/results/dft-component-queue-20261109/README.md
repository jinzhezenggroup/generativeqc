# Generic component exchange admission: normal DFT qualification

From the repository root:

```bash
PYTHONPATH=.:python python benchmarks/results/dft-component-queue-20261109/replay.py
```

Offline replay authenticates every retained original endpoint byte, unchanged
independent dense-GPU/grid-stability/CPU-GPU-spot gates, strict native E/F and
semantic work, actual cold dispatch, and every predeclared timing guard. It needs
neither GPU, reference execution, network nor Git-history recovery. Optimized
Python is refused rather than disabling acceptance assertions.

| Complete Direct E+F cold mean | Control | Candidate | Change |
|---|---:|---:|---:|
| A100 B3LYP96 |11.873728 s|10.340895 s|-12.91%|
| A100 PBE096 |11.762709 s|10.221247 s|-13.10%|
| A100 B3LYP48 |5.526261 s|5.472524 s|-0.97%|
| A100 PBE048 |5.469026 s|5.370808 s|-1.80%|
| 5090 B3LYP96 |17.396792 s|17.195726 s|-1.16%|
| 5090 PBE096 |17.008046 s|17.054149 s|+0.27%|
| 5090 B3LYP48 |8.543105 s|8.473478 s|-0.82%|
| 5090 PBE048 |8.319067 s|8.163289 s|-1.87%|

These are matching **complete normal images**, not the earlier interposer pilot.
Build 16924 freezes master 76bb87c76; normal A100 LSF 230562 and physical 5090
Slurm 16931 use fresh-process ABBA, two colds per arm and three own-density warms
per process. Cold includes calculator construction plus energy and analytic
forces; it is not an OS-page-cache reset or a compiler benchmark. The native
source/ELF identities are in `cohort.json` and in each endpoint receipt.

Both A100 96-atom gains clear `max(2%, twice summed relative MADs)`, observed floor 2%.
Every 48-atom cold, every 5090 cold and every paired warm pass 3% nonregression. Worst
paired warm ratios are 1.022951 on A100 and 1.010280 on 5090. The smaller changes
are guards, not statistically qualified speedup claims. Initial guess, target
Focks `[9,1,1,1]`, precision timeline and J/K/incremental/final-rebuild work match.
Whole-endpoint quartet/recurrence counters were not acquired.

All four additional M06-2X/omega-B97M-V 96/48-atom control/candidate pairs per card
pass complete cold and all warm guards and independent science. They are
single-pair guards, not ABBA promotions. **Omega-B97M-V cold is not materially
accelerated here.** Fresh same-allocation GPU4PySCF 96-atom pairs give native/reference
cold ratios 0.791620/0.922620 on A100 and 0.794536/0.828406 on 5090 for B3LYP/PBE0.
This establishes neither HF/DF performance nor all non-CBS README leadership.

## Raw-K, dispatch and deployment boundaries

Both actual normal ELFs pass independent raw-K/count/graph gates on A100:
18 definitions, 12 class/geometry inputs plus 6 reversed-cache layouts, 65 replicated
systems/definition, 864 repeated predicates and 54 unsupported refusals/image.
Replicas are not additional independent geometries. The producer uses normalized
Libcint full `int2e` and inventory-masked dense Einstein K, never generated scatter
as its oracle. The unchanged per-element gate is `2e-11+2e-10*abs(reference)`.
The exact fixture manifest, producer, driver and complete result logs are retained.
Generated binary fixtures remain in the ignored experiment owner and are
reproducible from the versioned `f_shell_numerics` inputs. Existing independent
raw-K arrays in `../dft-shared-rys-streaming-20261108/` remain offline references;
they are not relabeled as new normal-image executions.

On 5090, actual queue fatbin 16921 passes the same raw-K gates and 16923 passes
six classes times six graph density/reset predicates for each of memcheck,
racecheck and synccheck, with zero errors/hazards/warnings. These are **not
normal-ELF tests**: normal shared-XY SM80 wrappers contain no SM120 image.
Normal job 16927's control launch fails before endpoint timing and is preserved.
Separate zero-loss CUPTI cold owners prove six shared-XY classes each execute
nine times on A100 and zero times on 5090. Both A100 images use the same kernel
names; source/ELF identities distinguish their worker implementations. Intrusive
times do not enter ranking, and subsequent diagnostic warms are outside collection.

A100 230553 fails in Python codec startup before ELF reconstruction or GPU work.
The corrected worker copies the exact existing Python 3.12.16 standard-library
bytecode into a task-owned sourceless-import layout; it changes no shared runtime,
scientific package, native image or timing protocol. No clean timing sample is
discarded or retried. Build/cache/restore receipts and deployment limitations
remain retained; full compiler stdout, binary products, original generated
fixtures and returned archives remain in `.artifacts/dft-component-queue-20261109/`.

## Extraction and storage

```bash
PYTHONPATH=.:python python benchmarks/results/dft-component-queue-20261109/replay.py \
  --extract /tmp/dft-component-evidence
```

This extracts original receipts and measurement/reproduction scripts, not a
portable GPU installation. Scripts preserve measured cluster paths and need the
documented scheduler, CUDA/CUPTI tools, verified compiler cache and authenticated
normal source reconstruction. Reference preparation uses PySCF only outside
production. All real GPU work requires a finite scheduler allocation.

Every endpoint FP64 word and original text byte uses the existing v3 reader.
The prototype transport archives remain ignored; all 18 original text members
are stored in that same capsule with individual size/hash bindings and original
transport hashes. Offline replay needs no ignored archive.
The six older raw-J populations are losslessly repacked using that same reader,
including their complete original outer container, to fit the unchanged 85 MiB
aggregate/1 MiB-file/2 MiB-change limits. Their scientific gates and every original
member digest remain unchanged. No Release, external asset, tag, network
dependency or budget waiver is introduced.

Rationale: `.agents/notes/implemented/performance/2026-11-09-component-exchange-admission-queues.md`.
External evidence/decisions: issue 2976, especially comments 6731367810 and 6731535816.
