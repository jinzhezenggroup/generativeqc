# Direct DFT cold: ordered DIIS transport

Measured on 2026-10-23, RTX 5090 on n1/node1. Both arms include master
`ececb48cd56a00a56fbb6223dc53bc61dfe6e11d` plus PR #2702's public-AO Schwarz
specialization. Its 5,029 recorded inputs match the merged master
`7b9ad5762` byte for byte. The candidate additionally caches incremental DIIS Gram entries,
stages contiguous operands, and uses a full CTA for history transport and
extrapolation. Every FP64 dot and output coefficient sum keeps its original
scalar order. The small solve, normalization, retirement and final physical
validation retain their existing owners.

## Complete endpoint results

Three alternating baseline/candidate rounds, each in a fresh process, measure
constructor, preparation, SCF, analytic forces and host export. Each cold call
is followed by three warm calls from that process's frozen cold density. AOT
compilation and Python imports are outside the endpoint. Native execution never
uses an oracle density. Slurm job 13579 preserves its assigned GPU visibility;
OMP/OpenBLAS/MKL each use one thread. Other scheduled jobs may use other GPUs
on the node; this is not a claim of an otherwise idle physical host.

| Method, 96 atoms | Baseline cold (s) | Candidate cold (s) | Reduction | Cold SCF / reported Fock builds |
| --- | ---: | ---: | ---: | ---: |
| PBE | 20.839021 | 19.895562 | 4.53% | 11 / 11 |
| PBE0 | 24.073740 | 23.507947 | 2.35% | 9 / 9 |
| r2SCAN-3c | 46.060636 | 44.163734 | 4.12% | 10 / 10 |

Every cold gain exceeds the predeclared robustness floor
`max(2%, 2 * (baseline MAD / median + candidate MAD / median))`, which is 2%
for these populations. All individual samples, MADs and ranges are retained.
Warm medians (baseline/candidate) are 5.479339/5.480555, 6.960302/6.959683 and
15.813859/15.936351 seconds respectively. Every warm call reports one iteration
and Fock build. No warm speedup is claimed; the largest increase is 0.77%.
Detailed Coulomb/exchange and quartet counts remain unavailable rather than
being inferred from iteration counts.

A separate intrusive PBE0 profile measures twelve DIIS updates at 0.815841 GPU
seconds before and 0.180795 after: 0.133003 for pending dots and 0.047793 for
updates. This includes three warm calls and is mechanism evidence, excluded
from endpoint populations. The native work-counter tests establish one new
live Gram row per insertion, including the deferred singleton diagonal;
cache reads do not redo old-old vector dots. These results are additional to
the Schwarz fix on the stated integrated base, not a combined speedup against
an older master or a universal Direct-versus-GPU4PySCF ranking.

## Scientific and resource qualification

All **130 integrated E/F calls** pass independent references: 72 clean calls,
40 Standard 3/12/48-atom, Fast 96-atom and DF 96-atom holdout calls, and 18
cold/warm/moved calls for PBE0/12/96 and r2SCAN-3c/96. Holdouts have one warm
call per cold process; their single observations are regression checks. Static
checks reuse the explicitly audited dense references in
[`readme-final-20261022`](../readme-final-20261022/README.md), including CPU/GPU
spot checks and grid stability. Gates are 1e-5 Eh/atom, 1e-4 Eh/Bohr maximum
force error and 3e-5 Eh/Bohr RMS force error. Moved atom 1 is displaced by 0.02
Bohr in z; fresh tighter full-density GPU4PySCF references run after native
timing and pass the tighter 2.5e-6/2.5e-5/7.5e-6 gates.

The optional stricter baseline/candidate diagnostic remains 1e-9 Eh and 1e-8
Eh/Bohr. Fast 3c/96 fails its force diagnostic at 3.442e-8 while passing the
independent scientific gate. Comparable same-binary variation was recorded in
the earlier Schwarz campaign; its cause is not established. Whole-endpoint
bitwise equivalence is not claimed. No failure or slow sample was removed.

GPU comparisons against the original serial DIIS path pass bitwise state,
coefficient and output checks at 512/513 AOs, one/two spins, normalized and
unnormalized windows, history capacities three/eight, wraparound, retirement,
reset, inactive poison and invalid state. Independent long-double dots check
the raw cache. Both comprehensive native DIIS suites pass. Compute Sanitizer
memcheck and synccheck report zero errors with filters covering both new kernel
instantiations; the unchanged serial comparison is not instrumented. There are
189 focused host/resource checks and five receipt-reader checks. No UKS
molecular endpoint speed claim is made from the two-spin kernel checks.

KS automatically admits the cache at >=512 AOs and history >=2. It costs exactly
8 * history^2 resident bytes per system and 4 KiB shared staging per CTA, with
no extra resident matrix. Capacity queries and execution share admission.
Small KS and ordinary HF retain their defaults; explicit
`GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM=0` restores serial DIIS. Explicit `1`
retains its prior cooperative meaning unless the reduction selector is
`ordered`. See the current [ownership contract](../../../docs/developer/array_api_frontend.md)
and [decision note](../../../.agents/notes/implemented/performance/2026-10-23-ordered-diis-cold.md).

## Rejected experiments and provenance

The retained 200 exploratory E/F calls all pass independent scientific gates,
but automatic cooperative reduction caused PBE cold iterations 11/16/13 versus
11/11/11 and a 3c 13-iteration outlier versus ten. It was rejected. Existing
ordered caching gained only 1.24% on 3c. Operand staging with the original warp
update improved PBE/3c but PBE0's 1.72% remained below the 2% floor. The final
full-CTA transport addresses the remaining copy/extrapolation cost. These older
populations precede the latest master integration and are never pooled with it.
Eight final intrusive profile calls also independently qualify.

CUDA 12.9.1, GCC 13.3, Release/sm120, `--fmad=false`, and verified sccache 0.16.0
were used. The cache was reused. Exact build input manifests (5,029 baseline,
5,032 candidate), package hashes, commands' hashes, cache statistics, run scripts
and filtered profiler rows are retained. The benchmark `source_revision` names
the common base; candidate changes are bound by their manifest. Full checkouts
come from integration commit `87b5a1a2e` plus the candidate input overlay.

A first packaging shell failed after compilation and was repaired without
rebuilding the frozen baseline. A first endpoint import lacked the public
method manifest; the complete checkouts were installed before the clean run.
A local host selection lacked a native library; all 67 resource tests then
passed with the final image under Slurm. An earlier unfiltered history-64
sanitizer run timed out at ten minutes and is not counted as a pass. Failed
attempts remain in ignored raw storage.

- Tracked `summary.json.xz` retains qualification, all timings/work, diagnostic
  outcomes, source-manifest identities and provenance.
- Raw `records.json.xz` and `sources.json.xz` retain exact-text numerical receipts
  (including rejected experiments and moved references) and input manifests.
  Their checksums and locations are in the summary. They remain in ignored raw
  storage to respect the repository's aggregate evidence cap.
- Raw bundles, binaries, full compiler commands, profiler reports and failures remain at
  `n1:/data/jzzeng/direct-dft-cold-integrated-20261023/` and
  `n1:/data/jzzeng/direct-dft-cold-next-20261023/`, with local ignored copies.

Host-only requalification, including every historical repeat, after copying the
two raw bundles from the stated existing location to `.artifacts/remote/`
(or passing `--raw-directory`):

```bash
PYTHONPATH=python:. python benchmarks/results/ordered-diis-cold-20261023/recheck.py
```

Acceptance checks remain active with `python -O`. The reader validates receipt
and reference identities and recomputes independent gates; it does not pin the
current production source or make a new performance claim.
