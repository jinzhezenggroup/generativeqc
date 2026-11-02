# Bounded H100 Direct WB97M-V water48/96 evidence

Refs #2732, #2724. This family retains five immutable execution histories and
their source, control and numerical evidence. The two formal size cohorts have
72 complete E/F returns from 18 fresh serial processes. All original observable
and independent reference quarter gates pass. The sizes ran in separate
physical GPU/cache cohorts; their timings are never pooled.

The measured science is repaired source `cbd4d8074de35105717c8aac7cd1f86babbe90ec`,
with the separately authenticated portable CUDA sm90, untuned HF SDK from #2750.
Archive commit `7522994e…` supplies retained references, not the measured source.
Neither current master nor the original measured 5,852-file environment is
qualified by these measurements. This evidence does not establish hardware-only
causality, a generally fastest engine, production promotion, or parent completion.

## Reproduce without a solver

From this directory, use standard Python (no NumPy, CUDA, SDK or network):

```console
python -B reader.py
python -B -m unittest discover -s . -p test_reader.py -v
python -B reader.py --restore 2732-formal96-run --output /new/empty/destination
```

The default reader authenticates the six artifact byte inventories, decodes the
original IEEE energy/full force/timer/ordinal frames, recomputes reference errors,
checks original settings and phase namespaces, and verifies archived Root release,
API allocation, source/environment/DSO, physical device and process birth witnesses.
It imports no archived executable code. The archived platform witnesses prove
what was observed during execution; this is not a live platform query.

`summary.json.xz` is the losslessly compressed derived output for inspection.
Read it with `xz -dc summary.json.xz`; it is never read as an oracle.
New reader runs still emit ordinary JSON. The decompressed snapshot retains
SHA-256 `347da594e81adcec91e8a9161861d793b3cf0b12211fdef9a16f80bcff3959ea`.
The original endpoint timer statements, scientific expressions and each process's
immutable cold-density warm seed were preserved. Added retention occurs after
the timer. Removing observer AST additions recovers the original statements;
the exact archived source and derivation receipts remain in each job's inventory.

## Formal measurements

Seconds below use the original complete endpoint timers. Cold is the median of
three independent processes. Warm is the median of their three within-process
warm medians. All three warm arrays and both ranges remain in the reader output;
nine warm calls are never pooled into nine independent process observations.

| Size | Engine | Cold median (s) | Median of process warm medians (s) |
| --- | --- | ---: | ---: |
| water48 | Standard | 73.38147906400263 | 8.331081944983453 |
| water48 | Fast | 48.716772874991875 | 7.6576017959741876 |
| water48 | GPU4PySCF | 17.581941420037765 | 5.876181058003567 |
| water96 | Standard | 258.75514141470194 | 26.020890172570944 |
| water96 | Fast | 162.75577101763338 | 23.901594722643495 |
| water96 | GPU4PySCF | 42.263697946444154 | 13.648962665349245 |

Within each formal cohort the order is Standard/Fast/GPU4PySCF, then
Fast/GPU4PySCF/Standard, then GPU4PySCF/Standard/Fast. Each process performs one
cold and three warm complete E/F calls. Source-matched physical UUID, PCI,
allocation/cgroup, source, environment, code, DSO and cache-cohort identities are
held within each size. A new versioned release and one-shot leases bound each
execution; no automatic retry or replacement outcome was used.

## Scientific and work scope

Inputs are spherical def2-SVP, charge zero, spin zero, full moving-grid force
response and the frozen default level-3 XC/VV10 settings. Input SHA identities:

- water48: `17d44df7afa9996b0abf7df72301bf4758eadbde9f4523ce321001d14d2596ea`;
- water96: `03c18d243ced417acc7ae0dc324f54145f1665febded4f50c62f8d73b1d4d0e3`.

The exact original independent oracle blob has SHA
`d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4`.
Indices 29/39 supply size-matched full level-5/6 reference arrays. The reader also
checks the original water3 CPU33/GPU27 level-6 geometry and quarter-margin gate.
No new reference solve was performed for this publication. Every retained call
must satisfy energy/atom <=1e-5 Eh, maximum force error <=1e-4 Eh/Bohr and RMS
force error <=3e-5 Eh/Bohr. Independent grid/spot errors must fit one quarter of
these bounds. Missing or nonfinite values are never replaced by zero.

Native Standard cold calls report 28 iterations/29 primary Fock builds; Fast
cold calls report 16/17, with requested auto/effective 32-bit Fock work and strict
stages. Warm native calls report one iteration/Fock build; Fast warm effective
precision is 64-bit. Fast retains `fp32_gradient_not_qualified` and
`qualification_not_per_call_observable_audit`; an exposed SCF precision inventory
is not proof of all force operators. Stage, final audit and Huckel preparation
counts overlap primary work and must not be added into an invented work count.
Actual GPU4PySCF iterations/veff counts remain per call in the reader output;
operator precision, fallback, VV10-pair work and peak memory are unavailable.

Engine stopping policies differ. Native48/GPy grids have 539168/539392 allocated
points; native96/GPy grids have 1078336/1078528. Padding counts are not evidence of
different physical quadrature. The comparison does not equate SCF iterations or
infer missing telemetry from elapsed time. All costs are retained by scope in
`costs.json`; guarded process wall, supervisor wall and created-to-finished clocks
overlap and are not summed. Actual billing and GPU busy time are unavailable.
Normal native/JIT/CuPy/driver cache changes were retained without clearing,
repair, identity rewriting, forced misses or an invented cache-purity gate.
Snapshot content is not a compiler hit count or a complete final filesystem map.

## Negative and diagnostic histories

| History | Complete E/F | NOT_RUN | Interpretation |
| --- | ---: | ---: | --- |
| failed first water48 diagnostic | 2 | 10 | numeric gates pass; worker phase/index validator fails; child1/guard125, cleanup succeeds |
| successful water48 diagnostic | 12 | 0 | one independent process per arm; no formal statistics |
| water48 formal | 36 | 0 | separate accepted source-bound formal cohort |
| water96 diagnostic | 12 | 0 | one independent process per arm; no formal statistics |
| water96 formal | 36 | 0 | separate accepted source-bound formal cohort |

The 98 actual complete returns include 26 diagnostic returns, of which the first
two belong to a failed control outcome. Only the 72 formal returns enter formal
statistics. The historical HOST V2 receipt publication failed after controls:
27 actual control originals survive, while full-suite qualification remains
`UNAVAILABLE`. The subsequent receipt correction and original failure logs are
retained, without promoting Windows mocks or partial HOST records to device proof.

## Lossless storage and availability

`terminal-evidence.json.xz` stores 840 unique SHA-addressed original byte objects
(8,297,094 decoded bytes) plus all five job/path/byte/hash/POSIX-mode inventories
and original nonmember envelope fields. It restores 1,101 selected original
members (23,528,376 logical bytes), including every scientific raw return,
metadata, log, control and source member. All originals, including the full
34,271,880-byte five-job logical history, remain unchanged locally/remotely.

Exactly three first-pilot generated cache binary contents, totalling 10,743,504
bytes, are inventory-only: two `object.o` files and one `runtime.so`. Their exact
paths, sizes, SHA and POSIX modes are retained. Public restoration explicitly
refuses these contents. No public binary download, regeneration or original
transport-envelope reconstruction is claimed. The later four original collectors
already retained those binaries remotely with inventory-only local coverage.
Private paths and local aliases are not substitutes for public byte objects.

The complete scientific/runtime/control/source history and independent oracle
remain in-tree. Ordinary `reader.py` replay still checks every actual return,
force vector, source/device/release witness, work count and failed/NOT_RUN record.
It reports the optional historical HOST audit as `NOT_RUN_HISTORICAL_AUDIT`,
separately from numerical acceptance, using `historical-host-summary.json`.

Four superseded HOST producer/failure capsules (490,912 bytes), plus their
original artifact inventory, use the existing stock snapshot restorer at
already-merged `dd4fa29891e84f2a7be234c30c1dbc7a3fa2de89`. All five original
files were recovered and checked byte-for-byte before retirement. The snapshot
retains their exact path, length, SHA256, Git blob and mode. No scientific
payload, independent reference or actual runtime original was retired.

The historical audit restores 37 uncompressed local HOST files and checks four
standalone compressed originals, including the independent oracle that remains
in the current tree. Its local HOST mode scope is distinct from remote POSIX
witnesses. Run this optional audit explicitly:

```console
python tools/restore_retained_evidence.py --all --manifest benchmarks/results/h100-direct-wb97mv-20261024/historical-host-snapshot.json --output .artifacts/wb97-host-history
python benchmarks/results/h100-direct-wb97mv-20261024/reader.py --host-history-root .artifacts/wb97-host-history/benchmarks/results/h100-direct-wb97mv-20261024
GQC_H100_HOST_HISTORY_ROOT="$PWD/.artifacts/wb97-host-history/benchmarks/results/h100-direct-wb97mv-20261024" python -m unittest discover -s benchmarks/results/h100-direct-wb97mv-20261024 -p test_reader.py
```

Shallow clones may need an explicit fetch of the pinned published commit first.
Neither reader nor restorer fetches implicitly. Missing/corrupt HOST recovery
still fails closed; ordinary offline numerical replay needs no history fetch.
Storage classification and historical recovery are not whole-science approval.
No new recovery framework or nested compressed archive is introduced.

Both XZ decoders use a 96 MiB memory limit, a 64 MiB expanded envelope limit,
EOF and no-trailing-data checks. Paths, duplicate members/JSON keys, base64,
SHA/size identities, missing/unreferenced objects and excluded binary coverage
are checked before restoration. Restoration requires a new directory and uses
exclusive file creation. POSIX modes are applied only on POSIX systems; Windows
does not fabricate equivalent permissions. No retained cache, shared SDK or
production path is modified by this reader.

Publication, independent latest-head approval, required CI and normal protected
merge remain separate delivery gates. This family does not close #2724 or any
broader method qualification merely because these bounded formal gates pass.

The byte-object container uses schema v2: UTF-8 objects are stored as text and
other objects as base64. The reader authenticates the same original SHA-256
bytes and preserves all 840 objects, failures and numerical populations. This
lossless storage change does not rerun or reinterpret the historical campaign.
