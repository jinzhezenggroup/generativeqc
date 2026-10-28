# Direct PBE0/48 component observations and moved E+F

Refs #2678. This bounded cohort qualifies **PBE0, 48 atoms (16 O, 32 H),
Direct, def2-SVP spherical, neutral singlet**, using GenerativeQC Standard and
stock GPU4PySCF. WB97M-V and 96 atoms remain **INCOMPLETE**. This is numerical
and public-observable qualification, with descriptive timings; it promotes no
production default, README ranking or universal performance conclusion.

## Selected complete endpoints

All values below are complete host-observed E+F seconds. Cold includes endpoint
construction and preparation; warm/moved includes complete energy and forces
returned to the host. Instrumented populations are separate from clean timing
denominators. Three repeat values are retained even when the first is slower.

| Engine | Clean cold | Clean unchanged repeats | Moved RMS 0.01 Bohr repeats | Moved RMS 0.05 Bohr repeats |
|---|---:|---|---|---|
| Standard | 12.150991 | 2.507887, 2.503362, 2.492670 | 10.277169, 8.783838, 8.759549 | 12.315394, 10.810821, 10.878743 |
| GPU4PySCF | 10.727535 | 3.128085, 3.055981, 3.072959 | 7.878879, 7.939394, 8.315136 | 10.103652, 9.543442, 9.669293 |

Instrumented cold/unchanged values are Standard 12.698487 /
2.509145, 2.519277, 2.512102; GPU4PySCF 11.011442 /
3.132307, 3.096232, 3.055883. Four additional cold setups, excluded from these
28 timed samples, cost Standard 12.163416/12.173962 and GPU4PySCF
10.713631/10.828605 seconds. No interval totals are added across overlapping
process, API, or CUDA-event clocks.

Standard target SCF iterations/Fock builds are 9/9 cold, 1/1 unchanged, 8/8 at
0.01 and 10/10 at 0.05. Its separate cold Huckel preparation used seven
iterations/builds; these are not additional target Focks. GPU4PySCF iterations
are 9, 1, 7 and 9 respectively. Its observed `get_veff` counts are 10, 2, 8
and 10; these are effective-potential invocations including initialization and
audits, not Native Fock-build equivalents. Full work histories and the available
Native precision ledger are retained. Narrow unavailable primitive counters
are not replaced by inferred values.

Each moved amplitude has a new owner and uses only that engine's immutable own
cold density. Both amplitudes share the frozen displacement direction field;
RMS is `sqrt(mean_atom(sum_xyz(delta**2)))`. Remote execution uses the actual
frozen coordinates, not a regenerated RNG sequence. No reference density enters
Native production. Actual D0 byte identity is **UNAVAILABLE**.

## Component observations

The following are the instrumented **cold** observations, not comparable kernel
measurements or additive decompositions. Full cold and unchanged records are in
`summary.json.xz` and the original trace/API records in `selected.json.xz`.

| Boundary | Standard count / inclusive CUDA-event seconds | GPU4PySCF count / inclusive API host-wall seconds |
|---|---|---|
| Full-range K | `direct_k`: 9 / 3.278016 | `scf.get_k`: 10 / 3.883506 |
| J / joint J,K | `cuda_ks_exact_jk_shared`: 9 / 3.773548, joint | isolated J UNAVAILABLE; public `get_jk` observer not invoked |
| Semilocal XC | `cuda_ks_xc_shared`: 9 / 1.992501 | `scf.semilocal_xc`: 10 / 2.254590 |
| Force one-electron | `one_electron_stationary_pair_prepared`: 1 / 0.122010 | one-electron and Pulay: 1 / 0.069203 |
| Force J,K | `direct_jk_force`: 1 / 0.709121 | `force.jk`: 1 / 0.747669 |
| Force XC/grid/partition | independent isolated force XC region UNAVAILABLE | `force.xc_grid_partition`: 1 / 0.735318 |
| SR K, LR K, VV10 and VV10 force | not applicable to PBE0 | not applicable to PBE0 |

Native event spans can include stream idle time; nested resident/prepared and
class/root records are not summed. API wrappers pass original arguments, aliases,
return values and exceptions through and add no GPU synchronization. J cannot
be recovered by subtracting K from joint time. The measurements do not identify
a causal cross-engine bottleneck. Full primitive observer neutrality remains
**UNAVAILABLE**; authenticated available public semantics passed.

## Numerical acceptance and observer admission

Every selected endpoint and cold setup passes the original independent gates:
energy absolute error per atom <= 1e-5 Eh, force maximum <= 1e-4 Eh/Bohr,
force RMS <= 3e-5 Eh/Bohr. Every geometry has two adjacent tighter GPU grids
(4/5) and an independent CPU grid-5 endpoint; moved geometries also have CPU
4/5. Grid stability and finest CPU/GPU agreement use the unchanged quarter
margin. The replay recomputes these from numbers, ignoring stored pass flags.

The cold GPU reference is explicitly reused from
`../readme-final-20261022/oracles.json.xz`, bound by its original
`reference-reuse.json` entry. Its producer source is
`02337e0cf3d94915a352d5056f6d09d27fe353f2`, not the measured production source.
The replay authenticates the original receipt identity, controls and exact
ordered geometry before translating field names. The two displaced GPU
references and all CPU references were independently executed for this cohort.

The legacy exact E/force/full-history bitwise diagnostic remains **FAIL**, even
for clean/clean GPU repetitions (force differences around 1e-13). It is retained
unchanged. The separately reviewed admission requires source pass-through AND
exact available public semantic work/settings/provenance AND the original
independent gates AND the existing quarter numerical stability budget AND full
within-engine cache equality. It does not replace missing primitive/D0 evidence
or fit a new epsilon. Both selected clean/instrumented pairs pass that conjunction.

## Source, allocations and limitations

Measured production source is `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`.
The benchmark driver SHA-256 is
`15f1ed3fe62e2944efd58ccb5d1367926bcaf986a8a1f93d2edfd1c17bfc2fe3`.
This identifies the executed snapshot. The submitted driver additionally applies
the existing raw-output path guard during argument parsing, before reading
geometry; endpoint, observer and admission functions are unchanged. The executed
snapshot remains independently recoverable with `driver.patch`.
The SDK config is `7143f01546fdfc991a007d841421d18ef033a16d4949d788832af25edbb59069`;
core DSO is `688b0e152215809bf493d6388e08da52cbeebc22f1db34c938ca98df8c47e5db`.
The preflights independently checked all 12,353 Git source blobs/modes, 13,924
environment files and 1,919 source/installed SDK dependencies, including 24 AOT
DSOs and manifests. Source-tree Python plus CMake SDK was used, not a wheel.
GCC 13.3.0, CUDA 12.9.86, sccache 0.16, Python 3.12.3, NumPy 2.4.6,
PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 13.6.0 and Libxc 7.0.0 are frozen.
GPU4PySCF used the CuPy contraction engine (cutensor unavailable).

Native 14 timed endpoints are from Slurm 13615; GPU4PySCF 14 are from 13673.
Both used physical RTX5090 `GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`, PCI
`00000000:01:00.0`, SM120, driver 580.95.05, 32607 MiB. These are separate
allocations. Original Native CPU affinity is **UNAVAILABLE**; actual later
GPU4PySCF affinity observations and enclosing process times are retained.
Native and GPU4PySCF comparator cache inventories were separately stable at
421 and 423 entries, respectively. Their starting inventories are different.
Neither the allocation nor a byte census is a reservation for later work.

## Negative results and recovery

All actual submissions, failures and extra work are retained separately:

- CPU 13563 failed dependency packaging before SCI; 13564 supplied five CPU
  reference endpoints. Its terminal scheduler classification was unavailable
  after Slurm purged the job and accounting was disabled.
- GPU 13585 failed the SDK inventory path assumption before SCI; 13586 supplied
  two Native pilot endpoints and stopped at the old bitwise predicate.
- 13589 supplied three Native diagnostic endpoints. 13594 supplied three
  GPU4PySCF diagnostics plus four displaced GPU reference endpoints. Startup
  clean A grew the cache and is excluded as a comparator; clean B is retained.
- 13615 stopped after GPU4PySCF clean grew cache 421->422. Its eight GPU4PySCF
  timed samples remain cache-rejected; its qualified Native samples are retained.
- 13624 ended DEADLINE without allocation or SCI. 13667 supplied eight more
  numerically valid GPU4PySCF samples but failed cache 422->423; moved SCI did
  not run. Both cache FAIL results remain FAIL.
- Offline inspection of the two new 2380-byte cache metadata records found
  exactly one byte different, offset 1520, the CWD digit `formal1`/`formal3`.
  They describe empty SM120 programs, contain no kernel body, and are not SDK
  executables. The raw bytes and hashes are retained in the diagnostics logical
  record collection, using base64 only for non-UTF8 bytes. Reviewer code can
  decode both, check hashes and compare the bytes directly. The frozen CuPy
  compiler-source hash and `_preprocess('', ...)` investigation are retained.
- Separately authorized 13673 held CWD at existing `formal3` while all new
  output/temporary/lease paths used `fixedcwd1`. Actual child-CWD execution and
  before/after old-tree censuses were verified. It completed 0:0 in 134 seconds,
  04:00:38--04:02:52 CST on 2026-10-24, within the 04:10 deadline, no requeue.
  No cache mutation was waived and no hidden priming SCI was performed.

There were 44 formal timed attempts: 28 selected and 16 cache-rejected, plus
four cold setups. Eight diagnostics and nine independent reference endpoints
are separate work; process and population clock intervals must not be added
when they overlap. The two earlier immutable local partial archives remain
separate incomplete snapshots; they are not this publication.

## Retained bytes and reproduction

Run from the repository with NumPy available:

```sh
python benchmarks/results/direct-phase-moved-20261023/reproduce.py --output .artifacts/direct-phase-moved-replay
```

The output must not already exist. This performs no SCI or GPU operation. It
requires Python assertions: `-O`, `-OO` and active `PYTHONOPTIMIZE` are rejected
before restoration or module import, since the immutable retained audit uses
assertions for acceptance gates. Run the public replay entry point above. It
checks publication hashes, restores the numerical/protocol logical records with
their original bytes/SHA-256, and recomputes all selected admission/numerical
gates. `provenance.json` retains the measured commit, SDK/library identities,
package versions, inventory counts and original manifest hashes. Full source,
SDK and environment file lists and compiler commands are recovered separately
from the [verified history snapshot](../retention-granularity-20261024/README.md);
they are not inputs to the numerical audit. `selected`, `diagnostics`, `cpu`,
`execution` and `protocols` are logical JSON.xz collections of exact original
receipts, inputs, scripts, trace, logs and failure evidence; their member SHA
and sizes are independently checked. Old incomplete summaries keep their
original status and dates and do not supersede the selected audit.

Shallow output directories are supported. The wrapper runs a byte-identical
temporary copy of the retained audit at sufficient path depth for its eager
legacy CLI default, passing the actual records and repository paths explicitly.
The temporary copy is removed after execution; restored evidence is unchanged.

Collection-member Linux file modes were not observed and remain **UNAVAILABLE**.
The replay restores bytes only; it does not infer modes from Windows or umask,
and performs no `chmod`. This differs from the recorded Git/source modes and
environment executable-bit observations authenticated by the original Linux
preflights and retained inside the provenance manifests. Outer and member paths
must be canonical and contained; duplicate entries and existing output are rejected. XZ decoding is bounded to 96 MiB
decoder memory and 16 MiB expanded bytes per record, with truncated or extra
streams rejected. All selected bytes are validated before output creation.

The compact environment record identifies dependency versions, not redistributable
package binaries. Full source-file Git OIDs/modes/SHA and build commands are
retained in the pinned historical snapshot; SDK DSOs themselves were never
stored in Git. The CPU
dependency-closure source can be reconstructed from the measured Git revision
and its explicit 928-file manifest in the retained protocol records, verifying
each blob/mode. This restores
source files, not the bytes of an old tar container. `driver.patch` reconstructs
the then-untracked measured driver on the production revision. On Windows,
apply that patch with `git -c core.autocrlf=false apply` to preserve the measured
LF bytes; default working-copy CRLF conversion is not byte evidence. Existing Git
reference data remains required; no external archive, Release or expiring URL
is required. Restoring records alone does not reproduce hardware timings or
claim an identical future compute environment.

## Lossless source-patch storage

The original source patches are retained as XZ files. Before applying a patch,
restore its original filename with `xz -dc NAME.patch.xz > NAME.patch`.
The [storage inventory](../pr-merge-storage-20261028/manifest.json) records both
stored and original byte counts and SHA-256 values. Measurement identities,
all samples, and acceptance decisions are unchanged.
