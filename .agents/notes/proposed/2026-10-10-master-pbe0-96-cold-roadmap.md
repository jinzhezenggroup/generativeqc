# Proposal: prioritize MD-J work reuse on current-master PBE0 cold

Status: proposed; investigation complete, no production optimization implemented
Date: 2026-10-10

## Scope and identity

The user requested a current-master investigation, not another measurement of
an older composed branch. Fetching origin and a subsequent remote-head check
both identify `985caaf01688bd6df3388b2c7985af3176efbf9d`. An immutable Git archive
of that commit was built independently on n1/node1 in Release for sm_120.
C++ and CUDA launchers reuse ccache 4.5.1 and its existing shared cache. The
automatic host linear-algebra provider resolves to scalar in this environment;
the CUDA scientific consumers remain native. The library SHA-256 is
`5757f82f923cf30fb1519b7942a3388767008069cf96d22208cd75da94661751`.

The closing fetch advances master to
`37227b5384fac74e6122f7befc1086bbbd877181` through CC #2191 and MP2 #2188.
An explicit integration audit finds byte-identical CMake configuration, DFT,
SCF, integrals, Python runtime, relevant compiler owners and PBE0 benchmark
protocol against the measured snapshot. The changed materialization manifest
is an MP2 review inventory, not a PBE0 runtime input. The local checkout is
updated to that latest master. These unrelated changes do not justify another
whole GPU campaign or change the proposed priorities; the recorded timings
remain measurements of **985caaf01**, not a relabelled whole-37227b538 build.
The exact path/tree comparison is retained as `master-integration-audit.json`.

This is the README **exact-direct** PBE0-RKS E+F fixture, not PBE0-DF:
32 waters / 96 atoms, 768 spherical def2-SVP AOs, FP64, 2,359,296 unpruned
48x16x32 moving-grid points, three Becke iterations, native energy/density
tolerances 1e-12/1e-10 and screening 1e-12. Independent GPU4PySCF uses the same
basis and quadrature, stock DIIS, explicit full-density potential rebuilding,
energy/gradient tolerances 1e-12/1e-10 and screening 1e-14. Host force return and
moving-grid response stay inside both endpoints. Their stopping metrics are
not identical and equal Fock counts would not imply equal scientific work.

Cold means a fresh process, density and prepared owner with persistent disk
caches retained. Timing covers preparation through synchronized first public
energy plus host forces; imports, CUDA context, Calculator construction,
serialization and post-result teardown are excluded, as in the README.
All real-device work uses finite srun on `main`, `gpu:5090:1`, eight CPUs,
preserving Slurm visibility. No maintenance-node work or reference cold seed
is used. No optional J/K/XC schedule selector is overridden in clean runs.

## Clean complete-endpoint evidence

Slurm **6989** runs three alternating fresh-process reference/native pairs
in one 15-minute allocation on an RTX 5090. All six child outcomes and the
outer allocation exit zero. Retain every trajectory and every timing:

| Engine | Complete E+F samples (s) | Median (s) | Iterations / Focks |
| --- | --- | ---: | --- |
| Native | 93.476007, 97.304600, 93.359329 | 93.476007 | 17/17, 18/18, 17/17 |
| GPU4PySCF | 77.953171, 86.627211, 93.844865 | 86.627211 | 37/38, 42/43, 46/47 |

Native is 7.906% slower at the observed median; three pairs do not establish
universal relative performance. Do not remove the reference's faster 38-Fock
sample or normalize the complete endpoints by iterations/Focks.

| Host-wall scope | Native median (s) | Reference median (s) |
| --- | ---: | ---: |
| Preparation | 5.047504 | 0.763853 |
| SCF and energy publication | 71.224643 | 77.232753 |
| Complete public force consumer | 17.203860 | 8.630604 |

Independent native maxima are 8.18546e-12 Eh energy error and
3.77452e-11 Eh/bohr force error; maximum native physical residual RMS is
1.76985e-12. Every native endpoint converges, returns all 288 coordinates,
passes the existing 1e-8 Eh / 1e-7 Eh/bohr gates and uses no warm seed/fallback.

The earlier Slurm **6983** also retains six complete, independently accepted
endpoint arrays. Its native/reference medians are 93.510326/86.568638 s.
However, editing its live launcher caused a post-measurement shell EOF error
and outer exit 2. Its original launcher is reconstructed byte-for-byte against
the recorded hash and retained as `run-clean-executed.sh`. These observations
are quarantined from the primary completed-allocation population, not silently
deleted. Job 6989 uses a separate immutable launcher, retains script snapshots
and records each child outcome explicitly.

## Intrusive attribution, not another clean population

Slurm **6988**, a separate finite ten-minute allocation, collects native and
reference Nsight CUDA/NVTX traces and Python cProfile. Both endpoints and the
outer allocation succeed. Native's traced branch uses **19 Focks**, not the
clean median's 17, and takes 104.653071 s. Its independent E/F gate passes.
Device sums, Python inclusive time and overlapping CUDA API residence must
never be added to the clean wall table.

Native SCF/publication has 79.837004 s host wall and 75.212276 s device-kernel
time. The kernel interval union is also 75.212276 s: these measured kernels
are serialized, not hidden concurrent work. Attribution of this device sum:

| Executed work | Device seconds | Fraction of SCF kernel time |
| --- | ---: | ---: |
| MD-J uniform potential | 18.187250 | 24.18% |
| MD-J partially screened residual | 13.083780 | 17.40% |
| Other MD-J | 0.007009 | 0.01% |
| XC consumers plus AO evaluation | 24.238139 | 32.23% |
| Generated/native exact K | 17.448086 | 23.20% |
| Other kernels | 2.248013 | 2.99% |

**MD-J is the largest operator group (31.278038 s / 41.59%), not K.**
Its `(2,1)` residual worker alone takes 4.506742 s. Uniform `(0,0)`, `(1,0)`,
`(0,1)` and `(1,1)` potentials together take 10.644864 s. The diagnostic owner
reports 19 MD calls and 209,520,685 residual candidate capacity per Fock;
that capacity is not an accepted-pair, primitive, Boys/root or recurrence count.
Those executed-work counters remain unavailable and are the first next gate.

K has 21 complete class launches per Fock, 399 total. Classes containing d
account for 59.43% of its time, but dddd is only 0.070457 s. Do not repeat a
dddd-only or psss/psps-only campaign as if either explained the current cold
endpoint. Source dispatch establishes MD-J followed by independently generated
K; the old reducer that requires two identical generated J/K class passes is
not valid for this master composition.

XC `tiled_potential` and `tiled_density_product` take 10.662781 and 5.958897 s,
with 145,844 launches each, or 7,676 original tiles per Fock. There are 8,448
nonempty AO tiles per traversal; only 772 (9.14%) use compact contractions.
Each of the five compact stages has 3,420 launches, or 180 groups per Fock.
The existing group guard requires every nonempty tile to have 32--128 active
AOs; the actual map maximum is 536. Automatic batching is already enabled:
this is mainly its large/ragged-domain fallback, not a missing enable flag.

All SCF kernels total 795,737 launches. `cudaLaunchKernel` inclusive residence
is 73.543542 s, but includes device/backpressure waiting and overlaps those
75.212276 device seconds. It is not 73 seconds of removable CPU overhead.
SCF copies take about 1.8 ms device time; large-copy optimization does not
explain this region. Graph-only gains must be measured rather than inferred
from the launch count or API duration.

## Secondary force target

Clean force telemetry attributes medians of 5.755095 s to stationary integral
derivatives, 6.665814 s to semilocal geometry, and 1.624918 s to owner/artifact
setup. These internal phases cover less than the complete public consumer.
Map-discovery time is nested within the geometry work, not additive:
4,608 cold sampled-jet discoveries take a median 5.863617 s and generate
18,119,393,280 dense discovery jet values before the mapped consumer generates
4,114,391,040 jet values. The selected force map has 94,349,066,240
point/AO-square work versus the dense domain's 1,391,569,403,904.

The force trace has 11.704251 device-kernel seconds within 19.489891 s public
force wall. cProfile identifies 4,608 AO-map selection calls and substantial
snapshot/setup work; CUDA memcpy API residence includes waits, not just copy
bandwidth. Actual force copy durations total about 14.5 ms despite thousands
of calls. Target discovery/owner orchestration rather than asserting that
reported transfer bytes or synchronization counts are the dominant cost.

## Prioritized route and stop conditions

1. **P0: J-only source-work reuse and residual scheduling.** First census each
   Fock's candidate, admitted uniform/residual, primitive-product and recurrence
   work, including overlap of the two physical output orientations. The uniform
   one-warp-per-bra source computes oriented potentials separately; investigate
   sharing source/root work between admitted reciprocal orientations, beginning
   with the measured low-angular potential groups. Preserve each orientation's
   density/error predicate; do not assume every primitive pair executes twice.
   Then target the actual `(2,1)` residual producer/consumer. Preserve native
   Hermite algebra, existing density precontraction, normal K/XC ownership and
   the 128-MiB optional-storage/fallback contract. A bilateral source schedule
   needs bounded partial reduction and independent numerical gates; atomics,
   extra traversals or retained traffic may erase its source-work savings.
2. **P1: extend compact XC profitability to the real ragged AO domain.** Obtain
   actual per-tile extent/group rejection counts. Test contiguous qualified
   segments or resource-bounded extent buckets without merging AO maps or
   changing final per-entry tile order. The current batched launch geometry
   hard-codes 128: merely raising an admission threshold is incorrect. Charge
   full AO/work/local-potential/descriptor storage and keep the current fallback.
   Qualify new emitters/library providers on actual local extents, not the
   historical dense 768-AO synthetic crossover alone.
3. **P2: reduce cold order-2 AO discovery and force setup.** Preserve immutable
   geometry/grid/basis/derivative identities and the current sparse consumer
   domain. Do not reuse order-1 SCF maps as order-2 force maps. The previous
   forced 96-atom pre-AO producer remains unqualified for profitability and can
   produce denser work; the separate AO-reuse cold campaign failed its robust
   gain gate. Neither is an automatic replacement. Batched exact-map
   discovery or qualified source reuse is a separate bounded experiment.
4. K work remains worthwhile after these larger groups, with broad class
   coverage, not another global lowering/occupancy switch. Do not relabel
   isolated kernel savings, DF approximation, looser convergence, reference
   DIIS changes or an oracle-derived initial density as a direct cold win.

An engineering milestone is complete cold below 85 s on this frozen protocol,
not a forecast or a microbenchmark gate. Candidate promotion requires unchanged
independent E/F and physical-residual gates, full work/resource/fallback
accounting, and repeated interleaved cold/warm/changed-geometry endpoints.
Keep all SCF histories; accept a gain only above the existing robust noise gate
(at least 2% and twice-summed relative MAD). No source-count reduction alone
authorizes promotion. This investigation does not implement any of these changes.

## Retention and references

Local ignored evidence: `.artifacts/pbe0-master-cold-20261010/`, containing the
frozen source identity, build/cache/CMake receipts, exact launchers, all three
jobs' records, profiler reports, pstats, checksums and machine-readable summaries.
Remote evidence: `/data/jzzeng/pbe0-master-cold-profile-20261010-985caaf01/` on n1,
including immutable source, binary, both SQLite exports and Nsight reports.
`cold.py` implements only the cold subset of the unchanged README protocol;
`run-clean.sh` and `run.sh profile <reference-json>` are the retained launchers.
Reproduction must use finite srun and the recorded environment, not unset CVD.

- `src/scf/cuda/direct_jk.cpp`: independent MD-J/generated-K dispatch.
- `src/scf/cuda/direct_md_j.cu`, `src/scf/cuda/direct_md_jk.cu`: measured J owners.
- `python/generativeqc_compiler/dft/xc_point_batch_cuda.py` and
  `xc_tile_batch_cuda.py`: compact admission and bounded launch/scatter algebra.
- `python/generativeqc/_resident_ao_maps.py`: derivative-domain map identity.
- [Previous AO/discovery qualification](../implemented/performance/2026-10-08-pbe0-ao-reuse-and-csr-admission.md).
- [Current compact XC boundary](../implemented/performance/2026-10-08-xc-compact-batch-default.md).
- [MD-J promotion and rejected residual prefixes](../implemented/performance/2026-10-08-md-j-default-cold.md).
- [Benchmark stopping-metric contract](../implemented/compatibility/2026-10-09-benchmark-scf-metric-contract.md).

## Source-reuse follow-up (not yet promoted)

The first reciprocal MD prototype covered pair-angular totals zero/one and
reused one canonical `fill_coulomb` source for both independently admitted
output directions. Fixed-density J event time improved 14.32%, but clean
Slurm 7003 did **not** qualify complete cold: incumbent median 101.467395 s,
candidate 105.689671 s, a 4.16% regression. Incumbent Fock counts were
17/17/22 and candidate 17/19/20. The candidate's relative MAD made the robust
gain threshold 8.79%. All trajectories passed the independent E/F gate; no
extra-iteration history was discarded or normalized. This subset cannot
support a production promotion or PR.

The expanded experiment covers all unordered angular classes through two.
Angular-two tiles have 80 publication slots but only 64 source lanes, so a
strided publication loop is necessary; changing only an angular guard drops
the tail. The new integral-compiler emitter retains canonical native radial
mathematics, independent directional density/error screening, an 8x8 tile,
4096-worker cap, existing stream cursor and the 128-MiB optional-storage
fallback. The selector is captured by the resident owner, not polled at replay.

Same-density Slurm 7024 measured J medians 1.618226/1.290978 s, a 20.22%
diagnostic reduction. The actual uniform radial evaluations decrease from
1,926,600,288 to 1,001,504,816 (48.02%), while 1,926,600,288 density directions
and 23,506,259,200 Hermite summands remain unchanged. Both arms test
209,520,685 residual shell candidates, admit 12,903,112 shell tasks, and
contract 93,656,520 primitive radial sources. Thus the old residual candidate
field is not admitted task or primitive work. The diagnostic's independent
1200-byte counter buffer is charged within the same allowance; its downloads
and fences are excluded from clean performance populations.

Clean Slurm 7019 (three fresh processes per arm) measured incumbent/candidate
medians 98.179921/95.754976 s, a 2.47% complete cold gain exceeding the 2.00%
robust-MAD gate. Fock counts were 18/18/17 versus 19/19/18, all retained and
independently accepted. This is preliminary support for the broader source
schedule, not a claim that either source-count or event-time savings alone
qualify an endpoint or that these samples establish all non-regression gates.

Validation uncovered an important evidence trap: public host
`FockPlan.evaluate` uses the retained compatibility evaluator even when its
preparation diagnostic reports MD admission. The original 14 raw-J tests
therefore did not prove execution of resident MD. The corrected test-only
adapter calls the actual native-KS prepared device-value seam, adds no
production ABI, and verifies per-replay work census and immutable density.
Slurm 7015 passed all 16 actual resident tests, independent libcint J gates,
memcheck (zero errors), and racecheck (zero hazards/errors/warnings). Its later
large diagnostic failed because generic preparation had no extra MD allowance;
Slurm 7024 explicitly charged the bounded diagnostic plan and completed. The
failed harness runs 6992, 7008, 7010, 7015 remain retained, not relabeled passes.

The measured source closes over master `82c166caca181fc0df51b1ff9378b972e3b1eba8`.
Changes since the earlier 15bc69700 freeze add opt-in ordered DIIS and allocation
journals; default serial DIIS remains selected in this protocol. Final-source
default/opt-out cold qualification completed in Slurm 7045: medians
94.944027/89.159602 s, a 6.09% gain above the 2% robust gate, with the candidate's
22/17/17 Fock histories all retained. Broad Slurm 7033 completed all independent
E/F and cold/warm/moved/moved-warm non-regression gates. Integration master
`8eaaa66b3113d17bf2c492423aea0ec43ba6db4f` only changes CI/diagnostic tooling.
The [implemented P0 decision](../implemented/performance/2026-10-10-md-j-reciprocal-source-reuse.md)
and [complete retained evidence](../../../benchmarks/results/md-j-reciprocal-cold-20261010/README.md)
supersede this preliminary status, not the rejected v1 outcome. Keep the complete
goal as P0 -> P1 -> P2 -> P3 rather than treating P0 as completion.

An ignored P1 prototype prepares actual per-group AO launch extents instead
of the three hard-coded 128-AO launch bounds. It preserves original maps and
ordered scatter, admits bounded extents through 1024 and charges the existing
32-MiB arena before choosing a smaller group or the incumbent fallback.
Independent host planning/arithmetic tests pass, including the actual 536-AO
domain. GPU profitability and endpoint gates are still outstanding. Its files
are separate from the P0 working patch and carry no production claim.

Follow-up raw evidence is retained under
`.artifacts/pbe0-mdj-reuse-20261010/` and the n1 directory
`/data/jzzeng/pbe0-mdj-source-reuse-20261010-b74815dba/`, with immutable source
archives, build/cache receipts, launchers, complete JSON histories, work counts,
sanitizer logs and failures. P1 host evidence is under
`.artifacts/pbe0-xc-extents-20261010/`.
- `docs/performance_engineering.md`.
