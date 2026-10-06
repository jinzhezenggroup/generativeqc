# Experiment: prune the residual Direct-force operator call graph

Status: proposed; warp-page candidate rejected, CTA follow-up numerically qualified but not promoted
Date: 2026-10-05

## Scope

Issue #1892's P0-B lane concerns the unavoidable generic derivative fallback,
not another implementation of the hot generated shell-class consumers. P0-A
generalizes those consumers separately. Both lanes must be tested independently
and composed before a production performance claim.

The initial control is clean master `de18539de`. The isolated candidate freezes
the full-range operator in the existing bounded kernel template at the
full-range launch boundary. Combined and independent J'/K' outputs remain
distinct instantiations. Existing SR/LR/RSH launch boundaries retain their
runtime operator and exact fallback. This removes unreachable operator callees
without changing screening, task classification, queue geometry, recurrence,
source coefficients, atom scatter, or the number of domain scans.

## Evidence collection

`benchmarks/direct_force_resources.py` collects the final linked shared library's
CUOBJDump ELF resource records, architecture/image identity, demangled symbols,
and binary/output hashes. It does not load CUDA. Repeated architectures are not
coalesced, incomplete or duplicate selected records fail, and PTX is excluded.
CUDA's unrelated local helper symbols may repeat inside a linked image and do
not invalidate the selected kernel inventory. Input ELF checks reject unlinked
objects, archives, and standalone cubins; production device-link provenance is
still required. Focused host tests cover these boundaries. A complete existing
library has also been inspected successfully.

That historical library belongs to the older current-defaults campaign, not the
new control. Its restricted/unrestricted generic force entries have 255
registers and 90,696/90,680 bytes of linked stack respectively. These are static
reservations, not executed traffic, occupancy measurements, or a new performance
baseline. The hashed historical receipt is retained locally under
`.artifacts/p0b/historical-resource-receipt.json`.

The control and candidate build with the release sm_120 configuration and
verified CXX/CUDA ccache launchers. The candidate initially lives under
`.artifacts/p0b/radial/`; its preserved compile command uses the control's
generated headers and otherwise identical options. An unlinked candidate
object is not accepted as final resource evidence.

## Qualification still required

- Finish and seal clean control and candidate source/binary identities.
- Compare final linked per-kernel registers, stack, and local reservations.
- Run exact RHF/UHF fixed-density derivatives and hybrid RKS/UKS stationary
  J'/K' gates, preserving coefficients, force semantics, and physical closure.
- Exercise through-f, SR/LR/RSH fallback, generated-class disable/overflow,
  changed geometry, memcheck, and initcheck.
- Measure 48/96-atom complete HF and PBE0 cold, five-warm, moved, and
  five-moved-warm endpoints. Retain all actual SCF trajectories and independent
  exact-direct energy/force comparisons; do not normalize by iteration count.
- Profile actual class work and device time separately from clean endpoint
  timing. Record traffic, occupancy, and stalls only when measured, otherwise
  leave them unavailable rather than inferring them from register count.
- Record eligible-warps/issue efficiency, active lanes, barrier/wait/scoreboard
  stalls, and local load/store request counts for every scheduled candidate.
  Counter improvements without complete endpoint improvements are insufficient.
- Reassess the residual generic fraction after P0-A and qualify the composition.

Local node3 is maintenance-drained in Slurm. The available peer pool is node1,
node2, node4, and node5; it is not a requirement to execute on every host. Select
an available matching node, normally node1, and use another peer only when that
advances a distinct useful gate. Node1 exposes verified usable RTX 5090 resources;
node2 exposes PRO6000 and is not a matched 5090 control. Node4's scheduled NVML
smoke (job 627) failed with a driver/library mismatch; its CUDA driver smoke
(job 629) also failed (`cuInit=804`). The loaded module is 580.173.02 while the
system driver libraries are 580.178.04. Do not label node4 usable until an
actual scheduled CUDA check succeeds. A matching private library extraction is
being investigated without installing or replacing system drivers. Node5's
single 5090 is occupied by unrelated job 1449; the finite smoke job 1452 is
queued and must not preempt that work.

Every real-device command runs through the peer's main partition with the
actual GPU type (`--gres=gpu:5090:1`, or `--gres=gpu:pro6000:1` on node2), finite
time, and unchanged Slurm device visibility. The user's explicit peer-node
request includes node2; its functional evidence and timings remain separate
from the 5090 campaign. Offline compiler and binary inspection do not require
a GPU allocation. Node2's scheduled PRO6000 smoke (job 2389) preserved
visibility `2`; this is infrastructure evidence, not a numerical gate.

Node1 Slurm job 6019 completed a scheduler-assigned device smoke check on RTX
5090 UUID `GPU-57db1e6a-5ff9-570d-67b0-6965fb3a77e6`, driver 580.95.05,
with visibility `2` preserved. This verifies an available device, not a numerical
gate. A clean control source snapshot has been staged at
`/data/jzzeng/qc-1892-p0b-20261005/control` on node1; its fallback source digest
matches the local control exactly. The numerical binaries are not staged yet.

## Rejected alternatives and stop conditions

Do not revive the rejected thirteen-pass angular schedule: this candidate keeps
one traversal and does not rescan the domain per order. Do not assume that
smaller blocks, a register cap, or a smaller static stack imply an endpoint win.
Retain known losing generated classes such as dddd in their exact fallback.
If linked resources or complete endpoints do not improve, preserve the negative
result rather than promoting this operator specialization as an optimization.

## Initial linked component result

Node2 CPU-only Slurm job 2386 reproduced the production native device-link
object inventory for control and candidate, then wrapped each component in a
host ELF for offline inspection. These images are **not runnable endpoint
libraries** and must not be loaded as substitutes for a complete CMake build.
Explicit GCC 11 matches the compiled objects; node2's default GCC 12 lacks its
C++ frontend, so the first attempt failed rather than silently changing inputs.

All eight specialized full-range derivative entries retain **255 registers per
thread**. For both spins, combined full-range entries reserve 87,520 stack bytes
and separate J'/K' entries reserve 87,376. Control generic entries reserve
90,680 bytes with Force screening and 90,696 with Fock screening. The unchanged
runtime range fallback retains 90,680 bytes. Thus operator specialization alone
reduces the static stack by approximately 3.5–3.7%, but does not lower the linked
register requirement. This is neither executed local traffic nor a demonstrated
speedup; numerical, profiling, and full endpoint acceptance remain outstanding.

Control component SHA256:
`11c60e52b0e17b3202b52d0062857a6adca88ddd015d6f9063e0e278e201ebd6`.
Candidate component SHA256:
`99a3b3ac11529a3a841726156c5151846dddde3773dd2e0f9954ceb35c7ff81e`.
The authoritative input/link/resource receipts are retained locally under
`.artifacts/p0b/native-resources/{control,radial}/`; check their hashes rather
than inferring source provenance from these component filenames.

## October 5 NCU refinement: prioritize task shape, not a register target

The new #1892 comment 5994009310 (2026-10-05 12:01:16 UTC) adopts #1965's
Slurm 6016/6018 measurements on its frozen 96-atom PBE0 warm force producer.
Those reports describe a different retained campaign; they are not measurements
of this worktree's still-building control or the radial specialization above.

| Counter | Retained measured value |
| --- | ---: |
| Achieved occupancy | 16.67% |
| Warp cycles per issued instruction | 31.10 |
| Barrier stall cycles | 13.67 (approximately 44%) |
| Wait stall cycles | 8.23 |
| Short scoreboard stall cycles | 5.50 |
| Long scoreboard stall cycles | 1.49 |
| Active threads per warp | 17.91 / 32 |
| Local-load requests | 58,849,833,769 |
| Local-store requests | 27,175,631,760 |
| Scheduler cycles with no eligible warp (6016) | 93.57% |

Local requests are **not bytes** and do not distinguish intentional local arrays,
stack temporaries, and compiler spills. Specialized reference kernels can also
use 255 registers; register count alone is therefore not a causal explanation.
The 6018 report required 30 replay passes and inflated its selected endpoint
to 396.56 seconds. Its timings must not replace clean endpoint or clean device
trace timings.

The refined next steps are:

1. Favor existing exact class/source queues so each CTA drains more homogeneous
   work without rescreening the entire domain. Keep the mathematical integral
   owner shared with P0-A and preserve unsupported/overflow fallbacks.
2. Inspect recurrence/root scratch and thread-local array lifetimes separately
   from spills. The radial specialization is now only a narrow resource-control
   experiment; its modest stack reduction is not sufficient to prioritize
   production promotion.
3. Compare switchable 128/64-thread variants as scheduling experiments, not
   register/occupancy remedies. Retain 256 as the control and hold task admission,
   coefficients, output policy, source identity, and screening constant.
4. Require numerical closure, matched semantic counts, the expanded NCU counters,
   and clean 48/96-atom HF/PBE0 E+F endpoints with actual SCF histories. Reassess
   the residual after P0-A before spending effort on a now-removed hot class.

There is an important implementation invariant for the CTA experiment:
the current warp-drain stride is `kBoundedDirectThreads / kDirectQuartetThreads`,
while scalar tasks use `blockDim.x` and candidate chunks advance by the fixed
256-entry queue capacity with one candidate per lane. Merely changing the launch
block size would skip both candidates and queued warp tasks. Any smaller-CTA arm
must coordinate its chunk width and warp-drain stride and demonstrate complete
candidate/queue coverage;
all threads must still reach the CTA barriers. The fixed 256-entry queue remains
an independent bounded-storage contract, not a reason to repeat domain scans.

The authoritative 6018 report hash is
`40d50996c07f979e5aa65a6734f0f9bdb893f665e39fab9e8a27cb53dd418e50`,
and its native binary hash is
`86e4c80a5afcafe79489d6fafcc70ef7a7dbc48f3a2d31177d3d2dd4eb5352d6`.
The fetched issue/comment snapshots are retained locally under
`.artifacts/p0b/issue-refresh/`. No new additive-gap attribution or speedup is
claimed by incorporating this mechanism evidence.

## Current control and independent-page experiment

The complete control CMake library is now sealed, with SHA256
`daf5d5b0f3bdca15bdf0e56243f0163455767333c2fa8ff6b9285f4a561ce2af`.
Its final resource receipt confirms 255 registers and 90,680/90,696 stack bytes
for Force/Fock-screened generic derivative entries in both spins. Node1 job 6039,
explicit step 0 with unchanged visibility `1`, exits zero for the independent
through-f J/K value/derivative, SR/LR, indexed/triangular prefix-budget and batch
gates. This qualifies the control, not an optimization candidate or endpoint.

Node4 job 631 now verifies the private 580.173.02 library path: NVML succeeds,
`cuInit=0`, and the scheduled visible device count is one. No system driver was
installed or replaced. Its already-running control memcheck job 633 remains a
distinct useful gate. The user clarified that peers are alternatives, not an
all-host obligation; unnecessary pending node5 jobs 1454/1455 and duplicate node2
control job 2393 were canceled. Subsequent deployment defaults to node1 only.
These cancellations are not numerical failures. Foreign jobs were not changed.

A separate full CMake candidate lives at
`/data/jzzeng/qc-1892-p0b-warp-pages-20261005`. Its switchable full-range experiment
gives each warp a disjoint 32-entry queue and independent claim on the existing
global block/page cursor. All candidate predicates, canonical orientation,
generated coverage, recurrence, source coefficients and scatter remain shared.
Warp barriers replace CTA barriers only for that experimental instantiation;
each accepted domain page is still claimed once, without per-angular rescans.
The ordinary schedule and range seams retain their existing behavior.

The host scheduler model passes 4,488 candidate-domain shapes and 290 queue-tail
cases, including indexed empty tails and unindexed tight-budget recovery. The
native cursor assertion remains exact: it accounts for eight independent
terminal warp claims per CTA only in the experimental full-range arm; LR retains
one terminal claim per CTA. An incremental build recompiles the changed native
test before sealing the final candidate. The production source in this worktree
remains unchanged.

## Sealed candidate and executed qualification

The isolated complete CMake candidate library SHA256 is
`b2d9886bde67dc01f764a38f8e60dfc2461382607e1f5fe4a8eae2912b3abab3`.
The source inventory differs from the control only in
`src/scf/cuda/direct_bounded_fallback.cu` and
`tests/native/test_cuda_fock_provider.cpp`. Its full-range warp-page entries
retain 255 registers/thread, reserve 87,520/87,376 stack bytes for combined/
separate outputs, and use 4,192 static shared bytes. The ordinary generic arm
remains available. This experiment couples independent warp scheduling with
full-range operator specialization; it does not isolate their individual effects.

The first deployment failed before device testing because an unanchored rsync
`build` exclusion also omitted ten tracked historical benchmark receipt files.
Job 6042 exits nonzero on the source-manifest check; it is not a numerical
failure. Restricting exclusions to the checkout root preserves the complete
sealed source inventory. Failed setup records are retained, not discarded.

| Node1 job | Explicit step | Gate | Result |
| --- | --- | --- | --- |
| 6043 | 0 | Native independent full/SR/LR J/K, both spins, through-f, indexed/unindexed prefix budgets and batch | exit 0; full derivative error at most approximately 3.89e-16 |
| 6045 | 0 | Public 3-atom PBE0 cold/warm/moved/moved-warm, independent GPU4PySCF, both frozen libraries | all four endpoints per native arm pass |
| 6052 | 0 | Hybrid RKS/UKS snapshots, independent fixed-density full-range J'/K' against Libcint, CPU/CUDA SR/LR derivatives | 14 tests pass, no skipped tests |
| 6054 | 0 | Memcheck on the actual changed full-range through-f path | exit 0; zero errors |
| 6055 | 0 | Initcheck on the actual changed full-range through-f path | exit 0; zero errors |
| 6056 | 0 | Synccheck on the actual changed full-range through-f path | exit 0; zero errors |

Node4 control memcheck job 633 also exits zero with no errors. Range-only
sanitizer runs cannot qualify this change: that mode does not instantiate the
experimental full-range schedule. Candidate sanitizer drivers therefore use
`--through-f-response`, not `--range-response-only`.

The PBE0 smoke retains identical native cold/warm/moved/moved-warm iteration
counts `15/1/10/1` and matching exported Fock-build counts. Maximum independent
energy error is approximately 9.67e-13 Eh and maximum force error approximately
1.35e-11 Eh/bohr. Its cold timings include runtime compiler-cache population
and are not an optimization speedup claim.

Matched 48/96-atom HF and PBE0 complete endpoints now run on one assigned GPU
in node1 job 6058, explicit finite `srun` step 0. Each arm retains one cold,
five fixed-density warm, one moved, and five moved-warm calls. The independent
oracle and scientific protocol are common to each pair; 96-atom process order
is reversed. The retained HF direct/DF endpoint harness, not the older
`readme_hf_scaling.py` comparator, supplies its existing moved-geometry protocol.
The source-archive adapter substitutes only the HF harness's two Git provenance
queries using verified snapshot identities; it does not replace SCF or force
owners. An earlier endpoint setup job 6053 failed because its independent
reference arm requested a nonexistent reference-source identity. The corrected
driver uses the actual control source identity; both attempts are retained.

Native RHF full iteration histories are not exported by this retained API, and
its simple `fock_builds` query returns null in this campaign; those counts must
not be inferred from iteration counts. Raw per-call RHF iteration counts remain
available; PBE0 retains its complete native KS diagnostic history and exported
work counts. Additional precision/operator-work diagnostics or device traces
can supply the missing RHF work evidence. No complete matched speedup, NCU
improvement, residual fraction after P0-A, or P0-A composition acceptance is
claimed at this checkpoint.

## Completed matched endpoint decision

Node1 job 6058 terminates with exit zero. All 96 native complete endpoints
(four method/size cells, two arms, twelve calls each) pass independent energy
and analytic-force gates. Each pair uses one assigned GPU in the same finite
Slurm step, the same scientific protocol, and the same independent oracle.
The raw timing medians are:

| Endpoint | Control warm (s) | Candidate warm (s) | Candidate change | Control moved-warm (s) | Candidate moved-warm (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HF, 48 atoms | 0.760433 | 0.759547 | -0.12% | 0.759165 | 0.758554 |
| HF, 96 atoms | 3.016947 | 3.002926 | -0.46% | 3.019599 | 3.017882 |
| PBE0, 48 atoms | 11.795183 | 13.011753 | +10.31% | 11.788955 | 13.000003 |
| PBE0, 96 atoms | 40.105534 | 41.053754 | +2.36% | 40.109866 | 41.062091 |

PBE0 warm and moved-warm calls have matching actual one-iteration, one-exported-
Fock-build branches in both arms. The HF differences are sub-percent and not
treated as a material improvement. Native RHF's simple build query remains
null in these clean records.

Cold and moved numbers are retained without iteration normalization. In
particular, PBE0 48-atom cold improves from 114.636307 to 107.694825 seconds but
executes 27 versus 25 SCF iterations/builds, so this is not evidence for the
force schedule. PBE0 96-atom cold executes 29 versus 32 iterations/builds, and
its moved call 13 versus 14. HF 96-atom cold likewise differs by one iteration
(25 versus 26). These actual branch differences must not be hidden by scaling
the times or discarding samples from accuracy gates.

Maximum independent native energy and force errors across this complete matrix
are approximately 1.04e-10 Eh and 2.73e-11 Eh/bohr, respectively, well inside
the retained 1e-8/1e-7 endpoint gates. Complete native PBE0 trajectories remain
in the raw records; the reviewed summary preserves actual per-call work and
branch matching, without claiming missing RHF histories.

**Decision:** do not promote the coupled warp-private-page/full-range-
specialization candidate. A reduced static stack and eliminated CTA barriers
are not performance acceptance: both qualified PBE0 sizes regress. Keep the
ordinary source and all generated/resource/unsupported fallbacks unchanged.
This decision rejects this measured scheduling candidate, not every possible
warp-scoped schedule or independent operator specialization.

Authoritative receipts are
`.artifacts/p0b/campaigns/6058-endpoints/` and the separately verified
`.artifacts/p0b/campaigns/6058-endpoints-summary.json`. Node1 job 6065 now
collects separate single-warm Nsight Systems traces and partial native HF work
telemetry. Those instrumented timings cannot replace this clean matrix.
The HF trace's final-density 55-class census is identical between the two
arms, and both report a valid native one-build strict stage with zero
post-SCF builds. The higher-level operator inventory remains explicitly
incomplete, with no full SCF timeline; do not relabel it as complete telemetry.
No instance of the changed generic kernel is observed in either retained HF
warm trace, so the HF comparison supplies compatibility/non-regression evidence
rather than speed evidence for the changed consumer. PBE0 source-device
attribution is now sealed below; the actual qualified post-P0-A residual census
and any subsequent P0-B candidate remain separate pending work.

Earlier focused host validation passes 29 tests, including strict rejection of a
missing selected resource line even when other entries are complete. Linked
resource collection never loads CUDA; it is not a device numerical gate.

The subsequent resources/Schwarz suite passes 27 tests, including two new CLI
boundaries: reject reviewed `benchmarks/results/` destinations before starting
binary inspection, and retain raw linked SHARED/STACK/LOCAL fields in an allowed
receipt. The collector reuses `benchmarks._retention.raw_output_path`; reviewed
publication still goes through `tools/evidence.py publish`. A real control
recheck after this guard change exactly preserves the original binary hash,
CUOBJDump stdout hash and all selected resource rows. These collector changes
do not alter or rebuild scientific production sources.

## Sealed single-warm device attribution

Node1 job 6065 completes with exit zero in explicit finite Slurm step 0. The
four Nsight Systems reports, SQLite exports, scalar records, driver snapshots
and checksums are retained under `.artifacts/p0b/profiles/6065-single-warm/`.
The read-only verifier `.artifacts/p0b/verify-single-warm-profile.py` checks all
twelve independent cold/prime/profiled-warm scientific gates, exact sealed
binary identities, the original report/SQLite/JSON hashes, actual one-build
PBE0 warm branches and matching grid/resource policies. Its derived receipt is
`.artifacts/p0b/profiles/6065-single-warm-summary.json`.

The public 96-atom PBE0 warm call contains exactly one changed generic force
launch in each arm:

| Measured quantity | Control | Warp-page candidate |
| --- | ---: | ---: |
| Generic force device interval (s) | 6.511770065 | 7.454649380 |
| All warm kernel intervals summed (s) | 37.026190353 | 37.991517791 |
| Exclusive stationary derivative wall (s) | 7.582477540 | 8.524317879 |
| XC geometry drain wall (s) | 21.866818864 | 21.911787815 |
| Registers/thread | 255 | 255 |
| Threads/CTA | 256 | 256 |
| Runtime application static shared bytes | 3,084 | 3,168 |

The generic device interval worsens by 0.942879315 seconds, or 14.48%, while
the exclusive derivative wall worsens by about 0.94184 seconds. This identifies
the observed consumer behind the separate clean 96-atom warm regression of
about 0.94822 seconds; it does not turn the intrusive device interval into a
clean endpoint sample or prove a stall mechanism. Both profiled PBE0 cold
trajectories execute 26 builds and their warm replay executes one build.
Clean campaign 6058's different cold histories remain unchanged in its raw
receipts. All compared force policies use dense AO work, 2,359,296 grid points,
256-point tiles (9,216 tiles), 10,752-point chunks (220 chunks), and the same
536,870,912-byte additional device budget.

The actual NVTX ranges are `p0b/profiled-warm` and
`p0b/public-stationary-force`; the verifier checks that the generic launch lies
within the public force range. The composite public force path does not call
the patched snapshot method, so the attempted
`p0b/full-range-Jprime-Kprime` label is absent. Do not claim a producer NVTX
range that was never executed. The measured kernel identity and the existing
exclusive force-work ledger provide the attribution instead.

Nsight warns that CUDA Event completion tracing may add overhead and false
stream dependencies. Summed kernel intervals are not whole-endpoint wall time.
Achieved occupancy, eligible warps, issue efficiency, active lanes, stall mix
and executed local load/store requests remain unavailable for this candidate:
no NCU capture or privileged profiling permission is claimed. In particular,
the zero CUPTI `localMemoryPerThread` column cannot establish zero local state,
as the exact module probe below demonstrates.

## Linked versus driver resource semantics

The apparent 1,024-byte shared discrepancy is resolved by a dedicated
attribute-only real-device probe: node1 job 6083, finite five-minute
`srun --partition=main --gres=gpu:5090:1`, step 0, exit zero. It loads the
sm_120 image extracted from each exact sealed final library, queries the exact
mangled kernel with `cuModuleGetFunction`/`cuFuncGetAttribute`, and queries
`CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK` (111). It launches no
kernel and provides no timing, occupancy, spill or numerical evidence.

The assigned RTX 5090 reports exactly 1,024 driver-reserved shared bytes per
block. Both extracted ELF symbol tables expose `.nv.reservedSmem.cap` with
value `0x400`. Their shared-section sizes agree with the CUOBJDump resource
receipt; the application attributes independently agree with the runtime
trace:

| Kernel | CUOBJDump linked SHARED | Driver application shared | Measured reserve | Driver local bytes/thread |
| --- | ---: | ---: | ---: | ---: |
| Control restricted generic Force | 4,108 | 3,084 | 1,024 | 90,680 |
| Candidate ordinary restricted Force | 4,108 | 3,084 | 1,024 | 90,680 |
| Candidate experimental combined Full | 4,192 | 3,168 | 1,024 | 87,520 |
| Candidate experimental separate FullSources | 4,192 | 3,168 | 1,024 | 87,376 |

Thus the retained modules account for the discrepancy as application shared
plus the measured driver reservation, not a second unidentified kernel. Do
not generalize by subtracting 1,024 from every architecture or tool receipt.
The driver local reservation agrees with linked STACK, even though the linked
LOCAL field and this CUPTI export's local-memory field are zero. These are
different reporting surfaces; none is a measured local load/store request
count or proof that the storage represents register spilling.

Reproduction scripts, job/step receipts, exact ELF section/symbol listings,
module/binary hashes and driver results are retained locally under
`.artifacts/p0b/resource-semantics/`; job results are in `probe-6083/`.
The exact extracted module bytes remain on node1 under
`/data/jzzeng/qc-1892-p0b-20261005/resource-semantics/{control,candidate}/`.
Their SHA256 values are respectively
`0fc205aac99d475836e719b091b2f00a72b267cc901afb607f4cbc478f42020b`
and `85e28ac4747d34cc966db883cee0b6e659e42f43f2a86b74a30ea67d5ff815c9`.
GNU readelf emits warnings for CUDA-specific metadata; those stderr files are
retained rather than misrepresenting the tool as a complete CUDA ELF decoder.
The sealed executed probe snapshot, not subsequently formatted helper copies,
is authoritative for job 6083.

## Fresh P0-A observation, not qualified composition

Read-only inspection of P0-A's completed node1 job 6079 observes the queue-v3
snapshot, not its older foundation or mutable current checkout. Its exact
profiled library hash is
`df9789b572c9c46c3baf129decf3171c474886a0c84a69e3d7dac17afb1cf1fd`,
source identity
`17e54eaf17d6ba0ac79804138c42e1994fe9b8dcefb6fa9aa84004c951103398`.
All twelve retained public PBE0 endpoints pass their independent oracle gates.
Only the first original warm call is captured, with one iteration/build; its
cold and moved histories execute 26 and 14 builds. The finite Slurm step and
driver/source/binary checks remain with the copied receipts.

The fresh trace still contains one ordinary generic force launch lasting
6.575867768 seconds (255 registers, 256 threads/CTA). Sixteen class-specific
weighted-force launches are also observed, summing to 0.000278174 seconds.
Kernel presence alone does not establish admitted/executed descriptors,
profitable hot-class migration, queue residency, or absence of whole-class
overflow. The high-level force ledger is not a complete per-class descriptor
census. Do not infer such counts from launch duration or its zero descriptor
fields. Do not present this distinct allocation as a matched comparison with
6065, or compute a post-qualified-migration residual fraction from it.

The observation is retained at
`.artifacts/p0b/p0a-observations/v3-6079/`, including original reports, scalar
histories and derived `direct-kernels.json`; `evidence-observed.sha256` seals
the local observation. The P0-A owner remains responsible for queue
qualification and complete paired acceptance; this lane does not modify that
checkout or duplicate those experiments. A qualified composition is still
outstanding.

## Independent CTA-size follow-up

Following #1892 comment 5994009310, a new default-off experiment is isolated in
`/data/jzzeng/qc-1892-p0b-cta-20261005` on the same `de18539de` control. Its
only changed tracked file is `src/scf/cuda/direct_bounded_fallback.cu`.
`GENERATIVEQC_EXPERIMENT_DIRECT_FORCE_CTA_THREADS=64` or `128` selects a
smaller CTA only at the existing full-range force launch boundary; unset,
zero, invalid strings and nonstandard input shapes retain the ordinary launch.
SR/LR/value boundaries, generated selection, kernel/operator call graph,
coefficients, scientific owner, screening and scatter stay unchanged. The
ordinary 256-thread schedule is not replaced by default.

The fixed 256-entry queue and existing sixteen indexed 64-candidate pages are
preserved. For unindexed work, admission packet width and stride shrink with
the actual CTA. The warp drain likewise strides by the actual number of warps;
changing the launch alone would skip candidates and queued contractions.
Global page claims and one terminal claim per CTA are unchanged, so the
existing native exact cursor assertions still apply. This is a coupled
CTA-size/within-page packet experiment, not evidence for register capping,
operator freezing, or a change in the number of scientific domain scans.
The compile-time launch bound remains 256; achieved occupancy is not inferred
from a smaller runtime launch.

`.artifacts/p0b/cta/check-ownership.py` reads the actual source constants and
passes 6,150 host models covering every 0..1,024 product-tail size, indexed
and unindexed traversal, all three CTA sizes, and exact warp queue ownership.
These are algorithmic ownership checks, not CUDA numerical qualification.
A release sm_120 CMake build now completes with verified CXX/CUDA ccache
commands and checkout-root normalization. Its final library hash is
`cca2d24221b6c040a32907b70cb77590526a6dd891da1ba14352e9b8b4472098`;
the restricted generic Force resources remain 255 registers, 90,680 linked
stack bytes/thread and 4,108 linked shared bytes. These unchanged reservations
make no occupancy or traffic improvement claim. The compiler was allowed to
finish; it was not restarted due to an observation timeout.

Node1 job 6091's finite Slurm driver first gates ordinary/64/128/invalid-setting
through-f execution and memcheck/initcheck/synccheck of both actual variants.
At this checkpoint ordinary, 64 and 128 through-f execution pass; remaining
gates are still running, so the overall job is not relabeled as a pass. Job
6095 has an `afterok:6091` dependency and can run clean 48-atom PBE0 triage only
after qualification finishes successfully. It compares the sealed control
and the same binary's disabled/128/64 schedules, with five warm and moved-warm
samples and actual histories. Losing triage variants need not duplicate an
entire 96-atom matrix. A winning variant still needs full matched 48/96 HF/PBE0
endpoints, precise work/fallback coverage, device attribution, measured NCU
counters and qualified P0-A composition before promotion.

Node1 still reports `RmProfilingAdminOnly: 1`; ordinary permission to proceed
does not authorize sudo, a driver policy change, or inheriting another task's
privileged profiling authorization. No such operation is performed.

Job 6091 subsequently completes with exit zero. All four native through-f
settings (`0`, `64`, `128`, `invalid`) pass, including both spins, independent
J/K and full/SR/LR rows, indexed/unindexed prefix budgets and batch ownership.
Both actual reduced-CTA variants pass memcheck, initcheck and synccheck with
zero errors in all six runs. All ten output hashes and completion markers are
independently checked in
`.artifacts/p0b/cta/qualification/6091-summary.json`; complete original logs and
job/step/source/binary checks remain under its sibling `6091/` directory.
These gates qualify correctness and memory/synchronization behavior, not speed
or achieved occupancy. The dependent clean triage job 6095 is now executing;
no partial timing record is accepted as a completed four-arm comparison.

## Completed CTA triage and remaining acceptance

Node1 job 6095 completes successfully in explicit Slurm step 0. All 48 native
complete PBE0 endpoints pass independent oracle gates; the four arms share one
assigned RTX 5090 and identical scientific protocols. Each arm retains cold,
five warm, moved and five moved-warm calls and its full native KS history. The
read-only derived comparison is
`.artifacts/p0b/cta/triage/6095-summary.json`; original checked records and
runner/assignment/provenance receipts remain in its sibling `6095/` directory.

| Arm, 48-atom PBE0 | Warm median (s) | Change from sealed control | Moved-warm median (s) | Change from sealed control |
| --- | ---: | ---: | ---: | ---: |
| Sealed control | 11.467568662 | baseline | 11.453083117 | baseline |
| New binary, disabled CTA experiment | 11.470957819 | +0.03% | 11.459995016 | +0.06% |
| New binary, 128 threads | 10.930831928 | -4.68% | 10.921827428 | -4.64% |
| New binary, 64 threads | 11.429770067 | -0.33% | 11.387323663 | -0.57% |

Every warm and moved-warm sample has matching actual one-iteration,
one-exported-Fock-build work. The disabled new binary agrees closely with the
sealed control, so the 128-thread result is not merely a comparison against a
slower disabled rebuild. The 64-thread differences are not treated as material
or selected for a duplicated full endpoint campaign. This does not establish
that 64-thread CTAs always lose on other workloads.

The raw cold/moved branches are retained without iteration normalization:
control cold/moved execute 25/12 builds; disabled executes 26/12; 128 executes
25/13; 64 executes 26/13. In particular, the 128-thread moved call is slower by
about 5.69% while executing one extra SCF build. Neither that difference nor
the differing cold histories should be attributed to the force schedule by
normalizing timings or removing calls from accuracy gates.

**Decision:** retain 128 threads as a promising isolated candidate, not a
production default. New finite node1 jobs 6111 and 6112 separately collect
single-warm Nsight Systems attribution (sealed control, disabled rebuild,
128-thread variant) and the full paired 48/96-atom HF/PBE0 endpoint matrix.
The two jobs have independently assigned GPUs; profiler intervals cannot
replace the clean matrix. Matrix process order is reversed at 96 atoms and
the archived HF adapter substitutes only its two truthful Git provenance
queries using the new one-file source identity. No scientific owner is mocked.
At submission, these jobs are pending/running, not pass evidence. Their
drivers are retained under `.artifacts/p0b/cta/` and their exact job ids are in
`followup-submissions.txt`.

Measured NCU counters still need explicit privileged-profiler permission; a
targeted permission question is asked without executing sudo or changing driver
policy. Qualified P0-A composition, full endpoint acceptance and those counters
remain outstanding. Neither the 48-atom result nor the numerical/sanitizer
success completes P0-B or authorizes production promotion.

## Completed follow-up matrix and actual CTA attribution

The previously pending follow-ups subsequently complete with exit zero.
This is an additional checkpoint, not a rewrite of the submission-time status.
Node1 job 6112 retains all 96 native complete HF/PBE0 endpoints at 48/96 atoms,
with cold, five warm, moved and five moved-warm rows per control/candidate arm.
Every independent energy/force gate passes. Maximum native energy and force
errors are `1.0504663805477321e-10` Hartree and
`2.95813651352006e-11` Hartree/Bohr against `1e-8`/`1e-7` limits.

| Clean endpoint | Control warm (s) | 128-thread warm (s) | Change | Control moved-warm (s) | 128-thread moved-warm (s) | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF 48 | 0.751831379 | 0.751084708 | -0.10% | 0.750311397 | 0.751064248 | +0.10% |
| HF 96 | 2.976780333 | 2.969434790 | -0.25% | 2.976284504 | 2.977499500 | +0.04% |
| PBE0 48 | 11.569291767 | 11.040451683 | -4.57% | 11.567456890 | 11.038006328 | -4.58% |
| PBE0 96 | 39.966333989 | 38.445335492 | -3.81% | 39.916961476 | 38.396261070 | -3.81% |

All warm/moved-warm samples have one actual SCF iteration; all PBE0 warm rows
also have one exported Fock build. Native HF does not export full iteration
histories or Fock-build counts, so those unavailable quantities remain explicit
rather than inferred from the iteration count. HF remains compatibility
evidence, not a speed claim for the changed generic force consumer.

Cold/moved solver branches are retained without timing normalization:
HF-96 cold executes 25/26 control/candidate iterations; PBE0-48 moved executes
15/12 builds; PBE0-96 cold/moved execute 28/25 and 17/12 builds. The complete
raw times still belong to the endpoint record, but their differences do not
isolate CTA scheduling. PBE0-48 cold has matching 25/25 builds and only a
0.51% raw difference; this is not claimed as a material cold-endpoint win.

Separate intrusive job 6111 authenticates the source/library receipts and
original profiler outputs. Its actual 48-atom PBE0 trace contains one generic
restricted full-range force launch per arm, with grid X 1360. Sealed control
and the disabled new binary both use block X 256; the enabled variant uses
block X 128. Their GPU intervals are `2.252254201`, `2.252667086`, and
`1.700066311` seconds respectively. The only executed P0-B NVTX ranges are
`p0b/profiled-warm` and `p0b/public-stationary-force`. This attributes the
candidate effect to an actually changed launch; it does not measure achieved
occupancy, hardware stalls or executed local-memory traffic. Intrusive CUDA
Event tracing remains separate from the clean job 6112 endpoint timings.

**PR boundary and decision:** ship the CPU-only resource collector, its host
tests and reviewed evidence/rationale. Do not apply the private CTA experiment
to shipping CUDA source or change the production default. Its one-file source
patch is retained solely for reconstruction. The promising warm/moved-warm
PBE0 gains justify continued investigation, not completion of P0-B. Measured
NCU counters, qualified P0-A composition, full promotion identity/resource
gates and a strict interleaved promotion comparison are still outstanding.
No sudo, profiling-policy change, merge, release or issue closure is implied.

The selected immutable samples, independent reference values, whole endpoint
protocols, source reconstruction, linked resource/driver receipts and separate
profile/triage/qualification summaries are published through
`tools/evidence.py publish` under
`benchmarks/results/direct-force-resources-1892-20261005/`. Its CPU-only
`verify.py` authenticates all selected members, recomputes the 96 independent
native numerical gates and reproduces every clean median/actual-work comparison.
The publication explicitly records an inconclusive performance decision and
does not manufacture missing equation/IR/schedule digests. Original profiler
databases, scheduler receipts, logs and build products remain ignored under
`.artifacts/p0b/`; large debugging streams are not committed or uploaded.

The final focused collector/scheduling suite passes 27 host tests with the
verified ccache CXX wrapper. This is host validation, not an additional GPU
qualification. All measured device work above uses finite Slurm steps on n1;
the allowed hosts are alternatives, not a required multi-host matrix.

## References

#1892 P0-B; #1952; the retained order-five weighted-source campaign under
`benchmarks/results/pbe0-weighted-order5-20261005/`; the rejected angular-schedule
note under `.agents/notes/rejected/2026-10-04-tzvpd-angular-force-schedule.md`.
