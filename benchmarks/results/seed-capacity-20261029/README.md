# Native warm-seed capacity qualification

PR #2865 replaces precision-audit staging's Python AO/spin reconstruction with
`generativeqc_batch_query_warm_seed_capacity_v1`. The native prepared owner
supplies all-slot density/coordinate bytes and the largest density buffer before
seed export. Overflow and failures publish no partial receipt; only an absent
symbol permits compatibility. Query bindings share bounded native library
ownership with the SCF-default transport.

This is a scoped #2227 R2/R4/R6 cutover. It does not move seed histories, audit
scheduling, checkpoint formats, or whole-process/provider memory authority.
The envelope excludes descriptors, allocator overhead and validation scratch.
R3 is excluded and the parent remains incomplete.

## Source and validation

Baseline production source is `e8de3e4d9` (#2864). Candidate production source
is `115cc3c72`; later commits integrate the base's evidence and retain these
receipts only. `identities.json` binds full source IDs, CPU/CUDA binary hashes
and actual worker metadata. Builds use Release, GCC 11.4, CUDA 12.9.86/sm_120
and verified sccache 0.16.0. Shared-cache statistics are not exclusive task hits.
Full build logs, installed binaries and original Nsight databases remain in
ignored artifacts; database hashes are retained in `profile-summary.json`.

Final-source CPU validation passes **130 CTests** and **88 focused Python tests**.
Earlier integrated validation passes 103 tests plus 25 subtests; those receipts
retain their original stage. Tests cover actual restricted/unrestricted HF/KS
ragged owners before/after SCF, independent PySCF spherical/Cartesian AO counts,
exact/below-capacity staging and restoration, integer overflow, transactional
refusal and compiled owners that forbid materialization or scientific work.
The installed public-header SDK exercises RHF/UHF/PBE RKS/UKS with Python and
compiler absent from PATH. The GPU-linked SDK prepares the corresponding real
CUDA owners. Sphinx and repository ownership/contract checks pass.

Slurm **14843** uses node1/main/RTX 5090, a finite one-hour allocation and its
assigned `CUDA_VISIBLE_DEVICES=0`. All **19 independent-reference CUDA precision
batch tests pass** with `_compatibility_seed_capacity` poisoned. They cover
Direct/DF HF/KS, separate candidate/auditor histories, cold/frozen/moved states,
fitted audit restoration and failed-neighbor isolation. No GPU work uses node3.

## Complete audited endpoint matrix

RHF, UHF, PBE RKS and PBE0 RKS request energy or energy/forces. H2 is scalar;
the fleet is H2/water, using H2+ and neutral water for UHF. CPU uses STO-3G and
GridSpec(12,6,12); CUDA uses spherical def2-SVP and production Treutler level 3.
Each configuration uses explicit FP64, initial_guess=None, and an absolute
energy TargetAccuracy of 1e-7 Eh. This forces actual independently owned audits
and candidate seed suspension/restoration, including when forces are requested.

Two source-isolated persistent processes prime ordinary model/artifact caches.
Static Calculator construction/imports are outside timing. Every measured
protocol starts fresh computational owners and cold densities. Scalar timing
includes preparation, execution, publication and cleanup. Fleet timing covers
preparation/cold, two frozen replays, moved coordinates, two moved replays and
teardown, plus a separately retained complete protocol total. CPU uses fixed
CPU affinity 0. This small-system matrix is not universal resource-limit or
method qualification, and does not claim a general speedup.

Eight alternating ABBA/BAAB groups retain **1,792 complete calls per backend**
across both engines. Samples retain E/F, iteration/Fock/grid/J-K/policy work,
actual audit counts, candidate seed bytes and auditor seed bytes. Fixed gates
require identical semantic/audit/seed work, E <= 2e-9 Eh, F <= 2e-8 Eh/bohr,
and every median candidate/baseline ratio <= 1.05.

| Backend | Maximum time ratio | Maximum E difference | Maximum F difference |
| --- | ---: | ---: | ---: |
| CPU | 1.008437472446882 | 0 | 0 |
| CUDA | 1.0376111664979546 | 1.1368683772161603e-13 | 5.3290705182007514e-14 |

Nsight's complete untimed matrices match at **309,867 kernels**, **520 graph API
calls**, **20,444 explicit dynamic allocations**, and **119,110,081-byte
simultaneously live explicit dynamic peak**. Both finish at zero dynamic live
bytes. Static storage is separately 1,822,825 bytes; opaque provider/driver
storage is excluded. H2D is 26,762 copies / 161,982,388 bytes; D2H 29,638 /
97,814,148; D2D 8,900 / 918,214,144. No work/data-movement reduction is claimed.

`qualify.py` reproduces the paired matrix and `endpoint_worker.py --once` the
untimed profile matrix. CUDA always requires finite `srun`. These readable
copies may be formatted; the executed originals are retained in the controls.

## Unsuccessful setup and exact recovery

Slurm 14842 passed the installed CUDA SDK, then failed pytest collection because
the uploaded source lacked `benchmarks._cases`. Its log is retained unchanged.
A new immutable source directory adds the missing tracked benchmark inputs;
14843 reruns the entire owned GPU suite and qualification. No test, model or
threshold was removed. The original CPU SDK probe also supplied the wrong RHF
carrier for a named KS fixture; the corrected fixture uses PBE_RKS/PBE_UKS and
passes. Both original failure and final success receipts remain separate.

`controls.json.xz` stores lossless original UTF-8 logs, test/control receipts,
executed scripts and profiler CSVs with SHA-256. To stay within the unchanged
80 MiB repository evidence budget, raw samples use existing Git history:

- CPU samples: commit `d1387a772`, original `cpu-samples.json.xz` path here.
- CUDA samples: commit `647b6daa0`, original `cuda-samples.json.xz` path here.

`history.json` records full commits, paths, byte counts and SHA-256. Recover with
`git show COMMIT:PATH` and verify the checksum. Earlier SCF-default raw samples
remain recoverable through their own manifest. No Release, external archive,
parent closure or broader resource-authority claim is implied.
