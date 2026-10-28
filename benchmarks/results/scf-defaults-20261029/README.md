# Native SCF defaults qualification

PR #2864 moves the incumbent Python default-control decision into the native
catalog/grid owner. The versioned query selects the same qualified all-electron
CUDA RHF and exact PBE/PBE0/B3LYP/r2SCAN/M06-2X restricted KS default-grid contexts;
other contexts retain strict defaults and all explicit controls remain authoritative.
Only an absent protocol symbol uses compatibility. This is a scoped #2227 R4/R6
cutover, not a new numerical method, full precision lifecycle or parent completion.
R3 is excluded.

## Frozen source and installed product

Baseline is `8d6225b4fa49a82fdaca1abc1d56927008dde76c` (#2859). Final production
source is `e8de3e4d9`; subsequent commits only retain qualification evidence.
`identities.json` records full source IDs, both CPU/CUDA binary hashes and the
actual worker metadata. Release builds use GCC 11.4, CUDA 12.9.86/sm_120 and
verified sccache 0.16.0. Before/after statistics describe the shared cache, not
exclusive task hits. Full build logs and original Nsight databases remain in
ignored artifacts; the profile summary binds the databases by SHA-256.

The final source passes 129 CTests, 361 focused Python tests plus 25 subtests,
and Sphinx warnings-as-errors. Tests cover transactional ABI refusal, actual
catalog/grid policy, authoritative native errors, cache invalidation, explicit
query replacement, shared-image lifetime and concurrent binding retirement.
The installed public-header SDK runs with Python/compiler absent from PATH;
the GPU-linked SDK also passes in the finite allocation below.

Slurm job **14841** runs the final GPU qualification on node1/main/RTX 5090,
with a one-hour limit and assigned `CUDA_VISIBLE_DEVICES=0` preserved. All
**48 GPU tests pass**, including strict independent-oracle complete E/F,
Direct/DF HF/KS, default-grid warm/moved work, independent audit histories and
failed-neighbor isolation. The qualification plugin forbids the old Python
SCF-default predicate. No GPU work runs on node3.

## Complete endpoint matrix

Four configurations (RHF, UHF, PBE RKS, PBE0 RKS) request energy or energy/forces.
H2 is scalar; the fleet is H2/water. UHF uses H2+ and neutral water. CPU uses
STO-3G and GridSpec(12,6,12); CUDA uses spherical def2-SVP and the production
Treutler level-3 grid. Both use Fast with initial_guess=None.

Separate persistent baseline/candidate processes prime ordinary model/artifact
caches. Static Calculator construction and imports are outside timing. Each
measured protocol creates fresh computational owners and cold densities. Scalar
timing includes preparation, execution, publication and cleanup; fleet timing
includes preparation/cold, two frozen replays, moved coordinates, two moved
replays and teardown, with a separately retained complete protocol total.

Eight alternating ABBA/BAAB groups retain **1,792 complete calls per backend**
across both engines. CPU workers share fixed CPU affinity 0. Every sample retains
E/F and the actual iteration/Fock/grid/J-K/policy/gradient work. The fixed gates
are identical semantic work, E <= 2e-9 Eh, F <= 2e-8 Eh/bohr, and every median
candidate/baseline time ratio <= 1.05. These small-system receipts do not qualify
all methods or basis/grid/resource limits and make no general speedup claim.

| Backend | Maximum time ratio | Maximum E difference | Maximum F difference |
| --- | ---: | ---: | ---: |
| CPU | 1.0251511613143593 | 0 | 0 |
| CUDA | 1.0130533877965417 | 1.1368683772161603e-13 | 1.5876189252139739e-13 |

`final-samples.json.xz` contains a JSON map of original sample filenames to their
lossless decoded JSON text and SHA-256. `qualify.py` and `endpoint_worker.py`
reproduce the paired matrix; CUDA requires a finite `srun`. These readable copies
may be formatted; exact executed scripts recover with the control payload below.

Nsight reports identical work: **139,320 kernels**, **149 graph API calls**,
**5,440 explicit dynamic allocations**, **120,317,549-byte simultaneously live
explicit dynamic peak**, and zero final dynamic live bytes. Static storage is
separate at 1,823,689 bytes; opaque driver/vendor memory is excluded. H2D is
6,928 copies / 39,310,944 bytes; D2H 7,975 / 24,445,752; D2D 3,722 / 454,773,440.
`profile-summary.json` and the original CSVs retain these scoped measurements.

## Failed and superseded campaigns

All three unsuccessful CPU matrices are preserved: uncached queries reached
1.159752268299457, fixed CPU affinity 1.174134781324136, and per-Calculator answer
caching alone 1.1758822235658621. The tiny UHF scalar endpoint failed repeatedly.
Standalone timing and GC diagnostics remain diagnostic; they do not establish
a definitive garbage-collector timing cause. The accepted source bounds both
detached per-Calculator answers and shared typed query bindings. Strong library
ownership and locked retirement prevent stale handles; explicit supplied bindings
remain authoritative. No gate, method or property was removed to obtain a pass.

GPU job 14836 failed before candidate execution because the independent
GPU4PySCF M06-2X methane reference did not converge. The unchanged #2859 baseline
reproduced it in 14837. GPU Newton also failed in 14838. Independent CPU PySCF
Newton converged in 14839 with the same model, stock grid, conv_tol=1e-12 and
conv_tol_grad=1e-9; targeted CUDA E/F then passed. The test now records
`reference_solver="gpu-diis+cpu-newton"` when needed. This is independent oracle
recovery only; production CUDA execution remains required. Job 14835 was canceled
while pending with zero runtime, not executed or passed.

Job 14840 passed 48 tests and 1,792 calls on source `4ac102f67`, before shared
binding extraction, with maximum ratio 1.0096753249835795. Its receipts remain
separate and are not relabeled as final-source acceptance.

## Lossless historical recovery

The unchanged 80 MiB repository evidence budget requires keeping some raw
payloads in existing Git history. `history.json` binds each exact Git commit,
path, byte count and SHA-256; restore with `git show COMMIT:PATH` and verify the
checksum. No external archive, Release or tag was created.

- `prequalification-receipts.json.xz`: all unsuccessful CPU populations,
  diagnostic scripts and reference/setup failures, from commit `fe234d89e`.
- `intermediate-gpu-receipts.json.xz`: superseded source/job 14840,
  from commit `f00dcef34`.
- `final-controls.json.xz`: original scripts, final test/control receipts,
  independent references and profiler CSVs, from commit `82fb77ad6`.

Each payload is a JSON map containing lossless UTF-8 receipt text and decoded
SHA-256. Older #2859 raw samples remain recoverable through its own identity
manifest and unchanged source commit. Passing final measurements do not erase
or retrospectively validate any earlier failure.
