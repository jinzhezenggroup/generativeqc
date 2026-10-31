# Native warm-seed snapshot qualification

PR #2868 replaces precision-audit descriptor/NumPy staging with immutable native
host snapshots around the existing HF/KS warm-state representation. Capture
checks the all-slot bound before resident export and publishes no partial handle.
The snapshot survives source batch/context destruction. Restore reuses physical
validation and the target prepared resource owner; it preserves absent slots and
warm-update policy. Python retains same-model identity, resource reservations,
metadata and independent candidate/auditor history selection and scheduling.
Only an absent capture symbol permits descriptor compatibility.

This scoped #2227 R2/R4/R6 cutover excludes R3 and does not complete the parent.
It does not duplicate #2850 projection transactions or #2855 compatibility cleanup.
Numeric receipts exclude metadata, allocator overhead and validation scratch;
there is no whole-process or opaque-provider memory claim.

## Sources and validation

Production source is `f7c7d6ba9ee8983c6ab2e15217d5aca09107f9e4`, compared with
#2865 production `115cc3c72dc443fd50576cc369588d58f694d4d0`. Later commits update
an API declaration-count test and retain evidence. `identities.json.xz` is
lossless JSON binding source IDs, installed CPU/CUDA binary hashes, actual worker
settings and validation scope. Builds use Release/GCC 11.4, CUDA 12.9.86/sm_120
and verified sccache 0.16.0; shared-cache statistics are not exclusive task hits.

Final validation passes 131 CPU CTests, 163 focused Python tests (9 explicit GPU
skips), 93 documentation/API tests plus 104 subtests, Sphinx and repository hooks.
The four new native policy/capacity/snapshot fixtures use always-on checks under
NDEBUG. Fault-injected owners cover cap-before-export, partial failure, output
transactionality, source destruction, concurrent reads, repeated restoration,
failed restoration and exact/below resource admission. Actual changed C API and
snapshot-test translation units pass ASan, fail-fast UBSan and leak checking.
Older-library descriptor/checkpoint/precision compatibility remains tested.

The installed public-header `sdk.cpp` executes RHF/UHF/PBE RKS/UKS, captures at
exact/below limits, destroys source batches, repeatedly restores into fresh
owners, and reads/destroys snapshots after context destruction. Both CPU and
CUDA variants pass with Python/compiler absent from runtime PATH. SDK checks
also remain active with `-O3 -DNDEBUG`.

Slurm **14845** completes with exit **0:0** on node1/main/RTX 5090, using a finite
one-hour allocation and its assigned `CUDA_VISIBLE_DEVICES=0`. All 32 tests pass,
including the independent-reference CUDA precision suite and actual snapshot
lifecycle/CPU-CUDA moved/frozen E/F replay. Qualification poisons descriptor
capture/restore and Python topology reconstruction. No node3 GPU work is used.

## Complete endpoint and profiler gates

The unchanged #2865 protocol (`../seed-capacity-20261029/qualify.py` and
`endpoint_worker.py`) compares source-isolated workers. RHF/UHF/PBE/PBE0 request
energy or energy/forces, with explicit FP64 and an energy TargetAccuracy of
1e-7 Eh forcing independently owned audit suspension/restoration. Scalar calls
include preparation/publication/cleanup. Fleet calls include cold preparation,
two frozen replays, moved coordinates, two moved replays and teardown; complete
protocol totals are retained too. CPU uses STO-3G/GridSpec(12,6,12), affinity 0;
CUDA uses spherical def2-SVP/production Treutler level 3. Ordinary import/model
caches are primed; every measured protocol starts fresh owners and cold densities.

Eight alternating ABBA/BAAB groups retain **1,792 complete calls per backend**.
Semantic Fock/iteration/grid/J-K/policy work, audit counts and candidate/auditor
seed bytes match exactly. Gates remain E <=2e-9 Eh, F <=2e-8 Eh/bohr and every
median candidate/baseline ratio <=1.05. This matrix makes no universal speedup
or broader method/resource qualification claim.

| Backend | Maximum time ratio | Maximum E difference | Maximum F difference |
| --- | ---: | ---: | ---: |
| CPU | 1.0202404723601008 | 0 | 0 |
| CUDA | 1.0253428651089198 | 9.947598300641403e-14 | 6.927791673660977e-14 |

Nsight's complete untimed matrices are identical: 309,867 kernels, 520 graph API
calls, 20,444 explicit dynamic allocations and a 119,110,081-byte simultaneously
live explicit dynamic peak. Both finish at zero dynamic live bytes; static
storage is separately 1,822,825 bytes. H2D is 26,762 copies /161,982,388 bytes;
D2H 29,638 /97,814,148; D2D 8,900 /918,214,144. Opaque provider/driver memory is
excluded. No reduction in scientific work, device copies or peak storage is claimed.
`profile-summary.json` binds the original ignored Nsight databases by SHA-256.

## Failures and exact recovery

The first Release snapshot fixture segfaulted because NDEBUG erased API calls
inside assert, leaving a test sentinel intact. Four fixtures were changed to
always-on checks, their owning PRs updated, and complete Release checks rerun.
Earlier assertion-guarded CTest passes do not establish those checks. Initial
CPU resource-diagnostic expectations also assumed a CUDA scope or an SCF sample
where none exists; final tests require honest CPU observation and actual CUDA
prepared-ledger diagnostics. The API declaration census initially omitted four
new functions; the corrected exact census and all documentation helpers pass.
No numerical, resource or performance gate was relaxed.

Slurm 14844 exited during input integrity preflight because the manifest creation
command used an unavailable remote Python name. It executed no scientific test.
A new immutable directory with the explicit interpreter and complete input
manifest was used for 14845. Both job logs remain distinct; all 6,025 input checks
pass before and after the accepted job. Input/receipt hashes are in identities.

`controls.json.xz` stores lossless original UTF-8 logs, executed scripts, SDK
source and profiler CSVs with per-file SHA-256. `history.json` locates exact raw
CPU samples at commit `46c95408d` and CUDA samples at `0be3a9fcb`, with full Git
IDs, original paths, byte counts and SHA-256. Recover with `git show COMMIT:PATH`.
The unchanged 80 MiB aggregate evidence budget is preserved through exact Git
recovery and lossless compression of prior identity receipts. No Release, tag,
external archive, parent closure or broader resource authority is implied.
