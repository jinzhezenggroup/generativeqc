# D4 native capacity qualification — budget performance blocked

Native production source: `16a1c897f49962f100aa7e61b71ad462421a9afb`.
Final GPU test-only source: `6dbc152db1d5da649b56ee3ae3943890871c4ea8`;
`qualification.json` records the authoritative full revision resolved from Git.
The initial GPU job 14848 found an incorrect preferred-cache boundary test.
The repaired test enumerates actual declared alternatives and checks the
mandatory minimum; job 14849 passed 92 GPU Python checks, native D4 tests,
installed SDK checks without Python/compiler in PATH, and sanitizer checks.

The ordinary CPU and CUDA campaigns each contain 1,792 complete calls in eight
ABBA/BAAB groups. Maximum median time ratios are 1.0059381798803984 and
1.0118287490373845, passing the unchanged 1.05 gate. Their exact raw populations
are recoverable with `default-history.json`; ordinary CUDA work/storage match
in `default-profile.json`.

**The new budgeted path is not performance-qualified.** CPU and CUDA each ran
1,792 calls, maximum ratios 1.082475961756885 and 8.628875019676185. CPU E/F match
exactly; CUDA max E/F errors are 4.98e-13 Eh / 1.32e-13 Eh/Bohr. Published SCF
work matches, but actual CUDA kernels rise from 172,628 to 221,476. Explicit
dynamic peak falls from 119,201,475 to 106,994,261 bytes and final live storage
is zero. Different copy counts/bytes are retained in `failed-budget-profile.json`.
These failures must not be reclassified as accepted or hidden by a changed gate.

The independent profiler job 14852 uses the same immutable sources/binaries.
It identifies repeated large-library hashing (10.50 seconds in the candidate
profile) and optional-XC suppression under selected-inventory admission. Those
are separate costs. A later loaded-image hash fix needs its own source-bound
receipt and cannot by itself qualify the native optional-XC path.

CPU timings use matching stricter E=1e-12/D=1e-11 after default UKS closure
failed before timed collection; CUDA uses defaults. Both include construction,
scalar and ragged cold/warm/moved E/E+F endpoints. The baseline has no global D4
protocol, so its budget arm uses ordinary execution; candidate uses 2 GiB caps.
Python D4 correction construction is poisoned in both arms. Standalone CNOH
coverage remains, while stable water supplies the independent complete PySCF
finite-difference and pinned dftd4 reference after incumbent CNOH electronic
closure failed in both ordinary and corrected PBE.

All samples, including failures, are retained through existing Git ancestry.
No external archive, Release, tag or history rewrite is involved. Restore using:

```bash
python tools/restore_retained_evidence.py --all \
  --manifest benchmarks/results/d4-capacity-20261029/failed-budget-history.json \
  --output .artifacts/d4-budget-restored
```

The same command accepts `default-history.json` for the ordinary populations.
Both recovery manifests were verified locally. Exact commands and worker scripts
are retained; adapt immutable paths and execute every CUDA replay/profiler via
finite Slurm allocations, preserving assigned device visibility. Bulk logs/XML,
Nsight databases and failed preflight traces remain ignored, with receipt hashes.

## Loaded-image hashing diagnostic

Frontend `3643d2665` changes only retained library provenance; native source and
binary remain unchanged. Finite Slurm job 14856 compared both budgeted arms over
two ABBA/BAAB groups (448 complete calls). It preserved published work with
max E/F differences 5.69e-14 / 1.74e-14; scalar median ratios range from 0.1750 to
0.2159 and the largest case ratio is 1.01821. Library hashing in the complete
profile falls from 10.50 to 0.60 seconds. Raw samples and command/provenance are
retained as `image-identity-diagnostic*`. This isolates one overhead; it is **not**
a replacement for the original eight-group D4 budget gate, which remains failed.

## Frontend planning follow-ups

`frontend-planning.json.xz` preserves every new sample, source/library identity,
exact scalar replay script and verification receipt. Rebuilt source `9db31ac68`
passed 127 CPU CTests, but its eight-group full CPU budget campaign still failed
with 1,792 calls and maximum ratio 1.0817157850930228 (E/F exactly equal).

`c4ab3d38e` removes redundant scalar planning at the named prepared boundary; its
eight-group scalar diagnostic still failed at 1.055734386993908. `e31c100aa` also
avoids comparing one functional object with itself through duplicate serialized
provenance, retaining full comparison of distinct objects. Its scalar diagnostic
passes at 1.0431432846967614 over 256 complete calls, with identical E/F and
published work. These are scalar-only diagnostics, not complete fleet acceptance.
Native/build/compiler inputs are unchanged from the rebuilt `9db31ac68` image.

The scalar budget-boundary suite passes 69 checks. Model/semantic coverage passes
117 checks plus four post-SCF rejection checks after restoring exact pinned
reference inputs. The complete updated CPU budget campaign at frontend `e31c100aa` then passed
1,792 calls in eight ABBA/BAAB groups: maximum ratio 1.0454224827459262, identical
E/F and published work. It uses the same strict matched CPU controls and native
`9db31ac68` image as the preceding campaign; compiled inputs are unchanged.
All original failed campaigns remain failed.

## Current CUDA qualification

The rebuilt `e31c100aa` CUDA image completed both full campaigns in finite Slurm
job 14867 on node1 with its assigned device visibility. Each campaign contains
1,792 complete calls in eight ABBA/BAAB groups. Default execution passes timing
at maximum ratio 1.009080760957646 and preserves actual GPU work/storage.
Budgeted execution **still fails** timing at 3.4192218547904325. Its maximum
E/F differences are 5.12e-13 Eh / 1.22e-13 Eh/Bohr; published SCF work matches,
but actual kernels remain 172,628 → 221,476 and explicit dynamic peak remains
119,201,475 → 106,994,261 bytes, with zero final live bytes. Frontend planning
repairs do not repair selected-inventory optional-XC suppression.

Native D4 tests, installed independent-reference SDK execution with
`PATH=/nonexistent`, 93 GPU Python tests and memcheck (zero errors/leaks) pass.
The job exits 1 at the changed-work/storage profile gate after preserving both
timing populations and profiles. Since that gate bypassed the shell's normal
final integrity step, a separate metadata-only check verified all 13,618 frozen
inputs after failure. The completed Slurm receipt preserves the failed status.

`current-cuda-{default,budget}-samples.json.xz` retains every sample, and
`current-cuda-qualification.json.xz` records the source and binary identities,
both profile summaries, exact replay source, test/sanitizer/scheduler receipts,
and hashes of ignored bulk logs. These source-bound results supersede the
latest-status claims only; the original failed campaigns remain available.

`current-storage.json` records lossless encoding of three older result JSON
files to make room for the current complete populations. Original Git blobs and
decoded/encoded hashes preserve every value; historical reconstruction patches
and recovery manifests retain their original paths in the recorded Git trees.

## Earlier diagnostic storage

The six intermediate hashing/planning/current-CUDA evidence files named above
are retained byte-for-byte in the existing ancestor recorded by
`pre-optional-history.json`. Restore them with the same offline restoration
command, passing that manifest. Their failed timing/work gates remain failed.
This migration keeps the unchanged aggregate evidence cap while retaining the
next complete optional-XC qualification. No historical Git ancestry is rewritten.

## Rejected s/p-only optional-XC admission

`sp-only-cuda-qualification.json.xz` retains both full 1,792-call campaigns
and profiles for source `9d0df5629`, finite Slurm job 14903. Default timing
passes at 1.01182439, but budget timing fails at 3.47645361: the original
def2-SVP water member contains d shells outside that first promise domain.
Actual kernels are 172,628 versus 216,916; the budget regression is unresolved
at that source. Numerical/lifecycle checks and 138 Python GPU tests pass.
The srun exit is 1 and the final input-integrity check completes; the completed
scheduler record expired before retrieval, and accounting storage is disabled.
The running scheduler receipt and srun failure are retained without inventing
a completed receipt.

## Complete optional MD-J admission

Source `b2e5de7d1` repairs both s/p/d optional XC and MD-J admission. In finite
Slurm job 14909, both eight-group campaigns contain 1,792 complete calls. Default
timing passes at 1.007475284577579; budget timing **still fails** at
1.2613104189433968. Maximum E/F differences are 8.53e-14 / 5.11e-14 for the budget
population. Both profiles now exactly preserve actual GPU work/storage:
172,628 kernels, 5,648 allocations, 119,201,475 explicit peak bytes and zero final
live bytes. All transfer counts/bytes and static storage also match the baseline.

The source-matched Release CPU/CUDA builds, 127 CPU CTests, 138 GPU Python tests,
installed independent-reference SDK and memcheck pass. All 27,409 frozen inputs
were verified before and after the failed timing campaign. Completed scheduler
and srun receipts retain the failure, alongside every raw population and both
profiles in `md-complete-cuda-{default,budget}-samples.json.xz` and
`md-complete-cuda-qualification.json.xz`.

A subsequent cProfile diagnostic identifies redundant Python request generation:
56 of 72 candidate resource-request calls occur during retained replay, consuming
0.3641 seconds under the profiler. Native execution totals are comparable
(candidate 7.46 seconds; baseline 7.53 seconds). This identifies the next repair;
it does not convert the failed complete timing gate into a pass.

The complete MD-J populations/receipts above and the first retained-request
448-call diagnostic are now preserved byte-for-byte in ancestor `8f1c2b3f6`.
`pre-replay-diagnostics-history.json` lists every file, byte count and checksum;
offline restoration was verified before removing the working-tree copies.
Both budget timing gates remain failed. The retained-request diagnostic reaches
1.1008970446134168, with maximum E/F differences 9.95e-14 / 4.66e-14, and all 38
D4 GPU checks pass after repairing two mutation-test fixtures. Its source,
completed scheduler receipt, failures and exact runners remain recoverable.

`optional-cpu-budget-samples.json.xz` retains the earlier source `9d0df5629` CPU
budget qualification: 1,792 complete calls, eight groups, maximum ratio
1.045464979193522, and exactly matching E/F. It is a source-bound CPU receipt;
subsequent CUDA request/diagnostic reuse does not enable caching on CPU paths.

`diagnostic-cache-{samples,receipts}.json.xz` retains the next two-group diagnostic
at `8f1c2b3f6`, Slurm 14915: 448 calls, maximum ratio 1.073302195362344, E/F
errors <=8.53e-14 / 2.45e-14, 82 relevant GPU tests passing. The timing gate
still fails; this is not eight-group acceptance. Cold/scalar preparation remains
the next optimization target after removing repeated diagnostic serialization.

`catalog-carry-diagnostic*.json.xz` retains the subsequent two-group failure at
`5e42386b2`, Slurm 14916: 448 calls, maximum ratio 1.079240692779798, E/F errors
<=8.53e-14 / 3.16e-14 and 82 passing GPU tests. A warmed profile separates native
preparation from frontend overhead and identifies the generic diagnostic copy.

The diagnostic-cache and catalog-carry populations/receipts are retained in
verified ancestor `1e214273b` through `intermediate-replay-history.json`. Their
failed gates remain failed. Every restored byte was verified before migrating
these four intermediate files to make room for complete eight-group results.

## Complete replay/diagnostic qualification

Production `1e214273b`, Slurm 14918, completes both eight-group campaigns with
1,792 calls each. Default timing passes at 1.0130209401642423. Budget timing
**fails** at 1.0586252112560126, confined to three scalar-complete cases; all
batch cold, warm and moved phases pass. Both actual work/storage profiles
exactly match the original baseline. Native/installed independent-reference
checks, memcheck, 150 GPU tests and before/after integrity of 27,418 inputs pass.
Completed scheduler/srun receipts retain the failed status. Every population,
profile and exact runner is retained in `container-copy-cuda-*`.

The complete `1e214273b` populations and receipt are recoverable byte-for-byte
from `container-copy-history.json` (storage ancestor `a677a78c0`). Restoration
was verified before migration. Default remains passing and budget remains failed;
this retention change does not alter either acceptance result.

`resolved-options-scalar-{samples,receipts}.json.xz` records the scoped follow-up
at production `a677a78c0`, Slurm 14919. All 256 scalar-complete invocations in
eight ABBA/BAAB groups pass at maximum ratio 1.0475317675582805, E/F errors
<=8.89e-16 / 6.67e-16. All 112 relevant tests pass and both integrity checks and
completed scheduler receipts succeed. This scalar-only diagnostic is not a
replacement for complete fleet qualification.
