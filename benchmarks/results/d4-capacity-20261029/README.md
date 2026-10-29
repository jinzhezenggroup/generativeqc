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
reference inputs. Complete updated CPU and CUDA budget qualification remains
necessary; the original failed campaigns remain failed.
