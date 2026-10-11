# Native cross-size arena qualification

The measured source starts at master
`a70ebfc61644910a35916790febd51aaaffc293e`. Both libraries contain the same
compiler/runtime patch, with CMake cross-size arena selection explicitly OFF
and ON. Serial Lambda scratch is **OFF in both complete-endpoint variants**.
The baseline is the candidate-source equal-product plan, not an independently
built pristine upstream library. Scientific equations, FP64 precision and
operator order are unchanged.

Qualification is source-bound. Reusing this record for another revision, a
different provider/cache choice or combined serial scratch is not valid.
Cross-size CMake generation is default-on after the complete lifecycle comparisons
retained in `evidence.json.gz`; it is not justified by arena slot counts alone.
Serial scratch is not promoted.

## Scope and observations

- Device runs use finite Slurm allocations on n1 and n5, `main`,
  `gpu:5090:1`, without changing assigned device visibility. n3 is not used.
- Each node runs baseline/candidate native numerical, response and unpublished
  failure gates. n1 also runs memcheck and initcheck.
- Complete native RHF/source/CCSD/(T)/Lambda/source/orbital-response E+F calls
  cover cold, same-process warm and moved-warm. Scientific owners are rebuilt
  on every call, without prepared-force replay or response recycling. Atom
  zero moves by `1e-4` Bohr in z, including the auxiliary geometry.
- The small input is **24-AO water**, not 24 atoms. The large input is 230-AO
  ethane with 488 auxiliary functions. Independent PySCF 2.14.0 energy and
  directional finite-difference fixtures retain their existing gates.
- Complete-force comparisons use `atol=1e-10, rtol=0`, not bit-pattern equality:
  even repeated baseline force assembly is not bitwise deterministic. The
  small native Lambda bit-pattern gate remains unchanged.
- Comparisons require unchanged iteration/action/work/packing/transfer counts,
  retained-frontier and audit/replay selection, nonincreasing reported numeric
  requests and a 2% complete-endpoint timing-noise screen. This tolerance is
  not permission to retain a known systematic slowdown.

Matched cold Nsight traces observe requested-live-device peaks of
15,571,039,334 and 15,429,542,246 bytes, with 418 allocations and 337,393 kernel
launches in each variant. Observed memcpy counts/bytes, traced kernel
register/shared/local-memory resource histograms and complete static resource
dumps match. Only anonymous-namespace hashes are normalized in kernel names.
The observed pool-event reserved maximum is 704,643,072 bytes in both variants;
this is not a measurement of all CUDA Graph or opaque provider/context retention.

The existing allocation journal is also retained. Its scope excludes raw CC
CUDA allocations and opaque context/provider memory; the journal, CUPTI trace
and NVML process residency are distinct observations, not interchangeable
definitions of peak. Exact-child-PID NVML sampling runs throughout every
three-call lifecycle, at approximately one-second intervals. Its maxima are
**sampled process residency**, not continuous allocator high-water. Actual HBM
traffic and dynamic spill transactions are not measured; no HBM-traffic or
speedup claim is made.

Combined cross-size/serial-scratch initial-snapshot evidence is retained as
negative promotion evidence. Warm calls change admission, refuse a valuable
invariant frontier and increase packing and Lambda owned requests. Do not
reinterpret faster warm timing as a fixed-work storage-only optimization.
Serial scratch stays default-off.

## Reproduction

Recover the measured code from the full base commit and the `source_patch`
member of the compressed record. The patch predates default promotion; compare
explicit OFF/ON selections rather than relying on a build default. Use verified
ccache and the recorded GCC 14.3.0/CUDA 12.9.86 toolchain, Release, `120-real`,
ordinary shell AOT enabled and unrelated stationary CPU/CUDA force AOT disabled.
Keep the CPU capacity headers and CUDA pointers in the same CMake selection.

The record retains the exact benchmark drivers, Slurm/qualification/profiling
scripts, source patch, library SHA-256 values, endpoint JSONs, process-residency
samples, numerical/work/timing gates, kernel-resource histograms and test logs.
Adapt recorded node-local paths and runtime libraries to the reproduction host;
n5 uses pinned libstdc++/libgcc and a benchmark linked for its older glibc.
Never transfer the newer-glibc n1 executable to n5. Preserve the SONAME symlink
`libgenerativeqc.so.0 -> libgenerativeqc.so`.

Large `.nsys-rep`, SQLite and full static-resource dumps remain in ignored
`.cache/memory-planning/profile-master-arena/`, with hashes retained in the
record. No release, external archive or publishing workflow is created.

See `.agents/notes/implemented/performance/2026-10-11-memory-aware-native-arena-candidates.md`
for the lifetime/capacity proof, rejected alternatives and promotion boundary.
