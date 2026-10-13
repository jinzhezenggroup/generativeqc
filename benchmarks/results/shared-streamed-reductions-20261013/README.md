# Shared streamed reduction qualification

This is a default-off compiler slice for Refs #783. The retained status is
`NUMERICAL_PASS_ENDPOINT_RECORDED_QUALIFICATION_INCOMPLETE`; parent issue #783
and complete RCCSD(T)/force qualification remain open.

`delta-records.json.gz` binds the latest execution-key fix to actual H200
compiled artifacts, PTXAS gates, independent NumPy canonical (T) energies,
runtime control pages, invalid-map recovery and all timing samples. Its source
is the recorded dirty `fca37ebf` checkout reconstructed by `source.patch.gz`.
The subsequent integrated upstream changes do not alter the tensor JIT source
inventory or its emitted CUDA dependency closure. Final delivery still requires
the usual source/blob/CI/review and merge checks.

| Synthetic shape | Default full lifecycle | Unfused full lifecycle | Shared full lifecycle |
| --- | ---: | ---: | ---: |
| (4,8) | 52.560 ms | 47.679 ms | 46.793 ms |
| (8,16) | 111.687 ms | 200.884 ms | 122.368 ms |

Full lifecycle includes control/static input preparation, owner creation, all
uploads/runs/downloads and cleanup of one already compiled specialization.
Planning, compilation, retained-owner replay and transfer/run breakdowns are
separate raw fields. The 8x16 shared result is slower than default; neither these
sequential samples nor isolated launch/work improvements justify promotion.
Inputs are synthetic supplied tensors, not an end-to-end SCF/CCSD solve.

`prior-count-records.json.gz` and `prior-count-source.patch.gz` belong to the
earlier bounded source snapshot. Its first-page scalar-body counters are from
separate instrumented binaries, excluded from production timings. The later
key fix changes deduplication identity, not these scientific arithmetic bodies;
the snapshots and their library/generated hashes remain distinct. They show
18,419,712 to 7,507,968 entries at 4x8 and 246,448,128 to 102,531,072 at 8x16.
These are semantic body entries, not SM instructions or DRAM traffic.

Scoped CUDA allocator peaks, successful runtime/driver-only launch API symbols,
compiled register/shared/stack/spill rows and binary/generated-source hashes are
retained. Launch APIs do not enumerate device-side child/graph kernels; error
reset is a separate static-contract operation. Allocator callbacks do not prove
total driver-reserved memory or dynamic register-live high-water marks.

`private-cache.json.gz` retains actual verified sccache0.16/NVCC invocations and
per-invocation foreground-server stats. Combined NVCC `--shared` calls are
reported as non-compilation calls, with zero hits; shared-daemon zero counters
must not be substituted for private-server statistics. No cache is cleared or
weakened. Large compiled/raw archives are ignored local qualification artifacts;
checksums are in `summary.json.gz`, rather than published Release assets.

Reproduction is specified in `publication.json`. Use its recorded base and
reconstruction patch in an isolated supported Linux checkout and an explicitly
allocated, bounded CUDA Job. Preserve unsupported/skipped/negative results.
