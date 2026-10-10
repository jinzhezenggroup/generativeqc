# Bounded fresh expanded physical replay, CUDA DF energies only

Production pair: 52.150713116 to 51.149165444 seconds (1.92% shorter).
Replay: 9.163709378 to 8.080924573 seconds. The prototype pair is retained
separately (2.12% shorter); do not pool it with production or old #2211 samples.
Both arms keep 19 primal observations/evaluations and all independent gates:
total energy 1e-8 Eh, triples 1e-10 Eh, physical R1/R2 1e-10.

Physical replay tiles 488 to 31, GEMM calls 13176 to 961, accumulation launches
976 to 31, packing bytes 684660517632 to 515801545216. Exact contraction terms
are unchanged; four scalar contractions become GEMMs. Every complete endpoint
counter matches its independent graph-work prediction. Device capacity stays
8583749632 bytes; bounded host bindings add 113476 bytes. Production process
peak RSS adds 3239936 bytes, not a universal memory claim.

16 determinant-oracle CPU cases, 8 prototype and 8 production CUDA actions,
13 prototype and 13 production matched/fallback solver outputs, carried-sum
overflow checks and both representative tail memchecks are retained. All raw
inputs/outputs, accepted samples, recipes, actual compiler/link commands,
borrowed object hashes, failure diagnostics and hardware/toolchain receipts
are lossless UTF-8/base64 records in raw-receipts.json.xz (stdlib lzma + JSON).
NVML failed before timing; actual CUDA runtime device/driver queries succeeded.
No kernel-only, broad statistical, force/response or release claim is made.

Use tools.generativeqc_validation.record.load_publication_record and the offline
test to verify original errors, work, scope and exact receipt bytes. Apply the
gzip source patch to c820ad08de1ceb33bc19c79115e0af2d52b01784, or rebuild the affected CC source closure using
the accepted #2211 sibling recipe and the retained 21-target commands. The
immutable library base includes RHF #2205/#2206 and #2211; newer master changes
listed in validation.json.gz are not consumed by this endpoint. Source hashes
match every modified production file. Generation needs compiler dependencies
only, not a GPU/runtime/oracle. Run GPU work through finite srun on node2 with
main/gpu:pro6000:1, assigned visibility intact, and reuse ccache. Benchmark
outputs belong in fresh ignored directories, never in this retained bundle.
