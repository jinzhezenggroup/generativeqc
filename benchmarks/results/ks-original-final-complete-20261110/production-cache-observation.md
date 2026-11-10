# Non-fast-compile build cache observation

The 2026-11-10 build selects verified sccache 0.16.0 for CXX and CUDA. Actual
generated compiler commands are retained in the full scientific record; the
structured build summary retains the launcher version and before/after statistics.
Raw compiler warnings remain in the ignored local build log rather than being
published as run debris. No cache is cleared, disabled,
given sloppiness settings, or assigned a false source/build identity.

At 22:55 CST, an inspection reports 22 compile requests, 74 executed stage
requests, 19 hits and 39 misses. Piping that client's statistics through `head`
causes a stdout broken-pipe panic in the inspection client; it is not a compiler
failure. Later inspections write complete statistics to files instead.

At 23:13 CST, a new statistics response contains zero counters and the original
server PID1830222 is no longer present. The existing launcher clients remain
alive and ordinary NVCC child compilations are observed. This is a cache-server
reset/fallback observation, not evidence of OOM, an actor, or a scientific
failure. This task does not stop any shared server or unrelated compilation.
Counters across this discontinuity must not be presented as a complete build's
hit rate. Compilation and endpoint wall times are not performance promotions.

The native source inventory is independently checked against the clean 926
archive: all 2127 input files match byte-for-byte, producing native source
identity baf0bc3589f99d2caa332ba48e36a6a90425cf8f209ecf2422dfa3896876a563.
Actual linked-library bytes and runtime release/fast-compile flags must still
be verified separately before any scientific receipt is accepted.

The build completes successfully at 2026-11-11 00:17:40 CST. Final shared-server
statistics are retained in the build log and a separate complete statistics
file; they include other requests and the observed counter reset, not a
task-isolated cache hit rate. The linked library SHA256 is
9a50430e953fc7322edde32eaf63ec414a22a7b072b17eb2c9e0ba99325d56e8.
Its native source identity matches the complete 2127-input inventory. Runtime
build flags remain independently checked in each allocated GPU job.
