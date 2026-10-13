# Restricted GFN2 mixer readout qualification

Frozen candidate tree `0acc762c69b3dba9acd0d6e249464981c1307963`, based on
`2946eed598c6ea54c0b493c60f695c0efad7fc50`. This is a source-bound diagnostic
qualification on NVIDIA H200, not a provider promotion or parent-issue completion.
The [provenance receipt](provenance.json) records full source/library/executable/
oracle hashes, actual device UUID, compiler/cache/toolkit identities, commands,
canonical exits, runtime dependencies and resource limits. Later evidence/docs
commits do not relabel this whole-library identity as their own build.

All 216 alternating OFF/ON complete energy plus host-force samples passed the
unchanged independent tblite `5e-7` gates. Maximum errors were
`1.963371687452309e-9` Eh and `3.960852935913639e-8` Eh/bohr. SCC work/status and
E/F OFF/ON agreement passed the separate `5e-14` gate. Six raw lifecycle calls
retain first failure, recovery, failed replacement, replay, pre-reset rejection
and recovery after rejection. Expected failed calls retain their failure/sentinel
semantics; they are not scientific successes.

Preflight hash/dependency checks, the default real CUDA bootstrap, the complete
probe and independent reducer exited zero. Actual graph family was 1 (device-tail
graph). Submission count and allocator peak remain unavailable in the snapshot
API. The main and lifecycle journal total is **1,027,036 bytes**, below the shared
64 MiB journal limit; retained row counts and attempted invocation counts remain
separate.

Full endpoint seconds include ON synchronous readback. Warm means retained cache
with fresh SCC; moved-replay uses the nonrigid moved geometry with fresh SCC.
Cold constructor costs are charged separately in the raw/report paired totals.
The process/library caches are shared across cases, so cold rows are not separate
fresh-process starts. Single cold observations and their order effects do not
establish a speedup.

| Intrinsic AOs / geometry | OFF warm median | ON warm median | Median paired ON overhead |
| --- | ---: | ---: | ---: |
| 48 / original | 0.028423473 | 0.028477869 | +0.1914% |
| 48 / moved replay | 0.028400075 | 0.028452552 | +0.2075% |
| 192 / original | 0.085269804 | 0.085247355 | +0.0677% |
| 192 / moved replay | 0.085177284 | 0.085275848 | +0.1605% |

For invocations with full history eight, the median history-stage share of
**that invocation's measured residual/history/solve/combination cycles** was
79.33% at 48 AOs and 86.48% at 192 AOs (original/moved differ by less than 0.01
percentage points). These are statistics of per-CTA ratios, not sums of cycles
across SMs. They exclude unmeasured normalization/commit work and are neither full
mixer nor SCC fractions. **Mixer/SCC fraction: UNAVAILABLE.** No nominal clock-rate
conversion or whole-graph NCU attribution is used.

The seven small fixtures and 48/192-AO original/moved domains were qualified.
The historical independent **768-AO E/F FAIL remains unresolved and was not
rerun** here. Restricted single-system results do not qualify ragged/unrestricted
paths or establish a provider crossover. Refs #1882 and #560; both remain open.

The Job requested fair-scheduling HIGH/4 (10 was rejected by the live CLI), while
the platform reported NORMAL/20. The reason is unconfirmed. Resource was one
H200, 20 CPU/200 GiB, 15-minute maximum runtime, no retry; the live rate was one
point/instance-hour. Actual billed points are unavailable. Program ran from
04:10:54 to 04:11:22 on 2026-10-14 Asia/Shanghai. The task's CPU preparation
instance has its own 180-minute timer ending 05:48:16; no renewal is implied.

Compiler cache snapshots are retained verbatim in provenance. Counter resets/
discontinuities prevent attributing their hit/miss totals to the entire build.
Actual launcher invocation was verified. The build log also reports that the
server shut down unexpectedly and compilation continued locally; uninterrupted
cache service is not established. Cache use does not replace device or
scientific evidence. The failed CRLF export is retained separately in the private
`preparation-negatives.tar.gz` and original bad-export archives; Python 3.10
codegen failure is also retained in the full bundle. Resolution is documented in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-10-14-gfn2-mixer-endpoint-journal.md).

An independent CPU replay found lifecycle metadata type coercion could accept
float call IDs or bool/int substitutions. Both paths now share strict validation;
45 behavior cases and 73 host checks passed (one expected real-device skip).
The unchanged real journals produce the same semantic report. The provenance
receipt records this later consumer repair separately from the original GPU
snapshot; no duplicate GPU run or new library identity is implied.

To independently replay the publication on a CPU host, decompress the four XZ
files, verify stored/decoded hashes from provenance, then run:

```sh
python benchmarks/gfn2_mixer_endpoint.py report \
  /absolute/raw-endpoints.jsonl /absolute/oracle.json /absolute/rechecked.json \
  --repetitions 5 --failures /absolute/raw-lifecycle.jsonl
```

`report.json.xz` is a convenience rendering of this reducer, not an independent
source of truth. `raw-endpoints.jsonl.xz`, `raw-lifecycle.jsonl.xz` and
`oracle.json.xz` retain the complete vectors and executed receipts.
