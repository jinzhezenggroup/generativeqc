# Decision: strict ccache coverage for the CI native CLI host build

Status: implemented in this change; hosted-run timing qualification pending
Date: 2026-10-12

## Problem and evidence

The native CLI job in master CI run 38191372178, job 114628144419,
at source 531364e133ae145097b402d66fec5036ed051365 reset ccache counters
immediately before its CLI build. Its final report showed 254 cacheable calls,
254 hits, zero misses, and 272 uncacheable calls (526 total). A 100% hit
rate among cacheable calls therefore covered only 48.29% of compiler calls.

The same CLI build log contained 272 CXX compilation tasks for the main
library, including its PCH, plus 248 CUDA and six other CXX compilations.
The CLI preset enables host PCH and the strict ccache launcher does not
supply the PCH sloppiness settings required by ccache 4.14. This identifies
PCH as the strongly supported explanation for the uncached host work;
the original log did not include verbose reason counters, so the exact
reason was not directly observed there.

## Decision

Only the native CLI configure invocation in CI overrides
GENERATIVEQC_ENABLE_CXX_PCH=OFF. This applies to its release and fast tiers.
The ordinary release/developer presets and default CMake behavior are
unchanged. CPU/CUDA sources, numerical options, compiler identity, generated
source provenance and existing compile parallelism remain authoritative.

A non-gating, always-run verbose ccache statistics step immediately follows
the selected CLI build, making uncached reasons visible even when that
build fails. It respects the docs-only skip. Existing smoke, distribution,
aggregate checks and summary reporting remain intact.

## Rejected alternatives

Do not add pch_defines,time_macros sloppiness. The ccache manual explains
that it cannot fully detect macro definition and time-macro changes through
PCH. Weakening those checks would trade scientific build correctness for a
higher cache hit count and violate the repository cache policy.

Do not disable PCH globally or alter all consumer build settings to increase
hits. This repair is limited to a persistent-cache-oriented CI invocation.
Do not clear existing caches, replace compiler identities, or disguise source
and build paths.

## Cache lifecycle and tradeoffs

The workflow is already part of the shared CUDA snapshot key and refresh
inputs. This recipe change can publish a new immutable generation under
the existing trusted-master producer policy; no cache namespace bump or
permission change is needed. Existing valid objects remain available.

The first PCH-free host build can be cold and may take longer than a cold
PCH build. Subsequent unchanged host inputs become eligible for strict
compiler caching. No end-to-end CI speedup is claimed from source inspection
or small fixtures: configure, generation, archive/link, install and packaging
are not compiler-cache hits. Observe normal hosted runs before quantifying
the time saved.

## Validation boundary

Executable workflow fixtures exercise both compiler tiers, configure/build
failure propagation, preset override scope, verbose reason output and
skip admission. Existing cache lifecycle and CLI workflow controls remain
required. A bounded host-only compiler-cache fixture may establish actual
reuse and source/header invalidation; it is not CUDA, GPU or performance
qualification.

## Revisit when

A supported compiler-cache/PCH combination preserves all required input
invalidation without sloppiness, or normal hosted measurements justify a
different scoped build policy.

## References

- [Original CLI job](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38191372178/job/114628144419)
- [ccache 4.14 PCH requirements](https://ccache.dev/manual/4.14.html#_precompiled_headers)
- [Shared cache lifecycle](2026-10-11-shared-linux-compiler-cache-lifecycle.md)
