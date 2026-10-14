# Decision: bound CuMetal build checkout history

Status: implemented
Date: 2026-10-13

## Problem and decision

CuMetal job [115407741360](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38429139233/job/115407741360)
spent about 13 minutes checking out the repository. Its all-branches/all-tags
fetch started at 16:19:30 UTC, failed at 16:31:59 with RPC curl 56 / early EOF,
retried, and checked out the PR merge at 16:32:28. The 30-minute job then expired
during native compilation before the Apple-GPU tests. Compilation also had cold
cache misses; this change does not claim to solve that separate cost.

Use `fetch-depth: 1` only in `cuda-tests`, retaining the pinned checkout action,
default event ref/SHA, complete tracked tree and `persist-credentials: false`.
Do not substitute the PR head or current master for the event's merge tree.
The independent upload-only Codecov job keeps its existing full-history checkout;
its metadata/reporting behavior is not part of this build optimization.

## History-dependency audit

- CMake/pyproject versions are literal `0.1.0`; no describe/tag-derived version is
  used. CMake's generated-source registrations and `tools/run_codegen.py` consume
  checked-out files and imported local Python dependencies.
- `tools/generate_build_identity.py` hashes the manifest's paths and file bytes.
  Git metadata is deliberately excluded. The compiler package and registered
  generators do not resolve prior GenerativeQC revisions. Reference-acquisition,
  evidence-retention and paired-history benchmark tools are not invoked here.
- The ordinary and scheduled CuMetal runners, their selected tests and shared
  pytest fixture do not need repository ancestry. Optional autotune provenance
  uses only HEAD/status, which work in a depth-one checkout.
- Explicit workflow Git commands and one-shot diagnostics target the separate
  CuMetal repository. Its immutable commit, recursive submodules and source
  status checks remain unchanged, as do toolchain/cache identities.
- This job produces JUnit rather than coverage history. Upload occurs in the
  independent job. No selection, quarantine, numerical/dispatch gate, timeout,
  concurrency, permission, compiler flag or cache behavior changes.

## Evidence and limits

The regression fixture fetches real depth-one Git histories for master, PR merge
and merge-group ref layouts using the recorded event SHA even after the remote
ref moves. It verifies HEAD, clean status, complete tree and one reachable commit,
then executes the production codegen wrapper/build-identity generator. Full and
shallow identities match; either parent tree or changed source bytes differ.
Workflow negative controls reject full history, wrong source selection, sparse
trees and persisted credentials. The old checkout fails the positive contract.

A separate full-repository depth-one fixture at
`0e6dfb453fdcd8802a9310031a9664137c4a7b47` produced byte-identical build-identity,
matrix-product and solver-lowering headers compared with the same full checkout.
CMake configure and the real `generativeqc_build_identity_codegen` target also
passed from that shallow tree with verified ccache 4.14.1 configured as launcher;
the identity target compiles no native code and cache statistics stayed unchanged.
No native library, GPU campaign or hosted speedup was measured locally. Observe
natural hosted checkout and full-job timing separately after publication.

Revisit the depth if an invoked consumer begins needing ancestry. Add an explicit
bounded fetch for that consumer rather than restoring all branch/tag history by
default. The pinned action's [SHA refspec implementation](https://github.com/actions/checkout/blob/3d3c42e5aac5ba805825da76410c181273ba90b1/src/ref-helper.ts)
and [official checkout documentation](https://github.com/actions/checkout#usage)
describe the preserved event selection and depth semantics.
