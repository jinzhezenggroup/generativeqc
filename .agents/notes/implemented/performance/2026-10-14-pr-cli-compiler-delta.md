# Decision: retain one bounded CLI compiler delta per pull request

Status: implemented
Date: 2026-10-14

## Problem and evidence

The CLI qualification restores the shared trusted-master compiler cache, but
previously saved compiler work only on master. Compilations unique to a PR were
lost at the end of each ephemeral runner. Three consecutive #2498 CLI jobs
reported 550 cacheable calls, 295 hits and 255 misses despite restoring different
recent shared cache keys:

- [115693578863](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38515593178/job/115693578863): source fc21a40c, build about 27m28s
- [115702748922](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38516657187/job/115702748922): source 56358655, build about 28m50s
- [115712668166](https://github.com/jinzhezenggroup/generativeqc/actions/runs/38519750708/job/115712668166): source 87ead646, same 295/255 hit/miss counts

The first transition changes two source-identity inputs, the generated build
identity header, and 394 identity fields in the CLI method catalog. Recompiling
those consumers is correct. It must not be described as identical build input.
The second raw-head transition preserves all 1,735 source-identity inputs, the generated
build identity header, the CLI method catalog, production native/compiler sources
and build recipes. Its two changed native test files still require their own test
targets to rebuild. Sixteen complete geometry CUDA emissions (eight methods,
both spin modes, 6,862,042 bytes per revision) were byte-identical across all three
heads; that does not override conservative stationary AOT contract identities.

The actual checkouts were synthetic merges 4c7cde212, 7aa8df751 and
73ef4325d. The third also changes the base from 10140ffa to c8a9531b,
adding a real src/dft/cuda_ks.cpp change and therefore another build-identity
change. Raw-head equality is not equality of the historical compiler inputs.
No claim is made that all 255 misses were avoidable. In that actual second
merge transition, cuda_ks.cpp is the only changed source-identity input; the
catalog, tested geometry emissions and PBE stationary AOT contract are unchanged.
The build-identity header is included only by c_api_tuning.cpp, not the PCH.
This observed change has narrow known native/header fan-out, not a global
source-identity explanation for all misses. Per-TU historical cache keys are
not available, so the recoverable subset is unmeasured.

These observations establish missing same-PR reuse, not the exact identity of
every historical compiler invocation or the amount of recoverable wall time.
The trusted-master artifact fallback remains correct: it only runs on a genuine
Actions cache miss, never replaces a successful prefix restore, and does not
explain these repeated misses after shared-cache hits.

## Decision

Keep the existing trusted local cache and add optional file-backed secondary
storage only during the PR CLI build. ccache's normal strict compiler/source/
header/flag validation is unchanged. `CCACHE_NORESHARE=true` prevents local
trusted-cache hits from duplicating the full snapshot into the secondary cache.
Only misses write through to both local and secondary storage.

Use one immutable key per PR number and CUDA/ccache schema. There is no source,
run, attempt, or timestamp suffix and no restore prefix. A helper validates the
numeric PR against `refs/pull/<number>/merge`. Even a primary-key partial match
is rejected for consumption. Exact hits are read-only; later changed-source
misses stay in the ephemeral local cache. GitHub's merge-ref scoping prevents
other PRs, master and merge-group runs from consuming the delta; literal
`pull_request` guards and a disjoint key/path namespace provide additional
separation. No credentials or permission changes are introduced.

After an actual successful CLI build, trim only the secondary directory to
256 MiB, reject links/special files and verify its nonempty bounded contents.
Only then save it, including when a later unrelated check fails. Restore, trim
and save are optional and timed; they never replace or relax build, link, smoke,
package, scientific or required-CI checks. Master fast-seed handoff and trusted
snapshot publication are unchanged. No cache is cleared.

## Consequences and rejected alternatives

- 256 MiB is a conservative storage ceiling, not a measured optimal working set.
  It is approximately 5% of a 5 GB shared snapshot and may retain only a subset
  of useful misses. Natural CI must establish hit and end-to-end time effects.
- Immutable entries cannot absorb subsequent misses until GitHub expires or
  evicts them. This intentionally avoids one large snapshot for every push or
  run. Do not add run/head suffixes without reconsidering total cache pressure.
- The repository cache quota is still shared. One small entry per PR limits
  growth but cannot guarantee zero eviction pressure on trusted caches. No
  quota increase or explicit cache deletion is part of this change.
- Full per-PR 5 GB snapshots and rotating 1 GB deltas were rejected for storage
  churn. Cross-PR reuse or promotion into master/queue would violate the trust
  boundary. Source-identity weakening and ccache sloppiness are not alternatives.

## Validation and revisit

Focused tests execute the event/ref/key and failure guards, secondary-directory
validation and upload cap. Real tiny C++ compilations exercise local-hit
no-reshare, remote reuse, immutable read-only behavior, changed-source misses,
and preservation of existing master-handoff/header/compiler/flag behavior.
No new full native build, benchmark campaign or GPU qualification is required.
Measure later naturally triggered same-PR builds before making a speedup claim;
revisit the cap or transport only with those results and repository cache usage.

## References

- [ccache 4.14 storage interaction and remote file storage](https://ccache.dev/manual/4.14.html#_storage_interaction)
- [GitHub cache scope, immutability and shared quota](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
