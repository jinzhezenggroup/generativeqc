# Decision: keep native Python probe cache identities stable and persistent

Status: implemented
Date: 2026-10-04
Agent: dot

## Problem

Standalone Python test probes need separate compile and link invocations for
ccache reuse. Generated inputs also live below changing pytest session and
worker directories, outside the checkout. Setting CCACHE_BASEDIR to the checkout
does not normalize those inputs. Setting it to the pytest root alone is also
insufficient: ccache rewrites paths relative to the compiler's working directory,
which was still the checkout.

The common Python CI library cache is saved before pytest starts. Probe objects
created later therefore were not persisted, even when post-test statistics
showed successful cacheable compilations.

## Decision

Use the worker's pytest temporary root for both compile cwd and CCACHE_BASEDIR.
Make caller-relative source, output, and include-directory arguments absolute
without following symlinks before changing cwd. Preserve that spelling for
quoted-header lookup. Make discovered compiler/cache executable paths absolute
without resolving symlinks, preserving driver names such as clang++. Link
separately in the caller's original working directory.

Keep the immediate shared library cache save. Restore a separate probe cache per
Python shard before pytest, cap it at 256M, and save it only after job success
using the pinned actions/cache action's post step. Key it by shard and source
commit, with a same-shard fallback. Reset only its statistics after restoration,
so the post-test report describes the current job.

Optional native probes retain their prerequisite skip. The split-hybrid admission
gate uses a required fixture and fails if the compiler or ccache is missing.

## Invariants and rejected alternatives

- Source bytes, included headers, compiler identity and flags remain normal
  ccache inputs. Do not add sloppiness or weaken identity checks to create hits.
- Compile and link errors propagate; an existing executable cannot hide them.
- A second save under the already-saved common library key cannot persist new
  probe objects because GitHub cache entries are immutable. Keeping probe
  storage separate also avoids duplicating the large library cache per shard.
- No required test selection, numerical assertion, scientific tolerance or CI
  timeout changes as part of caching.

## Evidence and limits

With ccache 4.14, equivalent generated probes in different session/worker
directories missed under the old cwd/base combinations and hit with both rooted
at the worker temporary directory. Behavioral regressions also cover changed
source/header/flags, relative include and executable paths, prerequisite failure
versus skip, and compile/link failures. Workflow gates cover cache separation,
shard ownership, ordering, bounds and reporting. These establish cache behavior;
they make no end-to-end speedup or scientific qualification claim.

## Revisit when

Revisit if probe storage exceeds the cap, cache statistics show persistent misses
after compiler/input changes have been excluded, or new callers need other
working-directory-sensitive compiler options.

## References

- PR #1851
- tests/python/test_native_cxx_cache.py
- tests/python/test_ci_native_probe_cache.py
- https://ccache.dev/manual/4.14.html#config_base_dir
- https://github.com/actions/cache/blob/55cc8345863c7cc4c66a329aec7e433d2d1c52a9/action.yml
