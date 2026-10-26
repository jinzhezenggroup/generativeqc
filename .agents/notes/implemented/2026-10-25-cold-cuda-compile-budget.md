# Bound the full CUDA integration compile for source-heavy PRs

The #2789 native CI retry (run 40227770123, job 121297581787) spent its full
30-minute production compile phase doing legitimate CUDA compilation. Ninja
advanced from 583/647 at 22:06:24 to 602/647 at 22:10:18 UTC, eight seconds before
the phase timed out. There was no compiler error; the later missing radial
artifact was a consequence of never reaching that probe. The prerequisite had
completed separately in about five minutes.

The restored 5-GiB cache was full, with 11 cleanups and 138 hits / 523 cacheable
calls (385 misses). A second identical retry does not preserve PR-produced
objects: only trusted master publishes compiler snapshots. Repeating without a
changed input or cache source is therefore not a repair.

Allow 45 minutes for the same full library/runtime integration target, preserving
the 10-minute prerequisite and 10-minute setup/probe/diagnostic allowance with a
finite 65-minute job cap. The extra 15 minutes permits the still-progressing tail
and device linkage; actual CI must establish whether this is sufficient.
Keep every target, CTest/probe, parallelism setting, diagnostic allowlist and
master-only cache trust boundary unchanged. No cache capacity/quota change,
scientific threshold relaxation, benchmark promotion or running-CI cancellation.

Focused budget tests retain failure injection for too-small/unbounded job caps,
missing phase caps, skipped gates, changed target order and unsafe diagnostics.
Host test execution cannot substitute for the full CUDA build or queue checks.
