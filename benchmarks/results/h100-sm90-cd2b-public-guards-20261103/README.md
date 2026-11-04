# Frozen H100 public API guards and explicit fallbacks

These are actual measurements of native source
`cd2b4514be0a38c4920934481b1f02d157a694b4`, tree
`c76f5bd3cfa62416255c5d6914cf701d34e10a8d`, production core SHA256
`37b8bd02c1dddbf42385c494ae21ff51b50d874dae1dd0c01ab2b75fb0eb194e`.
The source-bound build, molecular DF44 and screening packets, and the supporting
native/reference/Nsight packets are prerequisites already retained in this repo.
Later main-branch compiler arithmetic and frontend/autotuning changes have no
measurement qualification from these frozen artifacts.

Run from the repository root with Python 3.12+ and NumPy 2.2.6:

```sh
uv run --no-env-file --no-project --python 3.12 --with numpy==2.2.6 \
  python -m benchmarks.h100_public_guards
```

This CPU-only command authenticates and extracts exact original packets in an
isolated temporary directory. It recomputes consumed prerequisite proofs and all
new accepted/failed facade proofs; their bytes must match the original proofs.
It never launches a solver, GPU process or compute allocation. Source snapshots
and independent frozen reference helpers come from local Git objects. Those
objects must be available in a non-shallow checkout. The two failed budget
attempts are authenticated and reported as retained failures; their old receivers
remain available, and are not claimed to have passed scientific acceptance.

| Actual attempt | Result and boundary |
| --- | --- |
| Budget5 | Four failed Direct branches, zero returned E/F; four public DF refusals. Original errors retained. |
| Budget7 | Three failed small-Direct warm branches, three retained cold E/F outputs; four DF and one large-Direct host planning refusal. |
| Budget9 | Twelve complete small-Direct E/F, six zero-budget/closed-owner controls, four exact DF planning refusals and one large-Direct host-plan refusal; all five refusals have zero E/F. |
| Original facade18 | Failed original native-only contract: 9/18 populations, 71 public E/F, 39 closed-owner controls, six directions. Three fresh independent reference E/F are retained. |
| Corrected public API contract | All 18 original physical groups represented by 22 actual component populations: 142 public E/F, 28 owners × three closed-owner checks, eight directions, four explicit unsupported-layout refusals. |

Budget9 corrects the observer's treatment of a legitimate warm-binding allocation
counter reset and its string-valued admission mode. It changes no native source,
capacity allowance, scientific input or numerical gate. Device ledger admission
does not establish a physical process-wide hard memory cap. Public DFT DF whole
request plans and the large-Direct host envelope remain unsupported; #2937 tracks
those limitations. Declared host plan quantities are not physical measurements.

The original facade observer intercepted five Fast Direct compatibility candidates
before their actual public calls. The corrected contract records those candidates
and their separate FP64 native audits explicitly. Four mixed-size DF prepare
requests with one incompatible global molecular auxiliary layout remain exact
zero-E/F refusals. Eight explicitly identified singleton owners preserve the same
physical cases, auxiliary primitives, cold/warm/moved controls and numerical gates.
No heterogeneous DF batch or batch isolation/performance qualification follows
from singleton execution. All actual owners receive all three lifetime controls,
so the corrected count is 84 rather than the original 72.

Across the corrected observations, 160 prepared item requests comprise 129
native-force-eligible and 31 compatibility requests. Returned public forces and
native audit results remain separately identified. Neither every Fast trial nor
every public output is promoted to a native-only force path. All full precision
vectors, logs, plans, actual work/capability records, original failures, mappings,
and exact staged producer versions are retained in the five archives.

The notebook archive records a real RUNNING allocation and terminal diagnostic
parent records, rather than inventing a terminal Job status. The first two
diagnostic parents bind staged versions and observed invocation provenance but
did not self-record the complete runner hash at invocation. The singleton parent
records its executed script hashes. Review reports retain this limitation.

The contract-scope and actual-result reports are same-family provisional reviews;
the requested reviewer model is gpt-6.1-sol and actual backend identity is
unverified. Their original full private audit input trees/traces are not bundled.
Public numerical and artifact acceptance comes from raw-data CPU replay, not
review status strings.

These guards provide zero clean timing votes. The original fixed 30-case final
contract remains 180 fresh processes and 720 complete E/F calls, three pairs per
case, at least 20 wins with median speedup >1.05 and all three pairs >1, all 30
scientific gates passing, and complete-set geometric mean >1. Final majority
performance and current-main integration remain separate work.

Historical storage: the original guard archives are recovered byte-exact from already merged Git using `historical-snapshot.json`. Existing cohort bindings, failures and CPU proof outputs remain unchanged. No additional scientific acceptance.
