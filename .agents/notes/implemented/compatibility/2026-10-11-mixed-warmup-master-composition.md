# Decision: compose mixed warmup with native arena and serial scratch owners

Status: implemented
Date: 2026-10-11

## Problem

PR #2278's reviewed `cfd12b89d19670cc8607c64fd7004998c786e49e`
intersects master `34dd51ed3ebc5b03b17f095de2cd7b39903923c8` in eleven files.
The branches independently extended the native Lambda owner, the generic symbolic
arena, benchmark positional controls and a test probe's bit mask. Subsequent
master changes through `4a78a93d72b10bd2eda318a145b3b2f9aa9d1540` compose without
changing these owners. Selecting either entire side would discard valid work or
silently alias unrelated selectors.

## Decision and compatibility

Keep master's benchmark argument 28 for serial scratch and 29 for lifecycle;
append CCSD and Lambda mixed warmup at 30 and 31. The native force function keeps
master's serial-scratch argument immediately after its reference-experiment
pointer, followed by the two separate mixed selectors. All three options remain
default-off. The test adapter preserves serial scratch at bit 4096 and moves the
three branch-only mixed controls to 8192, 16384 and 32768. Historical benchmark
runners remain byte-for-byte source-bound and must run on their recorded source;
the current command-line layout is documented separately.

The compiler preserves master's symbolic-arena v2 capacity groups and shared
lifetime analysis together with the reviewed uniform FP32/FP64 storage parameter.
Cross-size selection continues to cover matching capacity queries and emitted
pointers; retained frontiers keep their exclusive storage. No equation, order of
arithmetic, logical scientific hash, or FP32 qualification tag changes.

## Scratch ownership

PR #2278 intentionally shares the matrix and audit/primal-replay scratch lease
by default, independently of either mixed option. This is its previously
reviewed narrower storage change, not promotion of master's broader all-stage
serial-scratch experiment. With serial scratch off, expanded/scalar storage
remains separate and the matrix lease is the final allocation before the
optional retained-core suffix. Audit admission can safely enlarge that last
lease. With serial scratch on, the existing capacity binder places the full
maximum scratch in an aligned prefix and shifts every persistent suffix pointer.

The mixed input tail remains bounded by the conservative maximum matrix/audit
capacity, even if the larger serial arena or total allocation could fit more.
Its lower bound includes mixed auxiliary, ordinary core, retained-core dynamic
and tau-action scratch high-water marks. Inputs/cuts/seeds/frontiers never alias
this tail. Finishing the single-use epoch is still mandatory before exact replay,
independent audits or parameter/factor/force VJPs. Refusal restores bounded strict
execution; numerical failures are not converted into retries or publication.

The PR's expanded Lambda default-off audit now also discovers master's existing
serial-scratch flag. Register it as `needs-qualification` without changing its
default, eligibility or scientific semantics.

## Validation and limits

Production-extracted host probes cover the actual planner, persistent pointer
bindings and mixed admission under both scratch/precision selectors, audit on/off,
ordered allocation rollback and nontrivial 256-byte alignment tails. They verify
allocation/capacity equality, disjoint simultaneously live persistent slots,
scratch separation and input-tail canaries, including a retained-core dynamic
high-water mark larger than the ordinary core's. Optional descriptor and storage
refusals stay bounded; a synthetic numerical failure must propagate. Every
benchmark selector combination, malformed token, argument bound and probe-mask
combination is independently exercised. Existing strict solver/admission,
Lambda lifetime, arena/code-generation and retry regressions remain authoritative.

These host checks are not real-device numerical or performance qualification.
No new GPU run, full build, benchmark campaign or cache clearing is performed.
Original source/binary hashes and all historical receipts are retained unchanged;
they do not qualify this integrated source or a new default promotion. Enforced
CI and the ordinary merge queue still decide merge eligibility.
