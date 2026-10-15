# Decision: make guarded VV10 rotating-partner traversal the default

Status: implemented
Date: 2026-10-15

The architecture scope below records the initial promotion. It is superseded by
`2026-10-15-vv10-packet-portable-default.md`; historical measurements remain bound
to their original source and library identities.

## Decision and scope

After reviewing the 96-atom off/on measurements, the user explicitly requested
a PR and default promotion. Use the packet kernel when the existing automatic
parallel admission succeeds: molecular VV10, sm_120, at least 65,536 nominal
grid points, admitted numerical domain and disjoint output leases. Unset and
empty selectors both use this policy. This does not expand architectural,
scientific, precision, memory or small-grid admission.

`GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel` restores the previous four-row
bilateral kernel; `ordered` retains an explicit ordered fallback. The existing
`parallel-wide` experiment remains explicit-only. Numerical rejection still
selects the ordered computation. No scalar pair roots, precision, thresholds,
resource allowances or optional component-reuse identities change in promotion.

This supersedes the default-off decision in
`2026-10-15-vv10-warp-packet.md`, not its historical measurements or limitations.
The reduction order is not bitwise deterministic, as with the existing bilateral
atomic schedule. All lanes, including inactive/padded partners, still participate
in the entire rotation before original-partner publication.

## Evidence and remaining limits

The opt-in implementation passed 752 scheduled admission/oracle cases, both
memcheck and synccheck, native lifecycle contracts, complete independent molecular
E/F and finite-difference tests, and the complete original 3/12-atom populations.
The full fixed-grid VV10 consumer improved by approximately 12.7% at 320k points;
the 12-atom changed-geometry endpoint improved by 4.0% in the retained campaign.
These observations do not imply equal whole-SCF speedups.

The original-limit 96-atom campaign observed cold E/F 1325.684 versus 1277.316 s
with identical 36-iteration/36-Fock/four-SCF-pair-sweep counts. Five-call warm
medians were 59.591 versus 59.689 s. Both arms timed out in `moved/0` under the
unchanged 1800-second whole-population limit after 6/12 calls. All returned calls
passed independent E/F gates, but **the 96-atom endpoint remains unresolved**.
This single fixed-order cold comparison is descriptive, not a repeated causal
speedup estimate. Promotion is an explicitly requested bounded policy decision,
not evidence that the interrupted moved geometry or full 96 population passes.

Historical opt-in archives remain unchanged in
`benchmarks/results/vv10-warp-packet-20261015/` and
`benchmarks/results/vv10-warp-packet-96-20261015/`. Their `default_enabled=false`
describes those exact measured sources; do not rewrite it to describe this PR.

Promotion adds executable independent-oracle coverage for unset/empty selectors
at 65,535/65,536/65,537 nominal points. A sparse physical domain bounds quadratic
work while preserving active zero weights, signed weights and screened responses.
Current source-matched device/endpoint qualification is retained separately.

### Promotion qualification

Rebase onto master `cb8781fb1` and a complete cached Release rebuild produce
native source identity
`9f72f8400dd4c6a2fdc8967a91b855f30c3765db1c9a49c54e7c9bea1fcfcbb4` and
library SHA-256
`6d9c1fd0cb2face995e15ff9881615417c9bf8495880be29f8ec8296f2b1eb80`.
The unscheduled host passes 238 tests (56 optional device cases skip). A fresh
exclusive finite n4 Slurm 955 allocation passes 752 admission/oracle cases,
47 runtime/lifecycle cases, six independent complete molecular E/F/finite-
difference cases, and six automatic-boundary cases each under memcheck and
synccheck with zero errors. With no pair-selector override, the original
3/12-atom protocols pass all 24 complete cold/warm/moved/moved-warm calls under
unchanged grids, reference gates, replay counts and point deadlines.

Raw deployment/build/cache/scheduler receipts, full endpoint results and the
historical verifier are retained in
`benchmarks/results/vv10-packet-default-20261015/`. These are current automatic-
dispatch qualification, not a new repeated off/on performance comparison. The
96-atom timeout remains the historical partial population described above.

## Revisit conditions

Retain the explicit prior/ordered routes for regression isolation. Profile warm
and changed-geometry complete paths before assigning the remaining approximately
60-second warm interval to any unmeasured component. Revisit this bounded default
if representative accepted-domain endpoints regress or independent numerical,
fallback or sanitizer gates fail. Broader hardware admission and completion of
the original-limit 96-atom population require separate evidence.
