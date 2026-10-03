# Decision: explicitly configure local AO maps in composite resident forces

Status: implemented (private opt-in caller; complete endpoint qualification pending)
Date: 2026-10-03

The composite force owner now accepts an explicit positive active-AO cutoff and
host cache allowance. Defaults remain dense. Its existing planner first admits
full AO and geometry scratch; only the remaining enclosing host budget can be
assigned to the shared `ResidentAoMapCache`. No assumed mean AO count reduces
any capacity. A zero or insufficient map allowance selects dense AO execution.

The cache domain includes exact basis/geometry/grid identities, actual native
resident-grid pointer/device, order-2 jets and point-tile boundaries. It is
released alongside geometry owners, rebuilt on allocation/domain changes, and
reset only for work accounting on warm calls. Source tokens remain validated
by the native snapshot owner. Both semilocal and nonlocal consumers receive
the same selected lease; an empty AO map still executes grid/weight responses,
VV10 seed handling and complete point accounting. Snapshot feature seeding and
the bounded two-pass capability fallback are unchanged.

Seventy-nine focused host routing/resource/cache tests passed, including
selected/empty maps reaching both consumers and rejection of mismatched actual
grid leases. All prek hooks passed. These establish caller plumbing only; no
new cutoff or default is scientifically promoted. Complete cold/warm/moved
energy/force measurements and constrained-budget controls remain necessary.

## Complete caller qualification

The source-identified composition on actual master e3f66381a, head
12926aa2d2fd1c27ba98a6de1b86b43e8acb875c, passed all seven complete
independent WB97M-V E/F, changed-geometry rebuild and stale-snapshot cases
with explicit cutoff 1e-16. The same seven cases passed with a zero cache
budget, verifying zero discovery and dense point/AO-square work on every
successful execution. n1 RTX 5090 Slurm job 5568 exited zero; sparse and
zero-budget suites took 178.70 and 177.00 seconds respectively, without skips.

The native library SHA-256 was
fee43f0f019149484ac73dd68a2acacf0a3a696f23fe97fe57bcb6df7cca1e89.
All 1,350 manifest inputs independently reproduced its source identity
26f67b33878cbab454f6e2e12a008669bcbf6cb47e752712047d192c9ee55919.
This composition contains separately developed WB97M integral work, its
persistent-claim barrier repair, and the native explicit SCF map API (unused
by ordinary SCF in this build). It is not a fresh standalone whole-library
claim for this small stacked caller PR. Raw logs, per-call work, source archive,
and build receipts remain ignored under the composition's
`.artifacts/wb97m-active-ao/` directory. Matched 24/48-atom cold plus three-warm
endpoint runs are in progress; no performance advantage or default is promoted.
