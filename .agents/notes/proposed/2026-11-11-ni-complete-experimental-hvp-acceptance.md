# Ni complete experimental HVP: accepted action, production integration still required

Status: proposed
Date: 2026-11-11

## Scope

The previously negative canonical NiH3 doublet protocol now returns one complete
experimental B3LYP/VWN-RPA nuclear HVP and passes the pre-frozen independent
action gates. This is not a full native Hessian, installed public endpoint,
mixed-ECP result, D3 composition or molecular TS acceptance.

The experiment combines the committed signed spin-gap/point-panel response and
one-electron second-integral reuse with the still-scratch directional-first,
first-only and shared-spin lowering in
`2026-11-11-factorized-directional-first-and-shared-spin-rhs.md`. The latter
still intercepts private emitters and rewrites source in an isolated process;
it is evidence for integrating explicit compiler/runtime operations, not an
acceptable production implementation.

## Identity and independent gates

The geometry, native default grid, spherical external def2-SVP, neutral charge,
doublet occupations (16,15), 46 public AOs, Direct/no-DF method and gates remain
unchanged. The gate SHA256 is
`8f15ecb93a8095c07861ad85e24498e670323306a8f5578ae3201d88a96cda95`.
The run records parent `e532f5f9459eb3ab75f69a1ded05c94353d9ebd6` plus the
sealed one-electron runtime source subsequently committed in `6d2fe42dae`.
Keep that original parent/dirty-source provenance; do not restamp it as a clean
later-commit or installed run. The native library SHA256 is
`cca47c31b6031dd9c5d3673847a70094003f5fd1407e6e980df3b7a71adbb49b`.
Every recorded numerical source/script/binary hash still matches at completion.

Native SCF takes 123 iterations with energy `-1509.5720618893429`. An independent
cold state matches with density error `7.872758001070679e-12`, energy error
`2.7284841053187847e-12` and orbital-gradient norm `4.003441143467981e-11`.
Both independent induced-Fock checks pass before the HVP. Independent states
are validation only; they do not seed the production SCF or response.

The collector verifies hashes for all 72 selected independent displaced-state
receipts/arrays, including the separately retained cold retries. It compares
the returned action against each of the three full independent finite-difference
matrices, without averaging, symmetrizing, projecting or selecting a best step:

| Independent step | Maximum matrix-action error |
| --- | --- |
| 0.001 | 1.256752251155291e-7 |
| 0.0005 | 3.237057666360954e-8 |
| 0.0003 | 9.918640075934348e-9 |

All pass the unchanged `1e-5` raw independent action gate. Native action
translation error is `2.0192181260370035e-15`, below `2e-8`. The true coupled
response residual is `2.83622151735528e-12`, below `2e-10`, after 23 solver
iterations. Every declared scientific component is present, their sum matches
the raw action, and displaced production SCF count is zero. One vector does not
establish native bilinear symmetry, full raw matrix/H@v parity or root continuity
for all native columns; those remain required.

## Work and timing boundaries

The complete native-plus-independent-validation process finishes with status 0
in **2777.97 seconds** (46:17.97), peak RSS **3,090,760 KiB**. Its native HVP
assembly timer is **2535.189372 seconds**. The process includes independent
validation and has warmed artifacts from preceding protocols; neither number
is a cold installed-endpoint benchmark or a paired speedup. The 2 GiB grid
workspace is not an aggregate Python/compiler/process-RSS cap.

Returned grid work is 48 actions, all 48 reusing the already built point panel,
with no panel build or fallback during the HVP. Keep solver iterations and grid
actions distinct. First-source counters report 26,199 scalar-direction calls,
52,398 raw-first calls, 1,456,983 primitive/geometry records and 28,151,469
component evaluations across the complete action.

Explicit second-integral boundaries report:

| Source | Seconds | Primitive records | Executions |
| --- | --- | --- | --- |
| T/V | 370.489800 | 10,431 | 2,427 |
| Coulomb | 59.008282 | 485,661 | 26,199 |
| Full-range exchange | 54.938349 | 485,661 | 26,199 |
| Overlap/Pulay | 0.700397 | 591 | 165 |

These do not include each source's complete relaxation/first-integral work and
must not be summed as exclusive endpoint timing. The grid-work proposal
`2026-11-11-composite-grid-hvp-structural-zero-work.md` records independently
verified redundant work; this trial does not uniquely attribute every remaining
second of runtime.

## Receipts and next gates

- `/tmp/gqc-ni-complete-source-reuse-probe/` retains the original source
  provenance, complete receipt and raw component/action arrays.
- `/tmp/gqc-ni-complete-source-reuse-probe.log` and `*-time.txt` retain the full
  process log, explicit source markers, exit status and process peak memory.
- `/tmp/gqc-ni-complete-source-reuse-acceptance.py` and its log reproduce the
  independent checks; `independent-action-acceptance.json` records every input
  digest and explicitly leaves public/full-matrix qualification false.
- An ignored local recovery archive under
  `build/qualification/ni-hvp-prototype-20261111/` retains the exact scripts,
  binary, oracle point files, action arrays and receipts without external
  publication. Its manifest records all member hashes; archive SHA256 is
  `18b76192924bcfd87857d0c352f1eba887b70bc635f390fce3c6893f13c7b1d9`.

Integrate explicit first-only/directional compiler outputs, an admitted native
operation and validated one/two-spin panels before claiming a production
result. Then require complete raw matrix/independent column, H@v, bilinear,
translation, state/lifecycle and installed endpoint receipts. Retain all earlier
failed/time-limited protocols. No issue in #2844--#2848 closes on this experiment.
