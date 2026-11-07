# Decision: preserve occupied support through the MP2 second pullback stage

Status: implemented, scoped CPU qualification; issue closeout incomplete
Date: 2026-11-06

## Problem

The [compact first-stage decision](2026-10-11-compact-factorized-mp2-first-stage.md)
omitted structurally zero first-stage work, but the second buffer still reserved
the full third-MO-slot domain. Its third-stage consumer multiplied those stored
virtual zeros by finite coefficients. This is a targeted #1626 extension, not
retirement of the five surviving production staging sites or the dense oracle.

## Decision and invariants

When the existing compact first layout is selected, allocate
`di*dj*O*N` second-stage doubles instead of `di*dj*N*N`. All factorized Coulomb,
exchange and correlation contributions have an occupied third MO slot, including
the Coulomb/exchange overlap. Preserve that exact support and the order of all
remaining reductions. The third-stage allocation, local weights, callbacks and
atom scatter are unchanged. Dense weights and the explicit staged layout retain
their full allocation/reduction domains. Keep the conservative admission plan;
an admitted bound is not a measured endpoint peak.

Fusing the second stage into every third-stage consumer was not selected: the
second result is reused across subsequent shells, and a naive fusion repeats
its producer. No complete-endpoint profitability evidence justifies removing
that reuse. No approximation, threshold, new oracle dependency or CUDA claim is
introduced.

## Evidence

Both ordinary and opt-in work-instrumented pullback suites pass 48 tests each.
They execute the real translation unit, compare compact/staged weights and
gradients exactly, use an independent direct tensor oracle, and contract real
PySCF ERI center derivatives for all four slots, repeated atoms and Cartesian
d/spherical d/f shells. Instrumentation remains opt-in; ordinary CI does not
fingerprint production function bodies.

The Cartesian-d/s probe (`case=5`, `N=7`, `O=2`, non-sparse) measures 4,802
third-stage accumulations, versus 16,807 for the explicit staged route. Local
accumulations remain 16,807. Measured simultaneous shell payload is 30,240
bytes versus 40,320 for the prior compact-first-only implementation and 53,040
for the full staged route, and total requested common
payload is 49,864 versus 78,424 bytes. This comparison includes the previously
implemented first-stage compaction; it must not all be attributed to this
second-stage change. The additional second-stage live-payload reduction for
the maximal `di=dj=6` pair is exactly `6*6*7*(7-2)*8 = 10,080` bytes.
Third-stage work changes from `N^5` to `N^4*O` across ordered shell triples;
first/second producer work and callback counts do not change.

Complete H2/LiH/water STO-3G cold and prepared-warm CPU E+F publications match
the original library exactly and independent, same-primitive PySCF 2.14.0
oracles at 1e-9 energy / 2e-6 force gates. PySCF's built-in basis has different
coefficient rounding, so the oracle imports the actual primitive inventory
rather than weakening the same-Hamiltonian gate. The CPU build uses Release,
CUDA disabled, packaged CPU force AOT disabled, and verified sccache 0.16.0;
it is not the default packaged-derivative performance configuration. A pinned
CPU ABBA observation retains complete endpoint timings separately from the
component work/allocation probe. No endpoint speedup is established.
The measured source patch, library identities, individual timings, complete
observables, oracle results and component records are retained in
`benchmarks/results/mp2-factorized-second-20261106/qualification.json`.

## Remaining scope

The source-matched scanner still exposes five common-pullback sites and the
dense orbital-oracle site. The current whole-tree inventory additionally
includes three UMP2 sites, not resurrected RCCSD(T) projections. Do not call
this compaction a six-site fusion/retention decision or close #1626 from an
inventory count. Each remaining disposition still needs its justified
alternative and representative complete-endpoint comparison.

For #1629, the existing source ratchet gains CPU-only replay of retained raw
journals against the independently pinned prepared-capture contract/policy.
It does not add a recorder, create host-transform lineage from equal bytes,
authenticate build identity offline, or grant a zero-round-trip assertion.
Synthetic CI controls cover hot iteration/tile work versus publication,
policy/workload changes, lost records and poisoned summaries. During the initial
CPU-only run, the local Slurm controller exposed only a drained GPU node; no
real-GPU execution was attempted at that stage, and no scheduler visibility
was overridden. That local inventory did not describe the other SSH nodes.

## Remote GPU follow-up

The follow-up uses SSH host `n4` and its own Slurm controller, node `node4`,
partition `main`, `gpu:5090:1`, preserving assigned device visibility. Job 1504
checks actual H2D/D2H/D2D bytes, stream fences and an event wait across explicit
preparation/iteration/tile/publication/observer regions. Job 1505 exercises the
production source-boundary wrappers in a deliberately synthetic helper DSO:
one allowed publication is within budget, while iteration/tile work and an
extra publication fence are rejected. CPU CI replays these retained raw
journals, not their saved diagnostic summaries. Neither helper is a production
scientific endpoint, and neither grants a residency PASS or scientific payload
lineage. The raw records, exact helper/collector sources and independently
frozen source/build provenance are retained in
`benchmarks/results/residency-n4-20261106/`.

Job 1506 subsequently builds on the same frozen source with the actual native
CUDA DSO: all four H2/LiH/water/F-HeH DF compatibility cases pass their committed
independent energy/amplitude fixtures, with intact source/API/activity joins.
This exercises the production CUDA producer and D2H host consumer; subsequent
scientific H2D consumption and payload dependencies remain unknown, not a
qualified #369 round trip. The prepared UHF history passes the original energy
and force gates against both ordinary CUDA and native CPU FP64 histories;
its collection exits 2 (`INCOMPLETE`) as designed. The RHF history fails the
independent moved-water force gate, confirmed in diagnostic job 1507:
`5.604596453601118e-9 > 1e-9`, despite agreement with ordinary CUDA within
`2.8866e-15`. The follow-up retains the failure and raw result comparisons;
it neither relaxes the gate nor claims the underlying numerical cause is fixed.
All observed source/activity/graph-inventory records are intact, but scientific
lineage, full implicit/graph-wait coverage and a matching independently reviewed
prepared-workload ratchet remain unqualified. These observations do not close
either issue or establish a complete-endpoint GPU speedup.

## Reproduce

```console
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/python/test_mp2_derivative_pullback_probe.py
PYTHONPATH=python:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GENERATIVEQC_MP2_WORK_PROBE=1 python -m pytest -q tests/python/test_mp2_derivative_pullback_probe.py
PYTHONPATH=python:. python -m pytest -q tests/python/test_cupti_source_ratchet.py tests/python/test_cupti_source_capture.py tests/python/test_residency_receipt_audit.py
```

Revisit producer fusion only with retained reuse/work evidence and independent
complete-endpoint acceptance; revisit residency acceptance only with actual
source-owned scientific links and qualified graph/implicit-wait coverage.

## Subsequent qualification

The remaining-scope statements above describe the initial campaign. The later
[six-site decision](2026-11-06-mp2-six-site-dispositions.md) supplies all six
measured retention dispositions with matched helper controls, complete endpoints
and simultaneous storage. The
[scientific lineage decision](../architecture/2026-11-06-scientific-df-residency-lineage.md)
supplies actual metric host-transform links and named HF replay-boundary
qualification. Historical failed records are not rewritten; global implicit/
graph-wait coverage remains unqualified even when the declared observed scope
passes its own contract.
