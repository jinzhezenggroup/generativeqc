# Decision: do not globally downgrade scalar optimization from subset timing

Status: rejected
Date: 2026-11-04

The first bounded Ni trial still exhausts its 1800-second whole-probe limit in
cold one-electron second-source assembly. An inventory of this sentinel needs
218 component/output artifacts across 52 S/T/V source classes, including overlap
for the later Pulay contribution. Bounding each scalar tile is not a bound on
aggregate compilation work.

A local experiment compiled the same first eight f/f nuclear-attraction
components and six HVP outputs through the full native Plan. The emitted source
is 375,727 bytes. Explicit `-O1` took 25.66 s, versus 33.38 s for normal `-O3`;
both independently match the analytic Libcint two-record contraction to
6.44e-14 absolute. Actual artifact flags are retained; the old probe's printed
O1 label is not authoritative for its explicit O3 run. The O1 experiment used
the existing opaque-option combined compile/link fallback, whereas the ordinary
O3 run retains separately cacheable object compilation.

Do not turn this one subset into a global compiler-policy change or an endpoint
speedup claim. It establishes neither the critical high-order subsets nor warm
execution cost, aggregate cold work, cache effectiveness, or a complete Ni HVP.
Production compilation remains O3 and the same finite deadlines remain in force.

Revisit a bounded scalar-optimizer choice only with full endpoint and independent
numerical evidence, explicit flag identity, genuinely cacheable compile phases,
and unchanged finite process ownership. Do not silently override flags through
a subclass or hide them from artifact provenance.

Local recipes/receipts: `/tmp/gqc-2843-2849/probe-ff-second-native-o1-v1.py`,
`probe-ff-second-native-o1.py`, `ff-native-o1.log`, and `ff-native-o3.log`.
