# Decision: extend automatic canonical reuse to bounded multi-packet orders

Status: implemented
Date: 2026-10-10

## Problem

The [order-five admission](2026-10-10-automatic-canonical-pair-reuse.md) removes
repeated primitive geometry and recurrence work without changing the canonical
AO predicate. Its single-packet bound does not cover every order-six/seven
quartet. Silently applying that consumer to those orders would omit components.

On fetched master `7f342546d887796e6a92a005ba033029ed73ce2a`, completed CUDA-event
source actions at the first 12-atom density total 20.708956 seconds. Orders six
and seven account for 5.162295 and 5.183564 seconds, respectively, across full
and LR actions: about half the measured source time, not half the endpoint.

## Decision

Admit canonical orders five through seven automatically by the same immutable
primitive-cache capability and remaining budget as the existing consumer. Do
not add an option, environment selector, basis name or molecule-size branch.
Order five owns one packet; order six owns two register slots per lane and
order seven owns three. Under the explicit s/p/d/f bound, the maximum component
counts are 162/324/648. Repeated physical pairs and partial tails reduce the
live domain but never weaken complete ownership. All slots share the incumbent
compiler-owned recurrence and publication/retirement barriers.

Four bounded prefix planes cover the supported angular blocks. Each
first/second pair-order combination has a unique total order, so the same plane
can hold disjoint first-order segments. Equal angular buckets own the sorted
symmetry-unique triangle. The impossible pair order seven is excluded before
indexing the eight-entry offsets table.

Budget accounting includes the fourth prefix plane, keys, bounds transpose and
sort/scan scratch. Established MD-J, geometry and derivative owners retain their
priority and lifetime. Allocation denial rolls back only the optional index.
The allocation fault fixture now records the end of derivative metadata:
denying a later optional index must retain already complete metadata, not demand
that it disappear because metadata used to be the last allocation stage.

## Scientific invariants

- Preserve every original component-level AO Schwarz test and positive J/K
  scatter. Shell maxima are conservative outer keys, not substitute gates.
- Preserve primitive ordering, raw coefficient multiplication, full/SR/LR
  identity, spin conventions and public Cartesian/spherical projection.
- Use the existing recurrence; do not introduce a second evaluator or alter
  compiler mathematics/generated source identity for the shared owner.
- Keep fixed screening, compensation, resident ERIs, combined full-J/range-K,
  paired RSH, other angular orders, mixed precision and forces with their owners.
- Missing cache/budget, allocation failure and unsupported resources retain the
  incumbent bounded consumer. Execution neither allocates nor rereads policy.

## Rejected alternatives

- Automatically promote orders three/four too. A numerical-gated prototype's
  first-density source times rise from 0.972357 to 2.220674 seconds at order
  three and 1.796859 to 1.921670 seconds at order four. These small component
  domains do not amortize this serial shared-preparation/256-lane schedule.
  Retain the incumbent route rather than accepting a small-system regression.
- Promote orders eight through twelve without new resource qualification. Their
  larger component domains need more register slots and independent endpoint,
  recurrence-count and register/local-memory evidence.
- Set the legacy dense HF materialization flag by default. That is a distinct
  traversal with retained negative evidence, not this indexed canonical route.
- Relax the 12-atom reference convergence gates. Stock incremental GPU4PySCF
  did not converge in 100 cycles; rebuilding full Focks converges at the same
  thresholds and provides an independent physical E/F reference. Report that
  solver policy explicitly, not as an unchanged stock-reference speed result.

## Evidence and scope

Source snapshots, build recipes, compiler-cache statistics, profile logs,
independent reference journals and qualification results are retained under
`/data/jzzeng/wb97m-canonical-orders-20261010-7f342546d` and the ignored local
`.artifacts/canonical-orders-20261010` bundle. Rejected prototype profiles remain
separate from qualification of the selected default.

Matched complete E+F on one RTX 5090 allocation falls from 532.670114 to
347.357041 seconds for 12 atoms (one pair, 21 iterations/Focks in both arms).
The independent full-Fock reference accepts both arms; candidate errors are
5.287e-12 Eh and 5.271e-11 Eh/Bohr. Force time remains 38.877125/38.684722
seconds, so the gain is in SCF, not a different force cache or cheaper domain.
Three alternating 3-atom repeats give medians 12.742789/11.956485 seconds,
with all independent gates passing and 15 iterations/Focks in every sample.
Separate allocation-first-use samples are not pooled into those medians.

All 56 first-density source actions on that allocation total
20.838681/12.021160 seconds. Orders six/seven fall from 5.189928/5.201477 to
0.917266/0.657018 seconds; retained orders three/four remain essentially
unchanged. Orders eight/nine now total 5.113911 seconds, about 42.5% of the
candidate source time. This does not turn source timing into an endpoint
measurement, but gives a concrete next profiling target.

The [source-matched qualification receipt](../../../../benchmarks/results/wb97mv-canonical-multipacket-20261010/README.md)
owns the complete measurements, binary/source identities and retained raw
artifact checksums. The native endpoint remains 1.92 times the full-Fock
GPU4PySCF reference's 181.018748 seconds; do not call it a stock-default ratio.

Matched selected 96-atom full/LR order-six/seven source actions total
638.517805/251.360523 seconds (2.54x). Six requested J/K matrices remain
finite and agree within 3.320e-14. The scalar ABI additionally passes an
unrequested J buffer during K-only actions: do not require a candidate dump
for that buffer or treat it as a published J source. The complete 96-atom
endpoint and independent physical gate remain unavailable.

The fixture is the README water proxy with complete spherical def2-TZVPD and
strict FP64 exact Direct WB97M-V, not the OMol25 molecular distribution. A
selected-source 96-atom capture deliberately omits other sources and exits
before publishing E/F. It cannot establish a complete 96-atom speed ratio or
scientific acceptance. Full-Fock GPU4PySCF is an independently initialized
reference, never a production density/derivative consumer.

## Revisit when

Profile the remaining source orders, especially eight/nine, and the integral
force endpoint. Revisit lower orders only with a smaller cooperative
schedule; revisit higher orders with explicit complete packet ownership and
resource gates. Require independent E/F acceptance, semantic work counts and
complete endpoint timing before promoting another default.
