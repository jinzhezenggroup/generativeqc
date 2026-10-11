# Decision: bounded vectorized stationary Cartesian component pages

Status: implemented, opt-in qualification candidate
Date: 2026-10-11

## Problem

The equal-spin XC candidates reduce point-only work, but frozen phase-1
complete B3LYP and M06-2X water E+F endpoints take about 20 seconds. A scheduled
complete warm M06-2X cProfile (Slurm 7463, native binary SHA-256
`fd8c290fc28c210eb344a1fda5c654dbc7753ef1d6dafc758c03958a30a78d07`)
records 21.844 instrumented seconds and 42,961,555 calls. Component-mode
`integral_page` accounts for 20.217 cumulative seconds; its scalar `integral`
producer executes 666,432 times, with 1,066,527 `_append_task` calls. Individual
coefficient `np.prod` calls and derivative-kind encodes each execute over a
million times. This is Python task construction, not dynamic XC spill traffic
or a clean endpoint timing sample.

## Decision

Keep `GENERATIVEQC_STATIONARY_COMPONENT_PACKING=bulk` as an explicit candidate;
an unset selector or `scalar` retains the scalar component producer. Other
values fail closed. Cartesian/noncomponent pages retain their existing bulk
producer. Both bulk producers share one descriptor-capacity/primitive-work
packing routine.

The candidate retains linear AO topology: expansion counts, up to three
Cartesian label IDs and their exact signed FP64 coefficients. Per-operator
kind tables canonicalize each admitted component tuple once, using the actual
artifact's request indices and the existing derivative binding/encoder. They
retain center/axis permutations and the nucleus bit, not local-domain indices
or freshly invented dispatch metadata.

Mixed-radix decoding expands logical pages in task-capacity-sized windows.
The last AO varies fastest, exactly as scalar `itertools.product`. Ragged
rows contribute only real terms: no padded `page * 3**4` descriptor inventory.
Logical and expanded NumPy scratch each remain bounded by task capacity,
including capacities smaller than one d-shell quartet's 81 descriptors.
Coefficients use `charge * np.prod(terms)` in the original order, without
reassociation or pre-multiplying charge.

Host admission conservatively adds `8 * (48 * primitive_tile + 7 * nao)`
numeric bytes for transient scratch and retained topology. It reserves this
space even for scalar packing, so changing the selector after preparing an
owner cannot evade the budget. The existing kind-table reservation covers
the three rank-two operators and rank-four ERIs. Python headers and compiler
objects remain explicitly excluded, as in the existing resource contract.

## Invariants and fallback

- Public AO indices, nucleus/charge/source metadata and coefficient bits stay
  unchanged; Cartesian labels select derivative kinds, not replacement AOs.
- Descriptor order and capacity/work flush boundaries match the scalar
  producer, including partially occupied pages and partial expanded windows.
- Descriptor/primitive counts, native kernels, precision, grids, SCF stopping
  criteria and method/backend admission do not change.
- Excess work, invalid indices and unavailable kinds remain explicit errors;
  there is no CPU derivative or reference fallback.
- Geometry rebind retains only topology-dependent lookup data. No density,
  geometry or descriptor inventory is cached by this optimization.

## Rejected alternatives

- Full Cartesian page products: descriptor bounds do not justify amplified
  host scratch or padded semantic work.
- Unbounded memoized quartet descriptors: hide work and retain molecular
  four-index storage.
- Automatically replacing kernels/providers: native derivative admission is
  a separate ownership/numerics decision. This only speeds the existing
  admitted fallback's Python construction.
- Point-only promotion: the complete endpoint profile contradicts that
  bottleneck assumption.

## Evidence and remaining gates

The focused device-free suite passes 311 tests. Exact descriptor gates cover
every AO tuple in a signed ragged fixture for overlap, kinetic, attraction and
ERIs, capacities 1/7/64, primitive budgets 81/1024 and two charges. Kind
permutations, public indices, coefficient bits and flush boundaries match the
scalar producer, including a partially occupied cross-operator buffer.
Additional gates cover missing requests, invalid indices,
oversize work, scalar fallback, linear topology bytes and expanded-window
bounds. Resource tests retain the unchanged caps and one-byte refusal gates.

Complete source-matched E+F and independent molecular gates remain mandatory
before promotion. The initial pilot uses frozen phase-1 native/compiler inputs
plus this Python change only, with equal-spin XC off in both arms. It does not
measure phase-2 XC binding. Failed setup Slurm 749 (missing manifests in the
isolated Python snapshot) is retained, not a timing observation. Corrected
snapshot inputs link to the frozen source without changing it or its library.
Raw receipts/observations live under ignored
`.artifacts/equal-spin-xc/component-endpoint/`.

The corrected Slurm 750 pilot completes 22 observations each for B3LYP and
M06-2X on spherical def2-SVP water with pilot grid `(32,8,16)`, two co-resident
plans, profile selection off and warm-start updates disabled after each cold
call. Five interleaved pairs per warm/moved cohort give:

| Method | Cohort | Scalar / bulk medians, s | Reduction |
| --- | --- | ---: | ---: |
| B3LYP | warm | 19.6723 / 4.01366 | 79.597% |
| B3LYP | moved | 20.2951 / 4.31379 | 78.745% |
| M06-2X | warm | 20.0215 / 4.04142 | 79.815% |
| M06-2X | moved | 20.1127 / 4.46881 | 77.781% |

Every paired iteration/Fock count and selected stationary semantic work count
matches; both methods retain 1,066,527 descriptors, 5,659,930 primitive records
and 262 primitive pages per endpoint. Every available cold/warm/moved sample
passes the independent matched-grid PySCF 2.14.0 / Libxc 7.0.0 molecular
reference. Maximum energy/force errors are 1.99e-13/2.17e-9 for B3LYP and
7.11e-14/1.61e-9 for M06-2X. This measures Python packing, not phase-2 native
binding or a production-default promotion. Cold observations include shared
persistent artifact-cache history and are not isolated cold ablations. The
r2SCAN-3c cohort, serial sanitizer endpoints and separate WB97M-V control
remain pending at this evidence checkpoint; do not qualify them from these
two completed method cohorts.

## Evidence follow-up

The checkpoint above is historical. A later comparator validates all retained
r2SCAN-3c samples against its complete independent electronic/D4/gCP reference
(maximum energy/force error 1.71e-13 / 1.30e-9); the independent finite-difference
uncertainty is below 1.65e-9. Slurm 752's complete B3LYP/M06-2X memcheck and
initcheck both report zero errors and pass independent molecular gates. These
receipts retain the frozen native-library and component-only Python source
boundary, not the later registry-wide XC sources. The separate WB97M-V control
does not establish a packing speedup, and its original semantic projection
contains missing route-specific fields; no work-count gate is promoted from it.

The [functional-independent optimization note](2026-10-11-functional-independent-xc-optimization.md)
records the subsequent source-bound capacity helper/host-equation updates.

## Revisit when

Complete cold/warm/moved cohorts, semantic counts, independent energy/force
references and larger mixed-shell domains justify promotion, or a separately
qualified native provider removes the component-task fallback.

## References

- [XC binding diagnosis](2026-10-11-native-xc-functional-binding.md)
- [Task pages](2026-09-21-stationary-task-page-schedule.md)
- [Native demand](2026-10-04-stationary-native-root-demand.md)
- [Qualification](../../../../docs/maintainer/xc_equal_spin_qualification.md)
- `python/generativeqc/_stationary_cuda.py`
- `tests/python/test_stationary_component_page.py`
