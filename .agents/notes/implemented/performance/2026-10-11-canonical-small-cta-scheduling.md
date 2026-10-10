# Decision: small canonical CTAs for orders three through five

Status: implemented
Date: 2026-10-11

## Problem

After automatic canonical recurrence reuse through order nine, orders 3/4/5
accounted for more than half of the first-density source time in the retained
12-atom WB97M-V water proxy. Generic promotion into the 256-thread owner had
already regressed order three. The maximum Cartesian domains are only
27/81/162 components, so the large owner pays publication/retirement and
resource costs for many idle lanes.

## Decision

Use one complete 32-thread CTA for each admitted order-3/4/5 shell quartet,
with 1/3/6 independently screened component slots per lane. Preserve the
256-thread owners and 2/3/6/9 slots for orders 6/7/8/9. The compiler emits
these complete s/p/d/f capacity bounds; production and qualification consume
the same definitions.

Generalize the shared consumer's slot stride to its compile-time lane count.
The legacy tile base remains `tile * 256`, and the default lane count remains
256. Thus existing HF and packet callers retain their old ownership. The
canonical owner uses tile zero and consumes the complete domain exactly once.

This is a whole CTA, not a warp-private region inside a larger CTA. Keep
uniform admission, `__syncthreads_or`, publication and retirement barriers.
No new concurrency/lifetime protocol, algebra, global ERI tensor, density
screen, precision policy or environment option is introduced.

Bound the persistent launch by `min(first_count, 4096) *
min(second_count, 4096)`, capped at 4096. Screened rows cannot exceed this
dense shell-pair product. The clamped factors avoid overflow, require no D2H
count or synchronization, and do not change the persistent task coverage.

Prepare the additional order-3/4 prefixes within the same five bounded planes.
Allocation rollback and incumbent MD-J priority are unchanged. Missing cache,
capacity or lease keeps the existing canonical fallback. Source capability
and total angular order select the route, never method/basis/molecule size.

## Rejected alternatives

- Reuse the 256-thread owner for low orders: the new width sweep reproduces
  the old regression. Even with a prototype active-rectangle initializer,
  order-three/four sources take about 2.181/1.912 seconds versus the new
  baseline's 0.991/1.814 seconds.
- Use 32/64/128 threads for orders 3/4/5: this first production prototype
  improves the 56-action source total from 7.423 to 5.172 seconds, but the
  isolated width sweep favors 32 threads for all three orders.
- Clear only the active Hermite rectangle: independent poisoned-workspace
  component qualification passes, but matched 32-thread ABBA source totals
  are 4.840593/4.844134 seconds with full initialization and
  4.839005/4.850040 seconds with active-only initialization. There is no
  reproducible additional win. Retain the original initialization and API;
  do not expand an undefined-spare-capacity contract for noise-level timing.
- Keep a fixed 4096-CTA launch: the first all-32-thread implementation has
  a reproducible small-endpoint regression, 11.731132 to 11.981014 seconds
  (15 Focks in both arms). Its SCF/publication phase grows by about 0.3
  seconds while force time is unchanged. Small shell domains still launch
  thousands of idle CTAs. The dense-product cap removes that provably idle
  launch work without a molecule/size whitelist or additional host wait.
  It does not eliminate the small-endpoint regression: the matched first-density
  source total remains 0.245797 versus 0.266688 seconds, explaining about 0.3
  seconds across 15 Focks. Record that tradeoff rather than claiming universal
  acceleration or attributing it entirely to the grid.

Do not infer that every individual class becomes faster. In the fixed-grid
96-atom diagnostic, order three grows from 134.210102 to 144.350219 seconds,
while the six selected order-3/4/5 actions improve from 608.579418 to
391.283691 seconds overall. The new launch cap leaves these large launch
domains capped at 4096; it is not a remedy for that residual order-three
class/primitive-distribution issue.

The width/initialization sweeps use a diagnostic O2 interposer and omit public
E/F publication. They guide selection, not acceptance. Final acceptance uses
the actual Release production kernel, matched complete endpoints and an
independent physical reference.

## Invariants

- Preserve original component AO Schwarz admission, conservative shell keys,
  bit-preserving bounds transpose and equal-bucket sorted triangles.
- Preserve primitive traversal, individual coefficient multiplication order,
  full Hermite initialization, radial owner and positive independent J/K orbit
  scatter. Never reconstruct coefficients by dividing cached products.
- Cover every low-order shell composition and slot tail. In particular pppp
  needs three 32-lane slots and dppp needs six, not one legacy packet lane.
- Preserve RHF/UHF, full/SR/LR internal sources, public spherical projection,
  and the public ranged-K-only contract. Do not add public ranged J.
- Leave forces, mixed precision, fixed/compensated/resident and combined/paired
  ranged consumers with their existing owners.
- Keep peak storage charged and bounded; do not call an outer-domain count
  ratio a scientific work reduction when its units change from AO to shell
  quartets.

## Evidence

Source-matched qualification and exact numerical/timing/work receipts live in
[`benchmarks/results/wb97mv-canonical-small-cta-20261011/`](../../../../benchmarks/results/wb97mv-canonical-small-cta-20261011/README.md).
The measured upstream is `01eb694578dab5ba96175fbedcad67e1db6bac37`;
the matched baseline includes the pending order-5-through-9 work, not raw
master. That work subsequently merged as #2217; the follow-up integrates on
`2b68d10a9f43175d796500d7853ed90245f24315` with identical changed mathematical
sources. Independent Libcint and native analytic/component oracles, persistent
grid-one sanitizers, complete 3/12-atom E+F and selected 96-atom sources are
separate gates. The water proxy is not an OMol25-distribution sample.

The follow-up was subsequently integrated and qualified on
`dec25f5922f8a63c56ecd9f6f89eb345595f3755`, including a complete twelve-atom
physical gate and unchanged semantic work/force traffic. These integration
samples do not replace the frozen matched throughput arms. A final correction
to the stale dispatch-width comment changes the embedded source identity:
among 489 compiled objects, only `src/api/c_api_tuning.cpp.o` changes; all
scientific objects remain byte-identical. Do not assume comment-only edits
preserve the complete library hash, and do not relabel one library's timing
as another's. The receipt records the object bridge and both identities.
Four native suites and independent complete three/twelve-atom physical gates
are repeated on the final submitted library; semantic work and force traffic
still match the frozen arm.

Complete 96-atom E+F timing and independent physical acceptance remain absent.
GPU4PySCF comparisons use an explicitly full-Fock independent solver policy,
not its stock incremental default. Neither selected-source timing nor the
width sweep establishes an unqualified engine comparison.

## Revisit when

Measure broader hardware/basis distributions or a changed angular capability
before altering the table. Orders 6/7 may warrant a separate measured schedule;
do not extrapolate the 32-thread rule to their larger per-lane slot demands.
Reconsider initialization only with an independently qualified, measurable
complete-endpoint benefit.

## References

- [Current recurrence contract](../../../../docs/developer/direct_pair_recurrence.md)
- [Earlier excluded lower-order schedule](2026-10-10-bounded-canonical-multi-packet-reuse.md)
- [Order-eight/nine qualification](2026-10-11-automatic-canonical-orders-eight-nine.md)
