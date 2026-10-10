# Prepared Direct primitive-pair recurrence

## Automatic canonical J/K values

Canonical Cartesian Direct sources automatically reuse the prepared primitive-pair
cache for total-angular-order-five shell quartets, including spherical public
projections. This route is independent of the legacy HF value switch below:
neither an unset switch nor `GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=0`
disables its admission. It applies by source capability, not by method, basis
name, molecule size, or a WB97M-V-specific selector.

Shell-pair maxima of the original AO Schwarz bounds provide conservative keys.
Angular/system sorting and bounded row prefixes skip rejected shell quartets;
each remaining component still uses the original AO predicate. The column-major
view borrowed by the generated HF helper is a bit-preserving transpose of the
canonical row-major bounds, not a recomputed bound or a symmetry assumption.
Full J/K, J-only, and standalone full/SR/LR K publish separate positive sources
through the existing compiler-owned recurrence and orbit scatter.

Admission borrows an existing immutable geometry cache and charges the complete
index, sort/scan workspace and bounds view to the provider budget, including the
preparation peak. Missing storage, allocation failure, or index capacity keeps
the incumbent canonical consumer. Execution allocates nothing and does not
reread environment controls. Other angular orders, fixed screening,
compensation, resident-ERI replay, combined full-J/range-K and paired RSH actions
retain their existing consumers. Forces and mixed precision are unchanged.

The [default-admission decision](../../.agents/notes/implemented/performance/2026-10-10-automatic-canonical-pair-reuse.md)
records qualification and the intentionally retained fallbacks.

## Legacy HF and dddd qualification

The internal HF/dddd qualification switch
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=1` is frozen when a CUDA Direct
batch/provider is prepared. It selects strict FP64 full-range value consumers
that share cached primitive-pair geometry, Hermite preparation and a Coulomb
simplex across the components of an admitted shell quartet. It is default-off.

The HF angular launcher uses the first queued packet of each admitted order
5--12 shell quartet to consume the complete component domain. Queue compaction
admits all packets together on the same precision/source route. The DFT native
dddd stream uses the same generated consumer for its positive J-only and K-only
outputs. Public spherical densities and matrices retain their existing
Cartesian projection owners.

Each component retains its exact AO Schwarz predicate. A complete shell with
no admitted components performs no recurrence. Shared preparation publishes
before readers consume it, and all readers retire before the next pair product
or shell replaces the workspace. The shared workspace is bounded below 48 KiB;
component values use finite per-lane register slots, with no global ERI tensor.

The candidate preserves the authoritative recurrence and scalar contraction
formulas, primitive traversal, individual coefficient multiplication order and
axis Gaussian arithmetic. It does not recover individual primitive coefficients
by dividing cached products, because zero/underflow can erase those inputs.

Mixed precision, missing primitive-pair storage, the separately qualified
reachable-Coulomb/Hermite-convolution value schedules and SR/LR operators retain
their existing owners. The value switch does not select derivatives or establish
an endpoint speedup.

The separate preparation selector
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_DERIVATIVES=1` is also frozen at preparation.
It selects dddd tasks in the stationary DFT `Full` (Combined) and
`FullSources` (Separate) bounded scheduler,
including its angular-partition qualification mode. The existing FP64 `Dual3`
Hermite response algebra borrows one scalar order-9 Coulomb simplex per
primitive-pair product for all six component packets and independent atoms.
`Separate` retains J'/K' density channels; `Combined` precontracts the caller's
signed J/K coefficients into one density weight before consuming the derivative. The existing Coulomb IR defines its spatial responses
as the next Cartesian states; a view exposes those responses to the same
contraction body. Bra Hermite responses survive all ket products. Distinct atom
seeds collapse repeated shell centers; only N-1 atoms are differentiated and
translation recovers the last. Each component retains its Schwarz and
independent source-density gates.

The derivative workspace is below 32 KiB, uses no global derivative tensor and
has a separate kernel specialization so the disabled path retains its resource
footprint. Missing primitive-pair data/offsets, reachable/convolution derivative
schedules, non-dddd shells and other radial operators use existing consumers.
The value and derivative switches are independently selectable.

`GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES=1` selects a separate
force schedule, frozen at preparation. With angular force partitioning and a
proved s/p/d basis, order-seven tasks use a dedicated CTA consumer. The
lowering also supports the three order-six classes for native qualification;
production order-six tasks retain the incumbent consumer. These classes share
cached geometry, axis-parallel Hermite responses and a
compiler-generated level schedule for the existing degree-eight Coulomb DAG.
Each lane holds at most three weights per source and contracts them directly
into atom gradients; a CTA reduction limits global force atomics to at most
three per unique atom per source. The omitted atom follows from translation.

The ordinary bounded full-range force route also uses that order-seven consumer
when the prepared maximum shell angular momentum is exactly two and the
materialized derivative cache/recurrence and cooperative selectors admit it,
and the caller lends the same plan's immutable class-major shell-pair topology.
The original shell domain is partitioned into three sequential owners:
128-lane generic orders zero through six, 256-lane class-major cooperative order seven,
and 256-lane materialized dddd. All three retain the original class masks,
canonical physical orientation and exact scientific screening. The same cursor
is reset on the caller's stream between owners; no new retained queue or pair
cache is allocated. The order-seven producer claims pages of at most 32 dp kets
per dd bra in the same physical system, including partial tails. Under the
proved s/p/d bound, dd x dp is exactly the order-seven domain; it does not scan
the complete shell-pair triangle again. Ordered indices are canonicalized by
physical pair ID before the unchanged Force predicate and generated-class
exclusions. Force-owner density bounds are passed explicitly, never borrowed
from the Coulomb-owner bounds stored in the topology. Submission/reset failures
stop before subsequent owners.
Missing coherent class-major topology or explicit cooperative disable (`0` or
`none`) keeps the previous two-owner
generic-plus-dddd control. These native selectors are enabled when unset.

The cooperative workspace is bounded below 44 KiB. Its kernel omits the generic
per-AO recurrence frame rather than carrying it in an unused runtime branch.
Mixed f bases, an unproved angular bound, missing pair caches, custom bounded
launch shapes and reachable/convolution derivative selections retain existing
consumers. The angular route keeps independent materialized dddd and cooperative
order-seven selections; the bounded split requires the materialized cache guard
as well. The weighted summation order changes, so independent force oracles
and complete endpoint measurements remain prerequisites for default promotion.

`tests/python/test_direct_pair_materialized_cuda.py` emits and checks the
generated consumer on a GPU. Set
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_CUDA_TEST=1` inside a finite Slurm job
with `--partition=main --gres=gpu:5090:1`. To check the actual DFT native launcher
and its fallback work counts, also set
`GENERATIVEQC_PAIR_MATERIALIZED_DFT_STREAM_OBJECT` to its compiled
`direct_bounded_dddd.cu.o`. Compilation uses ccache and retains commands and
before/after statistics in the pytest temporary directory. The same executable's
`--derivatives` case checks raw component derivatives, independent host source
and signed combined contractions, a one-channel output canary, repeated atoms,
same-pair domains and exact preparation counts.
`tests/python/test_materialized_coulomb_response.py` checks every degree-0--8
spatial response against differentiation of the authoritative Coulomb IR with
its existing Boys leaf rule.

`tests/python/test_cuda_hybrid_snapshot.py::test_separate_full_range_derivatives_match_libcint`,
enabled by `GENERATIVEQC_RESOURCE_CUDA_TEST=1` inside Slurm, exercises both
derivative selections with one- and two-oxygen Cartesian/spherical systems and
independent Libcint J'/K' responses. Two oxygens supply nonzero dddd derivatives.

`tests/python/test_direct_pair_materialized_runtime.py`, enabled with
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_RUNTIME_TEST=1` in the same Slurm
environment, compares the prepared public Cartesian/spherical RHF/UHF J/K
outputs with independently normalized Libcint ERIs. This oracle is confined to
qualification; production does not import or execute it. Complete energy/force
endpoints and actual work counts remain the promotion gates described in
[`performance_engineering.md`](../maintainer/performance_engineering.md).
