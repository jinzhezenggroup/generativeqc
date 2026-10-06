# Prepared Direct primitive-pair recurrence

The internal qualification switch
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

The separate default-off switch
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
