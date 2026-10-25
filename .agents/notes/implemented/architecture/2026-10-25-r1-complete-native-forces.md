# Decision: source-complete native forces for the named KS catalog

Status: implemented
Date: 2026-10-25

## Problem

The native energy catalog in #2782 still left CUDA DF, several spin/model
combinations, ECP and imported CPU point programs behind Python force consumers.
Fixture-sized admission envelopes also rejected valid calculations despite
complete derivative inventories and sufficient storage. This implements the
force slice of #2227 for those energy-admitted programs; it does not close the
resource, policy, response or retirement work in the parent tracker.

This supersedes the force exclusions and historical size envelopes in
[native meta forces](2026-10-25-native-meta-forces.md). The public contract now
lives in [the native SDK guide](../../../../docs/user/native_cpp.md).

## Decision

Keep the prepared KS owner as the authority for the actual model, spin,
approximation, angular inventory, precision, final-state token and capacity.
Do not add a blanket method force flag. Python validates its model identity,
asks each prepared native context for its properties and copies native outputs.
A conservative constructor-level capability cannot veto an admitted context.

CPU LDA, B3LYP UKS, PBE50 and imported first-order point programs reuse the
existing moving-grid rho/gradient/tau derivative traversal. No new functional
formula is introduced. Direct ECP adds the same convergence-checked residual
operator derivative used by the energy provider, and nuclear attraction and
repulsion use ionic charges.

CUDA DF submits resident final densities to the existing three-center/metric
response algebra. RKS borrows one matrix; UKS packs its two device matrices on
the provider stream. Compact J/K derivatives enter the same device source slots
as Direct. The standalone one-electron bridge borrows that persistent stream,
contracts Hcore/Pulay, incorporates the ECP residual with the correct gradient
sign and drains before freeing its temporary buffers. It never destroys a
borrowed stream.

WB97M-V preserves its declared model: fitted full-range J/K plus exact LR by
default, or explicitly selected fitted LR with its own metric derivative.
Independent radial providers retain their own streams. An event orders final
D/W reads; the source consumer drains a previous stream before another provider
replaces or recomposes slots. CPU and CUDA share the existing scalar SR/LR
recomposition through `mixed_rsh_exchange.hpp`. VV10 and all geometry sources
still feed the compiler-defined ordered sum. Token, source-mask, finiteness and
terminal-publication checks remain mandatory.

The native profile inventory adds MN15, including its DF SCF provider. This does
not enable the separate local-AO optimization. D3/D4/gCP corrections compose once
with the complete electronic force, including r2SCAN-3c Direct and DF.

## Capacity and retained boundaries

Remove electronic atom/AO/primitive and grid-work fixture ceilings. Use checked
integer products and explicit numeric budgets. The shared CPU range runtime
formerly hid an 8-atom limit in a 24-element stack array; its transactional
accumulator now scales with the molecule, and the caller charges that additional
numeric storage. Quartet pages remain bounded, metadata conversions remain
validated and work counters reject overflow before publication.

ECP staging still obeys generated signed-int offset and rounded-launch bounds;
its quadrature convergence and angular expansion contracts remain unchanged.
The CUDA stationary arena is dynamic, while individual geometry schedules keep
their actual scratch, task and lane bounds and bounded fallback. gCP likewise
uses representability/allocation admission. D4's actual fixed EEQ pivot array
and D3's separate provider capacity remain real constraints, not electronic
force ceilings.

Absent energy providers remain absent: CPU M06-2X/MN15, CUDA PBE50, ECP+DF and
unlowered catalog graphs are not admitted by name. Unsupported mixed precision
or custom scaled graphs retain their existing qualification boundaries.

## Evidence and reproducibility

Qualification uses independent PySCF E/F and AO-transformed final D/W for every
named electronic program, both spins and Direct/DF on available backends.
The tests prohibit both Python scientific force consumers and cover ragged H2 /
water batches, empty beta, warm replay, moved geometry, malformed neighbors and
recovery. E/F/D/W limits are 2e-8 / 3e-7 / 4e-7 atomic units. Explicit fitted LR
uses an independently fitted LR oracle. ECP checks independent libcint LDA/PBE
and two-step reconverged full-energy derivatives for all admitted families.
A nine-atom WB97M-V trimer independently tests the former size envelope.

Native transport tests cover every linked profile, incomplete/duplicate sources,
stale/failing ownership, exact/below-floor capacity, and allocation beyond the
old atom/AO/primitive ceilings. Independent r2SCAN-3c correction totals and
native-only composition checks cover the corrected graph. CLI subprocesses run
with Python and compilers absent from PATH. All real GPU work runs through
finite Slurm allocations on node1/main/gpu:5090:1 with assigned visibility.

The initial remote rebuild exposed a provenance pitfall: `rsync -a` preserved
old source timestamps, so an expanded generated C++ profile layout did not
rebuild all consumers. Freshly touching the changed sources and rebuilding
resolved the resource-shape mismatch. Subsequent synchronization uses content
comparison without timestamp preservation; final receipts include binary and
source identity. Compiler caches are reused, never cleared to hide this issue.

Exact commands, numerical receipts, paired complete-endpoint timings and known
baseline failures are recorded with this PR's qualification evidence. Earlier
failed runs remain failed and are not relabeled as final-source acceptance.

The first full-endpoint preflight exposed a real warm WB97M-V DF regression:
the new resident source transport called generic fused RSH workers, while the
previous host-output path used the packaged omega=0.3 LR workers. The resident
isolated LR branch now shares that exact specialized launcher with the host
consumer and writes directly into its LR source slot. General RSH retains its
existing fused fallback. This is a transport/schedule repair, not a different
radial equation or a cached-force shortcut.

Final frozen qualification is retained in
[`r1-native-forces-20261025`](../../../../benchmarks/results/r1-native-forces-20261025/README.md):
192 ABBA/BAAB cold/warm/moved comparisons pass with a maximum median ratio of
1.00637 and equal SCF work. Full E/F samples, source/binary identities, compact
provider counters and a reproduction wrapper are retained. This compares
implementation `685478ca1` with parent `df2ded411`; the later parent CI-budget
fix does not change numerical source. The CUDA API AUTO-provenance assertion
fails on both versions and remains unchanged. Provider counters do not certify
whole-process RSS or simultaneous complete-device memory.

Post-campaign exact-budget ECP tests exposed a separate planner defect. The
shape adapter must reduce per-atom core counts to the scalar ABI field. More
subtly, energy setup retires SCF storage before ECP work, but a stationary ECP
derivative retains SCF and force arenas while allocating the same two-grid
scratch and standalone one-electron bridge. The native fleet now charges that
overlap, including the compact Hcore/Pulay outputs, instead of treating it as
retired setup storage. The old estimate caused UKS to reject its batched ECP
allocation and retry a smaller radial tile even with the advertised budget.
The retained fallback is unchanged; the corrected plan passes both spins and
representations with zero rejected allocations. Slurm job 14348 records explicit
owned-device plus provider-reservation peaks below the planned limits; it is
separate from the frozen all-electron timing campaign. Rebuilt CPU CTest remains
120/120, and the public budget/replay/failure tests and resource transport gates
exercise the follow-up.

## Rejected alternatives and invariants

Copying Python force equations into native code would create another scientific
authority; sharing the point programs, integral derivatives and scalar
recomposition avoids that. Removing every numeric limit would also be wrong:
fixed kernel index types, actual provider arrays and explicit budgets remain
checked. Reintroducing fixture-sized force refusal to hide a slow schedule would
confuse scientific admission with performance planning.

Revisit admission when a missing energy/derivative provider is implemented, and
requalify any changed precision, approximation or default schedule independently.
Do not infer universal convergence or grid accuracy from force capability.
