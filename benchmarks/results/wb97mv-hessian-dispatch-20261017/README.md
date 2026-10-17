# Bounded WB97M-V Hessian dispatch optimization

The four paired cases and ethane controls use the same-node RTX 5090 for a
linked C++ CUDA compiler-AD Hessian. The corrected DF/ECP qualification and
native-only methanol followup use a separately identified n2 PRO 6000 allocation.
This is
not a general large-molecule or statistically established performance promotion.
The numerical decision is accepted only for the covered comparisons. The formal
performance envelope deliberately remains not-run: isolated peak memory,
comparable clean compile cost and complete solver histories are not available.

## Complete endpoints and phases

All cases use direct RKS WB97M-V/STO-3G spherical orbitals, self-consistent VV10,
identical moving primary/NLC/CPHF 8/4/8 grids, energy/density tolerances
1e-13/1e-11, screening 1e-14, and response atol/rtol 1e-12/1e-11.
This deliberately coarse diagnostic grid is not a chemical-convergence claim.
Each endpoint includes a fresh process, fresh SCF, full host Hessian, JSON and
teardown, under a 180 s deadline and 24 GiB sampled device-memory watchdog.

The table gives medians of three artifact-primed **fresh-process** samples per
engine; the separately retained first sample is excluded. Native/reference order
alternates between repetitions within each case.

| Case | Native endpoint (s) | GPU4PySCF endpoint (s) | Native / reference | Native Hessian (s) | Reference Hessian (s) |
|---|---:|---:|---:|---:|---:|
| water-dimer | 10.504804 | 25.758098 | 0.4078 | 8.389073 | 2.984565 |
| formaldehyde | 5.225690 | 37.399466 | 0.1397 | 3.051830 | 7.250810 |
| water | 2.749883 | 23.197878 | 0.1185 | 0.907326 | 2.088331 |
| h2 | 1.258217 | 19.064626 | 0.0660 | 0.171121 | 0.498217 |

Do not hide the Hessian-only gap behind the complete endpoint. Reference SCF
cycle counts and SCF wall times differ substantially from earlier campaigns;
raw phase times and work are retained for every sample. Faster native complete
endpoints do **not** establish Hessian-only parity or a pure-kernel speedup over
GPU4PySCF. Native preparation is separately reported and included in endpoint
wall time; native Hessian time includes response execution and host matrix copy.

The unchanged pre-optimization native deployment is also run on the same node
and GPU class after the full-projection intermediate campaign and before the
final tiled campaign: first 94.075937 s, primed 94.050333 s for water dimer. There
are only two baseline samples and this old/new order is not randomized. All
old/new native matrices pass 3e-12 absolute/relative gates; semantic SCF cycles,
CPKS actions, exact primitive components, directions and mixed contractions
match. This is an observed improvement, not a confidence-interval claim.

Every completed cross-engine matrix passes the unchanged 3e-6 Eh/bohr^2
absolute gate. The largest error is 3.65129841362e-08 Eh/bohr^2. The native child
executes with Python and executable discovery disabled and PTX JIT disabled;
there is no runtime Python, runtime compilation, host numerical fallback,
displaced SCF or intermediate panel download. Independent component oracles and
coupled optimized/ordered parity cover RKS/UKS, DF+ECP and spherical basis.
This followup does not claim fresh complete molecular energy finite differences.
The ethane native-only storage control additionally compares the final complete
matrix against the 111.660121 s full-projection fallback endpoint at unchanged
3e-12 absolute/relative gates and identical semantic work. It is not an
independent GPU4PySCF comparison.

## Memory and bounded larger controls

| Case | Native sampled total-device GiB | Reference sampled total-device GiB |
|---|---:|---:|
| water-dimer | 16.853 | 2.692 |
| formaldehyde | 16.827 | 2.692 |
| water | 16.800 | 2.706 |
| h2 | 16.784 | 2.683 |

These are sampled **total-device** used bytes, including contexts/reservations,
not isolated process allocation or an exact peak. The smaller Hessian-owner
arena is not total GPU memory. Polling at approximately 50 ms can miss spikes.
The large fixed native footprint is still present; this change does not claim
to diagnose or remove its origin. No CUDA stack-limit reduction is made.

Native-only larger controls retain the same admission, deadline and watchdog:

- methanol: journal status `completed`; native endpoint 10.846486 s.
- ethane: journal status `completed`; native endpoint 23.007458 s.

There is no new GPU4PySCF ratio for these two cases: earlier strict-reference
SCF failures remain in the preceding scaling publication. These small minimal-
basis examples are not practical large-molecule qualification. Resource/work
limits are not raised to force a larger case to complete.

## Dispatch and scientific invariants

The runtime retains optional strict canonical full/LR sources for induced
actions, batches compatible compiler integral selectors with compact primitive
records, projects shell groups concurrently and performs ordered warp-owned
reductions. VV10 pair tiles are 256 points; moving Becke tiles are 1024 points.
All optional storage and provider metadata remain charged to the existing
integral/aggregate budgets, with explicit bounded ordered fallbacks. Precision,
AD expressions, reference pair domain and numerical/solver gates are unchanged.
Source builds and still-required replayed quartets remain visible: zero induced
recurrence ERIs is not zero integral-source work.

`provenance.json.xz` separates instrumented baseline, compact-record and
full-projection intermediate profiles from clean timings. No final tiled-candidate
kernel profile is claimed; complete CLI phases/work are retained. Full-record intermediate batching could not fit
the dimer's 64 MiB integral budget and left the endpoint around 46 seconds.
Compact records enabled it without raising that budget. The compact-record
campaign was deliberately canceled during formaldehyde reference after all four
dimer repetitions; its checkpoint journal is retained as running, not relabeled
as a completed ladder. The final shared-incidence/warp and geometry-tile build
is separately identified and qualified. Its all-quartet projected storage still
missed ethane's budget, taking 111.660121 s with ordered fallback. The final plan
reuses a bounded ordered projected tile (16 MiB cap and remaining-arena bound),
carrying each output's original FP64 sum across tiles. It does not materialize
a whole public-quartet tensor or increase the 64 MiB integral budget.

The primary final qualification allocation exited on two DF/ECP test-path
admission assertions: its forced 512-byte test tile could not hold even one
p-shell group and correctly selected bounded fallback. Energy, HVP, Hessian
and residual were unchanged. Only the probe fixture cap was corrected to
32 KiB for the p-shell cases; production CLI/library bytes remain identical.
Fresh followup 3393 reruns both DF/ECP cases on n2. All failed-attempt receipts,
assertion traces, corrected receipts and probe hashes remain retained, rather
than relabeling the original allocation as wholly passing.

## Integrity and replay

Measured source is base `4dae97170434d4fb697f721c2f1009666786ff47` plus
`candidate-source.patch.xz`; the manifest's dirty flag refers to this production
and qualification overlay. Intermediate patches are preserved as exact strings
inside authenticated `provenance.json.xz`, not substituted for the final source.

Subsequent review repairs an optional LR-source metadata allocation failure so
the existing bounded GPU replay remains available. That admission guard is
host fault-injection tested; it has no fresh GPU qualification or timing campaign.
The archived source patch, matrices, qualification and provenance remain the
original measured `f4015ae9ae545ef5361083a5a38715acd588e4a7bdddfd22001ddb23222760df`
source, rather than evidence for the later repaired native source. The replay
below intentionally reconstructs that measured overlay. Its output-lifetime
repair writes default receipts under the caller's `.artifacts/benchmarks/` and
resolves relative `OUTPUT` paths before entering temporary source storage, so
both successful and failed receipts survive cleanup.

`samples.json.xz` preserves raw reports, full matrices, child stdout/stderr,
geometries, memory samples and coupled parity receipts. `qualification.json.xz`
contains recomputable explicit matrix gates. `provenance.json.xz` retains
compiler/cache/build logs, source and binary identities, finite allocation
commands, test/sanitizer output and diagnostic profiles. Full profiler traces,
binaries and caches remain ignored local artifacts, not external release assets.
Component memcheck/initcheck use a separate finite n2 PRO 6000 allocation and
the same compiled artifacts; those diagnostics are not RTX 5090 timing samples.

Set `SOURCE_CHECKOUT` to a repository containing the exact base,
`NATIVE_ROOT` to the measured linked CLI/library directory (including SONAME
symlinks), and `REFERENCE_PYTHON` to the recorded GPU4PySCF environment. Add the
CUDA and reference cutensor directories through `REFERENCE_LIBRARY_PATH` when
needed. The replay checks binary hashes and creates a private archived-source
overlay; it never modifies the source checkout. It does not rebuild binaries.
The complete cached sm120 orbital-f/auxiliary-g AOT build commands, versions
and cache statistics are retained separately. Match that toolchain when
rebuilding; a rebuilt binary is new provenance, not the measured artifact.

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=16G --time=00:15:00 bash /path/to/reproduce.sh
```

Select a compatible permitted node/GRES together; do not override Slurm device
visibility. Replay is syntax checked, not claimed as an additional timing
campaign. The actual retained allocation executed the authenticated driver and
qualification scripts in `provenance.json.xz`.

Current ownership and fallback contracts are in the
[Hessian developer guide](../../../docs/developer/dft_hessian.md); durable
rationale is in the
[Agent Note](../../../.agents/notes/implemented/performance/2026-10-17-bounded-wb97mv-hessian-batching.md).
