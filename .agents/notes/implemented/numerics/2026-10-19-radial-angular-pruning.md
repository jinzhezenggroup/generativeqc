# Decision: explicit, fixed-topology radial angular pruning

Status: implemented (opt-in; no production-default or headline-ranking promotion)
Date: 2026-10-19

## Problem

Rectangular atom/radial/polar/azimuth quadrature uses the same angular work in
every radial shell. Reducing that constructed domain can remove semilocal work
and nonlocal pair work, but changing only the independent comparator's `prune`
attribute cannot change its overridden atomic-rule constructor. Pruning also
needs to survive preparation budgets, snapshots, resident ownership and analytic
moving-grid response; dropping points only in one exporter is not a solution.

## Decision

Grid contract version 3 explicitly selects `radial-v1`, requires positive
element radii and fixes the radial-rank schedule documented in
`docs/user/ks_options.md`. Versions 1/2, their defaults and scientific identities
remain unchanged. The C descriptor layout is unchanged: version 3 identifies the
entire new quadrature, and Python snapshot decoding restores its pruning field.
Custom v3 specs do not inherit canonical unpruned GridPolicy provenance.

The rule chooses full, half or quarter angular orders using integer radial rank.
This keeps dry point-count/resource planning free of quadrature construction,
avoids root-roundoff-dependent shell boundaries and preserves topology under
nuclear displacement. Each shell has its own complete angular rule and weights;
this is not an empirical Lebedev/NWChem table or density-based screening.

The CUDA implementation passes at most five contiguous radial bands by value,
with up to three shared angular rules. The explicit 128/512 maximum orders let
those rules fit the existing bounded 256/1024 caches without enlarging the
device-layout ABI or uploading point-sized host indices. Unsupported orders fail
closed. CPU/CUDA allocation admission, stationary owner decoding and snapshot
exports use the actual pruned point count. Host setup bounds include shell
descriptors and concurrent angular rule vectors.

The diagnostic reference independently reconstructs the rank prescription and
installs it for both semilocal and nonlocal moving grids. Diagnostic CLI options
make pruning opt-in while explicitly retaining the historical radial scales.
This does not turn a matched-grid diagnostic into a production ranking.

## Rejected alternatives

- Change v1/v2 quadrature or enable pruning by default: breaks retained scientific
  contracts without method-, basis- and geometry-class qualification.
- Choose bands by floating-point radial roots: materializes Legendre rules during
  metadata-only resource planning and adds avoidable threshold-roundoff identity
  questions. Integer rank is sufficient for this deterministic opt-in rule.
- Subsample dense angular nodes while retaining their weights: does not produce
  the selected lower-order quadrature and can lose even low-order moments.
- Keep dense arrays with zero-weight points: changes neither constructed storage
  nor the nominal domain that downstream pair kernels may still traverse.
- Upload per-point shell/owner maps: unnecessary for a bounded band schedule.
- Reuse the unpruned product for force ownership or resource estimates: crosses
  atom boundaries incorrectly and can invalidate stationary capacity admission.

## Invariants and qualification

- Grid identity, native payloads and snapshots retain the exact pruning contract.
- Atomic measures are independent of geometry; point translation and Becke weight
  response remain the derivative sources for moved grids.
- Independent Gaussian volume/moment integrals and multistep finite differences
  test the quadrature and its geometry response.
- Compiled native CPU/Python parity covers odd/small orders, partial tiles and
  pruned/unpruned point ordering. Real-device tests cover coordinates, atomic and
  partition weights, response exports, resource accounting and owner cleanup.
- Optional independent PySCF LDA/PBE convergence uses the same declared energy
  and analytic-force gates as the unpruned profiles, with dense/ultradense
  reference stability and explicit point-reduction checks.
- No full VV10 endpoint speedup or general production accuracy is inferred from
  point reduction or these semilocal oracle cases. Broader qualification remains
  required before default promotion or headline publication.

## Retained qualification evidence

The 2026-10-19 CPU build used `build/pruning-cpu/libgenerativeqc.so`, Release,
CUDA/AOT disabled and verified ccache 4.5.1. Focused quadrature, identity,
snapshot, displaced public energy/force, reference-grid and documentation tests
passed (212 tests, 8 explicitly allocated-CUDA cases skipped). Host resource,
allocation-double and generated-kernel arithmetic tests passed (294 tests,
9 allocated-CUDA cases skipped). Both independent grid-convergence tests passed.
Public CPU LDA/PBE forces agreed with independent PySCF full-response forces on
both the original and displaced water geometries; restricted/unrestricted custom
SCF cases also covered both pruning choices.

`python -m benchmarks.grid_policy_convergence --include-pruning` with PySCF
2.14.0 retained these final integer-rank-policy results for STO-3G water:

| Profile | Unpruned points | Pruned points | Energy error (Eh) | Max force error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: |
| LDA standard | 82,944 | 41,760 | 6.9485e-7 | 2.9197e-5 |
| LDA tight | 153,600 | 77,100 | 3.7507e-7 | 1.1425e-5 |
| PBE standard | 82,944 | 41,760 | 6.8805e-8 | 2.0746e-5 |
| PBE tight | 248,832 | 124,416 | 3.0832e-7 | 4.0548e-6 |

Every profile passed its unchanged energy/force/point-cost gates against the
96x32x64 reference, whose stability against 120x40x80 also passed. These are
accuracy and constructed-work measurements, not native endpoint timings.
Raw local evidence is `.artifacts/pruning-20261018/convergence-final.json`;
earlier `convergence.json` used a discarded physical-node schedule and must not
be cited as evidence for the final prescription.

Slurm job 10255 on node1 (RTX 5090, CUDA 12.9.1, verified sccache 0.16.0,
finite five-minute allocation) passed the standalone compiled native CUDA grid
probe under compute-sanitizer memcheck: zero errors and zero leaked bytes.
The probe covers all three versions, small/odd orders, maximum pruned angular
orders, 512 radial nodes, translation, coincident centers, resident exports,
native weight response and exact allocation lifetime. At the maximum radial
order the rational tail reaches about 1e5 Bohr: host/device norm subtraction
needs a distance/separation-conditioned roundoff tolerance for dimensionless
Becke weights even in v1/v2. This does not change production numerics or relax
energy/force gates. The probe does not qualify full GPU SCF/force execution or
VV10. Raw logs, input hashes and cache statistics are retained locally alongside
the CPU evidence.

### CPU fresh-owner endpoint timings

A follow-up on 2026-10-19 measured the actual public CPU energy-plus-force
endpoint on the same STO-3G water geometry. Each timed sample constructs a fresh
`Calculator`, prepares, converges SCF, evaluates analytic forces and closes the
calculation. One untimed complete call per candidate warms process/artifact
caches; there is no cross-sample SCF-state reuse. Eight paired samples alternate
candidate order. Hardware was an AMD EPYC 7K62 on node3, with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, using the native `cpu_reference` backend
from the source-matched CPU build. No GPU was used on node3.

| Method | Unpruned median (s) | Pruned median (s) | Median speedup |
| --- | ---: | ---: | ---: |
| LDA | 1.735403 | 1.092696 | 1.5882x |
| PBE | 2.447289 | 1.501921 | 1.6294x |

LDA interquartile ranges were [1.733396, 1.740763] and
[1.091399, 1.092909] seconds; PBE ranges were [2.446320, 2.451025] and
[1.500121, 1.502549]. The constructed grid shrank from 82,944 to 41,760 points.
Every sample retained 10 SCF iterations and 12 physical Fock builds, and passed
the unchanged corresponding dense-reference energy/force gates. These are
CPU process-warm timings including fresh-owner setup, not process-cold timings,
GPU results, VV10 measurements or a comparison against another engine. They do
not establish the README production ranking or qualify larger systems.

The ignored local reproduction script and raw samples, including binary/input
hashes and per-sample physical work diagnostics, are
`.artifacts/pruning-20261018/endpoint_performance.py` and
`.artifacts/pruning-20261018/endpoint-performance.json`.

### GPU fresh-owner endpoint timings

The 2026-10-19 GPU follow-up used a source-matched Release build on node1 with
CUDA 12.9.1, SM 120-real, AOT orbital shells and native stationary CUDA enabled
for `lda_rks;pbe_rks`. Python shared stationary-force AOT and CPU-force AOT were
disabled. The verified C++/CUDA compiler launcher was sccache 0.16.0; neither
precision nor compiler-cache correctness flags were weakened. The first build
hit its 30-minute limit; incremental completion retained its objects and cache.
The full dirty source capsule, HEAD, input/binary hashes, build scripts/logs and
cache statistics are retained in `.artifacts/pruning-gpu-20261019/`. The built
library SHA256 is
`92e4508516078043a72030fcd8dbf8ed6bbe556b8eece29a0e1b3511c4fac447`.

All real GPU execution used finite `srun` allocations on node1, partition `main`,
`--gres=gpu:5090:1`, preserving scheduler-assigned `CUDA_VISIBLE_DEVICES=2`.
The device was an NVIDIA GeForce RTX 5090 (compute capability 12.0), with CUDA
runtime 12090 and driver API 13000. No GPU execution occurred on node3.

Each timed sample synchronizes the device, constructs a fresh `Calculator`,
executes public `singlepoint(..., properties=("energy", "forces"))`, including
batch preparation, converged SCF, analytic forces and prepared-owner cleanup,
then synchronizes again. One complete untimed call per candidate warms process
and artifact caches. Candidate order alternates; no converged SCF state is
passed between samples. Native/BLAS thread counts were one. The solver used
STO-3G, energy tolerance 1e-12 and density tolerance 1e-10. These are complete
process-warm endpoints, not process-cold or isolated XC-kernel timings.

Slurm job 10296 collected eight samples per candidate on the independently
qualified three-atom water geometry:

| Method | Unpruned median (s) | Pruned median (s) | Median speedup |
| --- | ---: | ---: | ---: |
| LDA | 0.295614 | 0.212991 | 1.3879x |
| PBE | 0.651860 | 0.428250 | 1.5221x |

LDA interquartile ranges were [0.295334, 0.297824] and
[0.210161, 0.215305] seconds; PBE ranges were [0.651408, 0.653786] and
[0.421824, 0.430146]. Both candidates retained 10 SCF iterations and 10 physical
Fock builds, with 82,944 versus 41,760 constructed points (49.65% fewer).
Every timed sample passed the unchanged independent dense-reference gates:
LDA energy 1e-6 Eh and maximum force error 5e-5 Eh/Bohr; PBE energy 1.5e-6 Eh
and maximum force error 3e-5 Eh/Bohr. Maximum pruned errors were respectively
6.8042e-7/2.9202e-5 and 9.6281e-8/2.0751e-5. The CPU PySCF reference and its
dense/ultradense stability qualification are the retained convergence evidence
above; there is no reference-engine work inside the timed native endpoints.

Slurm job 10297 additionally measured the nested `water-24` fixture from
`benchmarks.readme_hf_scaling.scaling_cases()` with six alternating PBE samples:
3.276269 to 2.245648 seconds, or 1.4589x. Interquartile ranges were
[3.268300, 3.288262] and [2.210827, 2.256865]. Both grids retained 12 SCF
iterations and 12 physical Fock builds; constructed points fell from 663,552
to 334,080. The two policies differed by 9.0464e-8 Eh in energy and at most
4.0462e-8 Eh/Bohr in forces. This is a timing diagnostic, **not independently
dense-grid accuracy-qualified**. A GPU4PySCF dense-reference attempt (job 10291)
failed to converge with unphysical energies; a retry (10293) behaved similarly
and was canceled. An independent CPU PySCF attempt timed out at 600 seconds
before completing the dense endpoint. Their logs/statuses remain retained;
neither failed reference nor pruned/unpruned agreement substitutes for the
missing independent qualification. The reference failure's cause is unresolved.

The local reproduction scripts are `endpoint_gpu.py` and `run.sh` under
`.artifacts/pruning-gpu-20261019/`; raw samples, diagnostics, scheduler metadata
and input hashes are `water-10296.json` and `water-10297.json`. The latter
explicitly records `dense_accuracy_qualified=false`.

### GPU qualification limitation: resource ledger

The follow-up allocated GPU SCF/resource test run (job 10298) selected
`test_custom_native_grid_matches_independent_scf_and_budget -k cuda` and ended
with eight failures, eight deselected. All LDA/PBE RKS/UKS cases, for **both
unpruned and pruned grids**, passed their preceding numerical/reference,
convergence, tile-invariance and warm-start checks, then failed the unchanged
test assertion equating preparation-ledger live bytes to planned resident bytes
at `tests/python/test_ks_options.py:524`. For example, unpruned LDA observed
1,047,044 live bytes versus 537,917,982 planned. The planner contains an existing
512 MiB generated-force staging reservation on the fallback route; this is a
possible explanation for much of the discrepancy, not a verified root cause.
No ledger test or resource-admission contract was weakened to hide the failure.
The complete log is `gpu-scf-oracle-tests.log` in the same local evidence folder.

Consequently, measured small-water GPU energy/force speedups are supported, but
the full GPU resource suite is not green. No VV10, 96-atom GPU, cross-engine
production ranking or general-default qualification follows from these results.
Defaults and the README production-ranking qualification boundary stay unchanged.

## Revisit when

The opt-in/default boundary in this initial implementation is superseded for
LDA/PBE by [qualified pruning defaults](2026-10-19-qualified-pruning-defaults.md).
That follow-up retains this rule and explicit v1/v2 compatibility, but changes
the canonical v3 profile maxima after independent 96-atom qualification. It
also resolves the ledger-test limitation above by testing admitted capacity
rather than requiring an unused force reservation to be allocated eagerly.

Representative semilocal/meta-GGA/VV10 energy-plus-force endpoints establish the
best independently qualified grid policy for each production workload, or a
new policy justifies a separately versioned shell schedule or larger rule caches.
