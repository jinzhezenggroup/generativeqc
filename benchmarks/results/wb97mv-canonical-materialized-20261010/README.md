# WB97M-V canonical component-reuse qualification

Historical opt-in qualification, superseded by the
[automatic-default qualification](../wb97mv-canonical-default-20261010/README.md).
The measurements and `default_promoted: false` in this snapshot remain unchanged;
they do not identify the later default binary.

Base: latest fetched master `4444d0376bb133f7981f44a263e2e409bfbfdd27`,
2026-10-10. Candidate binary SHA256:
`31dffff8697d949265993e1021a575a4bec2753d3215e29aac7ec91ce39addfa`.

The fixture is the README **water proxy**, not an OMol25 molecular-distribution
sample. Science is unchanged: complete spherical def2-TZVPD (including diffuse
and f), exact direct FP64 RKS WB97M-V, grid 48 x 16 x 32 per atom with three
Becke iterations, E/density/screen tolerances 1e-12/1e-10/1e-12, 100 maximum
iterations and VV10 density threshold 1e-8. Physical forces retain the explicit
4 GiB incremental admission budget from the original investigation.

## Result and limits

The new route is **qualification-only and disabled by default**. Select it before
preparing a Calculator/owner:

```bash
GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=1 <ordinary WB97M-V command>
```

It reuses the resident primitive-pair cache across all Cartesian components of
an indexed order-five shell quartet. Its original AO screening is unchanged;
an explicitly charged, bit-preserving bounds-layout view protects threshold
boundaries. Missing budget/cache/index capacity and unsupported consumers keep
the existing source. No CPU reference is introduced into production.

### Selected 96-atom source, not a physical endpoint

Same binary and Slurm GPU, policy 0 versus 1, complete selected `(3,2)` full J/K
and standalone LR K actions. Other source buckets are intentionally omitted;
the observer exits before publishing any physical energy or force.

| Source wall seconds | Scalar | Materialized |
| --- | ---: | ---: |
| Full J/K | 277.911619 | 156.057308 |
| LR K | 274.660582 | 159.299363 |
| Sum | 552.572200 | 315.356671 |

This is **1.752x / 42.93% less isolated source time**, not a 96-atom E+F speedup.
The scalar index has 26,062,583,408 AO entries per action; the materialized outer
index has 249,422,096 shell entries. These counts have different units. Exact
AO admission remains protected by native census and independent matrix tests.
The 2048 x 2048 selected Cartesian matrices agree with scalar output to
1.324e-12 (J), 3.731e-14 (full K), and 2.376e-14 (LR K) maximum absolute error.

### Complete fresh-owner 3-atom E+F

Three repetitions per policy, alternate order, same binary/GPU, existing caches
reused without clearing. Import/context/Calculator construction precede timing;
prepare, SCF, physical forces and synchronized host publication are included.
Teardown and serialization are excluded.

| Metric | Scalar | Materialized |
| --- | ---: | ---: |
| Median complete seconds | 12.740837 | 12.582382 |
| Iterations / Fock builds, every sample | 15 / 15 | 15 / 15 |

The small endpoint improvement is only about 1.24%. The scalar first-use sample
is 29.301302 s because force time is 19.931541 s; subsequent force times are
about 3.3 s. Do not attribute that first-use JIT/cache advantage to this value
optimization. All six samples pass the independent GPU4PySCF E/F gates
(1e-8 Eh / 1e-7 Eh/Bohr), with maximum energy error 8.413e-12 Eh and maximum
force error 1.822e-11 Eh/Bohr.

The unchanged stock 12-atom reference fails to converge. Its protocol is not
relaxed. A native/native larger comparison is not an independent oracle pass.
No complete 96-atom E+F result or independent 96-atom accuracy claim is made.

### Complete fresh-owner 12-atom native/native pair

One matched pair on the same binary/GPU, not a repeated median:

| Metric | Scalar | Materialized |
| --- | ---: | ---: |
| Complete E+F seconds | 582.965492 | 531.350324 |
| Prepare seconds | 0.977958 | 0.940432 |
| SCF/publication seconds | 543.416907 | 491.876176 |
| Physical force seconds | 38.570627 | 38.533716 |
| Iterations / Fock builds | 21 / 21 | 21 / 21 |

Both native solves converge. Complete time is 8.85% lower in this pair; force
time is unchanged, so the improvement is not a force-cache bootstrap advantage.
Energy differs by 5.685e-14 Eh and forces by at most 5.706e-11 Eh/Bohr between
the two native sources. This is a consistency check, **not** an independent
12-atom accuracy gate or a universal endpoint speedup claim.

## Validation

- Independent native CPU-ERI full/SR/LR matrices, RHF/UHF, J/K masks,
  spherical/Cartesian projection, signed/diffuse/screened inputs, two-system
  batches, displaced geometry, exact AO/radial census, frozen preparation
  selection, bounds-layout correspondence and bounded allocation fallback pass.
- Real-GPU pair recurrence tests pass for orders 5--12 and retained derivative
  consumers. A 162-component/16-primitive-product packet executes 16 Coulomb
  preparations, 2,592 contractions and 162 publications.
- Final memcheck, initcheck, synccheck and racecheck report zero errors/hazards.
  The lifetime arm forces a single materialized CTA through multiple tasks;
  this diagnostic launch configuration is not used for endpoint timing.
- 21 focused Python code-generation/dispatch tests and the 504-module compiler
  structure check pass. Formatting and diff checks pass.
- Static RKS register/stack reports change from 174 / 4,136 bytes to
  141 / 512 bytes, with 8,336 shared bytes. No achieved-occupancy or spill claim
  is inferred from these static reports.

## Reproduction and provenance

All GPU work uses finite `srun` allocations with scheduler visibility intact.
Run from the permitted host's Slurm controller, not the maintenance node's
controller. Native compilation and JIT use verified ccache 4.5.1, CUDA 12.9.1,
GCC 13.3, sm_120, without clearing or weakening cache identities. Before/after
stats and actual compiler commands are retained.

Ignored raw evidence is retained on n1 at
`/data/jzzeng/wb97m-canonical-opt-20261010-final-4444d0376/`, with local summaries
under `.artifacts/wb97m-canonical-opt-20261010/final-results/`. Its `input/`
directory contains the frozen master archive, exact candidate patch, observers,
bounded build/source/endpoint/sanitizer scripts and the summary reducer.
The earlier prototype is separately retained and is not substituted for final
evidence. An attempted n5 run rejects the n1 binary's newer glibc ABI before
execution and is not counted as a GPU test or endpoint.

Implementation rationale and promotion gates:
`.agents/notes/implemented/performance/2026-10-10-indexed-canonical-pair-materialization.md`.
