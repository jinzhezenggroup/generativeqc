# WB97M-V Hessian capacity: bounded source replay

Native numerical revision: `618f7e3ddf616f7960e9e3fc9df6aa31fda26107`.
The previous source-matched binaries rejected both cases before publishing a
Hessian. This bundle retains those failures, intermediate diagnostic failures,
complete successful native endpoints, and negative comparator/performance data.

## Native capacity

All cases use spherical direct RKS WB97M-V and deliberately coarse **8/4/8**
primary/NLC/response grids. This is not chemical quadrature convergence.
Native execution is linked C++/CUDA compiler AD, with no runtime Python/JIT,
displaced SCF, numerical CPU/PySCF fallback or posthoc symmetrization.
SCF energy/density, screening and response tolerances stay
`1e-13/1e-11`, `1e-14` and `1e-12/1e-11`.

| Native case | Atoms / spherical AOs | First / primed endpoint (s) | Sampled total-device peak (GiB) |
|---|---:|---:|---:|
| water-tetramer / STO-3G | 12 / 28 | 100.802116 / 100.742439 | 16.939636 |
| water-dimer / def2-SVP | 6 / 48 | 159.903344 | 17.006042 |

Every native sample stays within the unchanged **180 s endpoint** and **24 GiB
sampled total-device** bounds. Device peaks are sampled, not exact isolated
owner-arena allocations. Jobs run through finite Slurm requests on n1 RTX 5090;
assigned device visibility is preserved. The node is shared, not exclusive.

The exact source no longer requires all ordered shell descriptors to reside on
device. It replays bounded canonical-source/permutation tiles, retaining global
primitive/projection/reduction bounds. Moving-grid work uses device-checked AO
primitive incidences rather than every AO times every molecular primitive.
Full Hessians reuse contracted frozen Fock derivatives instead of rebuilding
four-index jets for solved density motion. See the current developer guide and
the bounded-source Agent Note for the contracts and explicit fallbacks.

## Numerical qualification

The tetramer complete analytic matrices agree with GPU4PySCF within `1.069e-7`.
For the def2-SVP dimer, all **73 independently converged CPU PySCF geometries**
produce an independent full Hessian from central gradient differences at
`0.004/0.002 Bohr` with Richardson extrapolation. Its maximum absolute error
against native is **`1.43809641973e-09`**, below the unchanged `3e-6` gate.
The independent CPU oracle takes 235.312 s outside native timing.

- 84 component tests pass, 8 skip unavailable adapter probes; independent
  libcint J/K derivatives, spherical projections and independent AO/Becke values
  cover streamed/tiled/scalar paths and resource/lifetime rejection.
- Four complete RKS/UKS direct and DF+ECP+spherical comparisons pass cached
  versus directional/physical/serial fallback at `3e-12` absolute/relative.
- The larger def2-SVP native full matrix agrees with a standalone native HVP
  column at `2.598e-14`; this is a schedule check, not an independent oracle.
- Memcheck and initcheck cover tiled exact sources, topology-checked moving
  grids and complete frozen-derivative composition: 22 reports, zero errors.
- Host closure gates admit 663,552 ordered groups within 64 MiB and reject
  global work/source overflow, missing closure and disabled fallback.

### Retained comparator disagreement

The stock GPU4PySCF def2-SVP analytic matrix differs from native by
`1.3952071e-5`, exceeding the unchanged `3e-6` gate; its next fresh SCF also
fails to converge. Do **not** turn the first timing into a qualified speed ratio.
A diagnostic dense solve of the same GPU4PySCF response operator reaches a true
residual of `6.27e-14` yet retains the analytic-matrix disagreement. Lowering
Krylov's dependence threshold alone is therefore not an explanation or fix.
Independent CPU gradients at displaced, independently converged geometries
provide a separate derivative oracle. These CPU calculations are qualification
only and are never called by the native endpoint. All negative attempts remain
in the compressed records/provenance.

## Performance is not solved

The accurate four-water paired run completes, but the primed full endpoint is
**100.742439 s native versus 14.227901 s GPU4PySCF: native is 7.081x slower**.
Its preparation-plus-Hessian versus reference-Hessian comparison is also
retained in raw phase receipts. One primed sample is not a statistical ranking.
The def2-SVP stock comparison is numerically unqualified, so no speed ratio is
claimed there. This change fixes measured capacity failures; it does not claim
larger-system superiority or completion of the requested performance work.

The 128 KiB optional dense CPKS cap is unchanged: these larger spaces still use
physical matrix-free GMRES. Work receipts expose the remaining induced actions,
exact source/kernel launches and moving-grid contractions. Source/work/arena
caps still reject unaffordable cases; these two scale axes are not an arbitrary
molecule-size guarantee. No time, memory, screening or numerical guard is raised.

## Stored data and reproduction

- `samples.json.xz`: complete original capacity/paired journals, matrices,
  failed attempts, phases, work counts, memory samples and independent oracle.
- `qualification.json.xz`: numerical acceptance scope and failed performance
  status. Numerical acceptance does not promote performance.
- `provenance.json.xz`: exact build/cache/qualification/sanitizer/diagnostic
  logs, source scripts, fallback receipts and the publisher.
- `reproduce.sh`: runnable from any directory. Set the reference interpreter,
  compiler-cache launcher, and both `GENERATIVEQC_CLI`/`GENERATIVEQC_LIBRARY` to
  the exact recorded source-618 artifacts; mismatched hashes fail before Slurm.
  The checkout must contain fixture driver revision `9abfe3b`; it is archived
  so fresh workers see the shared tetramer case. The verified native pair is
  staged and journals go under the checkout's `.artifacts/`. Native source
  identity remains `618f7e3`, not HEAD.

Raw reproduction scripts embedded in provenance preserve the measured host
paths/job ids. For a new oracle run, stage the new native comparison journal as
`.artifacts/candidate-9977/capacity.json` in the oracle working directory and
extract `cpu_oracle.py`/`cpu_oracle_full.py` there. Run the oracle separately with
the pinned reference environment and finite CPU Slurm time; do not insert it
into the native CLI or count its time as a native endpoint measurement.
