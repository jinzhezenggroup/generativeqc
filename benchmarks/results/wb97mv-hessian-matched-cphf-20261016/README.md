# Grid-aligned native WB97M-V Hessian comparison

Measured source: `17c539db5aa05dce9e2396aef814732f68cdfcb6`.
Slurm 3241 used an exclusive node2 allocation and one Slurm-visible NVIDIA
RTX PRO 6000 Blackwell with CUDA 12.9/sm120. Both engines ran canonical WB97M-V,
spherical STO-3G, direct RKS, self-consistent VV10 and full moving-grid response.
Primary, NLC and CPHF quadratures are aligned. The coarse 8/4/8 grid is an
implementation/latency diagnostic, **not chemically grid-converged**.

## Complete endpoint measurements

Each sample is a fresh process and fresh SCF through the full host Hessian and
JSON serialization. Native children have Python/compiler discovery unavailable
and PTX JIT disabled. Both engines have the same 600-second endpoint deadline.
First-observed samples are retained but excluded from the medians; two later
artifact-primed fresh processes per engine/case are interleaved. No persistent
compiler cache is cleared, and no empty-cache cold-start claim is made.

| Case | Native median (s) | GPU4PySCF median (s) | Native/reference time | Max matrix error (Eh/Bohr²) |
| --- | ---: | ---: | ---: | ---: |
| H2, 2 AOs | 1.743034371 | 5.473797673 | 0.318432371 | 1.834633556e-9 |
| Water, 7 AOs | 15.490663528 | 10.831725798 | 1.430119615 | 3.603285953e-8 |

Both pass the unchanged `3e-6 Eh/Bohr²` gate on every raw matrix entry across
every native/reference sample pair. H2 is about **3.14× faster**; water takes
about **43% longer**. These tiny workloads and two primed repeats do not establish
a statistically qualified default promotion, scaling result or universal speedup.
The publication accepts numerical equivalence and retains valid bounded ratios;
its broader performance-promotion stage remains unqualified.

`samples.json.xz` contains all twelve complete timings, matrices, commands,
environment/binary identities and native SCF/response/integral/grid work counts.
Reference response-action counts are unavailable, explicitly recorded as null.
`qualification.json.xz` retains independently recomputable acceptance records.
The shared publication manifest authenticates the stored bytes.

## Independent values-only molecular qualification

Slurm 3240 passed both actual benchmark full-matrix energy-oracle tests:
H2 has 36 raw entries and water 81. The unchanged `.004/.002` Richardson gate
is `atol=8e-7`, `rtol=1e-7`; observed maximum errors are `1.205120864e-10` and
`5.132580599e-8 Eh/Bohr²`. Both stencil matrices and every extrapolated entry
are retained, not only an aggregate pass flag. Libcint, upstream Libxc and an
independent direct-space VV10 expression supply values; no compiler derivatives,
native response or displaced production Hessian enters that oracle.

This oracle run used an explicitly retained test patch on production source
`f8ee2dd66299a5704626a662f6dbe05d7725e3ed`, not a clean-source17 rerun.
Its committed test module/probe and unchanged native library match the source17
bytes/content identity. Exact reconstruction and separate source/binary identities
are retained in the qualification record. Earlier full f/g AOT, RKS/UKS, Direct/DF,
ECP, Cartesian/spherical and public CLI/API qualifications remain in
`../wb97mv-hessian-default-fg-20261016/`; they are not relabeled as new runs.

## Why the older comparison failed

The old reference adapter configured primary and NLC grids but left GPU4PySCF's
independent default SG1 CPHF grid. Slurm 3239 changed only that response grid and
reduced the H2/water matrix errors from `9.3267e-5`/`0.00104567` to
`1.8346e-9`/`3.6021e-8`. The complete A/B matrices and own-primal diagnostic
energies are retained in the qualification record. This is workload/configuration
misalignment, not proof of faulty vendor kernels. Historical failed raw timings
and null ratios in `../wb97mv-hessian-native-full-20261016/` remain unchanged.

`reference-grid-control.py.xz` preserves the exact historical driver. Its
reproduction needs the pre-alignment sourcef8ee reference helper and canonical
native inputs reconstructed from the source84 publication, as detailed in the
manifest. It is not a source17 timing script. Decompress with `xz -dc` before use.

## Reproduce

Keep this bundle outside the checkout, restore its measured source, and run
`bash /path/to/reproduce.sh` with `REFERENCE_PYTHON`, CUDA and reference shared
libraries configured. The script verifies the compiler cache, enables the full
default AOT closure and uses finite Slurm allocation/steps. Build-time Python is
allowed; the measured native numerical runtime is Python-free. The standalone
reproduction script was not independently replayed.
