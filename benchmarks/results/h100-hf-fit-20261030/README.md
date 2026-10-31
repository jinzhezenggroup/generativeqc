# H100 HF fitting numerical preflight (2026-10-30)

The stock HF-DF auxiliary basis fails the declared energy gate on the sampled
three-atom water geometry. Explicit shared **cc-pVTZ-JKFIT** H/O tables pass all
energy and full analytic-force gates at 3, 48 and 96 atoms for native baseline
`15ab66a68a061355fda833a93e88484dbfee6360` and stock GPU4PySCF 1.9.0. This is a
static numerical preflight, **not a default promotion or a performance result**.

Both engines ran on physical H100 UUID
`GPU-b62d2463-27e7-bf5f-41a5-3ea1a4f99508`, SM90, MIG disabled, 80-GB HBM3,
driver 595.58.03 and CUDA 12.9. The finite qz build allocation provided 20 CPUs
and 200 GiB; each numerical endpoint used one math thread. Candidate compilation
was concurrent: **every retained second is excluded from clean performance**.

Each of the six explicit-fit populations has one complete cold energy/force
call plus three warm calls from its own frozen cold density: 24 E/F calls.
The stock-default GPU4PySCF failure adds four retained calls. No repeat was
removed. The stock reference retains its convergence, screening, DIIS and guess
defaults; the fitting table is its sole scientific override. Auxiliary-basis
response is enabled. Native endpoints block PySCF/GPU4PySCF imports.

| Observations | Maximum E error/atom (Eh) | Maximum force error (Eh/Bohr) | Maximum force RMS error (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| Explicit fit, 3 atoms, both engines/all repeats | 9.11968e-7 | 3.18395e-6 | 1.83863e-6 |
| Explicit fit, 48/96 atoms, both engines/all repeats | 2.38996e-6 | 4.46962e-6 | 2.21416e-6 |
| Stock-default fit, GPU4PySCF, 3 atoms | 1.85075e-5 (fails) | 8.88477e-6 | 4.64319e-6 |
| Unchanged gates | 1e-5 | 1e-4 | 3e-5 |

The independent dense direct targets remain in
[`readme-repaired-20261021/oracles.json.xz`](../readme-repaired-20261021/oracles.json.xz).
The receipt-bound version-2 reuse contracts retain their original producer,
controls, geometry, CPU/GPU spot check, software maps and scientific-input
review. Replay recomputes these gates and separately subtracts every full force
vector with `math.fsum` RMS arithmetic. It does not trust saved pass flags.
Integrity and numerical checks remain active with `python -O`; replay also
requires the exact seven unique scientific populations, with labels bound to
their method, route, size, engine and fitting contract.
The oracle producer audit's unresolved full-script reconstruction for 14
denser-grid **DFT** receipts does not apply to the HF-only observations here.

The retained CPU scan includes the unsuccessful default and def2-universal
113-function fits and both tested alternatives. The explicit 139-function,
maximum-angular-momentum-four fit was chosen for its energy accuracy margin,
before clean fitting-performance comparisons. PySCF 2.14.0 tables were exported
with exact float round-trip; native and reference inputs share those tables.
The CPU direct reference matches the retained independent CPU HF vector/energy.

`manifest.json` authenticates the stored records, exact input files and actual
reference/native adapter scripts. `observations.json.xz` retains full-precision
samples, work/settings, source/core/AOT identities, plans, package/device
observations, the CPU scan and runtime audit. Private shared-root prefixes in
string values and dictionary keys are replaced with
`<H100_PORTABILITY_ROOT>`; every number, sample, geometry and ordering is
unchanged. Original raw/plan SHA-256 values bind privately retained originals;
the public copies are explicitly transformed, rather than claimed byte-identical
original receipts. Retained adapter sources in `*.py.xz` decompress byte-identically
to the executed versions. Replay verifies their original identities. The native
adapter calls the pinned public benchmark driver.

From the repository root, with NumPy available:

```bash
python benchmarks/results/h100-hf-fit-20261030/replay.py
```

This performs no GPU work, network access or native compilation. To repeat GPU
measurements, use a finite allocation, the pinned source/core and isolated
GPU4PySCF1.9.0/CuPy13.4.1/cuTENSOR2.2.0 environment, the exact input files and
adapter `--plan`/`--output` interfaces. Create fresh plans with the real new
allocation and absolute paths; the retained historical plans are provenance,
not commands to replay into the original output directory.

Candidate #2888 qualification, moved geometry/lifetime/resource guards, final
production configuration and clean three-process timing remain pending under
[#2891](https://github.com/jinzhezenggroup/generativeqc/issues/2891) and
[#2892](https://github.com/jinzhezenggroup/generativeqc/issues/2892).
This evidence does not establish the parent majority target, close #2884 or
qualify chemistry beyond the sampled water-series HF task. Original defaults,
primary case counts and tolerances are unchanged.
