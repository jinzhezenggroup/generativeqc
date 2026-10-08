# MD-J default: 96-atom complete cold qualification

Production source is based on master
`ed8d21684910f65b656f5bf42c8ef0248aa7d40f`, measured on October 8, 2026.
This is the J-only production integration, not the earlier standalone/fused
MD J/K prototype. The MD samples remove `GENERATIVEQC_DISABLE_MD_J` entirely;
normal samples set it to `1`. K, XC, initial-density policy and finalization
remain normal production consumers in the same native library.

## Endpoint and result

- PBE0 RKS; 32 waters / 96 atoms / 768 spherical def2-SVP AOs.
- Actual native table-free grid: 24 radial x 8 polar x 16 azimuth = 294,912
  points, no pruning, matched to the independent reference prescription.
- Strict FP64, screening `1e-12`, energy tolerance `1e-11`, density tolerance
  `1e-9`, 150-iteration cap; energy-only endpoint.
- Three alternating fresh-process pairs; no CUDA priming, supplied density,
  reused calculator, warm start or retained owner. One Slurm job and binary.
- Timer includes new calculator, preparation, complete SCF and owner teardown.
  Python imports and process shutdown are outside the timer.

| Repeat | Normal seconds | Default MD-J seconds | Normal / MD Focks |
| --- | ---: | ---: | ---: |
| 1 | 73.958711 | 62.942511 | 17 / 16 |
| 2 | 74.428092 | 63.100250 | 17 / 16 |
| 3 | 74.273034 | 63.069519 | 17 / 16 |
| Median | **74.273034** | **63.069519** | 17 / 16 |

Median ratio is **1.177638x** (15.084229% less complete endpoint time).
This includes the different converged trajectories: it is not a same-Fock
kernel-speedup claim, nor a claim about force, response or warm endpoints.
Preparation medians are 3.328249 / 3.442795 seconds; MD setup is not omitted.

All six samples pass the independent energy gate `3e-9` hartree and physical
residual gate `1e-9`. Maximum energy error is `6.730261e-11` hartree; maximum
physical residual is `1.148153e-11`. Independent PySCF energy is
`-2441.5918489586534` hartree. Normal has zero MD calls; default MD has exactly
16 MD calls per complete endpoint. Grid visits are 5,013,504 / 4,718,592.
The retained dense angular residual probes are upper bounds, not accepted
quartet counts: 209,520,685 per MD Fock, 3,352,330,960 per MD endpoint.
Density-bound refreshes are 32 per MD endpoint.

## Qualification scope after resource repair

The source and binary receipts below identify the measured revision
`9c911d28004eb0baae07ef63d78ee9c8ffcda63f`. Subsequent host resource repairs retain
these original hashes; they are not receipts for a rebuilt binary. The repair
keeps the original optional allowance and numerical kernels for unbudgeted KS,
retains normal J under public resource ledgers, and uses the existing checked
optional-allocation rollback. Host policy/allocator tests qualify those changes;
no additional GPU performance or sanitizer campaign is claimed.

## Retained evidence

`paired-cold.json` contains all six samples and the acceptance verdict.
`input.json` contains the exact geometry and contracted shells;
`independent-oracle.json` retains the independent SCF receipt and history.
`provenance.json` identifies the measured binary and native-source hashes,
recipe and sanitizer qualification. No binaries or profiler archives are
tracked. Input and oracle have one additional trailing LF for repository
formatting; remove that LF to recover their recorded execution-byte hashes.

The measured library SHA-256 is
`119fc921535b5e6d05d11c3f69616e047f348ef14a5b27d312d5e880637011e8`.
The baseline is the optimized normal provider in this same library, not the
old scalar/fused prototype. This isolates the J-only selector; source changed
since the older prototype cohort, so its timing is not substituted here.

## Reproduce

Use CUDA 12.9.1, an sm120-capable compiler, CMake/Ninja, ccache and a Python
environment containing NumPy, PySCF and pytest. Configure/build normally:

```bash
cmake --preset cuda-release-sm120 \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DGENERATIVEQC_CUDA_COMPILE_JOBS=8 \
  -DGENERATIVEQC_AOT_COMPILE_JOBS=2 \
  -DGENERATIVEQC_AOT_SPLIT_COMPILE_THREADS=8
cmake --build --preset cuda-release-sm120 -j8
export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
export PYTHONPATH="$PWD/python:$PWD"
export GENERATIVEQC_BENCHMARK_SOURCE=ed8d21684910f65b656f5bf42c8ef0248aa7d40f
export GENERATIVEQC_TEST_MD_J_NORMAL=1
mkdir -p /tmp/md-j-default-cold
python - <<'PY'
from pathlib import Path
source = Path("benchmarks/results/md-j-default-cold")
for name in ("input.json", "independent-oracle.json"):
    Path("/tmp/md-j-default-cold", name).write_bytes(
        (source / name).read_bytes().removesuffix(b"\n"))
PY
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --nodelist=node1 --time=00:20:00 \
  python benchmarks/md_j_normal_cold.py \
  --input /tmp/md-j-default-cold/input.json \
  --oracle /tmp/md-j-default-cold/independent-oracle.json \
  --output /tmp/md-j-default-cold/paired-cold.json --repeats 3
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --nodelist=node1 --time=00:10:00 \
  python -m pytest -q tests/python/test_md_j_normal_cuda.py
for tool in memcheck racecheck; do
  srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
    --nodelist=node1 --time=00:15:00 \
    compute-sanitizer --tool "$tool" --error-exitcode 99 \
    python -m pytest -q tests/python/test_md_j_normal_cuda.py \
    -k '0.02 and unrestricted and spherical'
done
```

Preserve Slurm's assigned `CUDA_VISIBLE_DEVICES`. Use finite `srun` for every
real GPU command; do not override device visibility. Source hashes in
`provenance.json`, not just the base revision label, identify this patch.

To regenerate rather than reuse the independent oracle, run the following
CPU-reference recipe in a finite Slurm allocation with NumPy/PySCF available.
It uses the exported explicit basis rather than fetching a named basis and
uses the NumPy grid prescription independently of native CUDA quadrature:

```python
import json
from pathlib import Path

import numpy as np
from pyscf import dft, lib
from generativeqc import Atom
from generativeqc_compiler.dft import GridSpec, MolecularGrid
from tools.generate_validation_references import pyscf_molecule

payload = json.loads(Path("benchmarks/results/md-j-default-cold/input.json").read_text())
atoms = [Atom.from_value(atom) for atom in payload["atoms"]]
molecule, _, _ = pyscf_molecule({
    "atomic_numbers": [atom.atomic_number for atom in atoms],
    "coordinates": [atom.position for atom in atoms],
    "charge": 0, "multiplicity": 1, "basis_representation": "spherical",
    "shells": payload["shells"],
})
lib.num_threads(16)
assert molecule.nao_nr() == 768
molecule.incore_anyway = False
tiles = list(MolecularGrid(atoms, GridSpec(24, 8, 16)).tiles(16384))
reference = dft.RKS(molecule)
reference.grids.coords = np.concatenate([tile.points for tile in tiles])
reference.grids.weights = np.concatenate([tile.weights for tile in tiles])
assert len(reference.grids.coords) == 294912
reference.xc = "PBE0"
reference.max_memory = 2048
reference.direct_scf = True
reference.direct_scf_tol = 1e-14
reference.small_rho_cutoff = 0.0
reference.conv_tol = 1e-11
reference.conv_tol_grad = 1e-9
reference.max_cycle = 150
energy = reference.kernel()
assert reference.converged and np.isfinite(energy)
print(energy)
```

A CPU-only recheck of the retained complete-endpoint gates:

```bash
PYTHONPATH=python:. python - <<'PY'
import hashlib
import json
from pathlib import Path
from benchmarks.md_j_normal_cold import collect

root = Path("benchmarks/results/md-j-default-cold")
report = json.loads((root / "paired-cold.json").read_text())
assert hashlib.sha256((root / "input.json").read_bytes()[:-1]).hexdigest() == report["samples"][0]["input_sha256"]
assert hashlib.sha256((root / "independent-oracle.json").read_bytes()[:-1]).hexdigest() == report["oracle_record_sha256"]
checked = collect(report["samples"], report["oracle_energy"])
assert checked["performance_acceptance_passed"], checked["gate_failures"]
for key, value in checked.items():
    assert report[key] == value
print(checked["median_seconds"], checked["normal_over_md_speedup"])
PY
```
