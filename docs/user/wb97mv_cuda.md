# WB97M-V CUDA energy and forces

The Python `Calculator` and prepared-batch interface compose the complete
all-electron WB97M-V energy and stationary analytic forces on CUDA: semilocal
meta-GGA, 15% short-range exchange, 100% long-range exchange (`omega=0.3`),
and self-consistent VV10. Nuclear repulsion, overlap/Pulay, AO motion,
quadrature-point motion and Becke partition response are included. With explicit
`precision="auto"`, the SCF may lower only the independently qualified Direct
Coulomb J recurrence to FP32 compute with FP64 accumulation; SR/LR K, meta-GGA
XC, VV10, the final physical audit and the stationary derivative owner remain
strict FP64.

Numerical acceptance covers restricted H2, unrestricted H3, spherical def2-SVP
and def2-TZVP water, water with the full local spherical def2-TZVPD snapshot,
and unrestricted NH₂/def2-TZVP,
including independent GPU4PySCF forces and reconverged energy finite differences.
The retained [SVP qualification](../../benchmarks/results/wb97mv-cuda-20260926/README.md)
and [OMol25-level benchmark](../../benchmarks/results/omol25-wb97mv-20261001/README.md)
record completed endpoints and incomplete attempts separately; acceptance on a
small molecule does not qualify scaling at every HF water-cluster size.

```python
from generativeqc import Calculator

calculator = Calculator(
    method="wb97m-v", basis="def2-svp",
    basis_representation="spherical", device="cuda",
)
water = [("O", (0.0, 0.0, 0.0)),
         ("H", (0.0, 1.43, 1.11)), ("H", (0.0, -1.43, 1.11))]
result = calculator.singlepoint(water, properties=("energy", "forces"))
print(result.energy, result.forces)  # Hartree and Hartree/Bohr; coordinates in Bohr
```

Set `precision="auto"` explicitly to opt into component-wise SCF precision.
The default remains `"fp64"`; AUTO is not promoted as a default without a
matched endpoint performance win.

Use `method="wb97m-v-uks"` and an appropriate multiplicity for unrestricted
spin. Energy-only requests avoid derivative work. Forces are the negative
energy gradient. No CPU integral derivative, reference SCF or finite difference
is part of the CUDA production force path.

The through-f opt-in gates include def2-TZVP RKS/UKS and local def2-TZVPD RKS.
When the optional retained Direct shell-force owner is unavailable, the existing
bounded CUDA one-electron fallback stages native basis metadata. This is not a
CPU derivative or oracle fallback; its H2D work is reported explicitly.

The complete Python force consumer admits built-in STO-3G, def2-SVP and
def2-TZVP, or explicit/local all-electron bases through f angular momentum, in
Cartesian or spherical representation. A basis such as def2-TZVPD is not
bundled and must be supplied as a local basis record. ECPs remain outside this
**force** contract. FP64 energy and force calculations may select
[density fitting](dft_density_fitting.md) for the full-range J/K primary;
the omega-dependent long-range
exchange correction remains exact Direct, and VV10 remains the existing
grid/nonlocal consumer. This is a mixed provider composition, not
range-separated RI-K. CPU also supports mixed FP64 energy/forces in the bounded
s/p/d/f domain documented by the [DF interface](dft_density_fitting.md); it shares
provider validation and source composition, not CUDA resident storage.
Component-wise `precision="auto"` is an SCF policy only;
it does not lower stationary force arithmetic or expand the basis, ECP or
resource domain.

For the H/O-only benchmark snapshot in a repository checkout, load the exact
diffuse basis rather than substituting the bundled def2-TZVP name:

```python
from generativeqc import Calculator, load_basis

basis = load_basis("benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json")
calculator = Calculator(method="wb97m-v", basis=basis, device="cuda")
result = calculator.singlepoint(water, properties=("energy", "forces"))
```

Capability discovery is intentionally layered. ``method_capabilities()`` is the
backend-neutral registry view and therefore reports the DFT carrier as
energy-only. After backend and basis selection, ``calculator.capabilities`` is
the authoritative execution-context view; admitted CUDA WB97M-V RKS/UKS adds
``forces`` there, and ``PreparedBatch.capabilities`` mirrors the same record.
The native manifest/registry remains conservative because the complete force is
a Python-composed stationary endpoint rather than a public C force action. The
private snapshot bridge is not a public C force API.

## Hessian and HVP

WB97M-V also exposes `Calculator.hessian()` and
`Calculator.hessian_vector_product()` on **CPU or CUDA**, using compiler automatic
differentiation and implicit CPKS, not finite differences of forces. It includes meta-GGA kinetic-density
response, both range-exchange operators, self-consistent VV10 and moving-grid/
partition and overlap/Pulay terms. RKS/UKS, Cartesian/real-spherical AOs through f,
direct/full-rank DF, and all-electron/scalar-semilocal ECP combinations are
supported in strict FP64. DF differentiates the fitting metric as well as
three-center values; rank-truncated or near-cutoff metrics fail explicitly.
CUDA Hessian numerical stages are device-resident, including moving-grid/VV10,
DF metric response, ECP projector AD, first/second integrals and the coupled
RKS/UKS response solve. Install `generativeqc[cuda-hessian]` for this endpoint;
it fails explicitly rather than falling back to CPU differentiation. Host
orchestration, fixed metadata/quadrature setup and scalar convergence checks
remain normal host work. Completed HVPs/matrices are returned as detached NumPy
arrays; intermediate numerical panels are not downloaded. This is not a speedup
claim. Resident dense primal ERIs have a bounded device-streamed fallback when
the integral budget does not admit their reuse.
Return values are `result.matrix` with shape `(3N, 3N)`
or `result.value` with shape `(N, 3)`, in Hartree/Bohr².

For an explicitly bounded demonstration (this small grid is **not** a claim of
quadrature-converged vibrational frequencies):

```python
from generativeqc import Calculator, GridSpec, KsOptions

calculator = Calculator(
    method="wb97m-v", basis="sto-3g", device="cpu",
    ks_options=KsOptions(grid=GridSpec(
        radial_points=12, angular_polar=4, angular_azimuth=8,
    )),
    energy_tolerance=1e-13, density_tolerance=1e-11,
    max_iterations=240,
)
result = calculator.hessian(water)
print(result.matrix, result.diagnostics["raw_symmetry_error"])
```

For open-shell CUDA use `method="wb97m-v-uks"`, `device="cuda"` and pass
`multiplicity` and `charge` to the endpoint. Set `basis_representation="spherical"`
for real spherical AOs. Enable fitted J/K with `density_fitting="cuda"` (or
`"cpu"` on CPU) and an appropriate explicit `auxiliary_basis`; omitting it uses
the orbital basis as the fitting basis. An ECP-bearing `BasisSet` retains its
ECP Hamiltonian automatically. The auxiliary AO adapter currently supports
through f. Inspect convergence and execution diagnostics and converge the
orbital, fitting and quadrature choices for the observable of interest.

Converge the grid and SCF/response tolerances for quantitative use. The grid
workspace allowance defaults to 128 MiB and the ordered VV10 pair cap defaults
to 16,777,216 **per action**. Fine/default grids can exceed this work cap;
`max_nonlocal_pairs` is an explicit override, not an accuracy or performance
guarantee. The full Hessian currently uses scalar columns. See the
[second-order execution contract](../developer/dft_hessian.md) for complete
resource scopes, raw-matrix behavior and independent validation gates.

NVCC is required for the generated geometry and final-reduction modules;
set `CUDA_PATH` or `CUDACXX` when it is not on `PATH`. Cold timing includes
compilation when the artifact cache is empty. Prepared batches retain derivative
owners on unchanged geometry and rebuild them on geometric changes. A live
SCF-generation token is checked before and after force assembly; failures
discard Python-owned derivative scratch and preserve per-item error reporting.

Through-f direct workloads automatically select exact FP64 angular-bucketed
J/K contraction under the existing provider budget; no performance option is
required. When its complete resident inventory fits, the source reuses HF's
normalized Cartesian integrals and public/source projections rather than
repeating spherical component expansion inside each quartet. Shell-local
transform spans bound projection work by the small shell component count,
not a dense cubic AO contraction; Cartesian public bases bypass identity
projection entirely. Full J and K share each symmetry-unique source ERI. Native Schwarz keys are
sorted within angular buckets and inclusive row spans omit rejected quartets
before traversal, preserving the exact product-based screening predicate.
RSH derivatives reuse that geometry-bound schedule and the same compiler-owned
integral algebra. Screening is defined in the selected source representation,
with identical full-range Schwarz admission for its J/K and RSH force consumers.
Generated SPD workloads retain their existing HF source owner.
Within canonical Cartesian sources, total-angular-order-five values automatically
share primitive-pair geometry and recurrence work across a complete shell
quartet. The route covers full J/K and standalone SR/LR K, keeps exact AO
screening, and falls back when its separately budgeted index/bounds view or
borrowed cache is unavailable. It is not an opt-in and does not enable the
legacy dense HF materialization schedule. See the
[prepared-pair contract](../developer/direct_pair_recurrence.md).
If optional sort/scan storage does not fit, execution keeps dense canonical
contraction. If Cartesian metadata/projection storage does not fit, the smaller
public-AO canonical source remains; if its pair/matrix storage also does not fit,
or mixed-J is requested, execution keeps the generic bounded source.
No four-index tensor is retained. Work counting
is available only through a borrowed native test/profiler observer, not a user
performance switch. These changes do not establish 100-atom endpoint parity.
See the [scheduling decision](../../.agents/notes/implemented/performance/2026-10-01-default-screened-through-f.md).

The consumer has explicit limits of 128 atoms, 1024 AOs, four million quadrature
points, and angular momentum through f for the WB97M-V geometry composition.
The generic stationary integral-descriptor fallback remains qualified through
d shells. Additional derivative numeric storage is
bounded by 1 GiB device and 2 GiB host capacity; admission can fail below the
shape limits when its conservative inventory exceeds these allowances.
`KsOptions.nonlocal_memory_budget_bytes` defaults to 1 GiB and bounds the
nonlocal provider separately. Capacity is allocated for the actual grid, not
preallocated to this limit. An explicit smaller cap remains a hard limit. The
force planner queries its native pair/seed owner's exact required capacity and
includes it with all other live owners under the total force allowances above.
Reducing the AO tile cannot reduce this full-grid pair/seed allocation.
The composite force planner prefers 1024 AO points per tile and tries smaller
bounded tiles when either total would be exceeded. It uses the existing
cooperative Becke response when target shared memory permits, preserving the
scalar response otherwise. The force report records the selected tile size,
tile count and planned geometry/Becke concurrency. Tiling leaves the discrete
grid, density cutoff and FP64 response equations unchanged.
These bounds exclude existing SCF state, compiler processes, CUDA modules and
driver-managed recurrence stacks. Host work includes snapshot validation and
exports, tiling, and total-density/VV10-active-domain packing. The VV10 cutoff
is `rho >= 1e-10` on both pair legs, using the same active-branch convention as
SCF.

Matched GPU4PySCF comparisons must configure the same VV10 density mask in both
SCF and analytic forces. Native MolecularV1 and GPU4PySCF 1.8.1 both use
`rho >= 1e-10`; PySCF CPU uses `rho >= 1e-8`. GPU4PySCF additionally screens
`|weight| <= 1e-14`, while native consumers retain nonzero signed weights and
their derivatives. Matching the density cutoff and grid coordinates therefore
does not match every screening operation. The live OMol25 runner binds the
native density policy in its protocol and records the comparator's separate
weight screening. Historical receipts retain their original density policy;
the `1e-8 Eh` / `1e-7 Eh/Bohr` acceptance gates are unchanged. See the
[OMol25 protocol](../../benchmarks/results/omol25-wb97mv-20261001/README.md).

Finite memory bounds do not imply scalable endpoint work: the retained
generic exchange/derivative provider and quadratic VV10 pairs remain material
performance costs.

## Reproduce acceptance and timing

Run real-GPU commands through the local Slurm partition, preserving its device
visibility. Point the Python package and `GENERATIVEQC_LIBRARY` at the same checkout
and its Release CUDA build. The opt-in tests compare independent GPU4PySCF
energies and grid-responsive analytic forces, reconverged energy finite
differences, warm replay, geometry rebuild and failed-neighbor isolation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=02:30:00 \
  env GENERATIVEQC_TEST_WB97MV_CUDA=1 PYTHONPATH=python:. \
  python -m pytest tests/python/test_wb97mv_complete_cuda.py -q
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:15:00 \
  build/generativeqc_cuda_fock_provider_tests --range-response-only
```

The benchmark uses the HF README's water geometries, 3/6/12/24/48/96 atoms,
spherical def2-SVP, and three interleaved synchronized complete warm SCF plus
analytic-force repeats. Each engine restarts from its own fixed post-cold
density. Both use the explicitly stated moving atomic quadrature, including
partition response. Default quadrature is 48 radial × 16 polar × 32 azimuthal
points per atom. Every paired result must satisfy `|dE| <= 1e-8 Eh` and
`max|dF| <= 1e-7 Eh/Bohr`; cold and priming samples are also checked.

```bash
export README_BENCHMARK_PYTHON=/path/to/benchmark-env/bin/python
export GENERATIVEQC_LIBRARY=$PWD/build/libgenerativeqc.so
export README_BENCHMARK_OUTPUT=$PWD/.artifacts/readme-wb97mv
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:35:00 \
  bash benchmarks/run_readme_benchmarks.sh wb97mv
python tools/render_readme_wb97mv.py \
  --raw-directory .artifacts/readme-wb97mv \
  --destination .artifacts/readme-wb97mv-figures
```

The `wb97mv-reference` runner group measures GPU4PySCF independently when a
native failure or timeout prevents a paired result. `README_BENCHMARK_POINT_TIMEOUT`
sets the finite per-point limit (default 900 seconds). Failed, incomplete and
timed-out points remain in the retained evidence and never become accepted
timings or speedup claims. Use the retained result records, rather than mere
API admission, to assess qualification at a particular size.
