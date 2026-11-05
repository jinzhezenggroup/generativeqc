# Installed CPU full-range RKS scientific qualification

This capsule binds the public PBE0/B3LYP admission change to independent
matched-Hamiltonian tests and the actual installed spherical Ni endpoints.
It is scientific qualification, not a cold-start or production-scale speedup.
The machine-readable receipts, not a green general CI run, decide acceptance.

## Accepted primary results

`acceptance.json` independently replays the actual arrays from the completed
installed campaign; `installed-endpoints.json` retains every real column,
source inventory, solve residual, SCF count and endpoint measurement.

| Gate | Actual result |
| --- | --- |
| Complete raw matrix | 15/15 actual columns; no reconstruction or symmetrization |
| Independent raw matrix, three steps | `2.198280826481991e-6`, `5.500142754577553e-7`, `1.9837600140348854e-7` |
| Independent normalized gradient HVP, maximum | `2.6253144744138862e-8` |
| Independent six-group source error, maximum | `2.5684420900325615e-6` |
| Raw symmetry / translation | `2.2655287368333887e-11` / `4.415837378916876e-11` |
| Native normalized matrix/HVP, two directions | `1.6630080382786077e-11` / `1.6975789528092713e-11` |
| Native bilinear / rotation covariance | `3.47687672162181e-14` / `6.095827597110042e-14` |
| Maximum matrix response residual | `2.0991430929893723e-10` |
| Full SCF + response + raw assembly | `11503.235189544968 s`; 97 SCF iterations, 113 Fock builds, 15 nuclear solves |
| Sampled simultaneous matrix process-tree RSS | `1105362944 bytes` |
| Two additional complete public HVP calls | `741.0083797601983 s` / `739.6329485243186 s` |

Errors use Hartree/Bohr² with the normalization defined by the producers.
These are warm-artifact endpoint measurements, not cold-start timings. The
explicit compiler preparation and interrupted cold campaign remain separate.
The primary default-grid HVP has maximum independent error
`2.1822929249761588e-8` at all 59872 points; `grid-endpoints.json` retains its
full time, memory, resource refusal and rotated-action receipt.

## Domain and frozen gates

- CPU, direct, all-electron, restricted FP64; Cartesian or real-spherical AOs
  through f. The physical semilocal graph is PBE0 or Gaussian/VWN-RPA B3LYP,
  with one admitted full-range exchange fraction. No named-functional Hessian
  equations, native density seeds for the oracle, or production reference engines.
- The independent light-element tests cover H2 and asymmetric HeH+, canonical
  and 37/100 exchange, full matrices at three energy steps, analytic-gradient
  differences at three directional steps, frozen K nuclear terms and six
  independent gradient groups. Matrix/component/normalized-HVP gate: `1e-5`
  Hartree/Bohr²; directional energy-curvature gate: `1e-6`.
- NiH4(2−) uses the pinned external real-spherical def2-SVP basis: 51 AOs,
  not the 56-AO Cartesian replacement. The explicit qualification grid has
  5760 points. All 15 actual public matrix columns must complete.
- Independent Ni raw-matrix/action/component gate: `1e-5` Hartree/Bohr².
  Native raw symmetry, translations, normalized H@v and bilinear gate: `2e-7`.
  Actual response residual gate: `1e-9`. No post-hoc symmetrization, inferred
  missing columns or tolerance changes are permitted.
- The default grid retains all 59872 points. The 128-MiB AO/grid lease must
  reject explicitly; the positive test requests a separate 4-GiB lease.
  Default-grid references use Treutler's Bragg-radius partition adjustment,
  not the equal-radius prescription used by the explicit fixture.
- Finite-grid rotation covariance uses a pi/6 z rotation preserving the exact
  explicit quadrature set. It does not assert arbitrary orientation invariance
  or rotational zero modes at this nonstationary geometry.

## Inputs and source provenance

`provenance.json` binds the measured dirty base, production source inputs,
native library, repaired wheel, producers, references and retained results.
The installed production modules must equal the checked-in source bytes.
The wheel is an ignored local qualification artifact, not a published Release
or distribution. Its OpenBLAS dependency is bundled by normal auditwheel repair;
installed endpoint execution needs no `LD_LIBRARY_PATH` workaround.

The primary campaign measures dirty base
`72499de172f5f75a425b0dee8bff8427d7cc061e`, not an invented final commit.
While it ran, the PR fast-forwarded to upstream merge
`021ad2ff92076060ec2dea22522146cd6230bead`. Its native delta affects CUDA
Direct replay, the CUDA DF cold seed for at least 512 AOs, and an opt-in
validation-provider preference whose previous CPU default remains false.
The three changed production Python modules and the scientific JIT owners
remain byte-identical. The earlier native library/wheel identities and their
measurements are not relabeled as this later native build.

`integration.json` binds the separate fresh integrated wheel, 445 passing
installed-runtime/native-contract cases, eight explicitly Slurm-only GPU skips, independent
Ni public action/component checks, unchanged default-grid/resource checks,
native rotation covariance and parity with the actual earlier full matrix.
`integrated-action.json` and `integrated-grid.json` retain the fresh endpoint
measurements separately. The integrated action's first cold invocation returned
no result within its 3600-second guard (status 124); its full-call retry keeps
that same guard, source, budgets and tolerances and reuses retained artifacts.
This is not a cold-start success or speedup claim.
The additional validation-product/Huckel probes compile current checked-in CPU
contract headers; they are not misrepresented as public molecular endpoint calls.

Reference numerical arrays are retained without pickle/object arrays under
`tests/data/global_hybrid/ni-second-order`. The
original independent reference producers and their bounded retry policy are
retained for audit, including the distinction between native comparison
densities and independently converged oracle guesses. The fresh independent
center separately retains its actual density and stationarity residual.

## Reproduce

Build a CPU wheel with a verified `sccache` (or supported `ccache`) launcher,
CUDA/tests/CLI disabled, strict Release flags and an LP64 OpenBLAS provider.
Install the repaired wheel in an isolated prefix. Keep the repository's
`python/` directory out of `PYTHONPATH` for installed validation. The reference
environment uses PySCF 2.14.0 / Libxc 7.0.0; production does not import it.
The qualification environment also needs NumPy, SciPy, psutil and pytest;
these are experiment/test prerequisites, not a production oracle dependency.

```bash
sccache --version
ROOT="$PWD"
PREFIX=/path/to/installed-venv
CAPSULE="$ROOT/benchmarks/results/global-hybrid-public-20261105"
REFERENCES="$ROOT/tests/data/global_hybrid/ni-second-order"
RUN=/path/to/new-ignored-run
CACHE=/path/to/existing-jit-cache
mkdir -p "$RUN/references"
cp "$REFERENCES"/reference-*.npz "$RUN/references/"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 14400 \
  "$PREFIX/bin/python" -u tools/qualify_global_hybrid_ni.py endpoint \
  --basis-dir "$ROOT/tests/data/global_hybrid" \
  --references "$RUN/references" --output "$RUN/ni-installed" \
  --cache "$CACHE" --installed-prefix "$PREFIX"

for mode in reference endpoint; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 7200 \
    "$PREFIX/bin/python" -u tools/qualify_global_hybrid_ni_grid.py "$mode" \
    --basis-dir "$ROOT/tests/data/global_hybrid" \
    --references "$RUN/references" --output "$RUN/ni-grid" \
    --cache "$CACHE" --installed-prefix "$PREFIX"
done
```

The integrated producer samples summed simultaneous endpoint process-tree RSS
at 50 ms, including its own compiler descendants. This conservative sum can
double-count shared pages and miss short spikes; it is not unique physical
allocation or a provider capacity. Explicit sibling-process compiler prewarming
is recorded separately and is not included in that process-tree peak. Its
cost cannot be described as a free cold-endpoint cache hit.

`observed-action.json` records one additional fresh installed public HVP with
solver-owned operator/preconditioner action counters and independent component
checks. Its observer returns the original response object without replacing
any solve or physics. The saved native baseline also checks rotation covariance
against the separately executed rotated public action. These counters are not
an inferred multiple of iterations or a claim of observing every raw-matrix
matvec. The matrix receipt separately retains every column's actual solve,
iteration, residual and source inventory, along with complete endpoint SCF counts.

## Limits and negative evidence

Prior three-hour standalone and four-hour integrated scalar-matrix attempts
stopped after 14/15 columns. They are failures, not reconstructed full matrices.
The earlier 4102.21-second HVP is not the full Hessian and its mixed reference/
native process RSS is not an isolated installed endpoint measurement.
The first installed cold campaign was explicitly stopped after 3822 seconds
with one accepted column; it is not a completed endpoint. `preparation.json`
records 218 successful artifact validations across 52 classes with eight
workers in 199.8622414013371 seconds. The subsequent public matrix starts
again from column zero, retaining the original four-hour guard. Compiler
identities, optimization flags, leases, production budgets and numerical gates
are unchanged. The producer now checkpoints completed endpoint timing and memory
before starting the next independent HVP, so a later interruption cannot erase
an already completed matrix receipt.
The default-grid oracle's initial equal-radius trial compared different Becke
partitions and failed density/energy matching; fixing the reference prescription
must precede acceptance, not a looser matching gate.

The 25–40-atom installed Ni gate remains explicitly unresolved under #2843.
UKS, ECP, DF, D3(BJ), native TS optimization and frequencies are separate domains.
Through-f Hessian-source evidence does not extend a distinct first-order force
consumer beyond its own qualified angular domain. No issue is closed merely
because the public method selector exists.
