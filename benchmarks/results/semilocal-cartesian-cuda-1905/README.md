# Direct CUDA LDA/PBE RKS Hessian qualification (#1905)

`matched.jsonl.xz` records the whole-endpoint gate against the historical mixed
route: CPU strict SCF and stationary response with CUDA first- and second-order
integral providers. The native route uses CUDA for both SCF and the analytic
derivative. Each phase includes preparation, converged SCF, derivative work,
and publication. Warm and moved phases reuse the process and compilation cache,
not the converged density. Cold mixed timings include JIT.

The source base is `aff6ad925831960ccec147ccf49ec022b88ba0a2` (identical tree to
`3e6d07a5cbac7c8da39234cb272b3cf879026c91`) with the #1905 implementation patch.
The final `libgenerativeqc.so.0.1.0` SHA-256 was
`ac6c224f867677d4ac79336f95bc67d814ab1cc82c1f13392002d929772e81cf`.
All timed runs used the same RTX 5090 in `node4`, strict FP64, Cartesian STO-3G,
Direct integrals, and the fixed grid and GMRES controls in the
[water-dimer acceptance test](../../../tests/python/test_semilocal_cartesian_hessian_cuda.py).
The dense fixture is its first water molecule, and the HVP fixture is the full
asymmetric water dimer. The moved geometry displaces the coordinates by
`0.01` Bohr along the normalized, zero-translation seed-42 direction; HVPs use
the seed-1905 direction in all three phases.

Each row is one matched comparison. All 12 rows passed both the `1.05×`
wall-time limit and the `2e-8 Ha/Bohr²` maximum raw derivative difference.
Work fields report the native owner's executed directions, induced Fock
actions, primitive components and mixed grid contractions. The timings do not
claim a speedup over other hardware, grid sizes, or method domains. A separate
native-to-native comparison follows.

## Against the #3132 native CUDA endpoint

Slurm `1808` alternated the order of nine fresh process pairs per method and
endpoint on the same `node4` RTX 5090 (driver `580.178.04`). Both arms used
the same Python 3.12.12 environment, OpenBLAS, CUDA runtime, geometry, grid and
SCF/response controls. `native-ab.jsonl.xz` retains all 108 phase pairs with
matched semantic work. The baseline is the Release SM120 build from
`e495a8c8bf106b37d11c112d89c6d991659d4381`; its production code and build
files are identical to `aff6ad925831960ccec147ccf49ec022b88ba0a2`, with only
tests and provenance records differing. Its library SHA-256 is
`ba45c62432914aed3f77a2138d6999541ae78802d555209d5917b8345970f796`.
The candidate is the qualified library listed above. These are native
energy-only SCF preparations: they don't exercise the extra stationary force
profiles linked in the candidate or measure the force-to-Hessian borrow.

| Endpoint | Method | Median paired change (cold / warm / moved) |
| --- | --- | ---: |
| Dense | LDA | −1.85% / −5.23% / −5.60% |
| HVP | LDA | −0.66% / −2.49% / −2.52% |
| Dense | PBE | −1.22% / −4.68% / −4.98% |
| HVP | PBE | −0.25% / −1.77% / −1.35% |

The gate uses the median of the matched candidate/baseline time ratios
for each phase, not the ratio between separately ordered medians. No phase
regressed by 5%. Dense outputs were elementwise equal on these water calls;
the largest water-dimer HVP difference was `6.67e-15 Ha/Bohr²`. The receipts
verify equal completed directions, CPKS iterations, cached actions, physical
checks, induced Fock actions, one-electron/Coulomb primitive components and
XC mixed grid contractions for every pair.

## Public endpoint oracle and live memory

Slurm `1807` ran both public water-dimer oracle tests and both live-ledger tests
on `node4` using the same library SHA-256 above: four passed. Errors are maximum
absolute values in Ha/Bohr²; neither the dense nor HVP endpoint symmetrizes its
output. Gradient differences use fresh strict CPU SCF at both displacements.

| Test | LDA RKS | PBE RKS |
| --- | ---: | ---: |
| HVP vs CPU analytic | 2.5071056342085285e-12 | 1.6370238498097933e-12 |
| Raw dense symmetry | 3.143728333210305e-12 | 6.3106742054230835e-12 |
| Raw dense action vs HVP | 7.74432601380326e-13 | 1.5623613514037515e-12 |
| Raw bilinear identity | 6.588896095394148e-13 | 5.652978085635141e-13 |
| Reconverged gradient difference at 1.2e-3 Bohr | 6.689083490996062e-07 | 6.360903390678629e-07 |
| Reconverged gradient difference at 1.3e-4 Bohr | 7.635574672537615e-09 | 7.381758149183071e-09 |
| HVP ledger peak (bytes) | 77,073,749 | 77,172,053 |
| Dense ledger peak (bytes) | 77,122,741 | 77,221,045 |

The peak includes the concurrently live prepared SCF allocation on the same
ledger. The first rerun (`1806`) stopped at native-library loading because the
OpenBLAS directory was not on the loader path; it produced no numerical sample.
The qualified rerun used the benchmark's Python 3.12.12 environment and pinned
OpenBLAS and NVIDIA libraries. Its JUnit XML SHA-256 is
`fe0883dbaf4dee2f6abc2503427dec8c3d85ec802ed4e278727ff7a77b0b43a6`.

## Final source-format rebuild

The run data above retain their original binary identities. After repository
formatting, the Release SM120 rebuild reports native source identity
`88242d04b3daafa1d750a74a234f3d9045fd70ac8b18d67bf7c8e152ad29019a` and library
SHA-256 `2119434afed9e80d269466cbdf6d2797504686653eccc09a626f3dc785ae4a5e`.
Slurm `1809` checked the live ledger and force-center leases on the new
binary, then compared the twelve cold/warm/moved raw outputs to all nine
pre-format repeats. Dense outputs were identical and HVP maximum drift was
`7.55e-15 Ha/Bohr²`, comparable to pre-format repeat spread. Completed
directions, CPKS iterations, exact primitive counts and XC contraction counts
also matched. This was a source-format parity check, not a new timing cohort.

Slurm `1811` then ran all four public water-dimer tests directly on this final
library: four passed, none skipped. Across LDA and PBE, the HVP error versus
the strict CPU analytic owner was at most `2.52e-12 Ha/Bohr²`; the independent
reconverged gradient differences were at most `6.69e-7` and `7.64e-9 Ha/Bohr²`
at `1.2e-3` and `1.3e-4 Bohr`, respectively. The raw bilinear error was at
most `6.56e-13 Ha/Bohr²` and raw dense asymmetry at most `6.32e-12 Ha/Bohr²`.
The concurrent device-ledger peaks matched the four values above. The JUnit
XML SHA-256 is `b45ed9e0a64ae89147515681d03f011a4b7bbde44274370e2fa7897d728f4de9`.
The first attempt (`1810`) stopped during collection because the staged Python
source omitted the public method manifest; it produced no physics result.

## Transfers on the final binary

Slurm `2358` captured the native response call after accepted SCF for LDA and
PBE on the same RTX 5090 water dimer. Each capture includes response setup,
all graph executions and result publication, but not SCF. Nsight Systems
2025.2.1.0 CUPTI memcpy events and a separate CUDA Runtime interposition run
agreed exactly on host transfer counts and bytes for each method and endpoint.

| Endpoint (both methods) | H2D copies / bytes | D2H copies / bytes | Published result (bytes) | CUDA graph executions | Host copies overlapping a graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| HVP | 37 / 8,293,444 | 32 / 492 | 144 | 21 | 0 |
| Dense (18 directions) | 36 / 8,293,300 | 26 / 2,956 | 2,592 | 378 | 0 |

All H2D copies occurred before the first graph. The additional 144 B for HVP
is the input direction. Outside the requested result, D2H was bounded status,
audit and diagnostic data: 228 B before the first graph for either endpoint,
84 B (HVP) or 8 B (dense) between graph executions, and 36 B (HVP) or 128 B
(dense) after the last graph. There were no host transfers inside the
device-controlled Krylov graph and no downloads of intermediate scientific
tensors. The traces do not imply zero host synchronization around the solve,
nor are transfer bytes a measure of simultaneously live storage.

## Reproduce

The retained JSONL receipts are losslessly compressed; inspect them with
`xz -dc`.

Use a Release CUDA 12.9+ build with `GENERATIVEQC_ENABLE_HESSIAN_AOT_LEAVES=ON`
and an inventory matching the basis. To also qualify the production force-table
borrow, link both `lda_rks` and `pbe_rks` in `GENERATIVEQC_STATIONARY_AOT_PROFILES`.
Install the native runtime dependencies and the compiler dependencies for the
mixed route. Set `GENERATIVEQC_LIBRARY` to the absolute path to that build.
Verify a supported JIT compiler cache before running the mixed arm.

From the checkout root, run each arm in a fresh process and fresh output
directory. Use the same reserved GPU and pinned environment for all arms;
adapt the node name and GPU resource to the scheduler.

```sh
export PYTHONPATH="$PWD/python:$PWD/tests/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
driver=benchmarks/semilocal_cartesian_hessian_endpoint.py
out=$(mktemp -d)

# Run the public endpoints, independent CPU oracle, and live-device ledger gates.
srun --partition=main --nodelist=node4 --gres=gpu:5090:1 --nodes=1 \
  --ntasks=1 --cpus-per-task=2 --time=00:22:00 \
  env GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST=1 python -m pytest \
  -o junit_family=legacy --junitxml="$out/public-oracles.xml" \
  tests/python/test_semilocal_cartesian_hessian_cuda.py || exit 1

for method in lda-rks pbe-rks; do
  for endpoint in hvp dense; do
    fixture=water-dimer
    if [ "$endpoint" = dense ]; then fixture=water; fi
    for arm in mixed native; do
      run="$out/$method-$endpoint-$arm"
      srun --partition=main --nodelist=node4 --gres=gpu:5090:1 --nodes=1 \
        --ntasks=1 --time=01:30:00 python "$driver" --method "$method" \
        --endpoint "$endpoint" --fixture "$fixture" --arm "$arm" \
        --directory "$run" >"$run.jsonl" || exit 1
    done
  done
done
```

Compare `cold.npy`, `warm.npy` and `moved.npy` against the same phase of the
other arm. Compare the paired `wall_seconds` from the JSONL receipts, preserving
the work and transfer diagnostics; do not compare only kernel time.

For native-to-native comparisons, call the same driver with `--arm native`
for both libraries. Alternate the library order in one reserved GPU job,
passing each via `GENERATIVEQC_LIBRARY` in its fresh process. Use paths
`pair-N/METHOD-ENDPOINT-baseline` and `pair-N/METHOD-ENDPOINT-candidate`
with each corresponding `.jsonl` adjacent to its result directory. Run
`python benchmarks/semilocal_cartesian_hessian_compare.py <output-directory>`
to check the paired gates and emit the combined receipts.

To reproduce the transfer census, use Nsight Systems with whole-graph CUDA
tracing enabled. `--profile-response` captures only the first (cold) response,
including its setup and publication. From the same environment as above:

```sh
for method in lda-rks pbe-rks; do
  for endpoint in hvp dense; do
    run="$out/$method-$endpoint-response"
    srun --partition=main --nodelist=node4 --gres=gpu:5090:1 --nodes=1 \
      --ntasks=1 --time=00:10:00 nsys profile --trace=cuda --sample=none \
      --cpuctxsw=none --cuda-graph-trace=graph --capture-range=cudaProfilerApi \
      --capture-range-end=stop --output="$run" python "$driver" --arm native \
      --method "$method" --endpoint "$endpoint" --fixture water-dimer \
      --profile-response --directory "$run" || exit 1
    nsys export --type sqlite --output="$run.sqlite" "$run.nsys-rep" || exit 1
    sqlite3 -readonly "$run.sqlite" \
      <benchmarks/semilocal_cartesian_hessian_transfers.sql
  done
done
```

The [Agent Note](../../../.agents/notes/proposed/2026-11-22-device-controlled-dft-cpks.md)
documents the independent scientific, transfer and live-memory gates.
