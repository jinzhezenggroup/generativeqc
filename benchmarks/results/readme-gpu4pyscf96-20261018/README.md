# GPU4PySCF ωB97M-V water-96 supplement (2026-10-18)

Only the missing **GPU4PySCF Direct/96 and DF/96** populations are measured.
No native calculation, other functional, other cluster size, primer, profiler or
sanitizer is run. This supplements the mixed-source homepage rather than
relabeling it as a benchmark of current master.

The [Direct table](timings.md) and [DF table](df-timings.md) retain the complete
homepage timing/work inventory. The new [Direct](direct-reference.json.xz) and
[DF](df-reference.json.xz) raw archives retain every force
component, scalar observation, final SCF residual, work count and source-file
identity. Historical partial/failed reference attempts remain in their original
publications; they are not pooled into the new timing populations.

## Cold and warm results

Complete energy + analytic-force times, in seconds. Warm is the median and
full range of all five calls; no SCF-work branch is discarded.

| GPU4PySCF mode | Cold E+F | Warm E+F median [min–max] | Cold SCF iterations/J-K calls | Warm SCF iterations/J-K calls |
| --- | ---: | ---: | ---: | --- |
| Direct | 4544.557 | 493.399 [399.659–587.833] | 45/46 | 1/2, 2/3, 3/4 |
| DF, exact-direct LR correction | 4576.779 | 407.708 [407.557–408.678] | 45/46 | 1/2 |

All **24 complete E+F calls** pass convergence, finite-value, force-shape and
same-geometry repeat gates. The largest repeat differences are
`4.5475e-12 Eh` and `2.4099e-11 Eh/bohr`; these test reference repeatability,
not a new native accuracy qualification. All 71 retained consumer source-file
hashes in each population match the frozen consumer commit.

Changed-geometry endpoints use the same complete E+F definition:

| GPU4PySCF mode | Moved E+F (s) | Moved-warm E+F median [min–max] (s) | Moved SCF iterations/J-K calls | Moved-warm SCF iterations/J-K calls |
| --- | ---: | ---: | ---: | --- |
| Direct | 3698.941 | 400.979 [399.534–774.996] | 36/37 | 1/2, 5/6 |
| DF, exact-direct LR correction | 3719.997 | 502.136 [407.529–502.201] | 36/37 | 1/2, 2/3 |

Both external `srun` launchers return exit code **0**. Direct runs from
14:36:44 to 18:13:24 and DF from 14:37:17 to 18:08:18 on 2026-10-18 (+08:00).
The [manifest](manifest.json) retains the initial allocation receipts and final
execution timestamps; the raw journals finish at `measured` / `complete` with
zero-exit process outcomes and no point deadline. Slurm accounting is disabled,
so no unavailable final scheduler-state receipt is invented.

## Protocol

- **Source:** unmodified benchmark consumer at
  `69c4549f260e8c46cdd69f2f59e09b0a1f7d9c4a`. This identifies the consumer, not
  a newly measured native binary. GPU4PySCF 1.8.1, PySCF 2.14.0, CuPy 13.6.0,
  cuTENSOR and CUDA LibXC 7.0.0 identities are retained with the observations.
- **Hardware:** separate Slurm-assigned RTX 5090 GPUs on n1 / scheduler node1;
  Direct job **10159**, DF job **10160**. Both use partition `main`,
  `--gres=gpu:5090:1`, eight CPU threads and 64 GiB host-memory allocations.
  Device visibility is preserved; allocation receipts identify both GPU UUIDs.
- **Inputs:** the unchanged nested 96-atom water cluster, 768 spherical
  def2-SVP AOs, FP64, and the moving unpruned `(48, 16, 32)` Becke grid.
  Self-consistent VV10 uses the same `1e-8` active-density threshold for SCF
  and analytic forces, including grid/partition response.
- **DF Hamiltonian:** explicit cc-pVDZ-JKFIT auxiliaries fit full-range J/K;
  the long-range K correction and its analytic gradient remain exact-direct.
  Default GPU4PySCF fully fitted long-range exchange is not substituted.
  The old native DF/96 12-GiB workspace hint is native-only; GPU4PySCF does
  not consume it. Every scientific protocol field is otherwise identical.
- **SCF:** unchanged full-current-density potential construction, energy
  tolerance `1e-12 Eh`, orbital-gradient tolerance `1e-10`, and direct-integral
  screening tolerance `1e-14`. No tolerance, grid, precision or repeat count
  is relaxed. Work counts are retained, not normalized between engines.
- **Sampling:** one cold, five fixed-post-cold-density warm calls, one moved
  geometry and five fixed-post-move-density warm calls per engine. Every call
  includes complete energy + analytic forces and force export to host. Cold
  includes engine construction/preparation plus the first E+F call; warm
  statistics include all five samples and their full range.
- **No process timeout:** the Python benchmark is invoked directly, without
  `timeout` or a point deadline. Per repository GPU rules, each Slurm allocation
  has an explicit finite **48-hour** resource limit. This is not subtracted
  from timings or represented as a completed run duration.

## Historical native evidence

All 64 native populations and the other 62 reference populations are unchanged.
Native observations retain their original accuracy labels,
including the qualified r²SCAN-3c Direct overlay. Native DF/48 and historical
DF/96 remain labeled unchecked; the new DF/96 independent arrays do not silently
rewrite the gates or reference bindings of an earlier native run. Native DF/96
still has its complete 4717.471-second cold call and only four of five warm
repeats: no native DF/96 warm median is invented.

All other GPU4PySCF observations retain their original source, sampling dates,
timings, SCF work and failure outcomes. These mixed-source absolute endpoint
timings are not a balanced same-revision performance experiment.

## Reproduction and audit

Prepare an immutable `source/` checkout/archive of the consumer revision above
and the retained orbital snapshot at `inputs/def2-svp.json` under a campaign
directory. `reproduce.sh` selects the existing n1 GPU Python/CUDA environment;
it runs only one of the two requested populations and exclusively reserves a new
population directory. Existing directories (including partial runs), symlinks,
logs and job receipts are never overwritten. Use a fresh campaign root for a
new attempt. No native shared library or build is required.

The manifest retains the hash of the original measured launcher. The current
launcher adds output-retention guards only; it does not relabel those historical
measurements or their consumer source.

On n1, launch the two independent GPU allocations:

```bash
set -euo pipefail
export README_GPU4PYSCF96_ROOT=/path/to/campaign
(
    set -o noclobber
    target="$README_GPU4PYSCF96_ROOT/reproduce.sh"
    [[ ! -e "$target" && ! -L "$target" ]]
    cat benchmarks/results/readme-gpu4pyscf96-20261018/reproduce.sh \
        > "$target"
)
pids=()
for mode in none cuda; do
    label=$([[ $mode == none ]] && echo direct || echo df)
    (
        set -o noclobber
        log="$README_GPU4PYSCF96_ROOT/$label.log"
        [[ ! -e "$log" && ! -L "$log" ]]
        exec > "$log" 2>&1
        exec srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
            --cpus-per-task=8 --mem=64G --time=2-00:00:00 \
            bash "$README_GPU4PYSCF96_ROOT/reproduce.sh" "$mode"
    ) &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
test "$status" -eq 0
```

Retain completed evidence and regenerate the two homepage figures/tables.
Both populations are validated and all receipt files are read before publication; an existing
archive or manifest must be byte-identical. A different attempt needs its own
publication rather than replacing these historical records:

```bash
PYTHONPATH=.:python python -m tools.render_readme_gpu4pyscf96 \
    --raw "$README_GPU4PYSCF96_ROOT" --render
```

Omit both options for a CPU-only audit of the retained archives and all endpoint
gates. The reader verifies raw checksums, force shape/finiteness, convergence,
all twelve endpoint identities, same-geometry repeat accuracy, unchanged
scientific protocols and the exact supplement scope. It never executes either
electronic-structure engine. Native historical acceptance is not recomputed
against a substituted oracle.
