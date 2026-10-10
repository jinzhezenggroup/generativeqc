# WB97M-V / complete def2-TZVPD cold-96 investigation

The compact `evidence.json` separates completed partial CUDA source actions,
right-censored complete-endpoint attempts, and independent small E+F smokes.
It is not a README speed claim or a passed 96-atom numerical qualification.

The scientific case is the README water-96 proxy, not actual OMol25 molecules
or the ORCA integration grid. Numerical and work-count interpretation, source
boundaries, rejected shortcuts and the next route are documented in
[the proposed Agent Note](../../../.agents/notes/proposed/2026-10-10-wb97mv-tzvpd-cold96-source-reuse.md).

## Retained raw evidence

The node1 directories are:

- `/data/jzzeng/wb97m-cold96-master-20261010-15bc69700`
- `/data/jzzeng/wb97m-cold96-master-20261010-cd0eb059f`

Corresponding ignored local bundles are retained under `.artifacts/` with the
same basenames. Each remote directory contains the frozen master archive,
`env.sh`, `build.sh`, `cold.py`, `first_iteration.py`, `launch_observer.cpp`,
`reduce_profile.py`, and `results/`. The later directory also contains the
finite `latest-gpu.sh` validation recipe. Raw reports, SQLite exports, compiler
commands, cache statistics, journals and launch logs remain outside tracked
benchmark evidence, in accordance with the repository artifact policy.

The core library is Release/sm_120 with a verified ccache launcher and reused
shared cache. These runs do not build optional separate force AOT manifests
and do not clear persistent runtime JIT artifacts. A fresh owner/density
defines cold; imports, CUDA context, Calculator construction, JSON serialization
and owner destruction are outside the recorded endpoint timer.

## Reproduction

Run real GPU commands only through finite Slurm allocations on an allowed
node. For the frozen initial revision, from node1:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=32G --time=00:45:00 bash -lc '
  source /data/jzzeng/wb97m-cold96-master-20261010-15bc69700/env.sh
  "$PYTHON" "$ROOT/cold.py" native --output "$ROOT/results/native-repeat.json"
  '
```

Use `reference` instead of `native` for the independent GPU4PySCF arm.
The retained wrapper supplies an explicit 4 GiB incremental force budget;
this is a diagnostic resource admission, not a production default change.

`latest-gpu.sh` reproduces the later revision's all-row census, completed
first-three-launch capture, and small independent default/shared-route gates
under a single finite allocation. Its interposer exits only after the selected
completed activities or row-prefix reads. **COUNT_ONLY and SELECT intentionally
omit scientific source work:** their energies/forces must never be accepted.
Prefix-domain entries do not measure primitive/radial evaluations or FLOPs.

The complete 96-atom arms timed out at 45 minutes. Their endpoint timing,
iterations, Fock counts, independent numerical errors and native/reference
ratio are unavailable, not zero. Small smokes and isolated kernel ratios do
not fill those missing fields.
