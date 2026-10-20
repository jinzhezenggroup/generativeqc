# CCSD(T)/CBS convergence-only FP64 ablation

These earlier source-matched receipts isolate relaxed convergence in FP64.
They predate default FP32 DF-CCSD warmup and do not qualify the final mixed
Fast implementation. The separate [final campaign](../readme-cbs-fast-20261020/README.md)
retains the mixed implementation's paired timings and independent DF water checks.
Standard remains the default. See
[the public contract](../../../docs/user/ccsdt_cbs.md#fast-convergence) and
[the decision note](../../../.agents/notes/implemented/numerics/2026-10-20-ccsdt-cbs-fast-convergence.md).

This bundle retains complete endpoint samples, independent oracle errors,
semantic work, selected stopping limits and source/library identities. The
numerical acceptance is limited to the measured water algebraic basis pairs
and H2 DZ/TZ. Water's algebraic pairs are not physical CBS basis sequences.
The DF oracle reconverges PySCF energies and uses central finite differences;
its exact driver is embedded in `df-water-paired.json.xz`. Planar symmetry
supplies the zero y force components. Standard/Fast differences alone are not
independent oracle errors.

`evidence.json.xz` checks the recorded Fast scalar error metrics against the
published composite/component allowances. The native tests additionally check
all conventional water constituent forces and energy-only parity. `validation.json`
records test results. The caller must still check convergence; this is not a
per-call error certificate.

The n4 GPU runs used Slurm with finite limits and preserved device visibility.
CPU and CUDA Release builds used ccache 4.5.1. The measured base revision and
file hashes are in `source-manifest.json`; `qualified-source.patch.xz` reconstructs
those dirty source files from that base. No generated binary is tracked.

The complete cold call plus three replay samples per case are preserved.
Each replay recreates the native basis-group owners. These small observations
are not a statistically qualified universal speedup or default promotion.
The separate [README campaign](../readme-cbs-fast-20261020/README.md) contains
fresh matched Standard/Fast QM7 C1–C7 observations.

Reconstruct the recorded source patch before reproducing this ablation; current
DF Fast defaults differ. Conventional measurements use the matching native library:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 python benchmarks/ccsdt_cbs_endpoint.py \
  --policy fast --case water-algebra --device cuda --repeats 3 \
  --oracle --work --output .artifacts/cbs-fast-water.json
```

Repeat with `--policy standard` and `--case h2-dz-tz`. Run the independent
`tests/python/test_correlation_convergence.py` with the source-matched CPU
library; CUDA cases require `GENERATIVEQC_CBS_CUDA_TEST=1` inside Slurm.
`tests/python/test_full_df_cbs_native.py -k cbs_complete` uses
`GENERATIVEQC_FULL_DF_CUDA_TEST=1` and PySCF to qualify both fitted-reference models.

Validate the selected publication bytes with `python tools/evidence.py check`
after staging. For scripted reads, use the shared `load_publication_record`
reader from `tools.generativeqc_validation.record`.
