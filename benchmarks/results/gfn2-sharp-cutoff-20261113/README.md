# GFN2 sharp-cutoff oracle reconciliation

This bundle explains and reproduces the historical #560 768-AO discrepancy.
It does **not** change production science, widen tolerances, or assert complete
current-source product qualification.

## Scientific contract

The pinned native GFN2 parameters select sharp D4 cutoffs: two-body 50 bohr,
ATM 25 bohr. tblite 0.6.0 defaults to a 0.05-bohr dispersion cutoff switch.
Its constructor must receive `xtb_config={"smooth_cutoff": 0.0}` for a matched
independent comparison. Both references otherwise use accuracy `1e-4`, 300 K,
maximum 300 SCC iterations, and damping 0.4. The energy/force gates remain
`5e-7` Eh / Eh/bohr.

The original water-128 geometry's native ATM energy is
`0.023909817914186865` Eh, versus default tblite's `0.023908544763071` Eh.
This explains the `1.2731511e-6` Eh total-energy discrepancy without changing
SCC. Explicit sharp and default tblite atomic charges are identical. A compact
nine-atom sharp/smoothed control is retained separately in
`tests/data/gfn2_sharp_cutoff_tblite.json`; the force difference exceeds `5e-7`.

## Source-bound evidence

- Production commit: `dc68eb0b2b496de18fd765bda577d836487b2883`.
- Original campaign: n2 Slurm 2432, 2026-10-05/06. All original input and result
  bytes are retained compressed, including the mismatched original oracle.
- Corrected oracle: fresh independent tblite 0.6.0 calculations on November 13,
  2026, using the **exact original and sine-displaced geometries**. This is not
  the newer mixer's different moved-geometry recipe.
- Fresh same-binary replay: n2 Slurm 4456, one `gpu:pro6000:1`, `main` partition,
  finite ten-minute allocation, unchanged Slurm device visibility. Actual device:
  NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
  UUID `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, driver `595.99.02`.
- The replay executes native energy-only and complete host-returned energy/force
  endpoints at 48/192/768 AOs, each with cold, five retained-workspace/FRESH SCC,
  moved, and five moved/FRESH SCC calls. It uses the original frozen library,
  not a relabeled current-source library.

All **136/136 historical observations**, including **36/36 formerly failing
768-AO observations**, pass the matched independent gates. All **72/72 fresh
native replay calls** also pass. Across both sets, maximum energy error is
`1.1368684e-12` Eh and maximum force error is `4.5232191e-8` Eh/bohr. Native-only
maximum force error is `2.2218043e-8` Eh/bohr. This includes the original smaller
domains and seven small fixtures; peer agreement alone is not the gate.

Raw endpoint times, iteration counts, result fields, host force arrays, input
identities and binary hashes remain in the archives. No new speedup or measured
allocator peak is claimed. Semantic work is a complete fresh SCC endpoint with
recorded iterations, not an isolated kernel or an inferred inner-stage cost.

## Reproduction

Run the standard-library-only integrity and scientific reducer:

```sh
python benchmarks/results/gfn2-sharp-cutoff-20261113/reconcile.py
```

To regenerate independent references, install the pinned optional oracle and
write to a new file; the archived references are not overwritten:

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python \
  benchmarks/results/gfn2-sharp-cutoff-20261113/regenerate.py /tmp/matched-gfn2.json
```

`provenance.json` records archive hashes, oracle-library identity and settings.
The ordinary maintained qualification procedure now records and enforces the
model configuration; historical oracle files without that verified configuration
must be regenerated, not silently reclassified.

## Scope at initial collection

The initial current-source CPU nine-atom cutoff regression and molecular oracle,
force and lifecycle tests passed. The CPU 768-AO original-geometry probe
did not converge within 300 iterations; it remains a recorded failure, not a
scientific pass. A separate history-32 control converged the original geometry
in 131 iterations, but its moved calculation still exhausted the 300-iteration
budget; it does not qualify the full protocol. The full current-source CUDA
rebuild was interrupted and supplies no passing device evidence.
Those observations alone did not qualify current-source large CPU/CUDA
admission, complete lifecycle or memory behavior.

The subsequent [November 14 current-source campaign](../gfn2-current-source-20261114/README.md)
qualifies the restricted FRESH CPU/CUDA endpoints with separately retained
source, binary, controls, real-device lifecycle and memory/sanitizer receipts.
Its CPU history-64 qualification does not relabel either earlier history-8/32
failure as a success. These original archives and their provenance stay intact.
