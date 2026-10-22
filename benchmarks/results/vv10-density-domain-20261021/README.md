# Default VV10 density-domain qualification

Source `bb25db89e` aligns the molecular density cutoff with GPU4PySCF 1.8.1:
rho >= 1e-10 for energy, potential and force consumers. Native retains nonzero
signed weights; GPU4PySCF additionally drops abs(weight) <= 1e-14. Raw kernels and
historical receipts are unchanged. CPU PySCF retains its independent 1e-8 cutoff.

133 host tests and 39 Slurm GPU force-seed tests pass. Standard 3/12/48 and Fast 12
complete cold + three warm endpoints, all passing unchanged observable gates
against stable XC 5/6 and VV10 4/5 references with quarter-gate CPU/GPU spots.
See `endpoint-checks.json`; original numerical receipts, source/binary/cache
identities and launchers are losslessly retained in `receipts.json.xz`.

Standard 48 warm median is 7.035 s. This branch excludes the separate Fast warm
fix and does not establish all-method performance or universal Fast speedup.
