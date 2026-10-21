# Default auxiliary-g response qualification

Native source `33461b49e` and binary `a85e3399...` pass 34 Slurm GPU tests and
five complete PBE/PBE0/r2SCAN-3c DF populations. See `endpoint-checks.json` for
all-repeat errors, dense-reference stability, CPU spots and baseline comparisons.
`receipts.json.xz` retains original inputs/results, source identities, launchers,
failed packaging attempt, cache receipts and executed response traces.

PBE DF96 warm decreases from 129.270 to 29.708 seconds; PBE0 from 255.253 to
56.309 seconds. These are noninterleaved diagnostic populations. They establish
neither universal superiority to GPU4PySCF nor a resolved r2SCAN-3c DF96 or
WB97M-V DF96 work/resource limit. Cold includes construction and preparation;
warm is the median of three frozen engine-local cold-density E+F calls.
