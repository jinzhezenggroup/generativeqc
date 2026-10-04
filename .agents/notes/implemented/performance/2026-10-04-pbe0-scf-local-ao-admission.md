# Decision: qualify local SCF AO maps for exact-direct RKS-PBE0

Status: implemented, explicit opt-in; complete endpoint qualification pending
Date: 2026-10-04

## Problem and decision

Native SCF already has compiler-owned local density and potential contractions,
geometry-bound AO discovery, resource accounting and a bounded dense fallback.
Its ordinary admission admitted WB97M-V only. PBE0 still contracted every grid
block against every AO, even when its basis tails were negligible there.

Extend `GENERATIVEQC_CUDA_KS_ACTIVE_AO=1` to all-electron, exact-direct,
restricted PBE0 in device-fused FP64 execution. The predicate checks the PBE
semilocal family, exchange/correlation scales 0.75/1, full-range exchange
coefficient -0.125, and absence of DF, ECP, range and nonlocal corrections.
WB97M-V admission and the dense default are unchanged. This is not admission
for arbitrary PBE hybrids or unrestricted PBE0 SCF.

The existing owner discovers value/first-derivative AO maps at cutoff 1e-16,
once per prepared geometry/grid, and reuses them for XC builds. It gathers
the density submatrix and scatters local potential contributions through the
existing generated kernels. No XC algebra, precision, force AO maps, Becke
response, J/K screening or SCF convergence policy is changed here. Existing
host/device budget declines retain dense execution; selection is observable
through the existing AO diagnostic, rather than presumed from the switch.

Matrix work changes from `G M^2` to `sum_b G_b m_b^2`, with one-time
`O(G M)` discovery and geometry-bound CSR storage. These are work proxies,
not FLOP counts or a claim of a different asymptotic rate for every molecule.
Discovery and map storage must be counted again after a geometry rebuild.

## Evidence and scope

The motivating experiment is **not a build of this admission patch**. It used
the frozen #1841 library at `5de22fd9a377f43536a31866456f08c2187b9752`, source
identity `18f397771932744c7c438e2469241c024b09deb9d44a3a81167d294b6cd7c602`,
library SHA256 `ec184b64996c856932fe1d9401cda4c38b1a42fcdd20e625aa41a8bbcc09e81c`.
The probe called `CudaXcPlan` directly, bypassing SCF admission. n1 RTX 5090,
finite Slurm job 5745, 96 atoms, 768 spherical AOs, def2-SVP, 2,359,296 grid
points, 256-point tiles, scaled-PBE semilocal E/V only, fixed resident density.

Three alternating pairs after warmups gave dense/local medians
10.212886419 / 2.300512154 seconds. These include XC submission and E/V
readback, not geometry setup, SCF, J/K or forces. Energy disagreement was zero;
maximum potential disagreement was 4.440892098500626e-16. Eight independent
small CPU integration gates covered Cartesian/spherical, RKS/UKS component
execution, cutoff 1e-16 and entirely empty maps. UKS component evidence does
not qualify an unrestricted SCF trajectory.

Actual selected/dense matrix work proxies were 75,674,112,000 /
1,391,569,403,904. There were 9,216 tiles, 1,460,424 total active columns,
768 empty tiles and a maximum of 536 active AOs. Selected setup cost
1.899793115 seconds, including 1.898161623 seconds of discovery. The selected
plan reserved 69,340,448 device bytes and charged 56,699,912 host peak bytes;
these are whole-plan/discovery bounds, not additional bytes versus dense.

Raw receipts remain under
`/data/jzzeng/qc-pbe0-force-compact-pages-20261004/.artifacts/xc-local-5745/`
on n1 and its retained local copy. None of these timings is latest-master
complete E/F evidence. The production admission patch starts at master
`daa932853e37b742008b54cc6e348484f9ad7034`.

## Acceptance and rejected alternatives

The host capsule compiles the actual admission block with synthetic owner
facts; it tests policy only, not CUDA concurrency or chemistry. The native
`--pbe0-local-ao` cases independently integrate full-AO scaled PBE on CPU,
including off-diagonal positive densities, through-f AO tails, empty maps,
host-budget decline and output canaries. New-source GPU qualification, then
same-source/GPU cold, warm and changed-geometry complete E/F comparisons, are
required. Every call must satisfy the README energy/force gates (1e-8 Eh and
1e-7 Eh/Bohr); record retries, final builds and selection diagnostics, not just
the final successful iteration count. Discovery belongs in cold/moved timing.

Do not promote the switch from component timing. Do not multiply this result
by #1830 or #1833 speedups: those modify force execution, and their composition
requires fresh complete-endpoint evidence. Do not use this work to declare the
generic force consumer solved; #1841's compact schedule regressed despite a
smaller linked stack. It remains default-off while separate recurrence work
addresses that larger hotspot.

## New-source component qualification

The standalone native XC executable built from `c99e524b3` passed both
`--pbe0-local-ao` and the adjacent `--ao-discovery` gates on n1 RTX 5090 in
finite Slurm job 5749. The former includes the positive off-diagonal density
cases added by this patch. Compute Sanitizer memcheck and synccheck repeated
that gate with zero errors. Executable SHA256:
`c54f4cdb7a6ccf1dde4d3b5aaf5ac83f419f8b2621796196717070ca93ddf670`.
Raw receipts are `.artifacts/component-5749/` in the local/remote
`qc-pbe0-scf-local-ao-20261004` checkout. Documentation-only follow-ups do not
change the production source identity
`7f0770695009fee99ed7a4b08f55f1367eb52d81fbc204bf065aa47b26511358`.
This standalone executable does not exercise `CudaKsPlan` admission or a
complete SCF trajectory. Those still require the production-library endpoint
campaign; no new complete endpoint speedup is claimed here.
