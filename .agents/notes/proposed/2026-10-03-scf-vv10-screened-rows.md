# Proposal: skip MolecularV1 screened VV10 rows in resident CUDA SCF

Status: proposed; complete24, changed-geometry and explicit-budget48 gates pass
Date: 2026-10-03

## Problem and decision

The resident WB97M-V SCF owner always applies the MolecularV1 density domain
before evaluating VV10, in both ordinary and graph replay paths. Inactive rows
have negative-zero weights, placeholder density one, and zero gradient. Partner
compaction already removes them from the inner dimension. The caller omitted
the existing row-mask layout flag, so these discarded outer rows still visited
every active partner and computed potentials that weighted AO assembly drops.

Enable that existing flag for this owner. Its constructor already requires
WB97M-V MolecularV1 for resident nonlocal execution. Active positive-zero and
signed nonzero quadrature weights retain the established semantics, including
negative-weight underflow protection. Invalid inputs still set the domain's
sticky failure flag; skipping their row does not turn a failed call into success.
Stationary forces already mask screened rows and are unchanged by this patch.
There is no new density threshold, pair cutoff, kernel, allocation or CPU work.

## Evidence and limits

The host probe executes the production pair arithmetic for VV10 and rVV10,
including the feature-only SCF specialization. A test-only counter inside the
actual partner loop verifies that one screened row among three points changes
executed pairs from six to four, while retaining the total energy. Active
positive/negative/zero/subnormal weights retain energies, potentials and work.
All 33 signed-weight, screened-row, underflow and finite-difference tests pass
with a verified ccache compiler wrapper. This is scalar arithmetic qualification,
not CUDA scheduling evidence.

Node1 Slurm 5361 passes all six independent complete RKS/UKS WB97M-V tests,
including STO-3G, def2-SVP, TZVP, TZVPD and reconverged displaced energies.
The first library SHA256 is
`18a24079d31aa77aefa6c3d79e4a09adca26c7cc63b32228e1578b01a21597a5`.
An intrusive 24-atom def2-SVP, grid 48 x 16 x 32 one-iteration profile takes
16.137 s, versus 18.207 s in the earlier unmasked allocation; these separate
profiles are only bottleneck diagnostics, not controlled endpoint speedups.
Same-device complete endpoints under node1 Slurm 5359 are now complete:

| Complete SCF + force | SPD unmasked | Screened rows | GPU4PySCF paired with screened rows |
| --- | ---: | ---: | ---: |
| Cold seconds | 381.379 | 358.585 | 123.351 |
| Three-repeat warm median seconds | 67.887 | 65.668 | 27.747 |

The warm improvement is 1.034x with one SCF iteration in both native variants.
Cold native iterations differ (18 versus 19), so cold timings include different
SCF trajectories. Every one of the five screened-row/reference pairs passes:
maximum energy error 2.9104e-11 Eh and force error 2.9813e-10 Eh/Bohr.
These are 24 atoms, 192 spherical def2-SVP AOs and 589824 grid points. Native
remains slower than GPU4PySCF; there is no large-system advantage claim.
The all-sample 1e-8 Eh / 1e-7 Eh/Bohr gates remain unchanged.

The 24-atom GPU4PySCF reference converges with the repository's established
1e-11 energy / 1e-8 orbital-gradient norm controls. An earlier diagnostic's
extra-tight 1e-12 / 1e-10 controls stalled and are retained as failed reference
evidence. With the established controls and explicit shared VV10 threshold,
the unmasked native cold/priming calls agree within 3.21e-11 Eh and 2.99e-10
Eh/Bohr. This changes no physical acceptance gate and does not establish a
timing comparison across node1 RTX 5090 and node2 PRO 6000.

Source patches/archive, build/ccache receipts and raw reports are retained in
ignored `.artifacts/wb97m-vv10-mask/` and the authorized remote task directory
`/home/jzzeng/codes/wb97m-20261002/`. Molecular active-pair counts are not yet
exported; grid-square capacities must not be mislabeled as measured work.

## Larger-case resource and environment diagnosis

The default 256 MiB nonlocal budget rejects the 48/96-atom full grid during
preparation. Explicit 1 GiB SCF-budget diagnostics do not qualify default
scalability. The force driver independently limits its nonlocal owner to one
quarter of a 1 GiB total device allowance; its resident allocation is about
240 bytes per point, so that second limit also prevents these larger cases.
Subsequent diagnostic runs explicitly allow 4 GiB total force memory and
retain the 1 GiB nonlocal cap. Neither control changes scientific work.

Node2 job 2045 additionally failed during force JIT because the default GCC 12
installation lacks cc1plus. GCC 11 is complete and is selected through
NVCC_CCBIN for reruns. Node5 job 1387 was stopped after diagnosing the
predictable force-capacity failure. These are failed/cancelled records, not
timing or accuracy evidence. Default resource planning still needs its own fix.

## Changed geometry and larger complete endpoint

Node1 job 5376 compares water12/def2-TZVP on grid24x8x16 with the same
22/1/11 cold/warm/moved iterations. SPD takes 167.254/23.496/92.773 seconds;
screened rows take 166.992/23.622/92.352 seconds. Every sample, including fixed
force, passes independent CPU PySCF/Libcint (maximum energy error <3.76e-12 Eh,
force error <9.08e-10 Eh/Bohr). This small-grid case has no meaningful warm gain.

Node2 PRO 6000 job 2073 completes water48/384 spherical def2-SVP AOs and
1179648 points using explicit 1 GiB nonlocal / 4 GiB total force host/device
caps. All three reference pairs pass, maximum energy error 4.548e-11 Eh and
force error 1.706e-10 Eh/Bohr. Native cold/warm are 1763.499/269.213 seconds
with 27/1 iterations; matched GPU4PySCF are 481.848/161.427 seconds with 18/4
iterations. One timed warm sample is completion/accuracy evidence, not a robust
performance estimate. This separate device cannot be timed against node1, and
the explicit budgets do not establish default scalability. The native library
is the screened-row binary identified above. The capacity follow-up #1725
separately passes a default48 complete endpoint on node1.

The successful node2/n5 toolchain uses NVCC_CCBIN=/usr/bin/g++-11 plus explicit
CUDA 12.9 cudart and cuSolver preloads. Selecting gcc-11 without C++ runtime
linkage caused a subsequent JIT load failure; those earlier failed runs are not
validation samples. Device visibility always remains assigned by Slurm.

## Revisit when

If resident SCF admits other nonlocal density domains, require an explicit
inactive-row contract before keeping this flag. Row compaction, pair reduction
reordering and changes to scalar pair algebra need separate qualification.
