# Automatic native CUDA forces and bounded AOT

This campaign qualifies the shared C++ prepared force owner for automatic
rho/sigma/tau Libxc programs under #2227. Python is the API and independent
test harness; the installed CLI executes without Python or a compiler on PATH.
The production envelope is all-electron s/p, direct Coulomb, strict FP64,
device-fused XC and three Becke partition iterations. Automatic AOT remains an
explicit build option. Neither full-catalog compilation nor representative
numerical qualification grants Hessians, HVPs, nonlocal/exact-exchange terms,
empty-spin CUDA UKS, mixed precision or higher angular momentum.

## Numerical and lifecycle acceptance

Eleven registrations cover LDA_C_VWN_4, LDA_C_CHACHIYO, LDA_C_PZ_MOD,
GGA_X_APBE, GGA_X_PBE_SOL, GGA_X_RPBE, GGA_X_B86, GGA_C_APBE,
MGGA_X_LTA, MGGA_X_GVT4 and MGGA_C_TPSS. Each has an asymmetric H4/STO-3G
full-vector test in RKS and open-shell UKS. H4 replay/moved MethodIR cases and
neutral-water s/p cases cover LDA, GGA and tau-MGGA in both layouts.

The original 22 full-vector and 12 lifecycle tests retain 58 force endpoints
and 300 finite-difference directions. Every stencil reconverges native and
independent PySCF 2.14.0 / Libxc 7.0.0 energies on the exported moving grid.
The basis oracle reads the separately pinned BSE fixture. Gates are 3e-9 Ha
for center energies, 2e-7 for physical D, 3e-9 for F, 3e-5 Ha/bohr for
forces at a 1e-4 bohr central step, and 2e-8 for translation sums. Positive
low-density and populated near-iso-orbital points are measured independently
for polarized H4 meta-GGA cases. Controls are radial/polar/azimuth 24/8/16,
tile 37, Hcore seed, at most 300 iterations, energy 1e-13, density 1e-12 and
screening 1e-12.

After rebasing, all 58 native endpoints and every force component are executed
again, with fresh independent center SCFs. The original independent FD values
are explicitly reused: unchanged point expression/source and force plan identities, original receipt
hashes and force parity at 3e-8 bind the reuse. The compiler source closure
correctly invalidated the force artifact after an upstream RHF integral-AD
change. An explicit 22-profile source review proves the generated bodies differ
only in the identity literal; every new identity is bound to its newly executed
binary and object. The initial strict identity rejection is retained.
These are new endpoint measurements with retained reference stencils, not a
claim that the finite differences were rerun on the final binary.

Separate gates cover exact minimum workspace, one byte below it, mixed H4/water
batch failure and recovery, stale/closed handles, nonconvergence and receipt
revocation. The minimum schedule is explicitly reported as `minimum-workspace`.
Native CUDA DFT/KS tests, 91 automatic energy/domain/lifecycle tests, 22 installed
CLI force cases and memcheck complement the force matrix. The installed test
sets `CUDA_DISABLE_PTX_JIT=1`, removes tools from PATH, and checks an omitted
provider rejects instead of compiling on demand. Slurm device visibility is
preserved in every real-device run.

## AOT scale and execution identity

The catalog contains 34 LDA, 99 GGA and 64 tau-MGGA registrations. The source
census measures 788 CPU/CUDA/spin transport records, 625 distinct point-source
hashes, 21,755,951 point-source bytes and 394 native force variants totaling
60,153,143 source bytes. The largest point and force sources are 288,389 and
413,893 bytes. Canonical order-one worker Graphs are measured separately before
backend equal-spin specialization: 231 layouts and 1,045 derivative roots.

All 197 CUDA energy registrations and 394 native force profiles compile with
NVCC 12.9.86, sm120-real and verified sccache 0.16.0. The full build contains
394 force objects (398,123,664 bytes), 33 point objects including the registry
(144,591,064 bytes), and a 638,954,812-byte native stationary archive. The first build took
2,109.534 seconds with four compile jobs; the final rebuild takes 926.877
seconds with sixteen compile jobs and the retained compiler cache. All 429
compile commands invoke the verified launcher. Shared device-link/registry
objects are reported separately; these figures are not installed-core size or
an uncached compiler benchmark. Shared cache counters are observations across
the build interval and can include other compilation, not a per-job hit rate.

Point and force cubin records satisfy explicit ceilings of 255 registers,
4,096 stack bytes, 4,096 local bytes and 49,152 static shared bytes. Observed
point maxima are 255/768/0/0; force maxima are 255/1768/0/10472. The executed
Becke dynamic shared requirement is checked separately. Stack/local reporting
does not establish zero compiler spills or clear historical higher-derivative
spilling variants. Full-catalog compilation does not claim full-catalog
molecular numerical qualification.

`aot-audit.json.gz` binds 22 actually executed profiles and 58 receipts to their
loaded-library digest, point/force source identity, generated source, object,
cubin resources, workspace and selected schedule. Descriptor addresses, not a
caller-supplied filename, identify the loaded module. Work is retained-controller
work; complete all-attempt J/K counters remain in SCF/performance records.
The public force tests also prove that energy replay and failure cannot reuse
old force-completion evidence.

The existing real CHACHIYO/CHACHIYO_MOD CPU alias test executes one stored object
against both independent Libxc registrations while preserving their distinct
provenance. Existing closure and invalidation tests are rerun. The narrow
[NVCC persistent-cache contract](../../../docs/developer/libxc_cuda_aot_closure.md)
remains separate: this static sm120 package does not claim a universal CUDA
persistent-cache closure or runtime alias cache.

## Complete endpoint performance

ABBA uses separate processes on one Slurm-assigned RTX 5090. Each process uses
two independent prepared owners per method; each owner has cold setup plus
SCF plus host-return E+F, three complete warm calls and one moved call. A
disposable PBE owner warms process/driver state equally in both arms. The matrix
is PBE and automatic LDA_C_VWN_4, GGA_X_APBE, MGGA_X_LTA, each RKS/UKS:
320 calls, 24 method/phase comparisons. Every sample, median, MAD, iteration,
Fock build, J/K work and warm fallback is retained. The gate is candidate median
no more than 1.05 times baseline, with full energy/force parity.

The baseline is the earlier automatic-force implementation before canonical
closure repair and artifact diagnostics. It is **not** an automatic-force
endpoint admitted on master. Both native libraries use the final Python timing
harness. PBE supplies a curated control. The repaired LTA warm path avoids
roundoff-amplifying fixed-point corrections after an already valid canonical
proof. Original tolerances remain unchanged. This small-system matrix is a
regression gate, not a production ranking or large-molecule scaling claim.

## Provenance, negative results and reproduction

`manifest.json` records measured source and binary identities, original
reference provenance, build/cache observations, final test outcomes and compact
negative evidence. Initial LTA warm closure failure is retained. Correlation-only
water-ion Hcore trials for LDA and TPSS failed to converge; neutral-water UKS
qualifies p-AO transport, while open-shell H4 supplies the polarized gate.
Neither failed water-ion solve is promoted. Initial report serialization and
fixture-registration failures are distinguished from numerical acceptance.

Final native code is commit `4fb1bcde754cc145d8485682e7a87f53ba6439d3` on
`cd9ad801b`. Later audit/test/evidence changes do not relabel it as a new binary.
Apply `reference-native-source.patch` to that commit to recover the independent
FD producer. Apply `baseline-native-source.patch` **after that patch** to recover
the pre-closure performance baseline. The shared reconstruction avoids copying
the same upstream changes twice. These patches cover the archived
src/include/python/cmake/tools/upstream and root CMake inputs; benchmark harnesses
remain separately identified. The complete two-patch application chain was
checked.
No external release or binary publication is required to inspect the evidence.

Run the offline verifier from the repository root:

```sh
PYTHONPATH=python:. python benchmarks/results/automatic-native-cuda-20261027/verify.py
```

It authenticates retained files, reconstructs original reference-receipt hashes,
checks all force/FD values and actual artifact bindings, applies cubin/dynamic
shared limits, and recomputes every complete endpoint performance gate.
The retained `drivers/` directory gives the exact build and allocation recipes
for the owned node1 checkout. Adapt checkout/toolchain paths for another machine;
keep finite Slurm scheduling and the actual assigned visibility. Large binaries,
compiler streams and source archives remain in ignored run storage. These
receipts expose endpoint and compiler evidence for #1124; they do not close all
remaining parent #1118 or cross-method #2227 obligations.
