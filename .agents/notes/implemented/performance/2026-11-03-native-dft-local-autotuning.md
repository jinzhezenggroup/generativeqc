# Decision: extend local autotuning through the native DFT XC schedule owner

Status: implemented
Date: 2026-11-03

## Problem

Master already has typed grid/XC schedules, independent workload identities,
DFT endpoint acceptance and installable DFT-only profile bundles. Its public
source tuner nevertheless accepts only RHF/UHF and profiles HF shell classes.
Simply adding KS method names would fail: the native KS owner explicitly refuses
HF profiling flags. Kernel-only wins would also not establish DFT endpoint wins.

The shared frontend's compiler comparison also retained a fixed five-family
inventory after the package gained additional compiler owners. This compared
455 loaded files against 590 checkout files and rejected even a byte-identical
checkout. Compare the complete imported package inventory on both sides instead;
exclude the wheel's separately owned vendored `assets/` tree, but never skip a
new compiler namespace or weaken byte checks to bypass a real mismatch.

## Decision

Route the existing explicit compiler frontend to a separate DFT search after
the common source/toolchain/target/Release validation. The first search space
is the existing native device-fused XC point tile, bounded to 1..4096, with
the full functional, quadrature, charge/spin, screening and FP64 policy fixed.
Do not change native numerical implementation or pretend that this also tunes
DFT J/K, force tiles, or density fitting.

The worker opts into the native resource owner with an unbounded application
ResourceBudget; the native selected inventory still imposes a finite capacity.
Record actual KS semantic work and the native allocation ledger, not HF counters
or invented static operation counts. Since only runtime tiling changes, retain
the exact compiled library and bind source evidence to the verified complete
native source inventory. The bundle separately hashes that library. This is
not new generated-kernel PTXAS or measured-occupancy evidence.

An untimed, finite-timeout PySCF/libxc subprocess supplies an independent
energy/analytic-gradient oracle with Treutler grids and grid response. The
independent floors match existing native KS qualification: energy per atom
1e-5, maximum force 1e-4 and force RMS 3e-5 in atomic units. Baseline/candidate
parity remains independently tighter at 1e-9 energy and 1e-7 force/translation.
Every paired sample enters the gates; no selected-fastest-repeat promotion.

Each accepted incremental winner must beat the current winner at the complete
energy-plus-analytic-force endpoint with at least five alternating-order pairs,
matching SCF iteration branches and a significant >=1.02x median improvement.
Portable comparisons additionally require a final unmasked incumbent gate.
An explicit host-unfused incumbent is not eligible for this resident-only search;
do not silently substitute device-fused execution and claim an incumbent win.

Extend, rather than replace, compatible profile content: retain the incumbent
native library, HF winners and their original evidence, plus DFT winners for
other scientific workloads. Replace only the selected DFT workload entry.

Keep the execution fallback explicit: profile selection must not override an
explicit user point tile, and calculators with an explicit ResourceBudget keep
their bounded native/user tiling policy. A winner measured under a larger
capacity is not authority to enlarge a caller's resource promise. Its native
library may still retain previously qualified HF kernels.

## Rejected alternatives

- Reuse HF profiling for KS: the owner rejects those flags and they omit XC work.
- Search quadrature, screening or precision: changes the scientific problem.
- Include host-unfused XC by default: silently introduces CPU production work.
- Claim a faster tile from a component timer: complete E+F may be force/J/K-bound.
- Rebuild the native library per tile: no generated source or kernel changes.
- Install a DFT-only replacement over an incumbent HF bundle: loses unrelated
  previously qualified consumers and invalidates their performance evidence.

## Invariants and evidence

Tests cover CLI dispatch and budgets, removal of duplicate/incumbent tiles,
DFT versus HF profiling ownership, observed bounded memory, independent-oracle
failure, all-sample shape/value checks, schema-valid bundle publication,
incumbent preservation, exact-match retuning and final portable-baseline gates.
A real-device worker/oracle test is opt-in and requires a finite Slurm job.
Synthetic unit-test artifacts are nonexecutable and never installed into the
user's active cache. No benchmark speedup is inferred from their fake timings.

## Revisit when

Endpoint-selected winners also carry a separate selection-context hash over
full basis data, representation, batch size, MethodIR and SCF controls. The
compiler's grid/XC scientific identity deliberately excludes these profitability
inputs; it must not authorize transferring a timing win between AO workloads.
Direct-auto qualification does not select tiles for fitted or explicit-seed
calculators. Context evidence is checked at bundle admission and selection.

Add DFT J/K, force or generated-kernel candidates only when their owner exposes
appropriate measured work, exact executable schedule replay and independent
numerical/resource evidence. Preserve fail-closed runtime profile selection and
complete-endpoint qualification while expanding the search space.

The independent selector must match the declared composition, not a generic
family-name assumption: native `lda-rks`/`lda-uks` use Slater exchange plus PW92
correlation, not VWN. Composition tests compare every supported oracle selector
with the resolved semilocal FunctionalSpec and MethodIR-owned exact exchange.
The semilocal spec intentionally carries zero HF exchange; inspecting only that
spec would incorrectly reject otherwise matching hybrid reference compositions.

## Real-device qualification

Slurm job 1437 on n4's RTX 5090 uses a source-matched SM120 Release library with
fast compilation disabled. Six native energy/force/resource checks pass for
LDA-RKS, PBE-RKS, PBE-UKS, PBE0-RKS, B3LYP-RKS and r2SCAN-RKS. The native source
identity is also checked against the exact copied checkout inside the allocation.

A public-frontend H2/sto-3g spherical PBE-RKS search at Treutler level 3 accepts
the 512-point tile over 256 after five alternating-order pairs. Frozen-density
warm complete E+F medians are 36.391 ms versus 31.274 ms (1.164x; bootstrap 95%
lower speedup bound 1.163). Every sample retains 19,616 model grid points and
one observed Fock build; full KS/J-K diagnostics, independent numerical checks,
allocation receipts and all raw endpoint samples remain in the report. This is
a small-workload qualification, not a performance claim for arbitrary DFT inputs.

The installed winner actually replays at 512 points; an explicit 64-point user
request stays at 64, and a 1 GiB ResourceBudget retains the bounded 256-point
policy. All three native E+F results agree within the schedule-parity bounds.
Activation uses an isolated ignored qualification cache, not the user's default
profile index. Local evidence is retained under
`build/qualification-dft-autotune/{n4-report.json,gpu-current.log,cpu-tests.log}`;
the focused CPU suite passes 150 tests, with GPU and optional-NVCC cases skipped
locally and the six GPU cases qualified separately through Slurm.

The final source-matched qualification (Slurm job 1441) also tunes water with
PBE/PBE0 and def2-svp, plus a batch of two water dimers with PBE/sto-3g. It
retains every paired sample and reference in
`benchmarks/results/dft-local-autotune-20261103/xc-receipts.json.xz`.
The companion README describes the exact last-accepted paired edges rather
than multiplying separately measured improvements into an aggregate claim.
Final H2 activation revalidates the endpoint context as well as explicit-tile
and resource-budget fallback behavior. All caches remain qualification-local.
