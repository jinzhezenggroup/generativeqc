# Decision: authenticate and isolate the physical KS induced Fock

Status: implemented (component, not a complete Hessian)
Date: 2026-10-16

## Problem

The live grid lease and generic coupled CPKS solver do not by themselves supply
a physical WB97M-V density kernel. Replaying the original SCF Fock source can
mutate scratch, revoke an occupied-projection lease, or use density-dependent
screening. An occupied-factor fitted exchange action is not valid for a signed
or indefinite CPKS direction. Detached model labels cannot authenticate either
source lifetime or the actual accepted molecular F.

## Decision

Add a scalar-canonical-validated native source lease carrying actual prepared
primary/correction pointers plus accepted F/H and the existing owner/epoch/
generation. The nonmovable response owner retains the batch and reborrows both
Fock and grid bindings before execution/publication. It validates the complete
WB97M-V model and reconstructs the accepted F on CUDA before admitting actions.
The functional-neutral grid adapter remains functional-neutral.

Canonical reborrowing uses a cached host token by reference rather than rebuilding
and copying the occupied-population vector on each action. Model/basis/grid are
immutable for the native owner; the cache key checks its nonwrapping solve epoch
and final density generation, and every borrow still checks the live successful,
inactive, nonpending, nonfailed state plus scalar canonical readiness. This caches
identity metadata only, never replaces live-owner or numerical-validation gates.

Exact actions prepare independent canonical Direct storage on the borrowed
producer stream. A zero, geometry-only screening threshold and independent
compensation planes preserve a linear signed-density action without changing
SCF scratch. LR is exact exchange with the compiler-resolved LR-minus-SR
coefficient; the primary is full-range exact or matched fitted J/K.

For the current full-rank fitted domain, borrow only immutable integral-source
metadata and device whitening. Generate raw A into the response arena through a
new current-device-only source seam. The compatibility source API retains its
existing device-selection behavior. This seam does not modify fitted J/K scratch
or occupied-projection generations. Build-time TensorIR forms W-transpose-W
because the source applies W[out_aux,source_aux] to raw values. Then use the
existing stationary DF energy's density VJP, not a new fitted Fock equation.
Generic affine lowering adds a sum-reduction primitive to execute that VJP.

This is deliberately a bounded correctness schedule, not a scaling win. Storage,
Direct ERI upper bounds and compiler-enumerated per-map IR scalar-term visits
are admitted separately. Primal validation dependencies in the AD graph count
toward work. Missing full-rank immutable source capabilities or canonical Direct
storage fail closed; there is no Python, CPU or density-screened retry.

## Rejected alternatives

- Relabeling a PBE grid lease as stationary WB97M-V: storage authentication is
  not model authentication. The actual full F reconstruction is mandatory.
- Invoking the original source's general J/K builder: mutates provider scratch
  and can use density-dependent screening or device switching.
- Cloning the DF owner: stream creation/destruction and transferred source
  ownership introduce unnecessary lifetime changes; readonly metadata plus a
  separate arena avoids them.
- Applying occupied-factor RI-K to signed directions, fitting LR exchange, or
  replacing LR exchange by full exchange: different mathematical operators.
- Guessing W-W-transpose because whitening is nominally symmetric: the source
  layout, not numerical near-symmetry, determines the inverse composition.

## Invariants and evidence

- Production actions remain native CUDA with no runtime compiler/interpreter,
  device switch, hidden scratch allocation or host numerical contraction.
- Native H2 RKS and distorted H3 UKS execute exact and fitted primary variants;
  each includes exact LR exchange, semilocal XC, bilateral VV10 and coupled
  CPKS. F/H reconstruction, stale-owner rejection, capacity/work, alias and
  nonfinite publication gates are tested behaviorally.
- Independent reference tests use pinned PySCF 2.14.0/Libxc 7.0.0, raw/full/LR
  integrals and real-space VV10 potentials. Signed-density differences retain
  the separately documented FHC-free private reference handles; default Libxc
  handles remain untouched. No production formula or tolerance is relaxed.
- Fock value gate: absolute/relative 3e-9. Directional potential differences:
  absolute 3e-9, relative 2e-6 at 1e-3 and 3e-4 displacement. CPKS D/W gates:
  absolute 3e-8, relative 2e-6 at 1e-4 and 3e-5, with displacement convergence.
- The independent CPKS oracle uses potential differences and displaced
  generalized eigenprojectors to form a small dense fixed-point solve; it does
  not call native response, compiler AD or the production GMRES implementation.
- Initial Slurm node2/main/PRO6000 jobs 3100/3101 run with assigned visibility 0:
  four native combinations pass, then seven compiler/molecular tests pass.
  These are component qualification durations, not endpoint benchmark times.
  Final source-matched qualification receipts are retained under ignored
  `.artifacts/wb97mv-hessian-aot-20261015/fock-*` and must supersede these initial
  runs when source changes affect the executable.

Final source-matched Slurm job **3102** repeats the six lifecycle cases and the
integrated regression: **185 passed, four host-compiler/cache prerequisite
cases skipped** on n2 (those cases pass locally). All **16 sanitizer runs**
(memcheck/racecheck/initcheck/synccheck times RKS/UKS times Direct/DF) are clean.
The final native probe also exercises capacity/work, alias, nonfinite input and
revocation gates. Local CPU regression is **44 passed, 145 explicitly opt-in
skipped**; CUDA-disabled C++ syntax, ty 0.0.82, all-files prek, 555-module compiler
structure and 249-module shared-SCF structure checks pass. Readelf finds no
libpython dependency in the probe or library. Final binary SHA-256:

```text
b6ceb9b734941b0bf2dcfa28083f65915d662c54ce158fef2b5cae63bfe258e9  generativeqc_ks_wb97mv_grid_probe
6786d271518a383da6a3e2eb0bbe47efb1a7ad1b7c1e37b46ea9c2d82123bede  generativeqc_ks_resident_frame_tests
5d8b3db10e610c793c9478c4e7a5869690daafb712ad3cfc7c9d0b2ec456c6e0  libgenerativeqc.so.0.1.0
```

## Review follow-up: borrowed-work completion

Queued host emulation reproduced a lifetime failure after this qualification:
Direct work could retain the caller's density pointer when a later generated
map launch failed, while the physical action returned through its exception
handler without draining. A failure-only scoped stream guard now starts after
validation and is disabled after the existing successful publication fence.
Admission/capture rejection must not fence unrelated producer work; successful
actions gain no additional fence. The durable host regression compiles the
complete production controller and actual generated affine maps with queued
CUDA/provider doubles, covering 90 launch-rejection boundaries across RKS/UKS
and Direct/DF, unchanged output and retry. Removing the guard fails that test;
the repaired controller passes host AddressSanitizer/UBSan. The revoked-owner
probe also fences its sentinel upload before rejection so its default-stream
observation has an established baseline. These are bounded host lifetime
controls, not renewed real-GPU or molecular qualification; the Slurm receipts
and binary hashes above describe the preceding source. Arithmetic, generated
math, scientific gates and the incomplete-endpoint hold are unchanged.

## Consequences and revisit conditions

The component can now serve real molecular CPKS, but it does not supply nuclear
motion, moving Becke grids, full shell/component closure, DF metric second
response, ECP second response or a stationary Hessian assembler. The C++ CLI
still invokes its explicit Python worker. Keep the migration/PR incomplete.
Revisit the materializing fitted schedule when a compiler-owned bounded tiled
schedule retains signed-density semantics and complete endpoint qualification.
