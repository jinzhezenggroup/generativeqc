# Decision: bind automatic native catalog entries to their worker domains

Status: implemented
Date: 2026-10-27

## Problem

At baseline `c8ba02453`, the shared native catalog generator assigned
`libxc-7.0/work-semilocal-v1` to every automatic Libxc entry. The compiler and
native density-only LDA workers already require
`libxc-7.0/work-density-only-v1`, with distinct total-density restricted and
spin-resolved unrestricted layouts. This affected 34 registrations / 68 spin
entries.

The Python descriptor bridge correctly rejected those catalog entries because
their domains did not match the requested model, then used its explicit compiler
route. Native CLI preparation rejected the incorrect domain outright. GGA-only
catalog tests missed both effects. This was an R5/R6 integration gap under #2227
in the automatic-method work tracked by #1118/#1121.

## Decision and invariants

Use the compiler-owned `automatic_scf_domain` when generating each automatic
catalog row. The library, Python descriptor bridge, and CLI then share the exact
worker contract without copying a functional-name table or changing equations.

- Keep exact method identity, spin and numerical-domain matching mandatory.
- Do not weaken native domain validation to accept obsolete descriptors.
- Preserve custom-graph/older-library compatibility and explicit CUDA catalog
  rejection. This repair does not qualify a new backend, derivative or provider.
- Resource metadata still consults Python compiler plans. Proving native
  descriptor selection does not prove removal of every Python policy call.

## Evidence

Validation used a Release CPU build with verified sccache 0.16.0, scalar linear
algebra, and explicit compiler launcher. Generated Ninja compile rules invoke
that launcher. The automatic CPU registry is built normally; CUDA and stationary
CPU force AOT are disabled for this scoped energy/catalog qualification.

- Generator tests cover LDA/GGA/tau-MGGA domains and both spin identities.
- Compiled-library tests compare the full catalog with compiler/public model
  resolution, require automatic descriptor binding without Python lowering, and
  exercise budgeted RKS/UKS preparation. The compatibility and transactional
  rejection tests remain in place.
- The existing independent PySCF 2.14.0 / Libxc 7.0.0 CPU suite passes for three
  LDA, five GGA and three tau-MGGA registrations, both spins, cold/replay/moved
  geometries, full final Fock and fixed-density directional derivatives. Its
  nonconvergence and stale-snapshot checks also pass.
- Five focused CTests pass, including native catalog/API, CLI runtime and catalog
  generation. The installed CLI executes all three ingredient families with
  RKS, its alias, and UKS, with no interpreter/toolchain on PATH. The existing
  public DFT API C++ test also compiles against the installed SDK and executes
  with an empty executable search path.
- An isolated compiled negative control using the baseline catalog fails native
  binding for all 68 LDA spin entries; the repaired catalog binds all 68 without
  Python descriptor lowering. Baseline and candidate catalogs run in separate
  processes to avoid ELF interposition of their identically named C++ data.

The complete CPU lifecycle comparison uses all 22 functional/spin contexts,
nine alternating samples per route, and includes calculator construction,
preparation, cold/warm/moved SCF and cleanup. Energy and semantic work signatures
agree. Aggregate median time changes by +2.25%; the largest per-context median
increase is +4.77%, within the 5% gate. The comparator explicitly selects the
portable descriptor route using the **same compiled scientific library**; this
is a route comparison, not a separately rebuilt historical-master benchmark
or a speedup claim. Raw logs, scripts, cache statistics and samples are retained
locally in ignored `.cache/1118-native-catalog/`.

## Remaining scope

#1118 remains open. Native automatic CUDA catalog selection, CUDA forces,
artifact/resource qualification, and the combined product/backend evidence
catalog require their own acceptance. #2782/#2789 are separate open method/force
work; this repair does not depend on or claim their proposed capabilities.

## References

- #1118, #1121, #1123, #1124, #2227
- `tools/generate_cli_method_catalog.py`
- `python/generativeqc_compiler/xc/automatic_semilocal.py`
- `tests/python/test_native_ks_catalog.py`
- `tests/python/test_libxc_automatic_cpu_ks.py`
- `docs/user/native_cli.md`
