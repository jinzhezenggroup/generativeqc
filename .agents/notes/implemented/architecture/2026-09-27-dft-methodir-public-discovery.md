# Decision: derive DFT discovery from MethodIR, not the native ABI manifest

Status: implemented
Date: 2026-09-27

## Problem

`manifests/public_methods.json` historically owned both stable native ABI/provider
metadata and the `compiler_method` / `spin` binding for every public DFT
selector. Libxc-backed method generation and `METHOD_CATALOG` already own the
scientific DFT composition. Keeping the same names in both places made
`public_methods.json` a second DFT whitelist and allowed the two sources to
drift as functional coverage expanded.

## Decision

`public_methods.json` now owns only stable ABI IDs, provider families, declared
native properties, batch capability and compatibility aliases. DFT scientific
identity is resolved from the compiler MethodIR catalog, including generated
metadata derived from the pinned Libxc sources.

Public DFT selectors use the explicit `<method>-rks` / `<method>-uks` form.
Resolution first constructs the compiler-owned MethodIR and then applies the
existing native primitive-lowerer gates. A representable MethodIR therefore does
not become executable merely because it exists in the compiler catalog.

Existing DFT ABI IDs are retained for compatibility. Newly discovered DFT
selectors reuse the stable PBE RKS/UKS provider carriers because the semantic KS
descriptor already carries the complete MethodIR composition. PBE-D4 keeps its
dedicated carrier because that native ID still selects the separately qualified
D4 correction owner.

## Rejected alternatives

- Continue adding one `public_methods.json` row per Libxc functional. This
  duplicates scientific identity and turns ABI metadata into a growing
  functional whitelist.
- Allocate a new C enum value for every discovered DFT functional. The native
  ID is not the scientific identity once the KS descriptor carries MethodIR;
  expanding the ABI would add compatibility burden without adding semantics.
- Treat every MethodIR entry as executable. Compiler representation is broader
  than the qualified native lowerers and would silently overstate production
  support.

## Invariants

- Existing method ABI IDs are never renumbered or removed by DFT discovery.
- MethodIR is the single source for DFT scientific composition and spin-specific
  resolution.
- Native primitive, backend, basis, grid, derivative and production-domain gates
  remain fail closed.
- Explicit Libxc blockers remain blockers; discovery does not convert a blocked
  or unlowerable composition into a runtime fallback.
- Compatibility aliases may remain in the ABI manifest, but they must not
  reintroduce `compiler_method` or `spin` ownership there.

## Evidence

- `python tools/generate_method_manifest.py --check`
- `pytest tests/python/test_method_manifest_generation.py`
- `pytest tests/python/test_ks_execution_plan.py`
- `pytest tests/python/test_hybrid_cli_boundary.py`
- `pytest tests/python/test_split_hybrid_cuda_registry.py`
- Regression coverage proves `pbe50-rks` / `pbe50-uks` can be discovered
  without ABI-manifest rows and that compiler-known `scan-rks` remains rejected
  while its native lowerer is unavailable.

## Consequences

Adding a representable and natively lowerable DFT functional no longer requires a
parallel edit to the ABI manifest. The native ABI table remains intentionally
smaller than the public DFT discovery set. Compatibility IDs remain supported,
so this ownership cleanup does not itself break existing C/Python selectors.

## Revisit when

Revisit the carrier policy if the native DFT provider gains behavior that cannot
be expressed in the semantic KS descriptor, or if a future ABI introduces a
first-class generic DFT provider ID that can replace the compatibility carriers.

## References

- `docs/user/methods.md`
- `docs/public_methods.md`
- `python/vibeqc/ks.py`
- `python/vibeqc_compiler/method/spec.py`
- Refs #396 #739 #745

Agent: ChatGPT
Model: GPT-5.6 Sol
