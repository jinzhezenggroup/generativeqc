# Decision: compose imported XC through the ordinary native KS owner

Status: implemented
Date: 2026-11-02

## Problem

Automatic Libxc molecular registration excluded curated components and accepted
only a unit-weight, single-component semilocal graph. r²SCAN0 therefore had a
valid MethodSpec but no native lowerer. rSCAN exchange was additionally blocked
by a stale helper-name denylist even though the current importer could lower it.
Spatial XC optimizations alone did not remove these registration boundaries.

## Decision

The actual importer determines supported Maple operations. Independent E/vxc/fxc
fixtures in both spins still gate promotion into the imported inventory.

`generate_semilocal_compositions.py` projects supported electronic MethodSpecs
into rational component sums and full-range exchange metadata. The XC compiler
consumes this generated data without importing method policy. Codes occupy a
separate uint32 namespace derived from canonical rational components and the
exchange fraction; generation rejects collisions. Names and alias ordering do
not determine science or native codes. Atomic codes remain unchanged.

The point emitter composes the physical E/vxc outputs of existing component
lowerers inside a single traversal. Each component retains its own worker
thresholds, FHC policy, original-density energy weight and physical derivatives.
RKS uses the existing differentiated equal-spin specialization; any density-only
component uses its total-density worker. Exact exchange stays in the ordinary
native Fock owner and the stationary plan carries its derivative weight.

Native preparation matches the complete component/exchange recipe. Live and
detached final-state checks authenticate the same fixed hybrid fraction. CPU
and CUDA expression identities still have to agree. The independent exchange
metadata header does not depend on linked point-program definitions.

Public `libxc:` method recipes and ordinary selectors choose the same semantic
specialization. Curated and split-hybrid implementations remain preferred,
retaining their numerical domains, precision and response qualifications.
Automatic registration includes supported curated atomic components. A partial
family graph selects its atomic program, without adding a missing X/C partner.
The previous Python-only partial-PBE scale selection could not pass native
component matching and was not an executable specialization to retain.

The emitted point-capability query admits local AO selection for registered
automatic physical programs. Previously it returned false even though the
shared density/assembly owner and point ABI already supported the layout. AO
selection changes traversal and scatter, leaving component math unchanged.
Mixed-density precision, response and point batching retain separate gates.

## Rejected alternatives

- Adding rSCAN0 and r²SCAN0 to a native whitelist would duplicate method policy
  and require repeating the change for each recipe.
- Replacing curated evaluators wholesale with raw Libxc would discard existing
  domain continuations and independently qualified performance mechanisms.
- Applying one shared input floor to a sum changes component-specific density
  tails. Differentiating those floors also changes the raw Libxc vxc contract.
- Inferring an X/C partner from a registration name changes the requested model.
- Coupling exchange composition metadata to the complete generated CPU point
  header unnecessarily makes metadata-only consumers depend on point linkage.

## Invariants

Generic CUDA recipes retain strict FP64, device-fused, all-electron execution,
nonempty occupied UKS channels and explicit artifact/resource checks. The new
force scope retains Direct s/p admission. Matched full-range DF J/K energy is
separate from force admission. Neither a generated recipe nor an imported
formula grants range/nonlocal, ECP, mixed-precision or response capability.

Optional compact storage and bounded fallback schedules remain owned by the
existing XC runtime. Production does not call PySCF, Libxc or a Python science
fallback. Independent reference libraries are qualification tools only.

## Evidence

Local raw evidence is retained under `.artifacts/unified-libxc-evidence/`.
The permanent tests cover alias/renaming semantics, atomic selectors, generated
recipe reproduction, compiled physical points against independent Libxc and
complete cold/replay/moved energy/force endpoints against PySCF. GPU checks
require finite Slurm allocations. Existing admission, exchange-model and
stationary contract suites remain required.

## Revisit when

Additional ingredients, range/nonlocal primitives, arbitrary runtime recipes,
or new precision/derivative envelopes gain their own execution contracts and
independent endpoint evidence. Parameter-worker discovery currently applies to
atomic programs; composed literal workers retain component-local constants.
