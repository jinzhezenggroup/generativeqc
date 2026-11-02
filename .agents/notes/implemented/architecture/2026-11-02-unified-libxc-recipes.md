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

### Endpoint qualification

The implementation at `d577931c472595223f3511665aa7aba3fd42c1dc`, rebased on
`3b39557b9a0b098628089cfc7f39244fd726871e`, passed the final molecular gates in
Slurm job **15219** on node1 (SSH n1), one RTX 5090, CUDA 12.9.1, driver
580.95.05, one CPU math thread, NumPy 2.3.5 and PySCF 2.14.0. Slurm's assigned
`CUDA_VISIBLE_DEVICES=0` was preserved. The native library SHA256 is
`90dca25eb2cdeee060555816d1dbfe89c19fdc033fd8b2233200d202a7911412`;
the fully matched tracked-source manifest SHA256 is
`9b9fcd12160be130eb5918187c3da4fccd9150d62b6771dfef4246c406194181`.

- `test_unified_libxc_endpoints.py`: **24 passed**, covering CPU/CUDA, both
  spins, rSCAN0/r²SCAN0 Direct energy/forces and matched DF J/K energies,
  cold/replay/displaced geometries, and literal LDA/GGA registrations.
- MN15 native force routes: **4 passed**, both spins and Direct/DF, including
  replay, failure recovery, final-state checks and moved geometry.
- Compiled point qualification in job 15211: **22 passed**. The independent
  Libxc point gates cover spin tails and physical derivatives; no runtime
  reference engine enters production.
- Final shared-binding/selection regressions: **40 passed, 4 CUDA skips**;
  integration with the latest FP64 matrix policy: **7 passed**. Full pre-commit
  and the 422-file CUDA ownership inventory passed. `ty check` retains the
  same **159 diagnostics** as the updated base, with no added/removed messages.

Two planar water arrays (2×2 and 4×4, 8 bohr spacing) used spherical STO-3G,
Treutler level 0, Direct J/K and strict FP64. Each method/schedule pair measured
one cold call, three unchanged-geometry warm calls, a 0.02 bohr displacement
and three moved-geometry warm calls. All **64 complete energy/force calls**
passed the predefined `2e-8 Eh` / `3e-7 Eh/bohr` gates against independent
PySCF with the same quadrature. The largest observed errors were
`3.19e-12 Eh` and `2.52e-11 Eh/bohr`. These are matched-grid implementation
checks, not grid-convergence or production-accuracy claims.

The observed SCF AO-pair work per XC evaluation was:

| Atoms / AOs | Dense pairs | Local pairs, original geometry | Reduction |
| --- | ---: | ---: | ---: |
| 12 / 28 | 7,294,336 | 5,288,920 | 27.49% |
| 48 / 112 | 466,837,504 | 69,743,616 | 85.06% |

Complete warm energy/force latency, median [min, max] of three calls in ms:

| Atoms | Recipe | Dense | Local AO |
| --- | --- | ---: | ---: |
| 12 | rSCAN0 | 168.35 [166.63, 170.35] | 168.97 [167.36, 170.75] |
| 12 | r²SCAN0 | 83.48 [80.87, 83.49] | 83.37 [81.11, 83.89] |
| 48 | rSCAN0 | 493.45 [488.67, 510.89] | 494.19 [486.51, 508.54] |
| 48 | r²SCAN0 | 414.63 [395.61, 419.26] | 397.59 [393.18, 403.90] |

Work reduction does not establish a uniform endpoint speedup: most warm
latencies overlap. Cold timing includes Calculator construction, preparation
and the first complete call; process/runtime caches are shared between
sequential variants, so cold observations are not isolated speedup estimates.
Raw cold/moved timings, Fock-build/XC-evaluation counts, discovery traffic,
capacity observations, compiler-cache statistics and source-matched receipts
are retained in `.artifacts/unified-libxc-evidence/final/`. Earlier failed
qualification attempts remain separate from these final passing receipts.

Two integration failures are worth preserving: CPU forces correctly refused a
build with `GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF`; enabling the
packaged derivatives fixed that build prerequisite. Separately, the shared
automatic force registration initially published zero EXX in its model key.
The generated binding and derivative reservation now carry the same physical
EXX fraction, with compiled host tests for both spins. Widening a Python force
allow-list would not have fixed either condition.

## Revisit when

Additional ingredients, range/nonlocal primitives, arbitrary runtime recipes,
or new precision/derivative envelopes gain their own execution contracts and
independent endpoint evidence. Parameter-worker discovery currently applies to
atomic programs; composed literal workers retain component-local constants.
