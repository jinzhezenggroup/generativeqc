# Decision: prune verified-unused native interfaces, not low-reference code

Status: implemented
Date: 2026-11-08

## Problem and scope

The [initial Mulliken/MINAO cleanup](2026-11-08-unused-mulliken-and-minao-pruning.md)
was deliberately narrow. A broader pass against master `d52d2ae2a` inventoried
6,511 tracked textual inputs and initially recognized 6,327 native function
definitions. Reference searches included maintained native sources, Python,
generators, build configuration and tests. Historical benchmark receipts and
upstream snapshots were excluded from caller counting, not deleted.

This inventory is a candidate finder, not a compiler/linker reachability proof:
overloads, templates, generated names, preprocessor branches, callback/address
uses and definitions missed by the parser require individual inspection. A
missing repository caller is insufficient for removing an installed public API.

## Decision

Remove the following private interfaces after checking their callers, consumers,
build ownership and retained replacements:

- GFN2 whole-batch AES2/ES2/ES3 energy accumulation and external-point-charge
  potential/energy wrappers. The production SCC path uses the retained
  per-system implementations; their arithmetic and validation remain unchanged.
- Unused common integral validation/shell-pair wrappers, the basis-plan byte
  accounting helper, H0 plan validation wrapper and SCC driver restart entry.
  Keep the independently retained basis-plan oracle in the native test fixture.
- Unused periodic-embedding view/workspace binding and batch-evaluation entry
  points, followed by their two newly unreachable private binding validators.
  Keep the per-system evaluator, active binding checks and error handling.
- Unused DF metric-pseudoinverse derivative wrappers, not the shared inverse
  response implementation or its numerical coverage; the unused Huckel
  maximum-AO helper; and the old DF budget convenience aliases.
- The `build_s_integrals` and `build_cartesian_integrals` compatibility aliases.
  Migrate the one live Cartesian test directly to `build_integrals`, preserving
  its derivative flag and numerical assertions rather than retaining a shim.
- The unused RCCSD generator helper `_label_kinds`. Keep `_kind`, which is used
  by the generator, and all generated scientific arithmetic unchanged.

Together with the initial pass, this removes 25 internal functions/aliases and
the duplicate MINAO header. It does not remove a scientific method, public C
entry point, production fallback, numerical test, frozen oracle or license.

## Rejected deletions and deferred candidates

An advisory name-based call graph marked live CPU code as unreachable. Manual
inspection found real callers of `adapt_ppps_queue_profile`,
`histogram_quantile`, `adapt_transfers`, `run_pbe_rks_nonlocal`,
`run_pbe_uks_nonlocal`, `spatial_grid_order`, `property_exception`,
`system_projection_failure` and `integral_ad_direct_motion`. All are retained.
The low-reference packet-ordering/finalization templates and dense linear
solver also have actual callers and are retained. Repeating the reference
inventory after pruning found no further individually verified CPU deletion;
this is not a claim that all dead code has been proven absent.

The following CUDA areas remain candidates for a separately qualified change,
not approved deletions in this change:

- `solve_lambda_cuda`, stream-specific direct-JK construction and the
  `copy_to_device` wrapper;
- unscaled direct-force task wrappers and the second-center primitive
  overlap/kinetic gradient wrappers;
- GFN2 whole-batch/standalone wrappers in AES2, ES2, ES3, point charges,
  Mulliken, occupations, pair lists, preprocessing, eigensolver, periodic
  embedding and SCC-related CUDA units;
- warm-field identity and interaction-mask helpers, plus CUDA test hooks.

Before deleting any of these, distinguish generated/template/device consumers
and test-only builds from genuinely unused code; compile the supported CUDA
configuration and retain scheduler-controlled real-device endpoint/error-path
qualification. Low reference counts do not authorize removing test hooks.
This checkout's local toolkit is 12.4, below the project's 12.9 requirement, and
the available Slurm inventory did not offer an allowed usable GPU node. No
local real-GPU test or CUDA performance claim is made.

Tracked content at the base is dominated by benchmarks (92,485,594 bytes), tests
(35,821,823 bytes) and native sources (19,085,272 bytes). Historical evidence,
independent oracles and vendor notices are not disposable size reductions.
Deleting tracked content does not reclaim historical Git objects.

## Qualification identity

Pruning changes the source inventory/native-input identity used by the Libxc
catalog gate even when capability labels are unchanged. Regenerate the retained
snapshot and its reference page with `tools/check_libxc_native_catalog.py
--write`, then run the checker normally. Keep the measured source revision,
measured library checksum and stale-evidence state unchanged: this operation
updates checked input identity, not scientific qualification. No product is
newly qualified and no capability gate is weakened.

## Evidence

The final tree recognizes 6,305 native definitions in the same textual scan.
For artifact comparisons, use the original master library and the final library
from the same GCC 11.4.0 Release, CPU-only build with stationary CPU force AOT
disabled. Ninja compiler commands invoke the verified sccache 0.16.0 launcher;
statistics are retained before/after builds without clearing shared caches.

| Artifact | Original master | After both passes | Reduction |
| --- | ---: | ---: | ---: |
| Changed native/Python code and data | 980,118 B | 856,057 B | 124,061 B |
| Shared library, unstripped | 23,010,872 B | 22,939,456 B | 71,416 B |
| Shared library, stripped | 22,104,016 B | 22,038,448 B | 65,568 B |

All 362 exported `generativeqc_` C symbol names are identical. Seven independent
tblite-fixture molecules plus triplet water, doublet OH and water dication have
exactly equal complete-endpoint energies, forces and iteration counts against
the original master build. Independent numerical-oracle tests remain required;
before/after parity does not replace them.

All five RCCSD generator outputs (CPU, CUDA, triples CUDA and CPU/CUDA Fock
outputs) are byte-identical before/after removing `_label_kinds`; this is a
one-off migration check, not a permanent production-source fingerprint test.
Native Cartesian integral, preliminary initial guess, density fitting, DF Fock
capacity and HF DF resource tests all pass.

The selected Python suites total 323 passed and 35 environment/backend-gated
skips. They cover independent tblite/xTB goldens, finite-difference forces,
retained-runtime/cache transaction behavior, electronic and RCCSD generation,
MINAO/Huckel admission, DF budgets/metric response and the Libxc catalog gate.

Less source reaches the compiler, but no controlled complete clean-build A/B
timing was performed. Incremental rebuild times, compiler-cache hits and object
size reductions are not a measured complete-build speedup.

## Revisit when

A concrete consumer requires a removed internal operation, or an environment
supports the deferred CUDA compile and real-device qualification. Introduce the
needed operation with an explicit ownership/publication contract and executable
numerical/error-path coverage, rather than restoring historical compatibility
wrappers or relying on a regex reachability report.
