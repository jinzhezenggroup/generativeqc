# Decision: adapt Libxc work-domain points to shared resident spatial execution

Status: implemented
Date: 2026-11-01

## Problem

The initial generic spatial extension reused nonlinear point results across AO
pairs, but accepted only nine curated `interior-v1` components. Automatically
imported Libxc programs and the MN15/M06-2X split pairs already had CUDA point
lowerers, yet could not use this prepared spatial optimization. Requiring a CPU
`NativeContractionProgram` also excluded split pairs absent from its scalar API.

This extends the admission boundary described in
[the initial spatial residency decision](2026-11-01-generic-spatial-xc-residency.md).
Its interior API and explicit host alternative retain their original semantics.

## Decision

`SpatialContractionProgram` compiles existing Libxc point workers into the same
`GridXcKernel` ABI and uses the same `PreparedXCContractions` driver, point panel,
AO traversal, matrix assembly, orbital route and resource accounting. A CPU
scalar artifact is unnecessary. Device artifact mismatch and host schedules
fail before borrowing device resources. The descriptor does not add runtime or
method admission, exact exchange, nonlocal operators, forces or response.

Ordinary imported components delegate representability and work-domain handling
to `emit_automatic_point`; there is no new spatial functional whitelist. Each
component preserves its own screening, floors, Fermi-hole constraint and
original-density energy weighting. Restricted LDA uses the existing U1 worker;
folding polarized work would change the interval between total-density and
per-spin floors. GGA/meta-GGA share equal-spin arithmetic only after proving
physical input equality, keeping independently differentiated spin partials.

Registered split-hybrid pairs reuse the composed stationary point emitter's
qualified FP64 bounded integer-power lowering, work policy and Cartesian
coefficient conversion. MN15/M06-2X names explicitly select only their semilocal
pairs. Exact exchange remains with the method's J/K provider. Arbitrary modified
split pairs and operator-bearing FunctionalSpecs fail closed.

The outer adapter rejects nonfinite/negative densities and tau, nonfinite
Cartesian gradients and overflowing norms before any worker can sanitize them
into finite values. Worker vxc is evaluated at clipped work inputs; it is not AD
through clipping and does not grant a physical empty-spin particle-birth
continuation. Source bytes, rational coefficients and mathematical worker
contracts bind the artifact identity. The 11-row panel remains `88 * tile_points`
bytes, and all AO pairs reuse one point evaluation per collocated point.

## Rejected alternatives

- Enlarging the old interior whitelist would conflate Libxc worker semantics
  with interior formulas and could silently alter boundary energies/vxc.
- Constructing a fake CPU native program or routing device calls through host
  reference libraries would obscure ownership and provide no useful fallback.
- Duplicating grid/assembly kernels for imported components would make common
  work reductions depend on a functional's source again.
- Interpreting this fixed-density E/V endpoint as a native MN15 SCF speedup would
  exceed the measured scope. Native KS already has independently owned shared
  point/assembly paths and admission gates.

## Evidence

The source inventory successfully emits all 197 entries from the current
`automatic_registry_entries()`, with zero failures. This is structural coverage,
not 197 new numerical qualifications. Generation also passes with imports of
the public runtime, PySCF, Torch and CuPy forbidden.

Independent PySCF 2.14.0 / Libxc 7.0.0 tests cover imported LDA_C_PZ, GGA_X_B88,
GGA_C_LYP, MGGA_X_MVS, MN15 and M06-2X for both spin layouts. Automatic worker
boundary tests cover vacuum, empty spin, floor intervals, zero gradients and
Fermi-hole clipping. MN15 additionally passes all 50 adversarial split-point
inputs. M06-2X exact-empty-spin cancellation remains owned by the existing
113-bit oracle; the new binary64 endpoint tests use occupied spins and vacuum.
Work-domain energy directional derivatives are tested only on the interior.

Host/codegen/admission regression: 127 passed, two installed-native-AO cases
excluded; the additional compiler-only import test also passes. Compiler
structure: 580 modules, zero dependency errors. Raw current-source evidence and
finite Slurm runners live in ignored `.artifacts/spatial-libxc-pr-evidence/`.

Real GPU validation on n1 / RTX 5090 through finite Slurm allocations passes:

- 168 complete endpoint/lease cases, including 48 new Libxc cases, job 15069;
- all 48 new cases under compute-sanitizer memcheck, zero errors, job 15070;
- 538 runtime publication/lifecycle/precision regression cases.

GPU tests exercise both spin layouts, density matrices and orbital factors,
minimum one-point/one-AO tiles, ragged tasks, noncontiguous AO maps, expired
leases, missing/mismatched artifacts, vacuum and valid-density recovery. During
resident execution, host AO evaluation and host tile entry points are disabled.
Runtime compilation uses sccache 0.16.0; host adapter compilation uses ccache.

### Complete fixed-density endpoint timings

Job 15070 measures 32 analytic Gaussian AOs, 4096 points, two spin channels,
128-point tiles, full feature mask and no screening. Each warm median includes
source handling, collocation, XC, matrix assembly, transfers and statistics,
from five calls after one cold call. Compilation/preparation/cold calls are
retained separately; independent reference work is outside timing.

| Functional | Warm median |
| --- | ---: |
| LDA_C_PZ | 11.988 ms |
| GGA_X_B88 | 17.896 ms |
| GGA_C_LYP | 15.889 ms |
| MGGA_X_MVS | 55.728 ms |
| MN15 semilocal | 25.546 ms |
| M06-2X semilocal | 31.912 ms |

Every endpoint performs 4096 point evaluations, owns a 11264-byte point cache
and 4686080 total device bytes. Maximum independent energy error is
`4.37e-11` Eh and potential error is `1.64e-11`. These are absolute endpoint
measurements, not before/after speedups: MN15 had no equivalent prior prepared
spatial worker API. They do not measure full SCF or molecular forces. The six
JSON records retain binary SHA256, point identity, timings and work metrics;
`benchmark.py` and the finite runner reproduce the fixture.

## Catalog identity and preserved qualifications

The new compiler source changes 221 catalog identities without stage promotion
or demotion. Retain the exact preceding snapshot summary from `d23009932` before
acknowledging those transitions. The historical campaign bytes remain unchanged
and stale; this extension does not relabel them as current native qualifications.

## Revisit when

Additional input/operator layouts or physical boundary continuations are
qualified, or this API needs geometry/response. Extend the owning point/contract
lowerer before adapting it to this shared spatial ABI. Complete SCF performance
and method qualification remain separate from fixed-density point reuse.

## References

- `docs/developer/xc_contractions.md`
- `python/generativeqc_compiler/xc/spatial_program.py`
- `tests/python/test_spatial_libxc_codegen.py`
- `tests/python/test_spatial_libxc_cuda.py`

## Master integration after qualification

Integrating master `1e1ae33b5838d8abb87e366c5bc666b70d66e4cd` into
`18cbb12298889daa220b0d5e42e307053a2ddfb0` required resolving only the generated
catalog snapshot and rendered report. Production compiler/spatial XC source
merged unchanged. Regenerate the catalog against the combined native inputs,
retaining the exact preceding PR summary as the new previous snapshot. There
are zero capability identity/stage transitions in this integration. The prior
221 identity acknowledgements remain recoverable in `18cbb122`; they are not
new transitions against the updated previous snapshot.

The historical campaign and retained manifest bytes are unchanged, and remain
stale: no current-source native context is admitted. This integration does not
rerun or relabel the earlier GPU timings or numerical receipts. Catalog checking
passes under normal and optimized Python, compiler structure checks 580 modules
with zero errors, and the catalog/transition/schedule regression passes 103 tests.
