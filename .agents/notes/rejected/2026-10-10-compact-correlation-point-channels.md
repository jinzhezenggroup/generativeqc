# Decision: do not promote five-channel correlation AD as a PBE0 optimization

Status: rejected
Date: 2026-10-10

## Problem

The current qualified 48-atom PBE0/RKS endpoint's exploratory Nsight capture
attributes 0.449522 seconds to 2304 bulk geometry point launches. Correlation
depends on two densities and three total-gradient components, but the shared
point forward differential carries eight independent spin-feature directions.
Reducing that storage appeared to offer a bounded point-kernel optimization.

## Decision

Do not promote the compact-only prototype or build another complete AOT/native
endpoint for it. CUDA 12.9.1 already eliminates the three duplicate directions:
the actual eight- and five-channel probe kernels have identical instruction
words **and scheduling-control words**, not merely matching register counts.
This does not prove that every compilation context emits identical code, but
it removes the concrete mechanism that would justify an expensive endpoint
campaign. No endpoint gain is established and no PR is opened for this change.

The prototype remains ignored under `.artifacts/xc-geometry-stage/prototype/`.
It parameterizes the existing AD operators rather than duplicating XC algebra,
retains `Jet = PointJet<8>` and the original `evaluate()`/response interface,
and maps the three common correlation gradient partials to both spin outputs.
The final emitter trial selects compaction only for PBE geometry, not LDA.
No production source is changed by this experiment.

## Evidence

The profiled endpoint uses the previously qualified #2185 snapshot plus the
default-off vector-publication scaffolding, not a newly built master. Both
publication opt-ins remain unset. Native SHA256 is
`cade369caa8eaa81d9a3f89f29e235a766e18129f94e16b576d069c72b7e15bb`.
Slurm 6981 on node1 executes two priming calls followed by one complete
oracle-checked warm capture: 5.460968 seconds, energy error 5.46e-12 Eh,
force error 1.63e-11 Eh/Bohr, one iteration/Fock, residual 3.95e-13.
This intrusive capture is not promotion timing. The observed point resource
`localMemoryTotal` is not a measurement of executed spill traffic. The observer
start/finish API was not activated in this capture; the ineffective env var and
late `/proc/maps` read do not prove an actual loaded AOT pathname.

CPU comparison of the actual namespace-relocated original/candidate headers
passes 20,416 point and 5,104 unrestricted response cases, all nine outputs and
validity bit-identical. Default point/response layout is preserved. All 97
existing independently generated SCF-domain points pass the unchanged gate
`abs(error) <= 5e-10*abs(reference)+1e-322`.

Slurm 6984 compiles the CUDA probe successfully, then exposes a harness-only
LDA Libxc array-shape error. Artifact-only resume 6985 exposes a wrong-cwd SHA
check. Resume 6987 passes LDA bitwise/domain gates but stops at a raw Libxc
comparison. These are terminal failed qualifications, not successful GPU gates.
Their scripts/logs/binary are retained rather than overwritten.

CPU diagnosis retains all 2000 original random positive-spin points. Six LDA,
403 PBE and 396 PBE0 cases disagree with raw Libxc under the unchanged strict
relative gate; all are checked against the independently differentiated
original rs/t2 formula at both 100 and 200 decimal digits, yielding identical
rounded FP64 references. The incumbent passes all these independent references;
Libxc fails them. The discrepancies include highly polarized spin fractions
and are consistent with the documented need for a separate spin-boundary
reference contract, not a compaction regression. The first diagnostic's PBE0
oracle incorrectly removed LDA instead of PBE exchange; the retained corrected
v2 recomputes PBE0 while reusing the valid v1 LDA/PBE diagnosis. No acceptance
tolerance is relaxed and no difficult input is deleted.

Slurm 6990 reuses the identical CUDA probe artifact and passes:

- all original broad-domain/invalid-input baseline-versus-candidate bit gates;
- the 97 existing independent device SCF-domain points;
- all retained independently arbitrated random cases;
- 2000 additional declared-interior Libxc points per LDA/PBE/PBE0 variant,
  with spin fractions 0.15–0.85, directly under the same strict gate;
- memcheck and initcheck, each with zero reported errors.

The raw Libxc disagreements remain explicitly recorded in the GPU outputs;
they are not relabeled as passing raw comparisons. Device arrays, validity,
resources and diagnostics are retained before final numerical assertions.
There is no synchronization/communication change requiring another racecheck
or synccheck population.

Actual sm_120 probe resource/code comparison:

| Variant | Registers, both arms | Local bytes, both | Instructions, both | Code/control words |
| --- | ---: | ---: | ---: | --- |
| LDA | 78 | 0 | 8416 | identical |
| PBE | 116 | 0 | 18096 | identical |
| PBE0 | 116 | 0 | 18144 | identical |

Retained probe library SHA256:
`797d138a52cc6eebb16d75c6739fe6604f3c0b6b06527eb182c9e408695e2755`.
Normalized PBE0 instruction/control-word SHA256, both arms:
`283221652024a34e10120d348b3aca15ef04b989b28827cb9ff40de4464fd484`.

## Invariants and consequences

Preserve the shared scalar algebra, existing native response layout, finite
Slurm allocations/device visibility, independent scientific gates, and complete
endpoint promotion criteria. Source-level derivative/storage reductions are not
automatically executed-work reductions. Inspect actual generated code before
launching another expensive endpoint matrix.

Master is checked at investigation boundaries. #2176/#2186 are OpenBLAS build
changes, #2191 is DF-CC dressing, and #2188 is MP2 oracle admission. None changes
this PBE0 execution path; they do not justify rerunning old numerical matrices
or relabeling retained measurements as current-master builds.

## Revisit when

A production compilation context demonstrably retains duplicated executed
directions, or a distinct specialization removes genuine scientific work
(for example, exact restricted-spin reuse rather than AD-array width alone).
Require actual device/code evidence before complete endpoint promotion timing.

## References

- `.artifacts/xc-geometry-stage/current-state.md`
- `.artifacts/xc-geometry-stage/evidence/profile/`
- `.artifacts/xc-geometry-stage/evidence/host/`
- `.artifacts/xc-geometry-stage/evidence/cpu-libxc/`
- `.artifacts/xc-geometry-stage/evidence/device/static-sass-comparison.json`
- `docs/developer/xc_scf_domain.md`
- `docs/maintainer/performance_engineering.md`
