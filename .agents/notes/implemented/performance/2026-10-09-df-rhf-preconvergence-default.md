# Decision: default to a bounded DF density guess, not a DF-HF reference

Status: implemented
Date: 2026-10-09

## Problem

The [initial master-based experiment](2026-10-09-df-density-exact-rhf-experiment.md)
shows that a cheap DF density can eliminate seven expensive Direct Fock
builds without changing the accepted RHF reference or complete CCSD(T) forces.
Its two prototype repeat pairs alone do not qualify default promotion.

## Decision

Put provisional preparation in the complete native DF-CCSD(T) owner, before
its optional-response-cache retry scope. Prepare at most once per endpoint;
only a detached AO density crosses into the fresh unscreened FP64 Direct SCF.
The original exact convergence, canonical/reference audit, CC and response
equations are unchanged. Include all preliminary work in reference and complete
endpoint time, including refusal and discarded work.

Admit neutral singlet H/C systems with 200–400 spherical AOs and through-f
orbital shells, without ECP. Use separately bundled H/C `aug-cc-pvtz-jkfit`
raw exponents, native-normalized on the actual geometry. This is a resource
and validated-topology guard, not a molecule-name or benchmark-identity switch.
Other cases keep the old cold Direct policy. Explicit detached seeds take
precedence, including future compatible warm-state callers.

The guess uses FP64, 32 iterations and `1e-4` preliminary energy/density
tolerances. Cap its numeric budget at 512 MiB after charging live correlation
auxiliary metadata and optional response-cache storage; skip below 256 MiB
available. Release the DF source, Fock, eigensolver and DIIS owners before
Direct; charge the surviving density's capacity in every downstream phase.
DF refusal/nonconvergence and seeded Direct refusal/nonconvergence retain
bounded cold Direct fallbacks. The final exact solver must genuinely converge.

`GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS=direct` is an explicit control/opt-out;
unset or `auto` selects the guarded default. Invalid selectors fail closed.
No public ABI layout or method identifier changes. The existing public
DF-CCSD(T) provider does not support warm density retention or prepared batches;
this change does not advertise either capability.

## Basis provenance

H/C raw, single-primitive, unit-coefficient records were exported offline from
PySCF 2.14.0 `aug-cc-pvtz-jkfit`, using the existing benchmark metadata exporter.
The 230-AO geometry gives 484 JK functions, distinct from 488 correlation-RI
functions. The complete exported shell metadata SHA-256 is
`f3aac69713fb69c025580064892e192d0991e5638a67889f39948fc3b35d5de4`.
The compiled table stores one atom template per element, not geometry or
oracle results. Production does not import PySCF or build a reference on CPU.

## Rejected alternatives

- Full DF-HF: changes the reference Hamiltonian and requires an independently
  qualified DF response/gradient definition.
- Reusing the caller's RI correlation basis for JK: silently substitutes an
  unvalidated fitting role.
- All sizes/elements by default: small resident-ERI cases need no such work,
  and unqualified topology/resource regimes require separate evidence.
- Carrying DF Fock/orbitals/DIIS into correlation: violates exact-reference
  ownership and could bypass genuine Direct convergence.
- Repeating preparation after an optional response-cache retry: adds work and
  obscures complete endpoint accounting without improving the detached guess.

## Evidence

The reproducible benchmark exposes `auto-direct` for the actual native default,
`direct` for the old control, and `df-direct` for the configurable experiment.
All-repeat numerical gates retain every sample, independent of iteration counts.
The policy test injects only the DF boundary and verifies selection, raw basis
topology, work limit, memory bound, density guard, strict control preservation,
refusal and explicit opt-out. Native methane tests retain the original
small-system path and independent conventional-RHF oracle/fallback checks.

### Five paired complete endpoints

The default owner was frozen on master
`4e20f7a7bcfbff770fefd13fa2777f8ab309679c` plus the retained dirty source
patch. The full parent revision is verified in the retained provenance rather
than inferred from a later publishing checkout. Slurm 2770/2771 execute five
fresh-process pairs on three RTX PRO 6000 Blackwell GPUs on node2, CUDA 12.9.1,
driver 595.91.07, Release/portable_cuda, architecture 120, GCC 11.4.0, FP64 and
compiler cache enabled. Pair order alternates Direct/auto and auto/Direct on
each assigned device; every pair uses the same frozen executable/library.
Different GPU allocations are recorded rather than treated as one device.

The original 230-spherical-AO ethane input and explicit 64-GiB numeric budget
are unchanged. Correlation uses 488 RI functions; the native guess uses 484
independent JK-fit functions. Both schedules keep strict Direct tolerances,
CC DIIS 8, packed CC DIIS, FP64 triples, and Lambda audit cadence 30.
These are matched benchmark controls, not changes to public CC solver defaults.

| Pair | Direct RHF (s) | Auto RHF incl. DF (s) | Direct E+F (s) | Auto E+F (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 128.560525488 | 86.660445924 | 516.443619047 | 474.384852924 |
| 2 | 133.074999983 | 87.595100988 | 526.590930116 | 480.554258746 |
| 3 | 133.180936441 | 90.017702510 | 537.403089648 | 495.214363574 |
| 4 | 128.665782829 | 86.741658283 | 516.578786513 | 474.590338736 |
| 5 | 131.991587293 | 89.328538003 | 526.206855745 | 483.172343411 |
| Median | 131.991587293 | 87.595100988 | 526.206855745 | 480.554258746 |

Median RHF reduction is **33.635845%**, including median DF preparation
3.210506688 s. Complete cold E+F reduction is **8.675789%** (45.652596999 s).
Every individual pair improves the complete endpoint. Direct work is always
19 SCF cycles / 20 physical Focks versus 13 DF cycles followed by 12 Direct
cycles / 13 physical Direct Focks, including post-SCF work. DF physical Fock
counts are unavailable and remain explicitly null, never inferred from cycles.
There are no seed fallbacks, Direct execution retries, discarded primal
attempts or discarded resident-JK response attempts.

All 25 control/candidate numerical combinations pass. Maximum errors are
`4.2632564e-14 Eh` in reference energy, `1.7053026e-13 Eh` in total energy,
`5.9118934e-11` in density, `6.6755490e-12 Eh` in orbital energies, and
`3.2718928e-10 Eh/bohr` in full-component forces (gate `3e-9`). Fresh independent
conventional PySCF RHF checks every sample: maximum energy/density/orbital
errors are `5.4001248e-13`, `3.3612757e-10`, and `6.4694983e-10`, respectively.

Independent same-Hamiltonian correlation energy error is `2.2026825e-12 Eh`.
Two-step directional finite-difference force errors are `7.5603988e-9` at
`h=1e-4 bohr` and `3.2064529e-9` at `h=3e-5 bohr` (gate `3e-7`). Maximum
physical Lambda/Z response residual is `6.1163276e-13`, and translational
defect is `1.1106983e-12`. Directional finite differences are not an independent
analytic oracle for every force component.

Frozen probe SHA-256:
`a71929ccd7954df598e224bb0b5d5020700aadb95d9258a0f007229b3424dd1c`.
Frozen library SHA-256:
`d1a9ecb2a9781fa68f29b7a1c5eb9e2f604b53986fcc6a9ff6cea29abae00110`.
The original two prototype pairs and later integration binary are not pooled
with this five-pair cohort. The existing public provider has no warm-density
or prepared-batch mode; no warm-density/batch throughput claim is made.

### Changed geometry and constrained numeric budget

Slurm 2774 shifts the first hydrogen by `0.02 bohr` along x. Its complete
E+F pair is 552.116997698 s Direct versus 487.593827330 s auto, with RHF
155.094346049 s versus 90.906128643 s including 3.297560717 s / 15 DF cycles.
Actual Direct Focks fall from 23 to 13; neither path reuses an old geometry
or density. Total energy difference is `2.2737368e-13 Eh`, full-component
force difference `4.5079052e-10 Eh/bohr`, and fresh independent RHF
energy/density/orbital errors are `3.1263880e-13`, `3.8625128e-10`, and
`5.7825744e-10`, respectively. All physical-reference and response gates pass.

Slurm 2773 changes only the declared numeric budget to 8 GiB. Complete E+F
is 592.025127689 s Direct versus 549.864125003 s auto, with RHF
128.874705287 s versus 86.698087783 s including 3.181876310 s / 13 DF cycles.
Actual Direct Focks remain 20 versus 13. The tighter existing correlation/
response schedule costs more than the 64-GiB cohort on both paths; those
times are not pooled. Maximum force difference is `3.2716208e-10 Eh/bohr`.
The unchanged-geometry independent RHF gates pass. This is a numeric-budget
qualification, not an 8-GiB measured-VRAM assertion. The host-injected policy
also verifies tighter-budget preparation refusal and unmodified cold Direct
admission, plus a single preparation across a retained-response-cache retry.

Slurm 6803 on node1's RTX 5090 passes 87 focused policy/ownership/CLI tests,
including real native methane HF/E+F, fresh independent conventional RHF and
deliberately exhausted DF-guess/cold-fallback tests. This small-system run is
not pooled into the RTX PRO 6000 performance cohort.

### Larger source/planner case

Slurm 2772 uses the reproducible 322-AO neutral propane fixture with 686
correlation-RI functions, the same 64-GiB declared numeric budget and the
fixed 512-MiB preliminary cap. Complete **energy-only** endpoints are
1323.555844317 s Direct versus 1092.770877391 s auto. RHF is
578.678886719 s versus 356.398949598 s including 9.786017762 s / 14 DF
cycles. Actual Direct Focks fall from 21 to 14; no guess or target fallback
occurs. This exposes a larger/more expensive source regime rather than
extrapolating the 230-AO setup cost.

Maximum candidate/control energy and density errors are `1.2789769e-13 Eh`
and `4.3339422e-10`. Fresh independent conventional RHF energy/density/
orbital errors are `7.6738615e-13`, `5.8465572e-10`, and `8.4509466e-10`.
An independently rebuilt same-Hamiltonian PySCF DF-CCSD(T) energy differs by
at most `3.6521897e-12 Eh`; its independent CC replay residual is
`1.6432807e-11`. All gates pass. No larger-system force timing or analytic
force oracle is claimed from this energy-only pair. The independently rebuilt
changed-geometry correlation energy also passes at `1.8189894e-12 Eh`.

### Memory observation

Slurm 2775 runs a separate, serialized complete E+F pair with 100-ms assigned-
device `nvidia-smi` sampling. Observed VRAM high waters are 31,548,506,112
bytes Direct and 31,584,157,696 bytes auto (34 MiB difference). These include
CUDA context/libraries and are sampled high waters, not exact allocator peaks
or the declared numeric-budget measurement. The preliminary 512-MiB cap is
not a promise that complete correlated E+F fits in 512 MiB or 8 GiB VRAM.
Both endpoints pass the unchanged reference/force gates; their instrumented
times are not pooled with the five uninstrumented pairs.

### Latest-master integration

After fast-forwarding to master
`cdb2131a47aaeb003bbc85fc76cf172848b8044b`, a separate cached Release build
and Slurm 2777 on node2 repeat the complete 230-AO E+F pair. Direct RHF is
128.362663975 s versus 86.381578004 s including 3.209459416 s of preparation;
complete E+F is 516.234425654 s versus 472.112529135 s. Direct physical Focks
remain 20 versus 13, with no fallback. Maximum full-component force difference
is `3.2721137e-10 Eh/bohr`. Independent conventional RHF, same-Hamiltonian
correlation energy, and both directional finite-difference gates pass.

The same finite allocation passes **89 focused tests**, including real native
HF/E+F and independent oracle checks. Latest probe SHA-256 is
`00871fa563c51958d77536587c44918bcfda23f493a606b6c321d4d764a93bcf`;
latest library SHA-256 is
`8474db95295d0f2df5d1e6677c4f739757343b82567fe345c8afbecb5902d3e1`.
This integration pair has a different binary/source parent and is retained
separately, not pooled with the five-pair promotion cohort.

The compact [reviewed record](../../../../benchmarks/results/df-hf-preconvergence-default-20261009/publication.json)
retains every timing/force observation, full-precision all-pair numerical
errors, input/source/binary identities and the verified measured-source
reconstruction patch. Its accepted publication scope is numerical. Compiler
solver-primitive performance promotion is explicitly unassessed because the
endpoint does not export every SCF iteration trajectory; the default decision
uses the native method's complete-endpoint/work-count protocol above, not a
fabricated same-state kernel or solver-history gate.

## Revisit when

Expand the domain only after new independent reference/force gates and paired
complete endpoint measurements, including larger/planner-cliff and changed
geometry cases. Any future warm/batch support must preserve seed precedence,
geometry identity and complete live-owner accounting. A different accepted
reference must be a separately defined and validated method, not this option.
