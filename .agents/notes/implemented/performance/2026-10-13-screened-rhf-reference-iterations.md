# Decision: screen preliminary RHF reference iterations, not the final reference

Status: implemented
Date: 2026-10-13

## Problem

The [bounded whole-shell reuse investigation](2026-10-13-bounded-rhf-shell-reuse.md)
removed redundant recurrence preparation and filled measured sm120 f-shell
value-kernel gaps. Nevertheless, the conventional RHF reference used by DF-CBS
requested exactly zero screening on every SCF iteration. This bypassed existing
Schwarz and density-weighted J/K rejection even far from convergence. The
434-AO C7 fixture has 96,070,591 candidate shell quartets per Fock construction.

The scientific reference must remain conventional RHF: changing its final
operator without also changing the correlation/response contract would make
the published energy, orbitals and analytic derivatives inconsistent.

## Decision

Use the existing shell/AO Schwarz bounds and density-weighted Coulomb/exchange
gates during preliminary FP64 iterations. The internal
`ScfOptions::reference_iteration_screening_tolerance` defaults to `1e-12`;
zero disables this optimization. It applies only to physical-reference export
with quartet-direct execution, no persistent ERIs or active resident-value
lease, no mixed-precision Fock and no incremental J/K. Other routes keep their
existing policies. Invalid negative/nonfinite cutoffs fail closed.

Both stages share the original executable owner, geometry/primitive-pair
cache, Schwarz bounds and admitted numeric arena. Density-dependent bounds
are refreshed using the current density. No second SCF plan, geometry setup,
global numeric allocation or four-index tensor is introduced. A changed cutoff
invalidates the cached execution policy, including captured graph arguments.

The preliminary stopping floors are `100 * cutoff` for energy and
`1000 * cutoff` for density. These are not final acceptance tolerances. Reset
the previous-energy baseline and DIIS history, retain only the current density,
then run the original unscreened FP64 equations to the requested convergence
and physical/canonical reference gates. Each stage retains the existing finite
iteration bound. A failed or unfinished strict stage cannot publish a reference.

The complete iteration and FP64 Fock counts include both stages. This stage is
screened, not FP32: do not report its work as mixed precision. The optional
progress journal separately records `screened_fock_builds`,
`unscreened_refinement_fock_builds` and `iteration_screening_tolerance`; extra
physical finalization actions retain their independent existing count.
The journal's integer-only `number` API cannot represent a small cutoff;
serialize the cutoff as a round-trippable decimal `label` rather than silently
truncating it to zero. The GPU replay test also checks this observation.

## Invariants and rejected alternatives

- Do not relax final convergence, residual, canonicality or derivative gates.
  A per-quartet cutoff is not a bound on the accumulated total energy error.
- Do not publish the screened determinant after only one unscreened energy
  evaluation. The target density/orbitals must actually reconverge.
- Do not retain DIIS vectors from the screened operator during refinement.
- Do not use a second plan or CPU/PySCF seed to obtain preliminary convergence.
- Do not silently substitute DF-RHF. DF remains in the correlation consumers.
- Resident ERI refusal is legitimate resource policy, not a reason to increase
  the quartic cache allowance or change the scientific method.

## Qualification procedure

Build `tests/native/rhf_screened_reference_probe.cpp` against the same native
library and private headers. Under a finite Slurm allocation, set
`GENERATIVEQC_RHF_SCREENING_PROBE` to the executable and run
`tests/python/test_rhf_reference_screening.py`. The independent PySCF/Libcint
gate uses signed spdf He2 bases at near and separated geometries, automatic
and forced-bounded routing, cold/warm/changed-geometry replay, cutoff changes,
invalid cutoffs and incomplete strict refinement. Energy/density/orbital gates
are `2e-9 Eh / 3e-8 / 3e-8 Eh`; physical residual/drift must be below `1e-8`.
Oracles supply assertions only, never native input densities or integrals.

For the pinned C7 input in the preceding note, compile
`benchmarks/df_hf_preconvergence.cpp` against this same library. Compare full
`INPUT - OUTPUT auto-direct hf --reference-screening=1e-12` and
`--reference-screening=0` endpoints on the same device, keeping all other
controls fixed. Use `LD_BIND_NOW=1` to expose missing generated dispatch symbols
rather than letting lazy loading hide them. Recompile native probes after
private `ScfOptions` layout changes; old objects have an incompatible C++ ABI.

Also run the existing independent generated-Rys fixed-density matrix suite,
RHF/UHF E+F replay suite, and DF-MP2/DF-CBS energy/force/reuse gates. Disable
optional resident values explicitly in the screening-specific suite, while
retaining an unforced resident ethane control. Raw receipts and binaries remain
in ignored qualification storage rather than the tracked evidence budget.

## Source-matched qualification

The qualified sm120 library includes both the reference compensation repair
`dcc773b35` and the strict-link/value-only registry repair `89cda7cbf`. Its
SHA256 is `b3720cc8356c27966dfe6d06848d02edc6f7386247ea48b07bb801cd2e80bcbe`.
CUDA 12.9.1 Release compilation uses the existing eight-shard auto profile,
direct device linking, fast-compile off and sccache 0.16.0. Cache fallback
occurred during the large bounded translation unit; not all builds were hits.

- Node5 Slurm job 1508 passes all four independent screened-reference replay
  cases, 3168 Libcint fixed-density generated-Rys cases and 24 RHF/UHF E+F /
  native DF-MP2/DF-CBS force and reuse cases. Eager binding is enabled. The
  screening-specific cases exercise both automatic and forced-bounded routes.
- Node5 job 1509 passes unfiltered memcheck on the full forced-bounded
  screened-reference probe (zero errors), plus racecheck restricted to
  `enter_target_refinement_kernel` (zero hazards). This is not exhaustive
  racecheck of the full E+F domain.
- The focused final host suite passes 53 cases, with 16 opt-in skips; eleven
  native benchmark argument-rejection cases separately pass against the
  compiled executable without initializing a GPU. The larger preceding
  compiler/reference-capacity suite passes 99 cases with seven opt-in skips.
  These overlapping suites are not an additive count of distinct cases.
- Formatting, dependency/ownership/complexity/default-policy inventories,
  evidence retention and the warnings-as-errors Sphinx build pass.

An earlier GPU attempt found only a diagnostic cutoff truncated through the
integer trace API; numerical/reference/count gates had already passed. The
source-matched rerun above includes the decimal-label fix. C7 job 814 was
cancelled before completion to repeat against this corrected library; it is
not endpoint evidence.

### Complete C7 reference endpoints

Node4 Slurm job 815 compares screening enabled and disabled in the **same
library**, on the same RTX 5090 (`GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`,
driver 580.178.04, 575-W limit). The pinned QM7 index 5604 C7H16/cc-pVTZ input
and independent PySCF 2.14 oracle are unchanged from the preceding note:
23 atoms, 434 spherical / 485 Cartesian AOs, strict energy/density tolerances
`1e-12 / 1e-11`, FP64, final screening zero and a 64-GiB budget.

| Preliminary cutoff | Complete cold RHF seconds | Screened FP64 Focks | Unscreened FP64 Focks | Total |
| --- | ---: | ---: | ---: | ---: |
| Disabled (`0`) | 557.381505233 | 0 | 29 | 29 |
| `1e-12` | 346.823209130 | 23 | 7 | 30 |

This is a 1.6071x same-binary endpoint improvement (37.78% less time), despite
one additional Fock action. The 23 screened iteration scopes total 210.9851 s
(9.1733 s mean); that excludes strict refinement and is not the endpoint time.
Compared with the earlier **portable** whole-shell-reuse endpoint of
1348.6332 s, the cumulative tuned-kernel plus screening endpoint improvement
is 3.8885x. Do not attribute that entire cumulative gain to screening alone.
These are single cold endpoints, not timing medians.

Both runs have zero extra final Focks, execution retries, rejected retained
references or provisional DF cycles. The native provisional DF guess refuses
its 512-MiB allowance, and the canonical ERI source refuses its 8-GiB capacity;
there is no external/reference density seed. Both report exactly 2,003,893,596
reference-peak bytes and 2,230,584,163 reference-phase-peak bytes. Screening
introduces no additional numeric capacity compared with its disabled control.

Maximum absolute errors against the independent unscreened oracle:

| Cutoff | Energy / Eh | Density | Orbital energies / Eh |
| --- | ---: | ---: | ---: |
| `0` | 5.68e-14 | 1.62e-11 | 6.90e-11 |
| `1e-12` | 2.84e-13 | 1.76e-11 | 6.92e-11 |

The screened endpoint's physical commutator and canonical-density drift are
`6.01e-12` and `4.35e-12`; all independent and native acceptance gates pass.
These timings end at the **RHF reference**, not complete C7 CBS E+F or direct
large-basis CCSD(T). Small-system DF-CBS E+F is independently qualified above.

### Screening work and resident control

Node4 job 816 separately profiles the first two preliminary iterations. Across
all 55 shell classes, the accepted shell-quartet census decreases from
192,141,182 with screening disabled to 109,665,513 at `1e-12` (42.92% fewer).
This counts accepted shell quartets, not AO components or primitive recurrence
work. The diagnostic deliberately does not converge; its census is captured
before strict refinement and must not be extrapolated to all 30 endpoint Focks.

The same job checks unforced ethane/cc-pVTZ, 144 spherical AOs. Disabled/enabled
policy endpoints take 3.951539763 / 3.811852064 s, each with 13 provisional DF
iterations and 12 exact reference iterations/Focks, zero preliminary screening
and zero extra final Focks. The existing resident route correctly bypasses
this optimization. Both have identical independent E/density/orbital errors
`6.96e-13 Eh / 1.43e-11 / 4.17e-11 Eh`. These short single samples establish
no meaningful speedup or regression on the unchanged small-case path.

## Revisit when

Profile the screened stage and strict refinement separately. Further reducing
unscreened target work needs a scientifically qualified stopping/response
contract, not a claim that an integral cutoff alone bounds E+F errors. Consider
additional generated classes or screened scheduling only after measuring their
remaining complete-endpoint contribution.
