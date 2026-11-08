# Decision: project correlated factors before the first AO pullback

Status: implemented
Date: 2026-11-08

## Problem and measured boundary

The first complete pinned H2IO iodine-ECP CBS energy/force replay passed on
the stationary-density-product candidate, library SHA-256
`944f546b8d63c3407f45c201c2ea14e480c68a1ab182c847cc2657f8b3a4254b`,
Slurm job 4130 on n2's RTX PRO 6000. Native complete endpoint time was
2863.197875082027 seconds; maximum component/composite directional-force error
over both independent FD steps was `7.65392111539212e-9 Eh/Bohr`.

Its 97-AO correlated MP2 derivative still submitted 1,500,625 ordered native
shell quartets. Completed derivative telemetry attributed 1961.590311688
seconds to shell consumers and only 7.735879896 seconds to host pullback/scatter.
The same frame's stationary derivative submitted 198,765 canonical quartets.
The finite conventional source work, not another convergence failure or missing
device storage, was now the dominant complete-endpoint barrier. These are
source-matched measurements of the earlier binary, not a new speedup claim.

## Decision

Project the sparse Fock and occupied/virtual correlation MO cotangent entries
onto all eight exact ERI permutations before transforming the first AO slot.
The common coefficient matrix acts on all four slots, so projection commutes
with the complete pullback. Keep repeated MO-index contributions in the average;
multiply the final local AO cotangent by its distinct shell orbit, using the
existing canonical consumer. No symmetry of the supplied Fock/correlation
weights is assumed.

Use at most the existing admitted `max_shell_width*N^3` first-stage capacity.
Reserve it once before submitting any quartet, and reuse it across first shells.
If this optional reservation fails for a resource reason, retain the existing
compact ordered factorization without mixing partially accumulated canonical
and ordered gradients. Explicit compact and full staged layouts remain
independent validation paths. All remaining buffers retain their admitted
shapes; no molecular AO-rank-four buffer or new CUDA arithmetic is introduced.

The exact stationary-density-product specialization remains first choice for
its exact unit occupied-diagonal Fock and zero correlation cotangent. The new
projection handles genuinely correlated weights, never a cutoff approximation.

## Invariants and qualification

- Preserve exact conventional/ECP inputs, FP64, coefficient ownership, physical
  response gates, convergence limits and the unchanged 43-GiB endpoint budget.
- Resource failure must select a complete bounded fallback before the first
  two-electron callback. Numerical/backend failures must not become fallbacks.
- Native comparisons cover nonsymmetric Fock/correlation weights, dense expanded
  cotangents, ordered compact/full layouts, repeated shells/atoms and multiple
  occupied orbitals. The common-translation-unit probe additionally checks
  independent molecular ERI center derivatives through Cartesian d and spherical
  d/f shells and the production numeric staging bound.
- New default resource tests enforce ownership, cleanup and admitted capacity,
  not exact new allocation call counts or a production-source fingerprint.
- Complete pinned component/CBS energies, zero-net-force and both independent
  directional FD steps must pass on the actual new binary before claiming
  complete issue qualification or endpoint performance improvements.

## Rejected alternatives and revisit conditions

Increasing only the Slurm allocation hides symmetry-related source work.
An unconditional AO-rank-four cotangent would discard bounded staging. Retaining
eight ordered derivative evaluations to save a modest host staging buffer is
not the default when that same full buffer is already admitted; it remains the
explicit resource fallback. Revisit when complete profiles show projected host
staging dominates, retaining these independent gates and bounded fallback.

## Complete source-matched acceptance, November 8

All three original complete conventional iodine-ECP CCSD(T)/CBS energy and
analytic-force cases pass on production source
`d7231a911e166ffd0e867291ed980e72e31a0307`, library SHA-256
`ac41a8f7cf2af63e7b8110dfbec6f61e20d03f7f3d1019b11cedcb323c3cc69c`.
The subsequent lifecycle-probe and documentation changes do not change that
production source or library. The complete regression module SHA-256 remains
`9369a5f56d72663913b14ee9ec98cd6e02e34de4a8478cc437db513a07c1cef5`.
The retained full source manifest verifies every production/build input.
The Release build uses CUDA 12.9.1, `120-real`, fast compilation disabled and
verified sccache 0.16.0; actual launcher commands and before/after cache statistics
are retained rather than inferring cache hits from elapsed build time.

Jobs 4138, 4139 and 4140 each use n2's RTX PRO 6000 Blackwell Workstation
Edition, a finite four-hour Slurm allocation, preserved device visibility and
the unchanged 43-GiB correlation numeric budget. Every srun returns zero.
These times include complete native endpoint preparation, all five component
energies/analytic forces and composite construction, not just a derivative
kernel. Test time additionally includes independent oracle qualification.

| Original case | Job; DZ/TZ AOs | Native seconds | Test seconds | Max component energy error (Eh) | CBS energy error (Eh) | Max two-step directional-force error (Eh/Bohr) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hf_dz`, CH3IN2 | 4138; 80/171 | 9501.172732 | 10124.66 | 1.1313091e-8 | 1.1313602e-8 | 5.2000556e-8 |
| `response_dz`, CH2ClI | 4139; 65/131 | 6260.366798 | 6727.92 | 9.2159098e-9 | 9.2223900e-9 | 1.4488144e-7 |
| `hf_tz`, H2IO(-) | 4140; 47/97 | 1126.247521 | 1165.03 | 1.0954295e-9 | 1.0970780e-9 | 7.6537256e-9 |

Every component and composite passes independent PySCF 2.14 energy and
directional finite differences at both `2e-4` and `6e-5` Bohr. Unchanged absolute
gates are `3e-8 Eh` per component, `9e-8 Eh` for CBS, `3e-7 Eh/Bohr` for
directional force and `3e-8 Eh/Bohr` for net force, with no relative allowance.
The largest component/composite net-force coordinate across all three is
`5.935247952837397e-13 Eh/Bohr`. No basis, ECP channel, electron, convergence
ceiling, iteration cap or integral model is removed or relaxed. PySCF and
displaced energies are test-only oracles, never production force providers.

The high-basis work journals distinguish a logical AO-domain scan from its
individual device reads and retain the actual admitted MO capacities:

| Case | AO values in one primal scan | Device reads | Retained MO bytes | Canonical correlated derivative quartets | Correlated derivative seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hf_dz` | 855,036,081 | 16 | 4,563,543,240 | 1,788,886 | 2079.456130 |
| `response_dz` | 294,499,921 | 16 | 1,376,919,768 | 636,756 | 1129.727753 |
| `hf_tz` | 88,529,281 | 1 | 419,477,240 | 198,765 | 315.357377 |

The 171-AO force source uses axis tile 86; bounded admission does not imply one
contiguous AO read or constant transformation work. Its shared primal producer
takes 4430.446246 seconds and records 1,637,340,214,725 transformation FMAs.
The corresponding 131-AO producer takes 3306.249878 seconds and records
395,406,170,665 FMAs. These retained endpoint/work measurements are not an
arbitrary-element speedup or memory-peak claim. The measured-peak sentinel
remains unavailable where retained owners prevent exact allocator telemetry.

Additional current-source acceptance includes 214 focused host tests (11
explicit opt-in skips), four native host suites, nine independent-oracle CUDA
CBS/ECP regressions, both native CUDA convergence suites, repeated warm
tight-budget MP2 energy/forces and an independent F-shell MP2 energy/force
endpoint in finite Slurm job 4137. The nine lifecycle-harness tests separately
exercise resource-only rollback, batched/OVOV-only admission, cleanup, restored
budgets and the stale-CUDA-error negative control. CI run 42436541740 passes
all Python shards, CPU GCC/Clang, CUDA compile/resource gates and native CUDA
CLI qualification at the test-only head `81ca354f9`.

Raw logs, compressed work journals, exact source/library/cache identities and
per-case numerical summaries remain under ignored `.artifacts/issue-3034/`.
The public, checksum-pinned inputs and executable qualification live in
`tests/data/conventional_ecp_cbs/` and
`tests/python/test_conventional_ecp_cbs_regression.py`; the current runnable
procedure and coverage boundary are in `docs/user/ccsdt_cbs.md`. Earlier binary
results remain separate rather than being relabelled as this candidate. This
completes the original three-case acceptance boundary; it does not certify
arbitrary other elements, bases or geometries.

## References

- #3034 and PR #3046.
- [First-stage compact fallback](2026-10-11-compact-factorized-mp2-first-stage.md).
- [Second-stage compact fallback](2026-11-06-compact-factorized-mp2-second-stage.md).
- [Stationary specialization](2026-11-08-stationary-rhf-density-cotangent.md).
- `tests/native/mp2_derivative_pullback_probe.cpp`.
- `tests/python/test_conventional_ecp_cbs_regression.py`.
