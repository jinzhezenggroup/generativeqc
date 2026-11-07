# Decision: Share the auxiliary-g DF export domain with the generated source

Status: implemented
Date: 2026-11-07

## Problem

Issue #3029's full-DF unrestricted CBS route rejected unmodified Na
def2-universal-JKFIT and B cc-pVTZ-JKFIT bases at the raw Cartesian exporter.
The source's generated generic values and auxiliary-g coordinate derivatives
already existed, but the single/batch exporters retained an f-only preflight
and used the direct-HF topology packer's f-only mode. Raising only the shell
gate would therefore still reject the combined basis during packing.

## Decision

Share the source's host-only value-domain admission and freeze the resolved math
policy for each export, with a separate raw-export auxiliary-g ceiling.
Generic/auto exports admit orbital f / auxiliary g;
specialized polynomial/Rys/candidate diagnostics remain auxiliary-f-only.
Both raw exporters pack Cartesian normalized metadata with `DfValues`, avoiding
unused direct-HF shell-pair, quartet, resident-task and warm-density state.
They retain the existing generated value and `AuxiliaryGDerivative` evaluators.
No recurrence, spherical ordering, metric cutoff or scientific identity changes.
The source's separately qualified full-range auxiliary-h domain (#3015) is not
permission to admit h into unrestricted correlation response. A compiled host
preflight checks that source h remains available while raw exports reject it
without CUDA work; the generic-only g and orbital-f boundaries remain intact.

## Rejected alternatives

- Merely lifting the f-only gate leaves the direct-HF packer rejection intact.
- Enlarging direct-HF's global shell domain would admit unrelated four-center
  consumers without qualification; the explicit DF packing mode avoids that.
- Truncating fitting bases, using CPU/PySCF production integrals, or increasing
  scalar-work limits does not solve the missing export-domain contract.

## Invariants

- Orbital g, auxiliary h, and unsupported g diagnostic math remain rejected.
- Memory admission, finite work limits, metric rank and strict convergence
  policies remain independently enforced by the complete endpoint.
- Include all physical centers in both raw three-center and metric derivatives.
- Keep the bounded coordinate-tensor response; this does not promote the
  unrestricted route to resident or shell-streaming correlation response.

## Qualification

- `test_df_native_g_derivatives.py` compares single/batch raw exports and their
  spherical/mixed transforms against independently normalized libcint M/A/dM/dA
  for permuted, signed contracted f/g shells and multiple geometries.
- `test_open_shell_cuda_df_complete_cbs_force` checks every unrestricted
  companion and the assembled CBS energy/force against reconverged independent
  energies at two displacement sizes, for both reference Hamiltonians and
  nonzero molecular internal-coordinate forces; energy-only replay is separate.
- `test_physical_auxiliary_g_full_df_ump2` retains the complete Na DZ and B TZ
  fitting bases and checks UHF/UMP2 energies and analytic forces. Its opt-in is
  `GENERATIVEQC_OPEN_SHELL_AUXILIARY_G_CUDA_TEST=1`.
- `test_physical_auxiliary_g_full_df_cbs` separately retains complete Na/B DZ/TZ
  fitting bases and checks every requested component and assembled atomic CBS
  endpoint. Its long-running opt-in is
  `GENERATIVEQC_OPEN_SHELL_AUXILIARY_G_CBS_CUDA_TEST=1`.
  Atomic translations alone are not a nonzero molecular response qualification.
- Real-device qualification must run in a finite Slurm allocation. Independent
  PySCF is validation-only, never a production dependency or fallback.

The executable gates require raw tensors/derivatives within `2e-9` absolute
and `2e-11` relative error; molecular component energies within `1e-8 Eh` and
directional forces within `3e-7 Eh/Bohr` at steps `2e-4` and `6e-5 Bohr`.
Physical atomic component/composite energies use `3e-8 Eh`, forces use
`2e-8 Eh/Bohr`, and directional differences use `5e-7 Eh/Bohr` at steps
`1e-3` and `3e-4 Bohr`. Component backend/convergence assertions and energy-only
replay are separate from these numerical gates.

## Execution evidence and limits

The source-matched CUDA library was built with CUDA 12.9.1 for SM120 using
verified sccache 0.16.0, Release plus the project's fast-iteration CUDA option.
This is correctness qualification, not release performance/resource evidence.
The build uses base `17babea1beeced50a11254684b879cbf26157d72` plus this change;
the library SHA-256 is
`26f8a0c1b0768e627518b77718553ed1c4dce92327b6d97f59d0bb53b29c07d4`.
The identical library was copied to n1 with its hash verified before execution;
the local node3 Slurm GPU was maintenance-drained and was not used.

On n1's RTX 5090 under finite `srun --partition=main --gres=gpu:5090:1`
allocations, all four molecular CBS cases, eight raw-source/export cases, and
both physical Na DZ / B TZ UHF/UMP2 cases passed against PySCF 2.14.0/libcint.
The complete physical B DZ/TZ CBS energy/force gate also passed, including all
component energies and both directional-difference steps for the components
and the assembled force.
The native DF suite also passed its 48 weighted finite-difference gates and
RI-J/K/planning checks; native work-policy/status contracts passed. The local
host CBS algebra, composition and policy suite passed 240 tests. The standalone
`validate_df_source.py --derivatives` tier passed all 16 configurations across
spherical and mixed Cartesian/spherical g fixtures, four source mappings and
two tile sizes, including single/batch values and coordinate derivatives.

The complete Na physical CCSD(T)/MP2 CBS force qualification exceeded its
20-minute allocation, and a separate Na DZ CCSD(T) energy-only attempt exceeded
five minutes. Neither published a qualified complete result. These limits are
not auxiliary-g errors, numerical acceptance failures, or evidence of complete
Na CBS force qualification. The current-master Na gate is running in an explicit
four-hour allocation without weakening its work/memory/convergence policies or
dropping its oracle. Complete Na CBS qualification remains pending.

## Revisit when

Specialized auxiliary-g math or auxiliary-h correlation response receives its
own independent numerical and complete-endpoint qualification, or the bounded
coordinate exporter is replaced by a fully qualified weighted response owner.
