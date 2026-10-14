# Candidate: typed DF polynomials without replacing the force schedule

Status: proposed native integration; optional compiler lowering implemented
Date: 2026-10-14

The efficient shell/Rys lowering and private native transactional consumer are
now recorded in `../implemented/numerics/2026-10-14-df-shell-typed-runtime.md`.
That continuation does not promote the raw generic metric candidate or complete
HF-DF endpoint; the independent negative evidence below remains retained.

## Boundary

The HF-DF profiles in `2026-10-14-hf-budget-gradient.md` identify derivatives,
not XC, as the dominant small-system residual. Replacing efficient shell/Rys
owners with generic evaluators risks repeating the rejected direct-HF schedule
experiment. Type the existing mathematics, then preserve the incumbent schedule.

`emit_df_derivatives_cuda` and its host adapter now accept `scalar_type="float"`.
The scalar DAG emits float leaves, literals and temporaries. Axis polynomials,
moment products and local derivative coefficients/sums use float. Gaussian
geometry and Boys evaluation remain double until the explicit polynomial
boundary; returned channels widen to double. Metric, three-center and
auxiliary-g derivatives share the existing equations. Neither the metric inverse
nor SCF narrows. The inventory distinguishes this mixed arithmetic from FP64.

Default FP64 emission remains byte-identical to master `c2c670e2f`. A one-off
comparison records normal/auxiliary-g hashes
`71430c3d8929868c10156d36578f719bbd56d595b35c4ae56131930395f4ce04`
and `f30ef5a213c29bd7a87e01ffc8767dcbd0a3c0a4b815d5d65d1e518196ccb2c6`.
These are migration receipts, not permanent source-fingerprint assertions.

## Independent evidence and retained rejection

Compiled-host checks use independent libcint normalized contracted blocks:
signed contractions, all s/p/d/f classes, selected auxiliary-g blocks,
Cartesian/spherical outputs and coincident/asymmetric centers. Compiled type
assertions and results differing from rounded FP64 exclude storage-only float.

The initial run passed 11 tests and rejected a diffuse d/f metric block.
At exponent scale `1e-4` and separation scale `100`, raw float fails the unchanged
`atol=1e-5, rtol=3e-6` block gate; one violating entry differs by about `4.071e-5`.
Strict passes the independent check. Do not loosen the gate or hide this case.
A negative test retains the rejection; positive scaled fixtures use `1e-2`
and `1e3`. The combined focused suite passed 102 tests before adding the
arithmetic-inventory check. The original failed log remains under
`.artifacts/readme-budget-fast-20261013/df-typed-polynomial-host.log`.

Slurm 2967 on a PRO6000 subsequently passed 14 real-CUDA tests in 19.68 seconds:
independently contracted metric and three-center blocks, through f and selected
auxiliary-g cases, both geometries and representations. NVCC compiled through
verified ccache. Slurm 2966 failed the cache prerequisite because its hiredis
library directory was missing from the job environment; that failure remains
separate and no uncached build was substituted. The host combined suite then
passed 493 checks with 14 scheduled-only skips. These CUDA checks do not admit
the diffuse rejection, and they are not timings of an application endpoint.

This is not a full-gradient error bound or complete-endpoint qualification.
Native integration still needs conditioning admission, pre-scatter finite
recovery, actual work/retry counters, and complete independent cold/frozen/moved
HF-DF checks with the current generic/shell/Rys schedules. A scalar generator
does not choose an application's error allowance. No public or native force
path selects this candidate yet; HF-DF remains unfinished.

Do not turn the sampled exponent scales into an arbitrary method whitelist.
Conditioning admission belongs with the response weights and the application
allowance; otherwise individually small raw errors can still exceed the
complete contracted force budget.

The complete native HF-DF route starts in `src/scf/rhf.cpp`, calls
`execute_cuda_density_fitting_generated_force_response` in `df_force_response.cpp`,
then `execute_cuda_df_hf_gradient` in `df_gradient_bridge.cu`. Direct HF's
`cuda_rhf.cpp` mixed-force flag is gated on quartet-direct work and cannot enable
DF by itself. Preserve the bridge's generic/packet/Rys choice; its metric
derivative tiles and three-center shell consumers are distinct arithmetic owners.

## References

- `python/generativeqc_compiler/integral/df_derivatives_cuda.py`
- `tests/python/test_df_derivatives_fp32.py`
- `2026-10-14-hf-budget-gradient.md`
- `.artifacts/readme-budget-fast-20261013/df-typed-polynomial-host-final.log`
