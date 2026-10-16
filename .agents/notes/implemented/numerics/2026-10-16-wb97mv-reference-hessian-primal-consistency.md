# Decision: preserve the comparison gate after a reference primal self-check

Status: implemented; diagnostic finding, not a reference-kernel fix
Date: 2026-10-16

## Problem and decision

The independently energy-qualified native Hessian failed the unchanged 3e-6
Eh/Bohr² GPU4PySCF comparison. Matching base energies did not explain the 9.33e-5
H2 matrix difference. The installed reference source includes NLC response;
assuming an absent VV10 kernel is not a supported diagnosis.

Preserve the comparator, scientific kernels and tolerance. Independently check
GPU4PySCF's analytic Hessian against its **own** reconverged primal. No qualified
speed ratio follows from this diagnostic, and finite differences never enter
the production Hessian path.

## Evidence

Slurm job 3222 used node2/pro6000 and the maintained comparator's H2/STO-3G
spherical, 8/4/8 moving-grid/VV10 contract and tight SCF settings. Every displaced
point reconverged from a fresh reference engine. The coordinate is the last
atom's z in Bohr; E(0) is `-1.1447427860966495` Eh. Full grid response remained
enabled. Native children had Python/compiler discovery unavailable, with PTX
JIT disabled.

| Quantity | Eh/Bohr² |
| --- | ---: |
| GPU4PySCF analytic Hzz | 0.24709299711411498 |
| Native compiler-AD Hzz | 0.24699973004378661 |
| Reference primal Richardson, h=.02/.01 | 0.24699972475487422 |
| Reference primal Richardson, h=.004/.002 | 0.24699972778208235 |

The extrapolants differ by 3.03e-9. Native agrees within 5.3e-9; this installed
reference analytic entry differs from its own primal by about 9.33e-5. This
local coarse-grid result neither identifies the responsible source term nor
proves all reference Hessians wrong or qualifies water.

| h (Bohr) | E(-h) (Eh) | E(+h) (Eh) |
| --- | ---: | ---: |
| .02 | -1.1460420977251662 | -1.1433445784822662 |
| .01 | -1.1454055137884658 | -1.14405535242636 |
| .004 | -1.1450109207717631 | -1.1444706992721438 |
| .002 | -1.144877353072463 | -1.1446072311123154 |

Reproduce using `benchmarks.readme_wb97mv.reference_engine`,
`benchmarks._gpu4pyscf_grid.preserve_reference_grid_order` and the exact settings
in `benchmarks.compare_wb97mv_hessian_gpu4pyscf._reference_worker`.
Compute `(E(-h)-2*E(0)+E(+h))/h²`, then `(4*fine-coarse)/3`. Do not seed from a
native result or patch a reference response kernel to obtain agreement.

The dirty development deployment used CLI SHA256
`1f2a3b1275549a0f37215afc0d5ecd0723be5146b604204b48e75110b8e5468f`
and library SHA256
`659f52d9aeffb85b53144e6e6f3fdf048a3887115de97f4a117c638cc7bcce74`.
It is not a clean-source timing receipt. Raw inputs/results are retained in
`.artifacts/wb97mv-hessian-aot-20261015/gpu4-curvature.log` and the immutable
remote `native-api-20261016/.artifacts/slurm-3222/gpu4-curvature.json` deployment.

## Clean-source repetition and NLC-disabled control

Slurm 3227 repeated the diagnostic using clean native source
`84abba9dc59359767c06ad188cbff815b0ac7268` and the unchanged installed reference.
Every displaced reference calculation again used a fresh engine. A separate
control sets `engine.nlc=0`; it is explicitly **not canonical WB97M-V**, never
changes the official comparator and is not a timing/acceptance substitution.

| Case / reference NLC | Analytic Hzz | Own-primal .02/.01 | Own-primal .004/.002 |
| --- | ---: | ---: | ---: |
| H2 / enabled | 0.24709299711411498 | 0.24699972475487422 | 0.24699972778208235 |
| H2 / disabled control | 0.24620471280899386 | 0.24611144066377655 | 0.24611144395595788 |
| Water / enabled | 0.11694845124148423 | 0.11713954684466898 | 0.11713853886874404 |
| Water / disabled control | 0.11560702086223909 | 0.11579799950330501 | 0.11579718162124664 |

All entries use Eh/Bohr². The H2 discrepancy persists at about 9.33e-5 when
reference NLC is disabled; this excludes an absent VV10 contribution as its
dominant explanation for that coordinate, without identifying a faulty source
term. Water extrapolants differ by about 1e-6: this diagnostic is less precise
there, but still far from its approximately 1.9e-4 analytic/primal discrepancy.
It does not qualify every water matrix entry. No vendor kernel was patched.

The complete reference inputs, energies, source/script hashes and exact runnable
diagnostic are retained in
`benchmarks/results/wb97mv-hessian-native-full-20261016/`. The official matched
matrix gate still fails and all qualified performance ratios remain null.

## Additional isolation controls, not a replacement comparator

The unchanged installed reference was also tested on H2 with NLC explicitly
disabled. Slurm 3230 found that the own-primal full-Hessian discrepancy is not
exclusive to MGGA, range separation or VV10: the fine-stencil analytic-minus-
Richardson errors include approximately `2.7333e-5` for LDA/VWN,
`-3.9486e-6` for PBE, and `9.3269e-5` for WB97M-V without NLC. HF agreed to
approximately `6e-11`. These controls do not change the canonical comparison.

Slurm 3234 separately fixed the AO density matrix and displaced the last H z
coordinate. The reference `_get_exc_deriv2_grid_response` agreed with its own
fixed-density XC-energy curvature to at most `1.04e-10` for LDA/VWN, PBE and
NLC-disabled WB97M-V using the same `.004/.002` Richardson stencil.
`_get_vxc_deriv1_grid_response`, transformed into the fixed origin MO/occupied
frame, agreed with displaced numerical-integration potentials to at most
`5.69e-12`. Earlier long-double Becke-weight controls likewise did not expose
a weight-jet explanation. These are directional controls, not a proof that every
frozen term or geometry is correct.

Slurm 3238 scaled the positive density matrix by `1±h` rather than moving nuclei.
For LDA/VWN, PBE and NLC-disabled WB97M-V, the complete Fock contraction agrees
with the first electronic-energy derivative within `4.32e-13`, and the
`gen_response(hermi=1)` contraction agrees with the second derivative within
`1.07e-10`. HF's first-energy derivative agrees within `6.67e-15`; the generic
RKS/HF response interface raises an AO-shape `ValueError`, so its second-response
comparison is explicitly unavailable. The first Slurm 3236 diagnostic stopped
on that exception; the revised driver retains the unavailable HF result and
continues the other planned controls rather than silently declaring it passing.
This does not describe a failure of the separately tested HF Hessian endpoint.

Raw scripts/results remain in local
`.artifacts/wb97mv-hessian-aot-20261015/reference-{model-controls,weight-control,frozen-control,density-control}*`.
The fixed-density and density-scaled controls narrow future investigation but do
not identify a faulty full-Hessian assembly term or resolve the official matrix
gate. Do not introduce compensating native terms, patch vendor kernels, or claim
an accepted speed ratio from these diagnostics.

## Revisit when

Repeat the analytic/primal control with pinned upstream versions or an
independently verified upstream fix. Keep the official matrix gate and raw
complete-endpoint timings separate from any diagnosis.
