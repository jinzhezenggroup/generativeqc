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

## Revisit when

Repeat the analytic/primal control with pinned upstream versions or an
independently verified upstream fix. Keep the official matrix gate and raw
complete-endpoint timings separate from any diagnosis.
