# Decision: qualify LDA/PBE pruning defaults on the 96-atom endpoint

Status: implemented
Date: 2026-10-19

Supersedes the default-selection boundary in
[the initial pruning implementation](2026-10-19-radial-angular-pruning.md).
The integer-rank pruning algorithm itself is unchanged.

## Problem

The user requested default enablement after a demonstrated 96-atom GPU benefit,
with pruning also enabled in GPU4PySCF benchmarks. The initial small-water
qualification cannot certify a larger cluster, nor can PBE results qualify VV10.
The original 54/16/32 radial/angular maxima need more careful qualification:

- 96-atom LDA's old unpruned grid misses the retained per-water energy budget.
- Simply pruning 64/16/32 still misses that LDA energy budget (about 3.9e-5 Eh
  against 3.2e-5 Eh); fewer points is not sufficient evidence of a usable default.
- PBE pruning at 54/16/32 is faster and passes the raw error gate, but its
  maximum-force margin is smaller than the observed dense-reference variation.

Do not relax accuracy gates, discard failed candidates, or describe the fastest
insufficiently qualified quadrature as the best production configuration.

## Decision

`GridPolicy` defaults to version-3 `radial-v1` for its qualified LDA/PBE families.
The standard maximum orders are 64/20/40 for LDA and 64/16/32 for PBE. LDA's
standard and tight policies currently share 64/20/40; PBE tight remains
72/24/48. Raising the radial/angular maxima before pruning gives substantially
more force margin without losing the complete-endpoint performance benefit.

`GridPolicy(pruning="none")` retains the previous version-2 profiles, scientific
provenance and explicit unpruned reproducibility route. Explicit `GridSpec`
v1/v2 contracts and the native ABI defaults are unchanged. Canonical v3 grids
receive canonical policy provenance; custom v3 grids remain explicitly identified.
Named LDA/PBE KS defaults use the new policy, including the existing PBE
electronic policy consumer. Automatic libxc aliases keep their previous grids;
other methods do not inherit a pruning qualification from these two functionals.

Matched-grid benchmark `auto` selection uses the qualified PBE policy for both
engines and retains unpruned defaults for unqualified functionals. Explicit
counts retain the historical one-Bohr diagnostic radii. The separate
`benchmarks.grid_pruning_gpu` production-grid qualification driver instead uses
GPU4PySCF's own radial/Lebedev grids and real NWChem pruning. It independently
qualifies both engines against dense references rather than transplanting the
native angular rule and calling it GPU4PySCF's normal production configuration.

Both native and reference production candidates default to pruning in that
driver; unpruned candidates remain measured for comparison. Only candidates
passing the shared scientific gates can be selected. Version 2 of its report
also reserves margin for observed dense-reference variation, applied equally to
all candidates. This is a conservative acceptance rule, not a rigorous
discretization-error bound. The energy budget is extensive per water unit;
the maximum force budget is unchanged with atom count.

## Evidence

Slurm jobs 10323 (LDA) and 10324 (PBE), node1/main, each used one RTX 5090 and a
finite 20-minute allocation, preserving Slurm's device visibility. Six samples
per candidate alternate order after one complete untimed warmup. Every sample
constructs a fresh owner and includes preparation, converged SCF, analytic
forces, cleanup/garbage collection and device synchronization. No converged
density is passed between samples. This is process/artifact-cache warm, not a
process-cold or isolated XC/VV10-kernel measurement. Native and BLAS threads were
one; both engines used the same water-96 geometry and STO-3G AO space.

| Engine / method | Old/unpruned median (s) | Pruned median (s) | Ratio |
| --- | ---: | ---: | ---: |
| GenerativeQC LDA | 23.158432 | 21.044082 | 1.1005x |
| GenerativeQC PBE | 24.355397 | 14.995148 | 1.6242x |
| GPU4PySCF LDA | 8.611896 | 6.591725 | 1.3065x |
| GPU4PySCF PBE | 10.201068 | 7.813512 | 1.3056x |

Native point counts change from 2,654,208 to 2,467,200 (LDA) or 1,579,008
(PBE). GPU4PySCF level-3 grid lengths change from 1,691,392 to 1,078,528, with
its own independent construction and padding. Native Fock/iteration diagnostics
and independent get_veff/cycle counts are retained per sample. The old native
baseline is not accuracy-qualified under the conservative margin rule, so its
ratio is a before/after-default timing, **not** a comparison of two equally
qualified production grids. Both GPU4PySCF candidates qualify. GPU4PySCF is
faster than native on these measured complete endpoints; the pruning benefit
does not establish a native-over-reference speedup.

Independent GPU4PySCF full-density-rebuild SCF and grid-responsive gradients at
levels 8/9 passed reference stability. The unchanged 96-atom gates are 3.2e-5 Eh
and 5e-5 Eh/Bohr for LDA, 4.8e-5 Eh and 3e-5 Eh/Bohr for PBE. Maximum observed
pruned-native errors, including the observed-reference margin, were
2.1272e-5/1.4290e-5 (LDA) and 4.3117e-5/1.2172e-5 (PBE), respectively.
GPU4PySCF's pruned production candidates also pass the same gates. The initial
version-1 raw reports used raw errors for classification; their unchanged bytes
are retained, with the stronger version-2 classification recorded separately.

An earlier level-6/7 reference did not meet stability; level-7/8 did, and 8/9
provided additional qualification. Incremental GPU4PySCF SCF accumulated an
energy drift and failed the strict convergence gate; the existing independent
full-density-rebuild wrapper resolved that issue without changing tolerances.
These are separate from the earlier custom-dense-grid reference failure, whose
root cause remains unestablished. No failed reference replaces a qualified one.

The native Release binary is the source-matched CUDA 12.9.1 / SM120 build
documented in the initial note, with native LDA/PBE stationary CUDA enabled.
The default promotion changes Python policy/selection, not compiled quadrature
math. Input overlays, the original dirty source capsule and per-sample summaries
are retained locally in `.artifacts/pruning-gpu-20261019/compact-summary.json`.
Raw reports, scripts and a verified 357,965-byte archive remain alongside it.
The archive SHA256 is
`87b4cc59b062a2d02f28dd39dcaa24c7e82514c5a067e2136150509c4eab338e`;
the unchanged compact-summary SHA256 is
`e9e6b0e060fdaf96b82c4effba2f85a0e2a6f31aa8ef93485364300cdd7790c9`.
The previously published compact [summary](https://github.com/jinzhezenggroup/generativeqc/blob/ebf0aca81ffe0b6788469e780b7626ccad2f6813/benchmarks/results/radial-pruning-20261019/summary.json)
and [scope/reproduction note](https://github.com/jinzhezenggroup/generativeqc/blob/ebf0aca81ffe0b6788469e780b7626ccad2f6813/benchmarks/results/radial-pruning-20261019/README.md)
remain recoverable at that immutable commit. The summary is the unchanged
11,722-byte file with the SHA256 above, including all 48 timing samples and
the rejected old-native classifications. These historical records retain
their original source-capsule scope; they are not measurements of later
integration or runner repairs. The larger raw archive is still local-only.

The native library SHA256 is
`92e4508516078043a72030fcd8dbf8ed6bbe556b8eece29a0e1b3511c4fac447`;
the original source-capsule SHA256 is
`a99c821eeca51c0e6acb61a71630f5996694b7182c9bdf590640f2f6c97192bb`.

The repository's 64 MiB aggregate evidence budget was already nearly full:
even the new 14,764-byte compact report exceeded its remaining capacity.
Keep these new data in the verified ignored local store, not a relocated JSON
dump elsewhere in the source tree. This decision note retains only the scope,
acceptance rationale and measured summary. Do not enlarge the evidence budget,
delete unrelated historical records or publish a Release/archive implicitly.

## Resource qualification and limitations

The earlier eight GPU ledger assertions compared live preparation allocations
to a plan that also reserves a future 512 MiB generated-force staging arena.
A real-device probe confirmed the admitted ledger limit exactly matches the
planned peak, with live/peak usage below that limit and zero rejected
allocations. Energy-only preparation must not allocate an unused force arena
just to make a test equate reservation and use. The test now checks the native
admission receipt, planned limit, live/peak bounds and absence of rejections.
No production allocator or budget was weakened. All eight allocated RKS/UKS,
LDA/PBE, pruned/unpruned resource/reference cases passed in job 10325.

Focused CPU/compiler tests and independent RKS/UKS/Fe-grid tests qualify the
new policy. A broader CPU force-suite attempt had nine native-AOT capability
assertions fail because the qualification library was configured with CPU-force
AOT disabled; those build-dependent paths are not claimed qualified by that
run. Existing numerical and AOT capability tests remain intact.

No 96-atom VV10, hybrid/meta-GGA, larger-basis performance or whole-README
production ranking is inferred from this water/STO-3G campaign. Such method
defaults remain unpromoted and the broad homepage ranking remains pending.

## Integration compatibility checks

The frozen DFT-MP-v1 campaign explicitly selects `GridPolicy("tight",
pruning="none")`. Its original v2 manifest, grid identities, input hashes and
acceptance gates remain unchanged when public defaults move to v3. Heavy-element
translation checks now execute both legacy and pruned grids. Host setup bounds
include the added shell/angular storage; the independent LP64 inventory tests
retain exact byte expectations rather than confusing setup and retained bytes.

Decoder, Python-grid, emitted-layout and CUDA-grid body hashes are recorded as
current source provenance, not refreshed historical implementation acceptance
pins. Executable legacy/pruned point/weight/owner comparisons, emitted-layout
buffer/point counts, allocated CUDA tests and snapshot mismatch/provider checks
cover the behavior instead. New host wire cases explicitly exercise v1/v2/v3
prescriptions for both spins. ABI, immutable-input, native-provider and cache
integrity controls remain in force; a source-only capacity report is not a
substitute for numerical or real-device qualification.

The no-toolchain H2/PBE sentinel retains its 2e-9 Eh tolerance and also exercises
the explicit legacy grid. Its pruned-default oracle energy is
-1.1520643723572155 Eh from independent PySCF SCF on the identical prescribed
points and weights (zero density cutoff), rather than a native result copied
into the expected value. The native value differs by 2.84e-10 Eh.

## Revisit when

Additional methods/bases/geometries establish their own independent accuracy,
reference-stability and complete-endpoint performance gates. In particular,
qualify VV10's actual density-screened pair work and analytic response before
changing its defaults; point-count reduction alone is not sufficient.
