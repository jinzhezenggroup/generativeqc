# Decision: match PBE0 final-state admission to the prepared DF provider

Status: implemented
Date: 2026-10-01

The prepared CUDA PBE0 DF owner identifies matching DF-J/DF-K, but the final-state
model predicate admitted only exact-J/exact-K at PBE0 scales. This rejected the
otherwise supported force consumer before any independent force comparison.

Admit either matching exact/exact or DF/DF PBE0 pairs. Keep the spin-dependent
exchange fraction, semilocal 0.75/1.0 scales, full-range operator, no range/nonlocal
decoration, determinant identity and physical-state proof unchanged. Mixed
ownership and wrong fractions remain rejected. B3LYP DF admission is unchanged.

Native analytic fixtures cover RKS/UKS matching and mixed provider pairs and wrong
fractions. Source-executing host tests additionally reject wrong families, scales
and decorations. These are admission checks, not device numerical qualification.

The external RTX5090 receipt on PR #1654 reports the original failure and a
locally patched 82-pass/1-skip matrix, including PBE0 RKS independent energy and
forces. Its patch bytes are unavailable in this checkout: this independently
implemented repair is not relabeled as that exact tested patch or a new-head GPU
pass. UKS PBE0 coverage here is identity/rejection proof only. Preserve a real
device rerun requirement and unchanged scientific tolerances.

Reference: https://github.com/jinzhezenggroup/generativeqc/pull/1654#issuecomment-5942165254

Agent: dot
