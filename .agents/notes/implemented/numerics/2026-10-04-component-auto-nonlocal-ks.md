# Decision: component-wise AUTO for nonlocal KS

Status: implemented
Date: 2026-10-04
Agent: dot

## Decision

Resolve CUDA KS precision by scientific component through the shared
`ExecutionPrecisionSchedule`. A complete nonlocal composition can request AUTO
without lowering every operator: only the independently qualified Direct J
recurrence uses FP32 computation with FP64 accumulation. Density contraction,
exact exchange (including SR/LR K), tau, XC point algebra, VV10 and final audit
remain FP64. The stationary derivative owner consumes the strictly refined
final SCF state and keeps its existing FP64 arithmetic.

An explicit nonlocal-correlation census record makes that strict work visible.
Only the ordinary device-fused, device-resident nonlocal owner can publish a
complete census; host-unfused work and device-chunk execution retain incomplete
inventory status. Precision policy does not choose a backend vendor or replace
provider compatibility checks.

## Invariants and alternatives

Keep the public default FP64. AUTO must complete strict refinement and final
state validation before publishing energy or supplying stationary forces.
Existing functional coefficients, spin, backend, ECP, density-fitting, basis
and resource admission remain authoritative. In particular, fitted AUTO stays
rejected, and generic RSH still requires its compatible prepared Direct
provider. The independent removals of the whole-graph nonlocal AUTO veto and
the WB97M-V-only RSH veto must both survive integration.

A whole-graph precision veto unnecessarily excludes a qualified Direct J
component. Lowering the full graph would instead exceed existing qualification.
Keep the component contract separate from provider selection and future
execution filters applied to the shared schedule.

## Evidence and limits

Host regressions cover component admission and census, strict refinement/final
audit identity publication, precision decoding, replay exclusion and the
RSH/nonlocal submission join with failure propagation. Opt-in CUDA tests cover
RKS/UKS AUTO versus FP64 energy and forces through cold, warm and moved geometry.
Those new AUTO GPU cases were not run during this host-only review/integration.
Historical independent WB97M-V force qualification supports the unchanged
strict owner and is not evidence of AUTO execution on this PR head.

No endpoint speedup or default-policy promotion is claimed. A future default
change requires matched real-device numerical and complete-endpoint evidence.

## References

- Issue #1854; PRs #1867 and #1863
- `src/dft/cuda_ks_precision.hpp`
- `tests/python/test_ks_component_precision_census.py`
- `tests/python/test_wb97mv_complete_cuda.py`
- `benchmarks/results/wb97mv-cuda-20260926/README.md`
- `benchmarks/results/omol25-wb97mv-20261001/README.md`
