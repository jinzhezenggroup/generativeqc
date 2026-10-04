# DF-CCSD(T) analytic-gradient composition

The native internal components target the correlation-only DF Hamiltonian:
conventional all-electron RHF supplies the reference Fock/orbitals, while the
correlation two-electron interaction is density fitted. The complete native
DF-CCSD(T) force endpoint is not yet registered or qualified.

## Correlation response

`solve_lambda_parameter_response_cuda` differentiates the retained RCCSD core
and every auxiliary slice of the virtual residual. It returns retained
Fock/integral block cotangents and Q-major `df_bov/df_bvv` cotangents. The latter
cover the virtual residual only. Fresh primal replay and an independently
expanded Lambda action retain the ordinary CC residual acceptance gates.

`cc::triples::pullback_df_cuda` supplies all fixed-canonical-input (T)
cotangents. Its T1/T2 sources drive the corrected-Lambda solve. The full
`fock_response_df_cuda` supplies same-space Fock matrices, including internal
occupied/virtual degeneracies. These matrices **replace** the epsilon-diagonal
sources; adding both would count denominator response twice.

`pullback_df_factors_cuda` combines retained Gram-block and virtual-factor
cotangents. Compressed Bov includes both ov/vo sectors, so full symmetric
embedding assigns half to each. `pullback_df_source_cuda` then reverses the
original retained molecular source:

```text
B[p,q,Q] = sum_P A_MO[p,q,P] W[P,Q]
A_MO[p,q,P] = sum_mn C[m,p] C[n,q] A[m,n,P]
W = M^(-1/2)
```

The original metric/source owner and immutable frame must remain alive and the
nonzero source identity must match. The common fixed-rank spectral VJP retains
retained/discarded subspace motion and rejects unresolved cutoff crossings.
Raw A cotangents address full, unit-weight `[mu,nu,P]` entries, with no triangular
doubling. All callbacks are provisional until the entire reverse call succeeds.

`scf::CudaDfNuclearSink` consumes raw A rows and the metric cotangent directly on
the producer stream. It visits all nuclear centers without materializing a
coordinate-indexed derivative tensor. Setup is persistent; consume performs no
allocation, copy or synchronization. The producer owns its stream and must
outlive the sink. Call `finish()` only after successful producer completion.
The caller must bind the exact original orbital/auxiliary geometry and basis.

## Conventional-reference response

`hf::rhf_frame_response_cuda` accepts the complete full-MO Fock cotangent and
AO frame cotangent. Compiler-owned matrix maps derive the closed-shell density,
Fock projection, unrestricted frame reverse map, symmetric metric/Pulay
transport and orbital tangent through the shared TensorIR AD machinery.

The physical action remains `G(D)=J(D)-K(D)/2` from the unscreened exact CUDA
provider. It is used for both forward tangent and reverse density response;
substituting DF J/K here changes the method. A single provider stream owns the
matrix maps and resident signed density actions. Optional BLAS executes packed
matrix products with sticky finite audits; the same-arena scalar CUDA schedule
remains available.

GMRES applies the orbital operator on demand. Neither a full MO ERI nor a dense
`(occupied*virtual)^2` Hessian is constructed. The final Z residual is recomputed
with the scalar CUDA lowering, followed by full frame stationarity after the
Z seed is subtracted. Same-space stationarity never divides same-space gaps.
An occupied/virtual gap and residual checks qualify the local solve; they do
**not** certify global RHF stability or the minimum Hessian eigenvalue.

The reference nuclear branch contracts AO hcore and Pulay weights with existing
CUDA derivative providers. Its two-electron source `P:G'(D)` uses the bounded
polarization identity `E2'(D+P)-E2'(D)-E2'(P)`, with `E2(D)=D:G(D)/2`. Three
passes are independent of the orbital-response dimension. The result is an
**electronic gradient**: the final method must add the correlation-source and
nuclear-repulsion gradients, then negate once to publish forces.

## Qualification and remaining integration

The internal interfaces admit complete numeric payloads before execution;
outer callers must charge all other live owners. Matrix response reports zero
explicit Hessian elements, J/K actions, derivative passes, generated contraction
summands, BLAS calls and matrix-owner transfers. Integral-provider setup and
nuclear-consumer transfers are separate; these counters are not a complete
endpoint traffic ledger.

Relevant validation modules include `test_df_cc_lambda.py`,
`test_df_source_metric_response.py`, `test_df_nuclear_sink.py`,
`test_rhf_frame_response.py`, `test_rhf_frame_response_codegen.py` and
`test_rhf_frame_response_cuda.py`. Real-GPU tests require finite Slurm allocation.
The RHF module checks independent forces and nonzero-Z molecular energy finite
differences with both scalar and BLAS schedules.

The remaining method owner must compose all these branches on the identical
converged CC/reference/source state, retain complete phase admission, validate
independent complete DF-CCSD(T) forces, and measure cold hundreds-AO endpoints.
Passing any individual component does not qualify the composed force endpoint.
