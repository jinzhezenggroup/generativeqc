# Decision: share the complete contraction-region lifetime

Status: implemented
Date: 2026-10-05

The first prepared source-response implementation moved vendor calls behind the
tensor table, but its method emitter still generated provider-specific context
setup and rejection handling. Keeping that pattern would duplicate lifecycle
policy in each new consumer and leak provider choice back into method codegen.

Move that owner into `tensor::PreparedContractionRegion`. A method emitter now
supplies only the emitted semantic portfolio, descriptor factory, shape key and
borrowed stream/work counters. The tensor layer prepares the context, validates
descriptor count, handles optional rejection, reports resources and replays the
already-bound table. Factories are invoked during construction only and are not
retained for replay. Re-selection uses the same portfolio and precision gates.

The shared owner checks overflow-safe host storage and the minimum simultaneous
provider reservation before preparing a context. Prepared shape/lifetime checks,
sticky finite errors and synchronous resource cleanup remain in the underlying
table. Source-response algebra, traversal and public method API are unchanged.

The real source-response tests cover all three providers, a third-plan rejection,
exact minimum budgets, unit/nonunit and nonsymmetric inputs, callback failures,
sticky nonfinite results and the independent complete derivative expression.
This ownership cleanup does not qualify cuTENSOR resource profiles or promote
a provider. It extends the source-response and explicit-plan-provenance decisions.

References: #1886, #1887, #1890; `test_df_cc_source_program.py`;
`src/tensor/cuda_contraction_selection.cuh`.
