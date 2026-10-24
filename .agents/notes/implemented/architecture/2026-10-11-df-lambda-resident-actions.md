# Decision: expose the existing DF Lambda transpose as a borrowed device action

Status: implemented (default-off binding, not promoted)
Date: 2026-10-11

## Problem

Conventional CUDA Lambda could bind the shared resident controller, but DF still
staged dense seeds/results for every physical action. Copying the DF dispatcher
into another backend would duplicate expanded, staged, matrix-tail and audit
schedule selection and risk omitting auxiliary derivatives.

## Decision

Factor the existing physical dispatcher into `DFLambdaActions::transpose_device`.
The host-vector adapter stages its seeds, invokes that dispatcher, copies its
borrowed outputs and audits the sticky error. The same resident Lambda backend
instead unpacks weighted pair coordinates into those device seeds and consumes
the borrowed outputs on the DF owner's stream. Both retain the original
diagonal policy, terminal replay and independent expanded physical residual.
No scientific equation, reduction order, second solver or derivative IR is added.

The view uses an opaque stream pointer so existing source-only host probes do
not require CUDA headers. It is a per-call borrowed action view, not a prepared
generation or revocable scientific lease. The DF owner must outlive the backend
and every enqueued consumer; failure unwinding drains before destroying storage.

The vector provider remains a separate shared tensor context. DF admission must
charge its full allowance beside the action provider, apply the original device
ceiling, and add resident admission to the complete host-plus-device numeric
bound. Keep the conservative host-vector allowance even when a candidate is
selected so refusal retains the exact incumbent schedule and diagonal policy.

The native complete DF force owner exposes a trailing default-off selector and
propagates it through cold retries. Installed public defaults stay unchanged.

## Rejected alternatives

- A second DF dispatcher in the vector backend duplicates method ownership.
- Reusing host-vector transpose through device downloads is not residency.
- Sharing only the action provider allowance undercounts two live contexts.
- Treating fresh logical states in a warm process as prepared reuse invents a
  lifetime proof that this change does not provide.

## Validation and remaining gates

Native noninteracting references cover supplied/canonical denominators,
unsafe/pair-inconsistent fallback and two valid budgets for both bindings.
Real-device fixtures cover expanded/scalar/matrix-tail schedules, full parameter
publication, determinant multi-step factor directions, complete molecular E+F,
optional refusal and unpublished failure followed by successful recovery.
Source-matched measured receipts belong under `benchmarks/results/`; neither
fixture existence nor source generation is device-execution evidence.

Broad complete-endpoint performance/resource census, #933 prepared lifetime,
HF/CPKS/MP2 residency and legacy retirement remain independent #932 gates.

## References

- #932 and follow-up PR #2291.
- [Conventional binding rationale](2026-10-11-native-lambda-resident-binding.md).
- `docs/developer/response.md`.
